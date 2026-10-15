"""Would another RIME scheme serve this user better? Replay the typing log through each.

For every intent (intents.py), in time order: query the pinyin the user actually typed,
check whether the first candidate is what they wanted, then commit that word so the
engine's user dictionary learns exactly as it would in daily use. Every engine starts
from a fresh copy of its user directory, so they are compared under the same history.

Engines:
  frost_prelog : the installed rime_frost (白霜) with the user dictionary from before the log
  frost_fresh  : rime_frost with an empty user dictionary
  wanxiang     : 万象拼音 base + wanxiang-lts-zh-hans.gram, empty user dictionary
`log` = what the user really saw (installed engine, live learning).
"""
import os, re, shutil, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from intents import DATA, HAN, load_intents

AIME_SHARED = "/Library/Input Methods/AIME.app/Contents/SharedSupport"
PRELOG = os.path.join(DATA, "userdb_export", "rime_prelog")
WX_SHARED = os.path.join(DATA, "schemes", "wanxiang_base")

def fresh_dir(name, src=None, drop_userdb=False):
    d = os.path.join(DATA, f"cmp_{name}")
    shutil.rmtree(d, ignore_errors=True)
    if src:
        shutil.copytree(src, d, symlinks=True)
        if drop_userdb:
            for f in os.listdir(d):
                if f.endswith(".userdb"): shutil.rmtree(os.path.join(d, f))
    else:
        os.makedirs(d)
    return d

ENGINES = {
    "frost_prelog": (AIME_SHARED, lambda: fresh_dir("frost_prelog", PRELOG), "rime_frost"),
    "frost_fresh": (AIME_SHARED, lambda: fresh_dir("frost_fresh", PRELOG, drop_userdb=True), "rime_frost"),
    "wanxiang": (WX_SHARED, lambda: fresh_dir("wanxiang"), "wanxiang"),
}

class Rime:
    def __init__(self, shared, user, schema):
        self.p = subprocess.Popen([os.path.join(HERE, "build", "rime_cands"), shared, user, schema],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def ask(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().rstrip("\n")
    def first(self, py):
        out = self.ask(py)
        return out.split("\t")[0] if out else ""
    def commit(self, py, text):
        return self.ask(f"!{py}\t{text}") == "ok"
    def close(self):
        self.p.stdin.close(); self.p.wait()

ABBR = lambda py: not re.search(r"[aeiouv]", py) or not re.search(r"([aeiouv]|ng|n|r)$", py)
intents = [it for it in load_intents() if HAN.match(it["gold"]) and it["pinyin"].isalpha()]
half = len(intents) // 2

results = {"log": [it["cands"][0] == it["gold"] for it in intents]}
for name, (shared, make_user, schema) in ENGINES.items():
    eng = Rime(shared, make_user(), schema)
    ok = []
    for it in intents:
        ok.append(eng.first(it["pinyin"]) == it["gold"])
        eng.commit(it["pinyin"], it["gold"])
    eng.close()
    results[name] = ok
    print(f"done {name}", file=sys.stderr)

groups = {
    "全部": range(len(intents)),
    "后半段(学习过一半日志后)": range(half, len(intents)),
    "单字": [i for i, it in enumerate(intents) if len(it["gold"]) == 1],
    "多字词": [i for i, it in enumerate(intents) if len(it["gold"]) > 1],
    "简拼/不完整拼音": [i for i, it in enumerate(intents) if ABBR(it["pinyin"])],
    "全拼": [i for i, it in enumerate(intents) if not ABBR(it["pinyin"])],
}
names = ["log", "frost_prelog", "frost_fresh", "wanxiang"]
label = {"log": "实际(日志)", "frost_prelog": "白霜+旧词库", "frost_fresh": "白霜(空词库)", "wanxiang": "万象(空词库)"}
print(f"{len(intents)} 个词，按时间回放，首选正确率（同时学习）\n")
print(f"{'':<24}" + "".join(f"{label[n]:>12}" for n in names))
for g, idx in groups.items():
    idx = list(idx)
    print(f"{g}({len(idx)})".ljust(24) + "".join(f"{sum(results[n][i] for i in idx)/len(idx):>14.1%}" for n in names))

# where the two schemes disagree (second half)
by = defaultdict(list)
for i in range(half, len(intents)):
    a, b = results["frost_prelog"][i], results["wanxiang"][i]
    if a != b: by["wanxiang_better" if b else "frost_better"].append(intents[i])
print(f"\n后半段：万象对、白霜错 {len(by['wanxiang_better'])}；白霜对、万象错 {len(by['frost_better'])}")
