"""What if the user typed several words at once? Offline test on the real typing log.

Consecutive intents that were adjacent in the text (same app, no punctuation commit in
between, < 15 s apart, and the app's text before the caret ends with the previous word)
are joined into chunks of N words. The chunk's pinyin is the concatenation of what the user
actually typed for each word (abbreviations included); gold = the words the user ended up with.

Same static RIME copy (user dictionary from before the log, no learning during the run) for:
  word-by-word : each word's pinyin queried alone (corrections = words whose first ≠ gold)
  chunk        : the joined pinyin queried once (corrections ≈ words of gold the first
                 candidate gets wrong)
Engines: `moqi` = the schema as installed (zh-moqi.gram, 7 MB); `wanxiang` = same schema
with wanxiang-lts-zh-hans.gram (400 MB) and Wanxiang's own grammar settings; `wx_scheme` =
the complete 万象 base scheme (its dictionary + grammar) with an empty user dictionary.
Optional `--lm`: rerank the chunk's full-span candidates with the LM (+ RIME rank).

Usage: sim_chunks.py [--sizes 2 3 4] [--lm]
"""
import argparse, json, math, os, shutil, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from intents import DATA, HAN, load_intents

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
ap.add_argument("--lm", action="store_true")
ap.add_argument("--max-chars", type=int, default=10)
ap.add_argument("--full", action="store_true", help="use full pinyin of the gold text (pypinyin) instead of what was typed")
args = ap.parse_args()

SHARED = "/Library/Input Methods/AIME.app/Contents/SharedSupport"
PRELOG = os.path.join(DATA, "userdb_export", "rime_prelog")
GRAM = os.path.join(DATA, "grams", "wanxiang-lts-zh-hans.gram")
WX_PATCH = """patch:
  grammar/language: wanxiang-lts-zh-hans
  grammar/collocation_max_length: 6
  grammar/collocation_min_length: 2
  grammar/collocation_penalty: -14
  grammar/non_collocation_penalty: -6
  grammar/weak_collocation_penalty: -100
  grammar/rear_penalty: -18
  translator/contextual_suggestions: true
"""

def engine_dir(name):
    d = os.path.join(DATA, f"sim_{name}")
    if not os.path.isdir(d):
        shutil.copytree(PRELOG, d, symlinks=True)
        if name == "wanxiang":
            os.symlink(GRAM, os.path.join(d, os.path.basename(GRAM)))
            open(os.path.join(d, "rime_frost.custom.yaml"), "w").write(WX_PATCH)
    return d

WX_SHARED = os.path.join(DATA, "schemes", "wanxiang_base")

class Rime:
    def __init__(self, name):
        if name == "wx_scheme":  # the full 万象 scheme (its own dictionary + grammar), empty user dict
            user = os.path.join(DATA, "sim_wx_scheme"); shutil.rmtree(user, ignore_errors=True); os.makedirs(user)
            cmd = [os.path.join(HERE, "build", "rime_cands"), WX_SHARED, user, "wanxiang"]
        else:
            cmd = [os.path.join(HERE, "build", "rime_cands"), SHARED, engine_dir(name), "rime_frost"]
        self.p = subprocess.Popen(cmd,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.cache = {}
    def cands(self, py):
        if py not in self.cache:
            self.p.stdin.write(py + "\n"); self.p.stdin.flush()
            line = self.p.stdout.readline().rstrip("\n")
            self.cache[py] = line.split("\t") if line else []
        return self.cache[py]

# ---- chunks ---------------------------------------------------------------------------------
intents = [it for it in load_intents() if HAN.match(it["gold"]) and it["pinyin"].isalpha()]
if args.full:  # what typing longer in full pinyin would look like
    from pypinyin import lazy_pinyin, Style
    for it in intents: it["pinyin"] = "".join(lazy_pinyin(it["gold"], style=Style.NORMAL))
def adjacent(a, b):
    if b["app"] != a["app"] or b.get("after_nonhan") or b.get("gap_ms", 1e9) > 15000: return False
    return b["ctx_src"] != "app" or (b["ctx"] or "").endswith(a["gold"])
runs, run = [], []
for it in intents:
    if run and not adjacent(run[-1], it):
        runs.append(run); run = []
    run.append(it)
if run: runs.append(run)

def chunks_of(n):
    out = []
    for r in runs:
        for k in range(0, len(r) - n + 1, n):
            ch = r[k:k + n]
            if sum(len(w["gold"]) for w in ch) <= args.max_chars: out.append(ch)
    return out

def word_errors(first, words):
    """How many of the gold words the chunk's first candidate gets wrong (position-aligned)."""
    gold = "".join(words)
    if first == gold: return 0
    pos, bad = 0, 0
    for w in words:
        if first[pos:pos + len(w)] != w: bad += 1
        pos += len(w)
    return bad

ABBR = lambda py: not any(v in py for v in "aeiouv") or py[-1] not in "aeiouvng"

engines = {name: Rime(name) for name in ("moqi", "wanxiang", "wx_scheme")}

lm = None
if args.lm:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, KVCache
    model, tok = load("mlx-community/Qwen3-0.6B-8bit")
    def lm(ctx, cands):
        ids = tok.encode("\n" + (ctx or ""))[-16:]
        cache = make_prompt_cache(model); last = model(mx.array([ids]), cache=cache)[:, -1, :]
        seqs = [tok.encode(c) for c in cands]; n = len(seqs); L = max(map(len, seqs))
        tid = mx.array([s + [0] * (L - len(s)) for s in seqs])
        msk = mx.array([[1] * len(s) + [0] * (L - len(s)) for s in seqs], dtype=mx.float32)
        first = mx.broadcast_to(last, (n, last.shape[-1]))[:, None, :]
        if L > 1:
            bc = []
            for c in cache:
                nc = KVCache(); nc.offset = c.offset
                nc.keys = mx.repeat(c.keys[..., :c.offset, :], n, 0); nc.values = mx.repeat(c.values[..., :c.offset, :], n, 0)
                bc.append(nc)
            allp = mx.concatenate([first, model(tid[:, :-1], cache=bc)], 1)
        else:
            allp = first
        lp = mx.take_along_axis(allp, tid[..., None], -1)[..., 0] - mx.logsumexp(allp, -1)
        s = (lp * msk).sum(-1); mx.eval(s); return s.tolist()

print(f"{len(intents)} words in {len(runs)} runs of adjacent words "
      f"(run length ≥2: {sum(1 for r in runs if len(r) >= 2)})\n")
for n in args.sizes:
    chs = chunks_of(n)
    if not chs: continue
    nwords = n * len(chs)
    print(f"== 每次输入 {n} 个词：{len(chs)} 串，{nwords} 个词"
          f"（含简拼的串 {sum(1 for c in chs if any(ABBR(w['pinyin']) for w in c))}）")
    print(f"   {'':<16}{'逐词:要纠正':>12}{'整串:首选全对':>14}{'整串:要纠正':>12}")
    real = sum(1 for c in chs for w in c if w["cands"][0] != w["gold"])
    print(f"   {'实际(日志)':<14}{real/nwords:>12.1%}" + ("   (实际打的是原拼音)" if args.full else ""))
    for name, eng in engines.items():
        word_err = sum(1 for c in chs for w in c if (eng.cands(w["pinyin"]) or [""])[0] != w["gold"])
        exact = chunk_err = 0
        lm_rows = []
        for c in chs:
            py = "".join(w["pinyin"] for w in c); words = [w["gold"] for w in c]
            cands = eng.cands(py)
            first = cands[0] if cands else ""
            exact += first == "".join(words); chunk_err += word_errors(first, words)
            if lm:
                grp = [k for k, x in enumerate(cands) if HAN.match(x) and len(x) == len(first)]
                # context = text before the chunk's first word (captured when it started)
                sc = lm(c[0]["ctx"], [cands[k] for k in grp]) if len(grp) > 1 else None
                lm_rows.append((cands, grp, sc, words))
        print(f"   {name:<14}{word_err/nwords:>12.1%}{exact/len(chs):>14.1%}{chunk_err/nwords:>12.1%}")
        if lm and lm_rows:
            half = len(lm_rows) // 2
            def acc(rows, lam):
                e = 0
                for cands, grp, sc, words in rows:
                    if sc is None: e += word_errors(cands[0] if cands else "", words); continue
                    j = max(range(len(grp)), key=lambda j: sc[j] - lam * math.log(1 + grp[j]))
                    e += word_errors(cands[grp[j]], words)
                return e
            lam = min([0, .5, 1, 2, 3, 4, 6, 8], key=lambda l: acc(lm_rows[:half], l))
            base = acc(lm_rows[half:], 1e9); with_lm = acc(lm_rows[half:], lam)
            nw = n * len(lm_rows[half:])
            print(f"   {name+'+模型':<14}{'':>12}{'':>14}{with_lm/nw:>12.1%}   (后半段; RIME 同段 {base/nw:.1%}, lam={lam})")
    print()
