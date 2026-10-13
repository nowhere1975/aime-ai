"""Offline eval: Rime's own first candidate vs. small-LM reranking with left context.

Protocol (word level, gold context):
  for each paragraph, segment into words (jieba); for every all-Han word w:
    - input = pinyin(w) (computed on the whole Han run so polyphones resolve in context)
    - Rime candidates (top 20) from AIME's librime + the user's real rime_frost config
    - rime@1  : Rime's first candidate == w
    - lm@1    : among Rime candidates with len == len(w), the one with max log P(c | left context)
    - combo@1 : lm score + lam * rime-rank prior   (lam tuned on even paragraphs, reported on odd)
  The left context is the true text before w (what the user has already typed), and the LM's
  KV cache is extended word by word, the same way an IME would after each commit.
"""
import json, math, os, re, subprocess, sys, time
from collections import defaultdict
import jieba, mlx.core as mx
from pypinyin import lazy_pinyin, Style
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")  # gitignored: corpora, Rime user copy, records
SHARED = "/Library/Input Methods/AIME.app/Contents/SharedSupport"
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-8bit")
ap.add_argument("--corpus", default="corpus.json", help="file in research/data/: [{src, text}, ...]")
ap.add_argument("--schema", default="rime_frost")
ap.add_argument("--max-words", type=int, default=3000)
ap.add_argument("--ctx", type=int, default=0, help="left-context tokens fed to the LM (0 = whole paragraph)")
ap.add_argument("--topn", type=int, default=20, help="only the first N same-length Rime candidates are reranked")
ap.add_argument("--out", default="records.json")
ap.add_argument("--report", action="store_true", help="only re-report an existing --out file")
ARGS = ap.parse_args()
MODEL = ARGS.model; MAX_WORDS = ARGS.max_words
HAN = re.compile(r"[一-鿿]+")
jieba.setLogLevel(60)

class Rime:
    def __init__(self):
        self.p = subprocess.Popen([os.path.join(HERE, "build", "rime_cands"), SHARED, os.path.join(DATA, "rime_user"), ARGS.schema],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def cands(self, py):
        self.p.stdin.write(py + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline().rstrip("\n")
        return line.split("\t") if line else []

model, tok = load(MODEL)

def extend(cache, text):
    logits = model(mx.array([tok.encode(text)]), cache=cache)
    last = logits[:, -1, :]
    mx.eval(last)
    return last

def lm_scores(cache, last, cands):
    seqs = [tok.encode(c) for c in cands]; n = len(seqs); L = max(map(len, seqs))
    ids = mx.array([s + [0] * (L - len(s)) for s in seqs])
    msk = mx.array([[1] * len(s) + [0] * (L - len(s)) for s in seqs], dtype=mx.float32)
    first = mx.broadcast_to(last, (n, last.shape[-1]))[:, None, :]
    if L > 1:
        bc = []
        for c in cache:
            nc = KVCache(); nc.offset = c.offset
            nc.keys = mx.repeat(c.keys[..., :c.offset, :], n, 0)
            nc.values = mx.repeat(c.values[..., :c.offset, :], n, 0)
            bc.append(nc)
        allp = mx.concatenate([first, model(ids[:, :-1], cache=bc)], 1)
    else:
        allp = first
    lp = mx.take_along_axis(allp, ids[..., None], -1)[..., 0] - mx.logsumexp(allp, -1)
    s = (lp * msk).sum(-1); mx.eval(s)
    return s.tolist()

def pieces(text):
    """Yield (kind, text, pinyin) in order; kind 'w' for Han words, 'o' for anything else."""
    pos = 0
    for m in HAN.finditer(text):
        if m.start() > pos: yield "o", text[pos:m.start()], None
        run = m.group(); py = lazy_pinyin(run, style=Style.NORMAL)
        i = 0
        for w in jieba.cut(run, HMM=False):
            yield "w", w, "".join(py[i:i + len(w)]); i += len(w)
        pos = m.end()
    if pos < len(text): yield "o", text[pos:], None

def main():
    corpus = json.load(open(os.path.join(DATA, ARGS.corpus)))
    import random; random.Random(0).shuffle(corpus)
    rime = Rime()
    records = []  # one per evaluated word
    t_score = []
    t0 = time.time()
    for pi, doc in enumerate(corpus):
        cache = make_prompt_cache(model); last = extend(cache, "\n"); left = ""
        for kind, w, py in pieces(doc["text"]):
            if kind == "w" and py and py.isascii() and py.isalpha():
                raw = rime.cands(py)
                same = [c for c in raw if len(c) == len(w) and HAN.fullmatch(c)][:ARGS.topn]
                rec = dict(par=pi, src=doc["src"], w=w, n=len(w), rime1=raw[0] if raw else None,
                           reach=w in same, cands=same)
                if len(same) > 1:
                    t = time.perf_counter()
                    if ARGS.ctx:  # what an IME would do: prefill only the last N tokens before the cursor
                        ids = tok.encode("\n" + left)[-ARGS.ctx:]
                        cache_s = make_prompt_cache(model)
                        last_s = model(mx.array([ids]), cache=cache_s)[:, -1, :]; mx.eval(last_s)
                        t = time.perf_counter()  # prefill happens after the previous commit, off the keystroke path
                        rec["lm"] = lm_scores(cache_s, last_s, same)
                    else:
                        rec["lm"] = lm_scores(cache, last, same)
                    t_score.append(time.perf_counter() - t)
                records.append(rec)
                if len(records) % 200 == 0:
                    print(f"{len(records)} words, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
            left += w
            if not ARGS.ctx: last = extend(cache, w)
            if len(records) >= MAX_WORDS: break
        if len(records) >= MAX_WORDS: break
    json.dump(records, open(os.path.join(DATA, ARGS.out), "w"), ensure_ascii=False)
    report(records, t_score)

def pick(rec, lam):
    if isinstance(lam, tuple): lam = lam[0] if rec["n"] == 1 else lam[1]
    if "lm" not in rec: return rec["cands"][0] if rec["cands"] else rec["rime1"]
    best = max(range(len(rec["cands"])), key=lambda i: rec["lm"][i] - lam * math.log(1 + i))
    return rec["cands"][best]

def acc(recs, f):
    return sum(f(r) == r["w"] for r in recs) / max(1, len(recs))

def report(records, t_score):
    tune = [r for r in records if r["par"] % 2 == 0]; test = [r for r in records if r["par"] % 2 == 1]
    lams = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
    lam = max(((a, b) for a in lams for b in lams), key=lambda l: acc(tune, lambda r: pick(r, l)))
    print(f"\nmodel={MODEL} corpus={ARGS.corpus} ctx={ARGS.ctx or 'full'} topn={ARGS.topn}  words={len(records)}  tuned lam(1char,multi)={lam} (on {len(tune)} words)")
    if t_score:
        ts = sorted(t_score); print(f"score latency p50={ts[len(ts)//2]*1000:.1f}ms p90={ts[int(len(ts)*.9)]*1000:.1f}ms  (cands/word avg {sum(len(r['cands']) for r in records)/len(records):.1f})")
    print(f"\nTEST set ({len(test)} words, held-out paragraphs)")
    print(f"{'bucket':<14}{'n':>6}{'reach':>8}{'rime@1':>9}{'lm@1':>8}{'combo@1':>9}")
    groups = defaultdict(list)
    for r in test:
        groups["all"].append(r); groups[f"len={min(r['n'],3)}{'+' if r['n']>=3 else ''}"].append(r); groups[r["src"]].append(r)
    for k in ["all", "len=1", "len=2", "len=3+"] + sorted(k for k in groups if not k.startswith(("all", "len="))):
        g = groups[k]
        if not g: continue
        print(f"{k:<14}{len(g):>6}{sum(r['reach'] for r in g)/len(g):>8.1%}"
              f"{acc(g, lambda r: r['rime1']):>9.1%}{acc(g, lambda r: pick(r, 0)):>8.1%}{acc(g, lambda r: pick(r, lam)):>9.1%}")
    # error reduction
    rime_err = 1 - acc(test, lambda r: r["rime1"]); combo_err = 1 - acc(test, lambda r: pick(r, lam))
    print(f"\nerror rate: rime {rime_err:.1%} -> combo {combo_err:.1%}  (relative reduction {(rime_err-combo_err)/max(rime_err,1e-9):.0%})")
    fixed = [r for r in test if r["rime1"] != r["w"] and pick(r, lam) == r["w"]]
    broke = [r for r in test if r["rime1"] == r["w"] and pick(r, lam) != r["w"]]
    print(f"fixed {len(fixed)}, broke {len(broke)}")
    print("fixed e.g.:", ", ".join(f"{r['rime1']}→{r['w']}" for r in fixed[:15]))
    print("broke e.g.:", ", ".join(f"{r['w']}→{pick(r, lam)}" for r in broke[:15]))

if __name__ == "__main__":
    if ARGS.report:
        report(json.load(open(os.path.join(DATA, ARGS.out))), [])
    else:
        main()
