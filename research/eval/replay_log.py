"""Replay real typing (from analyze_log.py's replay_cases.json) through context-aware reranking.

An *intent* is one word the user wanted:
  - a commit kept as is                          -> gold = committed text
  - a commit deleted right away (⌫ ≥ its length) and the same pinyin typed again
                                                 -> gold = the second commit ("上屏错误" redo)
For each intent the first attempt's real candidate page (≤9) and the real left context
(text before the caret, or recent commits where the app exposes none) are used.

Rerankers, compared with RIME's first candidate:
  lm      : log P(cand | context), Qwen3-0.6B
  fused   : lm − λ·log(1+rank)                       (λ per single-char / multi-char)
  +hist   : fused + μ·[(previous word, cand) seen earlier in this log]   (online, chronological)
λ, μ are tuned on the first half of the log (by time) and reported on the second half.
"""
import argparse, json, math, os, re, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-8bit")
ap.add_argument("--ctx", type=int, default=16, help="context tokens")
ap.add_argument("--adapter", default=None, help="LoRA adapter dir (e.g. data/ft/adapters_q06b8)")
ap.add_argument("--tiny", default=None, help="tinylm model dir (research/tinylm) instead of --model")
args = ap.parse_args()

HAN = re.compile(r"^[一-鿿]+$")
# ---- build intents -------------------------------------------------------------------------
sys.path.insert(0, HERE)
from intents import load_intents
cases = json.load(open(os.path.join(DATA, "replay_cases.json")))
intents = load_intents()

# ---- LM -------------------------------------------------------------------------------------
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
if not args.tiny:
    model, tok = load(args.model, adapter_path=args.adapter)

def lm_scores(ctx, cands):
    ids = tok.encode("\n" + (ctx or ""))[-args.ctx:]
    cache = make_prompt_cache(model)
    last = model(mx.array([ids]), cache=cache)[:, -1, :]
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
    s = (lp * msk).sum(-1); mx.eval(s)
    return s.tolist()

if args.tiny:
    sys.path.insert(0, os.path.join(HERE, "..", "tinylm"))
    from model import Scorer
    lm_scores = Scorer(args.tiny)

t0 = time.time()
for it in intents:  # rerank only the full-span group (see intents.py)
    if len(it["group"]) > 1:
        it["lm"] = lm_scores(it["ctx"], [it["cands"][k] for k in it["group"]])
print(f"scored {len(intents)} intents in {time.time()-t0:.0f}s", file=sys.stderr)

# ---- pick functions -------------------------------------------------------------------------
def pick(it, lam=None, mu=0.0, seen=None):
    cands, group = it["cands"], it["group"]
    if lam is None or "lm" not in it: return cands[0]
    l = lam[0] if len(cands[0]) == 1 else lam[1]
    def score(j):
        k = group[j]
        s = it["lm"][j] - l * math.log(1 + k)
        if mu and seen is not None and it["prev"] and (it["prev"], cands[k]) in seen: s += mu
        return s
    return cands[group[max(range(len(group)), key=score)]]

def run(items, lam=None, mu=0.0):
    """Accuracy over items in time order; the history bigram set grows online."""
    seen, ok, want = set(), 0, {it["i"] for it in items}
    for it in intents:  # walk the whole log so history includes the tuning half
        if it["i"] in want: ok += pick(it, lam, mu, seen) == it["gold"]
        if it["prev"]: seen.add((it["prev"], it["gold"]))
    return ok / max(1, len(items))

half = len(intents) // 2
tune, test = intents[:half], intents[half:]
grid = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
lam = max(((a, b) for a in grid for b in grid), key=lambda l: run(tune, l))
mu = max([0, 1, 2, 3, 4, 6], key=lambda m: run(tune, lam, m))

n_redo = sum(it["redo"] for it in intents)
reach = sum(1 for it in test if it["gold"] in [it["cands"][k] for k in it["group"]] or it["cands"][0] == it["gold"]) / len(test)
print(f"\n{len(cases)} commits -> {len(intents)} intents ({n_redo} were commit-delete-retype)")
print(f"tuned on first {len(tune)} intents: lam(1char,multi)={lam} mu={mu}; tested on last {len(test)}  (gold reachable {reach:.1%})")
print(f"\n{'':<22}{'首选正确':>8}{'需要纠正':>10}")
for name, acc in [("RIME 现状", run(test)), ("模型重排", run(test, (0, 0))),
                  ("模型+RIME排名", run(test, lam)), ("模型+排名+个人历史", run(test, lam, mu))]:
    print(f"{name:<18}{acc:>10.1%}{1-acc:>10.1%}")
base_err = 1 - run(test); best_err = 1 - run(test, lam, mu)
print(f"\n需要纠正的次数减少 {(base_err-best_err)/base_err:.0%}")

seen = set(); fixed, broke = [], []
test_ids = {it["i"] for it in test}
for it in intents:
    if it["i"] in test_ids:
        p = pick(it, lam, mu, seen); r = it["cands"][0]
        if r != it["gold"] and p == it["gold"]: fixed.append(f"{it['pinyin']}:{r}→{p}")
        if r == it["gold"] and p != it["gold"]: broke.append(f"{it['pinyin']}:{r}→{p}")
    if it["prev"]: seen.add((it["prev"], it["gold"]))
print(f"改对 {len(fixed)}: {', '.join(fixed[:40])}")
print(f"改错 {len(broke)}: {', '.join(broke[:40])}")
