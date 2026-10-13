# Optimized keystroke path: vectorized gather, one mx.eval, compiled-free. Measures floor.
import sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
model, tok = load(sys.argv[1])
def prefill(ctx):
    cache = make_prompt_cache(model); ids = tok.encode(ctx)
    last = model(mx.array([ids]), cache=cache)[:, -1, :]
    mx.eval(last, [c.keys for c in cache], [c.values for c in cache]); return cache, last
def score(cache, last, cands):
    seqs = [tok.encode(c) for c in cands]; n=len(seqs); L=max(map(len,seqs))
    ids = mx.array([s+[0]*(L-len(s)) for s in seqs]); msk = mx.array([[1]*len(s)+[0]*(L-len(s)) for s in seqs], dtype=mx.float32)
    bc=[]
    for c in cache:
        nc=KVCache(); nc.keys=mx.repeat(c.keys[..., :c.offset, :], n, 0); nc.values=mx.repeat(c.values[..., :c.offset, :], n, 0); nc.offset=c.offset; bc.append(nc)
    logits = model(ids[:, :-1], cache=bc) if L>1 else None
    first = mx.broadcast_to(last, (n, last.shape[-1]))[:, None, :]
    allp = first if logits is None else mx.concatenate([first, logits], 1)   # [n, L, V]
    lp = mx.take_along_axis(allp, ids[..., None], -1)[..., 0] - mx.logsumexp(allp, -1)
    s = (lp*msk).sum(-1); mx.eval(s); return s.tolist()
ctx = "关于下周的产品发布会，我们需要提前准备好演示材料和新闻稿，另外市场部那边也要同步一下宣传的节奏，"
for rep in (1, 2, 4):
    c = ctx*rep; cache,last=prefill(c)
    cands = ["时间","事件","实践","世间","时监","是","事","时","市"]
    for _ in range(3): score(cache,last,cands)
    ts=[]
    for _ in range(30):
        t=time.perf_counter(); score(cache,last,cands); ts.append((time.perf_counter()-t)*1000)
    ts.sort(); t=time.perf_counter(); prefill(c); pf=(time.perf_counter()-t)*1000
    print(f"{sys.argv[1].split('/')[-1]} ctx={len(tok.encode(c))}tok prefill={pf:.0f}ms  9cands p50={ts[15]:.1f}ms p90={ts[27]:.1f}ms")
