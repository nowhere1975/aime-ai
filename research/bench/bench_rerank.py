"""Benchmark: can a small local LM rerank IME candidates fast enough?

Per commit  : prefill left context once -> KV cache (can run async, off the keystroke path).
Per keystroke: score N candidates in ONE batched forward pass on top of the cached prefix.
Score = sum log P(candidate tokens | context).
"""
import sys, time, copy
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache

repo = sys.argv[1]
model, tok = load(repo)

def prefill(ctx):
    ids = tok.encode(ctx)
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    last = logits[:, -1, :]
    mx.eval(last, [c.keys for c in cache], [c.values for c in cache])
    return cache, last

def tile_cache(cache, n):
    out = []
    for c in cache:
        k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
        nc = KVCache()
        nc.keys = mx.repeat(k, n, axis=0)
        nc.values = mx.repeat(v, n, axis=0)
        nc.offset = c.offset
        out.append(nc)
    return out

def score(cache, last_logits, cands):
    seqs = [tok.encode(c) for c in cands]
    L = max(len(s) for s in seqs)
    pad = tok.eos_token_id or 0
    batch = mx.array([s + [pad] * (L - len(s)) for s in seqs])
    bc = tile_cache(cache, len(cands))
    logits = model(batch, cache=bc)                       # [N, L, V]
    lp_first = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)  # [1, V]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp, lp_first)
    scores = []
    for i, s in enumerate(seqs):
        t = lp_first[0, s[0]].item()
        for j in range(1, len(s)):
            t += lp[i, j - 1, s[j]].item()
        scores.append(t)
    return scores

# (left context, candidates in Rime's order, expected)
cases = [
    ("我们明天下午三点在会议室", ["开会", "开花", "凯慧", "开会儿"], "开会"),
    ("这个函数的返回值", ["类型", "累心", "雷星", "类星"], "类型"),
    ("昨天晚上下了一场大", ["与", "雨", "鱼", "语", "于"], "雨"),
    ("他在公司担任技术", ["总间", "总监", "总检", "宗建"], "总监"),
    ("请把文件发到我的", ["有些", "邮箱", "油箱", "有效"], "邮箱"),
    ("汽车的", ["邮箱", "油箱", "有些", "有效"], "油箱"),
    ("这道题的", ["解法", "接发", "姐发", "结发"], "解法"),
    ("他们俩是大学", ["同学", "同穴", "铜穴", "童学"], "同学"),
]

# warmup
c, l = prefill("你好"); score(c, l, ["世界", "时间"])

hits = 0
pf_times, sc_times = [], []
for ctx, cands, gold in cases:
    t0 = time.perf_counter(); cache, last = prefill(ctx); t1 = time.perf_counter()
    s = score(cache, last, cands); t2 = time.perf_counter()
    pf_times.append((t1 - t0) * 1000); sc_times.append((t2 - t1) * 1000)
    best = cands[max(range(len(cands)), key=lambda i: s[i])]
    hits += best == gold
    print(f"{ctx}|{best:<4} gold={gold} {'OK ' if best==gold else 'BAD'} prefill={pf_times[-1]:.1f}ms score={sc_times[-1]:.1f}ms")

# keystroke-path latency with a realistic page: 9 candidates, long context
ctx = "关于下周的产品发布会，我们需要提前准备好演示材料和新闻稿，另外市场部那边也要同步一下宣传的节奏，" * 2
cands = ["时间", "事件", "实践", "世间", "时监", "是", "事", "时", "市"]
cache, last = prefill(ctx)
ts = []
for _ in range(30):
    t = time.perf_counter(); score(cache, last, cands); ts.append((time.perf_counter() - t) * 1000)
ts.sort()
t = time.perf_counter(); prefill(ctx); pf = (time.perf_counter() - t) * 1000
print(f"\n[{repo}] accuracy {hits}/{len(cases)}")
print(f"ctx tokens={len(tok.encode(ctx))}  prefill={pf:.1f}ms  "
      f"score 9 cands: p50={ts[len(ts)//2]:.1f}ms p90={ts[int(len(ts)*.9)]:.1f}ms")
print(f"peak mem={mx.get_peak_memory()/1e6:.0f}MB")
