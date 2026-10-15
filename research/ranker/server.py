"""Local candidate-ranking service for AIME (prototype).

Listens on a Unix socket (~/Library/AIME/ranker/ranker.sock, owner-only) for JSON lines:
  {"op":"rank","ctx":"…text before the caret…","prev":"previous word","cands":["…", …]}
      -> {"i": index on the page to highlight (0 = keep RIME's choice), "ms": …}
  {"op":"commit","prev":"…","text":"…"}      (no reply; updates the personal history)
Scoring (see research/eval/replay_log.py): among the full-span candidates (Han, as long as
RIME's first), score = log P(cand | ctx) − λ·log(1+rank) + μ·[(prev, cand) committed before].
Nothing leaves the Mac; the model runs with MLX.

Usage: server.py [--model DIR] [--adapter DIR] [--lam1 2] [--lam 2] [--mu 4]
(a fused model dir — `mlx_lm fuse` — is faster than base + adapter)
"""
import argparse, glob, json, math, os, re, socket, sys, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-8bit")
ap.add_argument("--adapter", default=None)
ap.add_argument("--ctx", type=int, default=16)
ap.add_argument("--lam1", type=float, default=2.0, help="RIME-rank weight, single-character words")
ap.add_argument("--lam", type=float, default=2.0, help="RIME-rank weight, longer words")
ap.add_argument("--mu", type=float, default=4.0, help="bonus when (prev, cand) was committed before")
ap.add_argument("--dir", default=os.path.expanduser("~/Library/AIME/ranker"))
args = ap.parse_args()

HAN = re.compile(r"^[一-鿿]+$")
os.makedirs(args.dir, mode=0o700, exist_ok=True)
SOCK = os.path.join(args.dir, "ranker.sock")
HIST = os.path.join(args.dir, "bigrams.json")

def log(*a): print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)

# ---- personal history: (previous word, word) pairs the user committed ------------------------
bigrams = defaultdict(int)
if os.path.exists(HIST):
    for k, v in json.load(open(HIST)).items(): bigrams[k] = v
else:  # warm start from the typing log, if any
    prev = None
    for f in sorted(glob.glob(os.path.expanduser("~/Library/AIME/typing-log/*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            if e.get("ev") != "commit": continue
            t = e["text"]
            if HAN.match(t):
                if prev: bigrams[f"{prev}\t{t}"] += 1
                prev = t
            else:
                prev = None
log(f"history: {len(bigrams)} word pairs")
dirty = 0
def save_history():
    global dirty
    tmp = HIST + ".tmp"
    with open(tmp, "w") as fh: json.dump(bigrams, fh, ensure_ascii=False)
    os.chmod(tmp, 0o600); os.replace(tmp, HIST); dirty = 0

# ---- model ----------------------------------------------------------------------------------
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
model, tok = load(args.model, adapter_path=args.adapter)
log(f"model {args.model} adapter={args.adapter}")

_ctx_key, _ctx_state = None, None
_ctx_miss = False  # whether the last rank had to compute its context (for diagnostics)
def context_state(ctx):
    """KV cache of the context; reused across the keystrokes of one composition."""
    global _ctx_key, _ctx_state, _ctx_miss
    ids = tok.encode("\n" + (ctx or ""))[-args.ctx:]
    key = tuple(ids)
    _ctx_miss = key != _ctx_key
    if key != _ctx_key:
        cache = make_prompt_cache(model)
        last = model(mx.array([ids]), cache=cache)[:, -1, :]
        mx.eval(last, [c.keys for c in cache], [c.values for c in cache])
        _ctx_key, _ctx_state = key, (cache, last)
    return _ctx_state

def lm_scores(ctx, cands):
    cache, last = context_state(ctx)
    seqs = [tok.encode(c) for c in cands]; n_real = len(seqs)
    # fixed shapes (9 rows, length 1/2/4/8) so Metal kernels are reused instead of rebuilt
    L = max(map(len, seqs)); L = next(x for x in (1, 2, 4, 8, 16, 64) if x >= L)
    seqs = seqs + [seqs[0]] * (max(9, n_real) - n_real); n = len(seqs)
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
    return s.tolist()[:n_real]

_last_ctx = ""
def rank(req):
    global _last_ctx
    _last_ctx = req.get("ctx", "") or ""
    cands = req.get("cands") or []
    if len(cands) < 2 or not cands[0]: return 0
    group = [k for k, c in enumerate(cands) if HAN.match(c) and len(c) == len(cands[0])]
    if len(group) < 2: return 0
    lm = lm_scores(req.get("ctx", ""), [cands[k] for k in group])
    lam = args.lam1 if len(cands[0]) == 1 else args.lam
    prev = req.get("prev") or ""
    def score(j):
        k = group[j]
        s = lm[j] - lam * math.log(1 + k)
        if prev and bigrams.get(f"{prev}\t{cands[k]}"): s += args.mu
        return s
    return group[max(range(len(group)), key=score)]

# warm up the kernels so the first keystroke is not slow
rank({"ctx": "你好", "cands": ["世界", "时间"]})

# ---- socket loop ----------------------------------------------------------------------------
if os.path.exists(SOCK): os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
old = os.umask(0o077); srv.bind(SOCK); os.umask(old)
srv.listen(4)
log(f"listening on {SOCK}")
import selectors
sel = selectors.DefaultSelector(); sel.register(srv, selectors.EVENT_READ)
buffers = {}
while True:
    for key, _ in sel.select(timeout=30):
        if key.fileobj is srv:
            conn, _ = srv.accept(); conn.setblocking(False)
            sel.register(conn, selectors.EVENT_READ); buffers[conn] = b""
            continue
        conn = key.fileobj
        try: data = conn.recv(65536)
        except BlockingIOError: continue
        except OSError: data = b""
        if not data:
            sel.unregister(conn); buffers.pop(conn, None); conn.close(); continue
        buffers[conn] += data
        while b"\n" in buffers[conn]:
            line, buffers[conn] = buffers[conn].split(b"\n", 1)
            try: req = json.loads(line)
            except ValueError: continue
            if req.get("op") == "commit":
                t, p = req.get("text", ""), req.get("prev", "")
                if HAN.match(t) and p: bigrams[f"{p}\t{t}"] += 1; dirty += 1
                if dirty >= 20: save_history()
                # the next composition will most likely see this text before the caret:
                # compute its KV cache now, between keystrokes
                try: context_state(_last_ctx + t); _last_ctx += t
                except Exception as e: log("prefill error", e)
                continue
            t0 = time.perf_counter()
            try: i = rank(req)
            except Exception as e: log("rank error", e); i = 0
            ms = (time.perf_counter() - t0) * 1000
            try: conn.sendall((json.dumps({"i": i, "ms": round(ms, 1), "miss": _ctx_miss}) + "\n").encode())
            except OSError: pass
    if dirty: save_history()
