"""Train tinylm on the token files from prepare.py (MLX, Apple GPU).

Pretrain:  train.py --out data/tinylm/base --steps 40000
Fine-tune: train.py --out data/tinylm/personal --init data/tinylm/base --steps 3000 --lr 3e-4 \
                    --mix data/tinylm/personal_train.bin --mix-frac 0.5
Resumes from --out if it holds a checkpoint (weights + optimizer state + step), so it can be
stopped (Ctrl-C) and restarted. Progress goes to <out>/train.log.
"""
import argparse, json, math, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import TinyLM, Vocab, save

D = os.path.join(HERE, "..", "data", "tinylm")
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", default=os.path.join(D, "train.bin"))
ap.add_argument("--valid", default=os.path.join(D, "valid.bin"))
ap.add_argument("--pvalid", default=os.path.join(D, "personal_valid.bin"))
ap.add_argument("--mix", default=None, help="second token file sampled with --mix-frac")
ap.add_argument("--mix-frac", type=float, default=0.0)
ap.add_argument("--init", default=None, help="start from this model dir")
ap.add_argument("--vocab", default=None, help="vocab.json (default: --init dir, else data/tinylm)")
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--dim", type=int, default=384)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--eval-every", type=int, default=1000)
ap.add_argument("--bench", type=int, default=0, help="time this many steps and exit")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, "train.log"), "a")
def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a))
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

vocab = Vocab(json.load(open(args.vocab or os.path.join(args.init or D, 'vocab.json'))))
data = np.memmap(args.data, dtype=np.uint16, mode="r")
mix = np.memmap(args.mix, dtype=np.uint16, mode="r") if args.mix else None
rng = np.random.default_rng(int(time.time()))

def batch(src, n):
    idx = rng.integers(0, len(src) - args.seq - 1, n)
    return np.stack([src[i:i + args.seq + 1] for i in idx]).astype(np.int32)

def next_batch():
    if mix is None or args.mix_frac <= 0:
        b = batch(data, args.batch)
    else:
        k = int(round(args.batch * args.mix_frac))
        b = np.concatenate([batch(mix, k), batch(data, args.batch - k)])
    b = mx.array(b)
    return b[:, :-1], b[:, 1:]

if args.init:
    cfg = json.load(open(os.path.join(args.init, "config.json")))
    model = TinyLM(**cfg); model.load_weights(os.path.join(args.init, "model.safetensors"))
else:
    model = TinyLM(len(vocab), args.dim, args.layers, args.heads, args.seq)
nparams = sum(v.size for _, v in tree_flatten(model.parameters()))

def lr_at(step):
    if step < args.warmup: return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)
start = 0
ck = os.path.join(args.out, "ckpt")
if os.path.exists(os.path.join(ck, "state.json")):
    model.load_weights(os.path.join(ck, "model.safetensors"))
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(os.path.join(ck, "opt.safetensors")).items()))
    start = json.load(open(os.path.join(ck, "state.json")))["step"]
    log(f"resumed at step {start}")

def loss_fn(m, x, y):
    return nn.losses.cross_entropy(m(x), y, reduction="mean")

state = [model.state, opt.state]

def train_step(x, y):
    loss, g = nn.value_and_grad(model, loss_fn)(model, x, y)
    g, _ = optim.clip_grad_norm(g, 1.0)
    opt.update(model, g)
    return loss
step = mx.compile(train_step, inputs=state, outputs=state)

def evaluate(path, n=40):
    if not path or not os.path.exists(path): return float("nan")
    src = np.memmap(path, dtype=np.uint16, mode="r")
    r = np.random.default_rng(0)
    tot = 0.0
    for _ in range(n):
        idx = r.integers(0, len(src) - args.seq - 1, 64)
        b = mx.array(np.stack([src[i:i + args.seq + 1] for i in idx]).astype(np.int32))
        tot += loss_fn(model, b[:, :-1], b[:, 1:]).item()
    return tot / n

def checkpoint(s):
    os.makedirs(ck, exist_ok=True)
    model.save_weights(os.path.join(ck, "model.safetensors"))
    mx.save_safetensors(os.path.join(ck, "opt.safetensors"), dict(tree_flatten(opt.state)))
    json.dump({"step": s}, open(os.path.join(ck, "state.json"), "w"))
    save(model, vocab, args.out)  # the latest usable model, at the top of --out

log(f"params {nparams/1e6:.1f}M, vocab {len(vocab)}, tokens {len(data)/1e6:.0f}M, steps {start}->{args.steps}, "
    f"batch {args.batch}x{args.seq}")
t0, seen, avg = time.time(), 0, None
for s in range(start, args.steps):
    opt.learning_rate = lr_at(s)
    x, y = next_batch()
    loss = step(x, y)
    mx.eval(loss, model.parameters(), opt.state)
    l = loss.item(); avg = l if avg is None else 0.98 * avg + 0.02 * l
    seen += x.size
    if args.bench and s - start + 1 >= args.bench:
        dt = time.time() - t0
        log(f"bench: {args.bench} steps {dt:.1f}s, {seen/dt:,.0f} tokens/s, loss {avg:.3f}")
        sys.exit()
    if (s + 1) % 100 == 0 and (s + 1) % args.eval_every:
        dt = time.time() - t0
        print(f"step {s+1} loss {avg:.3f} lr {opt.learning_rate.item():.2e} {seen/dt:,.0f} tok/s", flush=True)
    if (s + 1) % args.eval_every == 0 or s + 1 == args.steps:
        model.eval()
        log(f"step {s+1} train {avg:.3f} valid {evaluate(args.valid):.3f} personal {evaluate(args.pvalid):.3f} "
            f"{seen/(time.time()-t0):,.0f} tok/s")
        model.train()
        checkpoint(s + 1)
log("done")
