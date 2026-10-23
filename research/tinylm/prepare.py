"""Build the character vocabulary and token files for tinylm (research/data/tinylm/).

Public text (raw/, see download in README): Zhihu answers, Chinese Wikipedia, THUCNews.
Personal text: research/data/ft/{train,valid}.jsonl (the same texts the Qwen LoRA used).
Each document becomes <bos> + its characters; files are flat uint16 arrays.
Full-width punctuation is kept as is: it is what appears before the caret when typing.

Usage: prepare.py [--vocab 8000] [--thuc-chars 200000000]
"""
import argparse, glob, json, os, random, re, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Vocab, BOS

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=8000)
ap.add_argument("--thuc-chars", type=int, default=200_000_000)
ap.add_argument("--valid-frac", type=float, default=0.005)
args = ap.parse_args()

OUT = os.path.join(HERE, "..", "data", "tinylm")
RAW = os.path.join(OUT, "raw")
FT = os.path.join(HERE, "..", "data", "ft")
WS = re.compile(r"\s+")
random.seed(1)

def clean(t):
    return WS.sub(" ", t or "").strip()

def wiki():
    for f in glob.glob(os.path.join(RAW, "wikipedia-cn-*", "*.json")):
        for r in json.load(open(f)):
            yield clean(r.get("completion"))

def zhihu():
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(RAW, "Zhihu-KOL", "data", "*.parquet"))):
        t = pq.read_table(f, columns=["INSTRUCTION", "RESPONSE"])
        for q, a in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
            yield clean(q)
            yield clean(a)

def thuc():
    # the file is grouped by category (sports first): sample lines across all of it, with
    # the same choice in both passes
    n, pick = 0, random.Random(3)
    for f in glob.glob(os.path.join(RAW, "THUCNews", "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if pick.random() > 0.3: continue
            r = json.loads(line)
            t = clean(r.get("text") or r.get("content") or "")
            n += len(t)
            yield t
            if n >= args.thuc_chars: return

def personal(name):
    for line in open(os.path.join(FT, name), encoding="utf-8"):
        yield clean(json.loads(line)["text"])

sources = {"zhihu": zhihu, "wiki": wiki, "thuc": thuc}

# pass 1: character counts
pub, per = Counter(), Counter()
for name, gen in sources.items():
    n = 0
    for t in gen():
        pub.update(t); n += len(t)
    print(f"{name}: {n/1e6:.0f}M chars", flush=True)
for t in personal("train.jsonl"):
    per.update(t)
chars = [c for c, k in per.most_common() if k >= 2]  # the user's own characters first
have = set(chars)
for c, _ in pub.most_common():
    if len(chars) >= args.vocab: break
    if c not in have: chars.append(c); have.add(c)
vocab = Vocab(chars)
tot = sum(pub.values())
cov = sum(pub[c] for c in chars) / tot
print(f"vocab {len(vocab)}; covers {cov:.4%} of public chars", flush=True)
json.dump(chars, open(os.path.join(OUT, "vocab.json"), "w"), ensure_ascii=False)

# pass 2: token files
def write(path, docs):
    with open(path, "wb") as fh:
        n = 0
        for t in docs:
            if not t: continue
            a = np.array([BOS] + vocab.encode(t), dtype=np.uint16)
            a.tofile(fh); n += len(a)
    print(f"{os.path.basename(path)}: {n/1e6:.1f}M tokens", flush=True)

train = open(os.path.join(OUT, "train.bin"), "wb")
valid = open(os.path.join(OUT, "valid.bin"), "wb")
counts = Counter()
for name, gen in sources.items():
    for t in gen():
        if not t: continue
        a = np.array([BOS] + vocab.encode(t), dtype=np.uint16)
        is_valid = random.random() < args.valid_frac
        a.tofile(valid if is_valid else train)
        counts["valid" if is_valid else "train"] += len(a)
train.close(); valid.close()
print(f"train.bin {counts['train']/1e6:.0f}M tokens, valid.bin {counts['valid']/1e6:.1f}M tokens")
write(os.path.join(OUT, "personal_train.bin"), personal("train.jsonl"))
write(os.path.join(OUT, "personal_valid.bin"), personal("valid.jsonl"))
