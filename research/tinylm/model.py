"""A small character-level language model for candidate scoring (MLX).

One token per character: Chinese needs no subword vocabulary, and a ~8k-character vocabulary
keeps the embedding table small (Qwen3's 151k-token table is most of a 0.6B model's memory
traffic for this job). Pre-norm transformer, RoPE, tied input/output embeddings.

A model directory holds config.json, vocab.json (list of characters) and model.safetensors.
"""
import json, os
import mlx.core as mx
import mlx.nn as nn

PAD, UNK, BOS = 0, 1, 2
SPECIALS = ["<pad>", "<unk>", "<bos>"]


class Vocab:
    def __init__(self, chars):
        self.itos = SPECIALS + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        get = self.stoi.get
        return [get(c, UNK) for c in s]


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(dim // heads)

    def __call__(self, x, mask):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.down(nn.gelu(self.up(self.n2(x))))


class TinyLM(nn.Module):
    def __init__(self, vocab_size, dim=384, layers=6, heads=6, max_len=64):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, dim=dim, layers=layers, heads=heads, max_len=max_len)
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = [Block(dim, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(dim)

    def __call__(self, ids):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(ids.shape[1]).astype(mx.float32)
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, mask)
        return self.embed.as_linear(self.norm(x))


def save(model, vocab, path):
    os.makedirs(path, exist_ok=True)
    json.dump(model.config, open(os.path.join(path, "config.json"), "w"))
    json.dump(vocab.itos[len(SPECIALS):], open(os.path.join(path, "vocab.json"), "w"), ensure_ascii=False)
    model.save_weights(os.path.join(path, "model.safetensors"))


def load(path):
    vocab = Vocab(json.load(open(os.path.join(path, "vocab.json"))))
    model = TinyLM(**json.load(open(os.path.join(path, "config.json"))))
    model.load_weights(os.path.join(path, "model.safetensors"))
    model.eval()
    return model, vocab


class Scorer:
    """log P(candidate | context) for a batch of candidates, summed over its characters."""

    def __init__(self, path, ctx_chars=32):
        self.model, self.vocab = load(path)
        self.ctx_chars = ctx_chars

    def __call__(self, ctx, cands):
        left = [BOS] + self.vocab.encode((ctx or "")[-self.ctx_chars:])
        seqs = [self.vocab.encode(c) for c in cands]
        L = max(map(len, seqs))
        # pad the batch to fixed shapes so the GPU kernels are reused across calls
        L = next(x for x in (1, 2, 4, 8, 16, 32) if x >= L) if L <= 32 else L
        n = len(seqs)
        rows = max(9, n)
        seqs = seqs + [seqs[0]] * (rows - n)
        ids = mx.array([left + s + [PAD] * (L - len(s)) for s in seqs])
        logits = self.model(ids[:, :-1])[:, len(left) - 1:, :]
        tgt = ids[:, len(left):]
        lp = mx.take_along_axis(logits, tgt[..., None], -1)[..., 0] - mx.logsumexp(logits, -1)
        msk = mx.array([[1] * len(s) + [0] * (L - len(s)) for s in seqs], dtype=mx.float32)
        s = (lp * msk).sum(-1)
        mx.eval(s)
        return s.tolist()[:n]
