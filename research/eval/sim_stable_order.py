"""Simulate ordering policies for RIME's first candidate group against the typing log.

Problem: RIME's user dictionary ranks by recency-weighted use, so after picking the second
homophone once it jumps to first place, and the next time the other one is wanted
("zai: 在/再" ping-pong). These policies only reorder the full-span group (intents.py);
everything is online and chronological: counts start from a user-dictionary export taken
before the log, then grow with each choice in the log.

  rime     : RIME's order as the user saw it
  unigram  : most-chosen word for this group (long-term counts), ties keep RIME order
  bigram   : most-chosen after the previous word if seen ≥ min_bi times, else unigram
  guarded  : only move a candidate above RIME's first when its score beats the first's
             by a margin (keeps RIME's order when the evidence is thin)

Usage: sim_stable_order.py [prior-export.txt]
"""
import math, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intents import DATA, load_intents

prior_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "userdb_export", "rime_frost_prelog.txt")
prior = Counter()
for line in open(prior_path, encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3 and parts[2].isdigit():
        prior[parts[0]] += int(parts[2])

intents = load_intents()

def make_policy(kind, min_bi=1, margin=0.0, alpha=1.0):
    def run(items_ids):
        uni = Counter(prior); bi = defaultdict(Counter)
        ok = 0; out = {}
        for it in intents:
            cands, group = it["cands"], it["group"]
            choice = cands[0]
            if kind != "rime" and len(group) > 1:
                g = [cands[k] for k in group]
                def uscore(w): return math.log(uni[w] + alpha)
                def bscore(w):
                    b = bi[it["prev"]] if it["prev"] else Counter()
                    tot = sum(b.values())
                    if tot >= min_bi and b[w]:
                        return 10 + math.log(b[w] + alpha) - math.log(tot + alpha)
                    return uscore(w)
                f = uscore if kind == "unigram" else bscore
                best = max(range(len(g)), key=lambda j: (f(g[j]), -j))
                if kind == "guarded":
                    if f(g[best]) - f(g[0]) <= margin: best = 0
                choice = g[best]
            out[it["i"]] = choice
            if it["i"] in items_ids: ok += choice == it["gold"]
            uni[it["gold"]] += 1
            if it["prev"]: bi[it["prev"]][it["gold"]] += 1
        return ok / max(1, len(items_ids)), out
    return run

half = len(intents) // 2
tune = {it["i"] for it in intents[:half]}; test = {it["i"] for it in intents[half:]}
print(f"{len(intents)} intents; prior entries {len(prior)}; tune first {len(tune)}, test last {len(test)}\n")

rime_acc, rime_out = make_policy("rime")(test)
rows = [("RIME 现状", rime_acc, rime_out)]
for name, kind in [("长期计数", "unigram"), ("长期计数+上一个词", "bigram")]:
    a, o = make_policy(kind)(test); rows.append((name, a, o))
# guarded: tune margin and min_bi on the first half
grid = [(m, b) for m in (0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0) for b in (1, 2, 3)]
m, b = max(grid, key=lambda mb: make_policy("guarded", min_bi=mb[1], margin=mb[0])(tune)[0])
a, o = make_policy("guarded", min_bi=b, margin=m)(test)
rows.append((f"带门槛(margin={m}, min_bi={b})", a, o))

print(f"{'策略':<26}{'首选正确':>8}{'需纠正':>8}{'纠正减少':>9}")
for name, a, _ in rows:
    print(f"{name:<24}{a:>10.1%}{1-a:>8.1%}{((1-rime_acc)-(1-a))/(1-rime_acc):>9.0%}")

# where the best policy fixes / breaks, and remaining ping-pong
best_name, best_acc, best_out = max(rows[1:], key=lambda r: r[1])
by_i = {it["i"]: it for it in intents}
fixed = [f"{by_i[i]['pinyin']}:{rime_out[i]}→{best_out[i]}" for i in sorted(test) if rime_out[i] != by_i[i]["gold"] == best_out[i]]
broke = [f"{by_i[i]['pinyin']}:{rime_out[i]}→{best_out[i]}(要{by_i[i]['gold']})" for i in sorted(test) if rime_out[i] == by_i[i]["gold"] != best_out[i]]
print(f"\n最好的是「{best_name}」: 改对 {len(fixed)}, 改错 {len(broke)}")
print("改对:", ", ".join(fixed[:30]))
print("改错:", ", ".join(broke[:30]))

def pingpong(out):
    last, n = {}, 0
    for it in intents:
        if it["i"] in test and out[it["i"]] != it["gold"] and last.get(it["pinyin"]) == out[it["i"]]: n += 1
        last[it["pinyin"]] = it["gold"]
    return n
print(f"\n来回翻转造成的错误(测试半段): RIME {pingpong(rime_out)} → {best_name} {pingpong(best_out)}")
