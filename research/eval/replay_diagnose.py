"""Diagnose RIME's first-candidate errors in the typing log.

flip-flop : RIME's first candidate is what the user chose last time for this pinyin, but
            this time the user wanted something else (recency learning flipping the order).
right-ctx : would the LM pick the right candidate if it could also see the *next* word
            (what a revisable draft could do), vs. left context only.
"""
import json, os, re, sys
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_log.py")).read().split("# ---- pick functions")[0])

last_choice = {}
err = flip = 0
for it in intents:
    py, first, gold = it["pinyin"], it["cands"][0], it["gold"]
    if first != gold:
        err += 1
        if last_choice.get(py) == first: flip += 1
    last_choice[py] = gold
print(f"RIME 首选错 {err} 次，其中首选 = 你上次对这个拼音的选择（来回翻转）{flip} 次 ({flip/err:.0%})")

# right context: score cand + next gold word
def score_with_next(it, nxt):
    cands = [it["cands"][k] for k in it["group"]]
    return lm_scores(it["ctx"], [c + nxt for c in cands])
tot = lo = ro = rime = 0
rows = []
for a, b in zip(intents, intents[1:]):
    if len(a["group"]) < 2 or not HAN.match(b["gold"]): continue
    if a["gold"] not in [a["cands"][k] for k in a["group"]]: continue
    if a["cands"][0] == a["gold"] and len(a["cands"][0]) > 1: continue  # focus: single chars + all errors
    tot += 1
    g = [a["cands"][k] for k in a["group"]]
    rime += g[0] == a["gold"]
    lo += g[max(range(len(g)), key=lambda j: a["lm"][j])] == a["gold"]
    sc = score_with_next(a, b["gold"])
    ro += g[max(range(len(g)), key=lambda j: sc[j])] == a["gold"]
    rows.append((g, a["gold"], a["lm"], sc, a["group"]))
print(f"单字词 + 所有首选错的情况 {tot} 个: RIME {rime/tot:.1%} | 模型只看前文 {lo/tot:.1%} | 模型看前文+下一个词 {ro/tot:.1%}")

# fused with RIME rank, half/half split (tune on first half, report second half)
half = len(rows) // 2
def acc(rs, which, lam):
    ok = 0
    for g, gold, l, r, grp in rs:
        v = l if which == "left" else r
        ok += g[max(range(len(g)), key=lambda j: v[j] - lam * math.log(1 + grp[j]))] == gold
    return ok / len(rs)
for which in ("left", "right"):
    lam = max([0, .5, 1, 1.5, 2, 3, 4, 6, 8, 12], key=lambda l: acc(rows[:half], which, l))
    print(f"融合({'只看前文' if which=='left' else '前文+下一个词'}) lam={lam}: 后半 {acc(rows[half:], which, lam):.1%}  vs RIME {acc(rows[half:], which, 1e9):.1%}")
