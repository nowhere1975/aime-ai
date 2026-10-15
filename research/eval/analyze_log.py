"""Typing-process metrics from AIME's typing log (~/Library/AIME/typing-log/*.jsonl).

Per commit (Chinese words only) it reconstructs what the user had to do:
  first    : the committed text was the first candidate on page 0 of the last shown list
  picked   : it was on the list but not first (number key / arrows / click)  -> 选错候选
  paged    : the list was on page > 0 when committed (or any page key was used) -> 翻页
  pinyin_bs: ⌫ pressed inside the composition (editing the pinyin)
  post_bs  : ⌫ right after the commit, before the next composition       -> 上屏错误
Also exports replay cases (context, candidate list, chosen text) for reranking eval.
"""
import glob, json, os, re, sys
from collections import Counter

HAN = re.compile(r"[一-鿿]")
files = sorted(glob.glob(os.path.expanduser("~/Library/AIME/typing-log/*.jsonl")))
events = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]

commits, cur, last_commit = [], None, None
prev_text, prev_t = "", 0  # previous commit of any kind (to tell where a clause breaks)
history = ""  # committed text, the fallback context where the app exposes none (terminals)
for e in events:
    ev = e["ev"]
    if ev == "start":
        cur = dict(app=e["app"], ctx=e["ctx"], snaps=[], keys=[], paged=False, pinyin_bs=0)
    elif ev == "key":
        if cur is None:
            cur = dict(app="?", ctx=None, snaps=[], keys=[], paged=False, pinyin_bs=0)
        cur["keys"].append(e["k"])
        if e.get("c0") and e["k"] in ("bs", "del"): cur["pinyin_bs"] += 1
        if e["k"] in ("-", "=", ",", ".", "pgup", "pgdn", "[", "]") and e.get("c0") and e.get("pg", 0) > 0:
            cur["paged"] = True
        if "cands" in e:
            cur["snaps"].append(e)
            if e.get("pg", 0) > 0: cur["paged"] = True
    elif ev == "pageclick" and cur:
        cur["paged"] = True
    elif ev == "commit":
        text = e["text"]
        if cur and cur["snaps"] and HAN.search(text):
            s = cur["snaps"][-1]
            cands, page = s["cands"], s.get("pg", 0)
            idx = cands.index(text) if text in cands else None
            first_idx = cur["snaps"][-1]["cands"][0] if cands else None
            rk = s.get("rk")  # the ranker moved the highlight here (see Apps/AIME/Ranker.swift)
            shown = rk if rk is not None else 0  # what Space would have committed
            rec = dict(app=cur["app"], ctx=cur["ctx"] if cur["ctx"] else history[-20:], ctx_src="app" if cur["ctx"] else "history",
                       pinyin=s["in"], cands=cands, page=page, text=text, idx=idx, rk=rk,
                       first=(page == 0 and idx == shown), paged=cur["paged"] or page > 0,
                       pinyin_bs=cur["pinyin_bs"], post_bs=0, nkeys=len(cur["keys"]),
                       t=e["t"], gap_ms=e["t"] - prev_t, after_nonhan=not HAN.search(prev_text or "x"))
            rec["picked"] = not rec["first"] and not rec["paged"]
            commits.append(rec); last_commit = rec
        else:
            last_commit = None
        history += text
        prev_text, prev_t = text, e["t"]
        cur = None
    elif ev == "post" and e["k"] in ("bs", "del") and last_commit is not None:
        last_commit["post_bs"] += 1

n = len(commits)
if not n:
    sys.exit("no Chinese commits yet")
chars = sum(len(HAN.findall(c["text"])) for c in commits)
per100 = lambda k: 100 * sum(1 for c in commits if c[k]) / chars
print(f"{len(files)} day(s), {n} Chinese commits, {chars} chars, avg {chars/n:.1f} chars/commit")
print(f"首选命中            {sum(c['first'] for c in commits)/n:.1%}")
print(f"选非首选 (选错候选) {sum(c['picked'] for c in commits)/n:.1%}   = {per100('picked'):.1f} 次/百字")
print(f"翻页               {sum(c['paged'] for c in commits)/n:.1%}   = {per100('paged'):.1f} 次/百字")
print(f"上屏后退格          {sum(1 for c in commits if c['post_bs'])/n:.1%}   = {per100('post_bs'):.1f} 次/百字")
print(f"改拼音(组字中退格)  {sum(1 for c in commits if c['pinyin_bs'])/n:.1%}")
ranked = [c for c in commits if c.get("rk") is not None]
if ranked:
    helped = [c for c in ranked if c["idx"] == c["rk"]]
    hurt = [c for c in ranked if c["idx"] == 0]
    print(f"模型调整高亮 {len(ranked)} 次：帮上忙 {len(helped)}（不调整就要手选），帮倒忙 {len(hurt)}（本来首选就对），"
          f"其他 {len(ranked)-len(helped)-len(hurt)}")
    print("  帮上忙:", ", ".join(f"{c['pinyin']}:{c['cands'][0]}→{c['text']}" for c in helped[:20]))
    print("  帮倒忙:", ", ".join(f"{c['pinyin']}:{c['cands'][c['rk']]}✗{c['text']}" for c in hurt[:20]))
print("上下文来源:", Counter(c["ctx_src"] for c in commits), " apps:", Counter(c["app"] for c in commits).most_common(5))
print("选非首选的例子:", ", ".join(f"{c['pinyin']}:{c['cands'][0]}→{c['text']}(#{c['idx']+1 if c['idx'] is not None else '?'})" for c in commits if c["picked"])[:600])
json.dump(commits, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "replay_cases.json"), "w"), ensure_ascii=False)
