"""Replay the latest typing-log day against the running ranker and report latency.

Sends a rank request for every composing keystroke and a commit message for every commit,
as AIME would, so the service's commit-time prefill is exercised too.
"""
import glob, json, os, socket, time
s = socket.socket(socket.AF_UNIX); s.connect(os.path.expanduser("~/Library/AIME/ranker/ranker.sock")); f = s.makefile("rb")
path = sorted(glob.glob(os.path.expanduser("~/Library/AIME/typing-log/*.jsonl")))[-1]
ctx, prev, rtts, moved, last_t = "", "", [], 0, None
slow = []
for line in open(path):
    e = json.loads(line)
    if e["ev"] == "start": ctx = e.get("ctx") or ""
    elif e["ev"] == "commit":
        s.sendall((json.dumps({"op": "commit", "prev": prev, "text": e["text"]}, ensure_ascii=False) + "\n").encode())
        prev = e["text"]
        time.sleep(0.05)  # the user needs at least this long before the next key
    elif e["ev"] == "key" and e.get("cands") and len(e["cands"]) > 1:
        t = time.perf_counter()
        s.sendall((json.dumps({"op": "rank", "ctx": ctx, "prev": prev, "cands": e["cands"]}, ensure_ascii=False) + "\n").encode())
        r = json.loads(f.readline()); rt = (time.perf_counter() - t) * 1000; rtts.append(rt); moved += r["i"] != 0
        if rt > 40: slow.append((round(rt), r.get("miss"), len(e["cands"]), max(map(len, e["cands"])), e["in"]))
    if len(rtts) >= 600: break
rtts.sort(); n = len(rtts)
print(f"{n} keystrokes: round-trip p50 {rtts[n//2]:.1f}ms p90 {rtts[int(n*.9)]:.1f}ms p99 {rtts[int(n*.99)]:.1f}ms "
      f"max {rtts[-1]:.1f}ms; >40ms {sum(r > 40 for r in rtts)/n:.1%}; highlight moved {moved/n:.1%}")
print(f"slow: {len(slow)}, context miss among them: {sum(1 for x in slow if x[1])}")
for x in slow[:12]: print("  ", x)
