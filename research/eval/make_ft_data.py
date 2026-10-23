"""Build a fine-tuning set from the user's own text typed BEFORE the typing log started.

Sources: data/personal.json (+ personal_chat.json if present), each item {text, ts}.
Anything at or after the first typing-log event is dropped, so replaying the log
afterwards never tests on text the model was trained on.
Writes data/ft/{train,valid}.jsonl as {"text": ...} (mlx_lm lora format).
"""
import glob, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
first_log = None
for f in sorted(glob.glob(os.path.expanduser("~/Library/AIME/typing-log/*.jsonl")))[:1]:
    first_log = json.loads(open(f).readline())["t"] / 1000
assert first_log, "no typing log found"

items = []
for name in ("personal.json", "personal_chat.json"):
    path = os.path.join(DATA, name)
    if os.path.exists(path): items += json.load(open(path))
before = [o for o in items if o.get("ts") and o["ts"] < first_log]
random.Random(0).shuffle(before)
n_valid = max(50, len(before) // 20)
os.makedirs(os.path.join(DATA, "ft"), exist_ok=True)
for name, part in (("valid", before[:n_valid]), ("train", before[n_valid:])):
    with open(os.path.join(DATA, "ft", f"{name}.jsonl"), "w") as f:
        for o in part: f.write(json.dumps({"text": o["text"]}, ensure_ascii=False) + "\n")
print(f"{len(items)} items, {len(before)} before the log -> train {len(before)-n_valid}, valid {n_valid}; "
      f"chars {sum(len(o['text']) for o in before)}; dropped {len(items)-len(before)} at/after log start")
