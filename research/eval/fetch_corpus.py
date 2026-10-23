"""Download a small public Chinese test corpus into research/data/corpus.json.

News (THUCNews) plus human-written answers from HC3-Chinese (open_qa, psychology, baike),
fetched through the Hugging Face datasets-server rows API (no full dataset download).
"""
import json, os, random
from collections import Counter
import httpx

random.seed(7)
c = httpx.Client(timeout=60)

def rows(ds, config, offset, n):
    r = c.get("https://datasets-server.huggingface.co/rows",
              params=dict(dataset=ds, config=config, split="train", offset=offset, length=n))
    r.raise_for_status()
    return [x["row"] for x in r.json()["rows"]]

out = []
for off in random.sample(range(0, 50000), 6):
    for r in rows("SirlyDreamer/THUCNews", "default", off, 5):
        t = r.get("text") or ""
        if len(t) > 200: out.append({"src": "news", "text": t[:800]})
for cfg in ["open_qa", "psychology", "baike"]:
    for off in random.sample(range(0, 1000), 3):
        for r in rows("Hello-SimpleAI/HC3-Chinese", cfg, off, 5):
            for h in r.get("human_answers") or []:
                if len(h) > 80: out.append({"src": f"hc3-{cfg}", "text": h[:800]})

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus.json")
json.dump(out, open(dest, "w"), ensure_ascii=False)
print(len(out), Counter(o["src"] for o in out))
