"""Turn a Markdown export of your own chat messages into an eval corpus.

Expected format (e.g. a Hermes export): lines starting with "**[HH:MM]** text", grouped
under "## YYYY-MM-DD" headings. Usage:
  python parse_chat_export.py export.md personal_chat.json
Writes research/data/<out>; cleaning rules are shared with extract_personal.py.
"""
import json, os, re, sys

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_personal.py")).read()
ns = {}
exec(src.split("# Claude Code prompt history")[0], ns)  # clean() / add() / HAN only

text = open(sys.argv[1], encoding="utf-8").read()
name = sys.argv[2] if len(sys.argv) > 2 else "personal_chat.json"
from datetime import datetime
day = hm = None
for tok in re.split(r"^(## \d{4}-\d\d-\d\d|\*\*\[\d\d:\d\d\]\*\*)", text, flags=re.M)[1:]:
    if tok.startswith("## "): day = tok[3:].strip(); continue
    if tok.startswith("**["): hm = tok[3:8]; continue
    if hm is None: continue
    ts = datetime.strptime(f"{day} {hm}", "%Y-%m-%d %H:%M").timestamp() if day else None  # local time
    ns["add"]("chat", tok.strip(), ts)
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", name)
json.dump(ns["out"], open(dest, "w"), ensure_ascii=False)
print(len(ns["out"]), "messages,", sum(len(ns["HAN"].findall(o["text"])) for o in ns["out"]), "Han chars ->", dest)
