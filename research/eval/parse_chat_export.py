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
for part in re.split(r"^\*\*\[\d\d:\d\d\]\*\*\s*", text, flags=re.M)[1:]:
    ns["add"]("chat", re.split(r"^## \d{4}-\d\d-\d\d", part, flags=re.M)[0].strip())
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", name)
json.dump(ns["out"], open(dest, "w"), ensure_ascii=False)
print(len(ns["out"]), "messages,", sum(len(ns["HAN"].findall(o["text"])) for o in ns["out"]), "Han chars ->", dest)
