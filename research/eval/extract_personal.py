"""Collect text the user typed themselves (Codex + Claude Code prompts) into research/data/personal.json.

Keeps only top-level user turns (no subagent threads, no injected context blocks, no pastes),
strips code/URLs/paths, and keeps messages with enough Han characters.
Output stays local in the eval directory.
"""
import glob, json, os, re, sys

HOME = os.path.expanduser("~")
HAN = re.compile(r"[一-鿿]")
out, seen = [], set()

def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"\[(Pasted text|Image)[^\]]*\]", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(~|/)[\w.\-/]+", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()

def add(src, t):
    t = clean(t)
    han = len(HAN.findall(t))
    if han < 6 or han / max(1, len(t)) < 0.3 or t in seen: return
    seen.add(t); out.append({"src": src, "text": t})

# Claude Code prompt history: `display` is what was typed; pastes live separately in pastedContents.
for path in [f"{HOME}/.claude/history.jsonl"]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        try: d = json.loads(line)
        except Exception: continue
        t = d.get("display") or ""
        if t.startswith("/"): continue
        add("claude", t)

# Codex sessions: only threads the user started, only their user turns, skip injected blocks.
INJECTED = ("<", "# AGENTS.md", "# Context from my IDE", "## My request for Codex")
for path in glob.glob(f"{HOME}/.codex/sessions/**/*.jsonl", recursive=True) + glob.glob(f"{HOME}/.codex/archived_sessions/*.jsonl"):
    with open(path, encoding="utf-8", errors="ignore") as f:
        try: meta = json.loads(f.readline()).get("payload", {})
        except Exception: continue
        if isinstance(meta.get("source"), dict) or meta.get("thread_source") not in (None, "user"):
            continue
        for line in f:
            if '"role":"user"' not in line: continue
            try: p = json.loads(line)["payload"]
            except Exception: continue
            for c in p.get("content") or []:
                t = c.get("text") or ""
                if c.get("type") == "input_text" and not t.lstrip().startswith(INJECTED):
                    add("codex", t)

for line in open(f"{HOME}/.codex/history.jsonl", encoding="utf-8", errors="ignore"):
    try: add("codex", json.loads(line).get("text") or "")
    except Exception: pass

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "personal.json"), "w"), ensure_ascii=False)
from collections import Counter
print(len(out), Counter(o["src"] for o in out), "han chars:", sum(len(HAN.findall(o["text"])) for o in out))
