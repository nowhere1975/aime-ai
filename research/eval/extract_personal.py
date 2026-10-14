"""Collect text the user typed themselves (Codex + Claude Code prompts) into research/data/personal.json.

Keeps only top-level user turns (no subagent threads, no injected context blocks, no pastes),
strips code/URLs/paths, and keeps messages with enough Han characters.
Output stays local in the eval directory.
"""
import glob, json, os, re, sys
from datetime import datetime

HOME = os.path.expanduser("~")
HAN = re.compile(r"[一-鿿]")
out, seen = [], {}  # seen: text -> index in out (keep the earliest timestamp)

def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"\[(Pasted text|Image)[^\]]*\]", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(~|/)[\w.\-/]+", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()

def add(src, t, ts=None):
    """ts: epoch seconds when the user typed it (None if unknown)."""
    t = clean(t)
    han = len(HAN.findall(t))
    if han < 6 or han / max(1, len(t)) < 0.3: return
    if t in seen:
        o = out[seen[t]]
        if ts is not None and (o["ts"] is None or ts < o["ts"]): o["ts"] = ts
        return
    seen[t] = len(out); out.append({"src": src, "text": t, "ts": ts})

def iso_ts(s):
    try: return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception: return None

# Claude Code prompt history: `display` is what was typed; pastes live separately in pastedContents.
for path in [f"{HOME}/.claude/history.jsonl"]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        try: d = json.loads(line)
        except Exception: continue
        t = d.get("display") or ""
        if t.startswith("/"): continue
        add("claude", t, (d.get("timestamp") or 0) / 1000 or None)

# Codex sessions: only threads the user started, only their user turns, skip injected blocks.
INJECTED = ("<", "# AGENTS.md", "# Context from my IDE", "## My request for Codex")
def open_session(path):
    """Codex compresses archived sessions as .jsonl.zst."""
    if path.endswith(".zst"):
        import io, zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")

session_files = [p for pat in ("sessions/**/*.jsonl", "sessions/**/*.jsonl.zst", "archived_sessions/*.jsonl", "archived_sessions/*.jsonl.zst")
                 for p in glob.glob(f"{HOME}/.codex/{pat}", recursive=True)]
for path in session_files:
    with open_session(path) as f:
        try: meta = json.loads(f.readline()).get("payload", {})
        except Exception: continue
        if isinstance(meta.get("source"), dict) or meta.get("thread_source") not in (None, "user"):
            continue
        for line in f:
            if '"role":"user"' not in line: continue
            try: d = json.loads(line); p = d["payload"]
            except Exception: continue
            for c in p.get("content") or []:
                t = c.get("text") or ""
                if c.get("type") == "input_text" and not t.lstrip().startswith(INJECTED):
                    add("codex", t, iso_ts(d.get("timestamp", "")))

for line in open(f"{HOME}/.codex/history.jsonl", encoding="utf-8", errors="ignore"):
    try: d = json.loads(line); add("codex", d.get("text") or "", d.get("ts"))
    except Exception: pass

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "personal.json"), "w"), ensure_ascii=False)
from collections import Counter
print(len(out), Counter(o["src"] for o in out), "han chars:", sum(len(HAN.findall(o["text"])) for o in out))
