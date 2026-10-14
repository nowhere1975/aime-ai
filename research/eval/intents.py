"""Shared: turn analyze_log.py's replay_cases.json into intents (what the user wanted).

A commit deleted right away (⌫ ≥ its length) and followed by the same pinyin is merged
with that retry: gold = the retry's text, candidates = the first attempt's page.
`group` = indexes of the full-span candidates (Han, same length as RIME's first) that a
reranker may reorder; `prev` = the previous intent's gold text.
"""
import json, os, re

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HAN = re.compile(r"^[一-鿿]+$")

def load_intents(path=os.path.join(DATA, "replay_cases.json")):
    cases = json.load(open(path))
    intents, i = [], 0
    while i < len(cases):
        c = cases[i]
        nxt = cases[i + 1] if i + 1 < len(cases) else None
        if nxt and c["post_bs"] >= len(c["text"]) and nxt["pinyin"] == c["pinyin"]:
            intents.append(dict(c, gold=nxt["text"], redo=True)); i += 2
        else:
            intents.append(dict(c, gold=c["text"], redo=False)); i += 1
    prev = None
    for n, it in enumerate(intents):
        cands = it["cands"]
        it["i"] = n
        it["group"] = [k for k, c in enumerate(cands) if HAN.match(c) and len(c) == len(cands[0])]
        it["prev"] = prev
        prev = it["gold"]
    return intents
