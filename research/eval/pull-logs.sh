#!/bin/bash
# Copy another Mac's typing log here for analyze_log.py (it reads data/typing-log/<host>/ after
# this Mac's own log). Run before each evaluation; only new or grown days are transferred.
#   pull-logs.sh user@host
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${1:?usage: pull-logs.sh user@host}"
DEST="$HERE/../data/typing-log/${HOST#*@}"
mkdir -p "$DEST" && chmod 700 "$DEST"
rsync -a "$HOST:Library/AIME/typing-log/*.jsonl" "$DEST/"
ls "$DEST" | tail -3
