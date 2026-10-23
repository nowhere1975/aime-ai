#!/bin/bash
# Start the local ranking service for AIME (foreground; Ctrl-C to stop).
# Uses research/data/tinylm/live (a tinylm model dir or symlink) if present — λ/μ as tuned by
# replay_log.py --tiny — else the fused fine-tuned Qwen, else the base Qwen.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"
TINY="$HERE/../data/tinylm/live"
if [[ -d "$TINY" ]]; then
    exec "$PY" "$HERE/server.py" --tiny "$TINY" --lam1 3 --lam 4 --mu 2 --cache-mb 64 "$@"
fi
MODEL="$HERE/../data/ft/fused_q06b8"
[[ -d "$MODEL" ]] || MODEL="mlx-community/Qwen3-0.6B-8bit"
export HF_HUB_OFFLINE=1
exec "$PY" "$HERE/server.py" --model "$MODEL" "$@"
