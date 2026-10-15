#!/bin/bash
# Start the local ranking service for AIME (foreground; Ctrl-C to stop).
# Uses the fused fine-tuned model if present, else the base model.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HERE/../data/ft/fused_q06b8"
[[ -d "$MODEL" ]] || MODEL="mlx-community/Qwen3-0.6B-8bit"
export HF_HUB_OFFLINE=1
exec "$HERE/../.venv/bin/python" "$HERE/server.py" --model "$MODEL" "$@"
