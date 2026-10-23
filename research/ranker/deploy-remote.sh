#!/bin/bash
# Deploy the ranking service (and optionally this branch's AIME build) to another Mac over ssh,
# with the same layout as here: ~/mycode/aime-ai/research on the remote.
#   deploy-remote.sh user@host          code, venv, the live tinylm model, personal history; start the agent
#   deploy-remote.sh user@host --app    also install clt-build/dist/AIME.app (asks for the remote sudo password)
# The remote needs Homebrew python3.12 and key-based ssh. Typing log and ranker are switched on there.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/.."
HOST="${1:?usage: deploy-remote.sh user@host [--app]}"
DEST="mycode/aime-ai/research"
LIVE="$(readlink "$R/data/tinylm/live")"   # e.g. ft500

echo "==> code and model ($LIVE) -> $HOST:~/$DEST"
ssh "$HOST" "mkdir -p ~/$DEST/data/tinylm/$LIVE ~/Library/AIME/ranker ~/Library/AIME/typing-log \
    && chmod 700 ~/Library/AIME/ranker ~/Library/AIME/typing-log"
rsync -a --exclude __pycache__ "$R/ranker" "$R/tinylm" "$R/requirements.txt" "$HOST:$DEST/"
# only what the server loads, not the training checkpoints
rsync -a "$R/data/tinylm/$LIVE/"{config.json,model.safetensors,vocab.json} "$HOST:$DEST/data/tinylm/$LIVE/"

echo "==> venv"
# --timeout: a stalled download otherwise hangs pip indefinitely
ssh "$HOST" "cd ~/$DEST && ln -sfn $LIVE data/tinylm/live \
    && { [[ -d .venv ]] || /opt/homebrew/bin/python3.12 -m venv .venv; } \
    && .venv/bin/pip install -q --timeout 30 --retries 5 --progress-bar off -r requirements.txt \
    && .venv/bin/python -c 'import mlx.core as mx; print(\"mlx\", mx.__version__, mx.default_device())'"

echo "==> personal history, switches, agent"
# copied only when the remote has none: once in use, the remote's own history is newer
ssh "$HOST" "test -e ~/Library/AIME/ranker/bigrams.json" \
    || rsync -a "$HOME/Library/AIME/ranker/bigrams.json" "$HOST:Library/AIME/ranker/"
ssh "$HOST" "touch ~/Library/AIME/ranker/ENABLED ~/Library/AIME/typing-log/ENABLED \
    && bash ~/$DEST/ranker/install-agent.sh && sleep 5 && tail -3 ~/Library/AIME/ranker/server.log"

if [[ "${2:-}" == "--app" ]]; then
    APP="$R/clt-build/dist/AIME.app"
    [[ -d "$APP" ]] || { echo "build it first: bash research/clt-build/build.sh"; exit 1; }
    echo "==> AIME.app (previous one kept as ~/aime-deploy/AIME-previous.app)"
    ssh "$HOST" "mkdir -p ~/aime-deploy"
    rsync -a --delete "$APP" "$HOST:aime-deploy/"
    ssh -t "$HOST" "P='/Library/Input Methods/AIME.app'; \
        { [[ -d \"\$P\" ]] && rm -rf ~/aime-deploy/AIME-previous.app && ditto \"\$P\" ~/aime-deploy/AIME-previous.app || true; } \
        && sudo sh -c \"rm -rf '\$P' && ditto \$HOME/aime-deploy/AIME.app '\$P' && chown -R root:wheel '\$P'\" \
        && pkill -f 'Input Methods/AIME.app/Contents/MacOS/AIME'; \
        codesign --verify --strict \"\$P\" && echo \"installed build \$(/usr/libexec/PlistBuddy -c 'Print AIMEResearchBuild' \"\$P/Contents/Info.plist\")\""
fi
echo "done. latency check on $HOST (needs a typing-log day there): .venv/bin/python ranker/bench_client.py"
