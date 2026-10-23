#!/bin/bash
# Run the ranking service in the background at login (and restart it if it exits),
# instead of keeping run.sh open in a terminal. Per-user LaunchAgent, no sudo.
#   install-agent.sh            install / reinstall and start
#   install-agent.sh uninstall  stop and remove
# Log: ~/Library/AIME/ranker/server.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="app.zool.inputmethod.aime.ranker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/AIME/ranker/server.log"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "uninstall" ]]; then
    rm -f "$PLIST"; echo "removed $LABEL"; exit 0
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PLIST")"
# ProcessType Interactive: a Background job gets throttled CPU/GPU, i.e. slower answers
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>/bin/bash</string><string>$HERE/run.sh</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>30</integer>
    <key>ProcessType</key><string>Interactive</string>
    <key>StandardOutPath</key><string>$LOG</string>
    <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL (log: $LOG)"
