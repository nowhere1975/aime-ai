#!/bin/bash
# Prepare the offline eval: build the librime driver against the installed AIME and
# copy the user's Rime data (so the eval never trains the real user dictionary).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
FW="/Library/Input Methods/AIME.app/Contents/Frameworks"
HDR="$ROOT/Packages/AIMEKit/Frameworks/CRime.xcframework/macos-arm64_x86_64/Headers"
[[ -d "$HDR" ]] || { echo "run research/clt-build/build.sh first (it fetches the librime headers)"; exit 1; }
mkdir -p "$HERE/build" "$HERE/../data"
clang -O2 -I"$HDR" "$HERE/rime_cands.c" -o "$HERE/build/rime_cands" "$FW/librime.1.dylib" -Wl,-rpath,"$FW"
rsync -a --delete "$HOME/Library/AIME/Rime/" "$HERE/../data/rime_user/"
echo "ok: $HERE/build/rime_cands, data/rime_user"
