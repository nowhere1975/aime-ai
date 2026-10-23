#!/bin/bash
# Build the AIME input method with the Command Line Tools only (no Xcode / XcodeGen).
#
# 1. librime prebuilt via scripts/fetch-librime.sh; its `xcodebuild -create-xcframework`
#    step needs Xcode, so the xcframework is assembled by hand when that fails.
# 2. SwiftPM builds the input-method executable (this package wraps Apps/AIME).
# 3. The official release bundle of the same version supplies resources, Settings app,
#    helpers and librime; only Contents/MacOS/AIME is replaced, then ad-hoc signed.
# Output: research/clt-build/dist/AIME.app (arm64 only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
XCF="$ROOT/Packages/AIMEKit/Frameworks/CRime.xcframework"
VERSION="$(sed -n 's/^ *MARKETING_VERSION: "\(.*\)"/\1/p' "$ROOT/project.yml" | head -1)"

if [[ ! -f "$XCF/Info.plist" ]]; then
  bash "$ROOT/scripts/fetch-librime.sh" || true
  if [[ ! -f "$XCF/Info.plist" ]]; then
    echo "==> assembling CRime.xcframework by hand"
    LIB="$XCF/macos-arm64_x86_64"
    mkdir -p "$LIB"
    cp "$ROOT/Vendor/librime/lib/librime.1.dylib" "$LIB/"
    rm -rf "$LIB/Headers"; cp -R "$ROOT/Vendor/.stage/headers" "$LIB/Headers"
    cp "$HERE/CRime.Info.plist" "$XCF/Info.plist"
    rm -rf "$ROOT/Vendor/.stage"
    grep LIBRIME_COMMIT "$ROOT/Vendor/librime.lock" | cut -d= -f2 > "$ROOT/Vendor/librime/.stamp"
  fi
fi

echo "==> swift build"
swift build --package-path "$HERE" -c release 2>&1 | grep -E "error:|Build complete" || true
BIN="$HERE/.build/release/AIME"
[[ -x "$BIN" ]] || { echo "build failed"; exit 1; }

REL="$HERE/.release/$VERSION"
if [[ ! -d "$REL/AIME.app" ]]; then
  echo "==> fetching official AIME $VERSION bundle"
  mkdir -p "$REL"
  gh release download "v$VERSION" -R zoolapp/aime -p "AIME-$VERSION.zip" -p SHA256SUMS.txt -D "$REL" --clobber
  (cd "$REL" && shasum -a 256 -c SHA256SUMS.txt --ignore-missing)
  ditto -x -k "$REL/AIME-$VERSION.zip" "$REL"
fi

echo "==> assembling dist/AIME.app"
rm -rf "$HERE/dist"; mkdir -p "$HERE/dist"
APP="$HERE/dist/AIME.app"
ditto "$REL/AIME.app" "$APP"
cp "$BIN" "$APP/Contents/MacOS/AIME"
# MENU_ICON=bookmark: the input-menu icon from before 0.1.6 (upstream switched it to 「艾」)
[[ "${MENU_ICON:-}" == bookmark ]] && cp "$HERE/AIME-bookmark.pdf" "$APP/Contents/Resources/AIME.pdf"
/usr/libexec/PlistBuddy -c "Add :AIMEResearchBuild string $(git -C "$ROOT" rev-parse --short HEAD)" "$APP/Contents/Info.plist"
xattr -cr "$APP"
codesign --force --sign - --entitlements "$ROOT/Apps/AIME/AIME.entitlements" "$APP"
codesign --verify --strict "$APP"
echo "==> done: $APP"
echo "install (needs admin, run in Terminal.app):"
echo "  sudo rm -rf '/Library/Input Methods/AIME.app' && sudo ditto '$APP' '/Library/Input Methods/AIME.app' \\"
echo "    && sudo chown -R root:wheel '/Library/Input Methods/AIME.app' && pkill -f 'Input Methods/AIME.app/Contents/MacOS/AIME'"
