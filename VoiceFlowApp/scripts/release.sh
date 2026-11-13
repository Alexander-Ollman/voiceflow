#!/bin/bash
# Build, verify and ship a VoiceFlow release to GitHub.
#
#   NOTARIZE_KEYCHAIN_PROFILE=<profile> VoiceFlowApp/scripts/release.sh 3.0.0            # build, verify, upload to the draft
#   NOTARIZE_KEYCHAIN_PROFILE=<profile> VoiceFlowApp/scripts/release.sh 3.0.0 --publish  # ...then publish as Latest
#   ... release.sh 3.0.0 --reuse-build   # re-verify and upload an already-built, notarized dist/
#
# Requirements (checked before anything is built):
#   - clean checkout of main, up to date with origin
#   - build.sh's bundle version equals <version>
#   - "Developer ID Application: Era Laboratories Inc. (JVSQ3LCY64)" in the keychain
#   - NOTARIZE_KEYCHAIN_PROFILE (preferred) or NOTARIZE_APPLE_ID
#   - a draft GitHub release v<version> (release notes live there)
#
# Installed copies update from the latest release's VoiceFlow.zip and reject
# anything not signed by team JVSQ3LCY64; the website's Download button uses
# VoiceFlow.dmg. Nothing is uploaded unless both pass Gatekeeper as notarized.
set -euo pipefail

VERSION="${1:-}"; shift || true
PUBLISH=""; REUSE_BUILD=""
for arg in "$@"; do
    case "$arg" in
        --publish) PUBLISH="--publish" ;;
        --reuse-build) REUSE_BUILD=1 ;;   # verify + upload the existing dist/ artifacts, no rebuild
        *) echo "unknown option: $arg"; exit 64 ;;
    esac
done
[ -n "$VERSION" ] || { echo "usage: $0 <version> [--reuse-build] [--publish]"; exit 64; }
TAG="v$VERSION"
TEAM_ID="JVSQ3LCY64"
IDENTITY="Developer ID Application: Era Laboratories Inc. ($TEAM_ID)"
REPO="Alexander-Ollman/voiceflow"
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(dirname "$APP_ROOT")"
APP="$APP_ROOT/build/VoiceFlow.app"
ZIP="$APP_ROOT/dist/VoiceFlow.zip"
DMG="$APP_ROOT/dist/VoiceFlow.dmg"

fail() { echo "✗ $*" >&2; exit 1; }
ok()   { echo "✓ $*"; }

echo "== Preflight for $TAG"
cd "$PROJECT_ROOT"
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || fail "not on main"
[ -z "$(git status --porcelain)" ] || fail "working tree has uncommitted changes"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || fail "main is not in sync with origin/main"
ok "clean main at $(git rev-parse --short HEAD)"

PLIST_VERSION="$(grep -A1 'CFBundleShortVersionString' "$APP_ROOT/build.sh" | tail -1 | sed -E 's/.*<string>(.*)<\/string>.*/\1/')"
[ "$PLIST_VERSION" = "$VERSION" ] || fail "build.sh bundle version is $PLIST_VERSION, expected $VERSION"
ok "bundle version $VERSION"

identities="$(security find-identity -v -p codesigning || true)"
[[ "$identities" == *"$IDENTITY"* ]] || fail "missing signing identity: $IDENTITY"
ok "Developer ID present"

if [ -n "${NOTARIZE_KEYCHAIN_PROFILE:-}" ]; then
    xcrun notarytool history --keychain-profile "$NOTARIZE_KEYCHAIN_PROFILE" >/dev/null 2>&1 \
        || fail "notarytool can't use keychain profile '$NOTARIZE_KEYCHAIN_PROFILE'"
    ok "notarization profile '$NOTARIZE_KEYCHAIN_PROFILE'"
elif [ -n "${NOTARIZE_APPLE_ID:-}" ]; then
    ok "notarization via Apple ID $NOTARIZE_APPLE_ID (notarytool will ask for the app-specific password)"
else
    fail "set NOTARIZE_KEYCHAIN_PROFILE or NOTARIZE_APPLE_ID"
fi

gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
[ "$(gh release view "$TAG" -R "$REPO" --json isDraft -q .isDraft 2>/dev/null)" = "true" ] \
    || fail "no draft release $TAG on $REPO (create it with the release notes first)"
ok "draft release $TAG exists"

if [ -n "$REUSE_BUILD" ]; then
    echo "== Reusing existing build (no rebuild); every check below still applies"
    [ -d "$APP" ] && [ -f "$ZIP" ] && [ -f "$DMG" ] || fail "no existing build to reuse"
else
    echo "== Building (build.sh signs, notarizes and staples the app and disk image)"
    bash "$APP_ROOT/build.sh"
fi

echo "== Verifying"
[ "$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")" = "$VERSION" ] \
    || fail "built app reports a different version"
codesign --verify --deep --strict "$APP" || fail "app signature invalid"
# Capture output before matching: with pipefail, `cmd | grep -q` fails when
# grep exits early on a match and cmd gets SIGPIPE.
for artifact in "$APP" "$DMG"; do
    info="$(codesign -dv "$artifact" 2>&1 || true)"
    [[ "$info" == *"TeamIdentifier=$TEAM_ID"* ]] || fail "$(basename "$artifact") not signed by $TEAM_ID"
    xcrun stapler validate "$artifact" >/dev/null || fail "$(basename "$artifact") has no stapled notarization ticket"
done
gatekeeper="$(spctl -a -vv -t exec "$APP" 2>&1 || true)"
[[ "$gatekeeper" == *"Notarized Developer ID"* ]] || fail "Gatekeeper does not accept the app as notarized"
gatekeeper="$(spctl -a -vv -t open --context context:primary-signature "$DMG" 2>&1 || true)"
[[ "$gatekeeper" == *"Notarized Developer ID"* ]] || fail "Gatekeeper does not accept the disk image as notarized"
# The zip the updater downloads must contain the same stapled app.
UNZIP_DIR="$(mktemp -d)"; trap 'rm -rf "$UNZIP_DIR"' EXIT
ditto -x -k "$ZIP" "$UNZIP_DIR"
xcrun stapler validate "$UNZIP_DIR/VoiceFlow.app" >/dev/null || fail "app inside VoiceFlow.zip is not stapled"
ok "app, zip and disk image are signed by $TEAM_ID, notarized and stapled"

echo "== Uploading to draft $TAG"
gh release upload "$TAG" "$ZIP" "$DMG" -R "$REPO" --clobber
ok "uploaded VoiceFlow.zip and VoiceFlow.dmg"

if [ "$PUBLISH" = "--publish" ]; then
    gh release edit "$TAG" -R "$REPO" --draft=false --latest --target main
    ok "published $TAG as Latest: https://github.com/$REPO/releases/tag/$TAG"
else
    echo "Draft is ready. Review it, then publish with:"
    echo "  gh release edit $TAG -R $REPO --draft=false --latest --target main"
fi
