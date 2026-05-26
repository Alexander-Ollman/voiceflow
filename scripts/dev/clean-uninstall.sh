#!/bin/bash
#
# clean-uninstall.sh — wipe VoiceFlow from this machine to simulate a fresh
# external user's first install. Use before each iteration of the
# notarized-DMG → drag-install → onboarding → dictate validation.
#
# Real users only experience this state once (the very first install). Dev
# testing needs to reach it on every iteration; this script is what does
# that. Not shipped to end users.
#
# What gets removed:
#   1. /Applications/VoiceFlow.app
#   2. ~/Library/Application Support/com.era-laboratories.voiceflow/  (models,
#      config, history) — Bonsai-8B-Q1_0.gguf lives here.
#   3. ~/Library/Application Support/VoiceFlow/  (legacy pre-bundle-ID dir,
#      if present)
#   4. ~/Library/Preferences/com.era-laboratories.voiceflow.plist
#   5. ~/Library/Caches/com.era-laboratories.voiceflow/
#   6. ~/Library/HTTPStorages/com.era-laboratories.voiceflow/
#   7. ~/Library/Saved Application State/com.era-laboratories.voiceflow*
#   8. ~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/
#      ← Parakeet weights. NOT under our app dir — HuggingFace's user-wide
#      cache. Skipping this means subsequent install iterations show the
#      Speech model already-installed and skip ~1.2GB of the onboarding flow.
#   9. Launchd registration for the app
#  10. TCC permissions: Accessibility, Microphone, ListenEvent (Input
#      Monitoring), ScreenCapture, PostEvent
#
# Also kills any running VoiceFlow / parakeet-daemon / llama-server processes
# so file replacement and a fresh launch work cleanly.

set -e

BUNDLE_ID="com.era-laboratories.voiceflow"
APP_PATH="/Applications/VoiceFlow.app"
HF_PARAKEET="$HOME/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2"

# Don't `set -u` — some optional paths are intentionally absent.

usage() {
    cat <<USAGE
clean-uninstall.sh — wipe VoiceFlow to simulate a fresh first-time install.

Usage: $(basename "$0") [--keep-hf-cache] [--dry-run]

  --keep-hf-cache   Don't delete the Parakeet HuggingFace cache (~1.2 GB).
                    Faster iterations, but doesn't fully simulate first-ever
                    install.
  --dry-run         Print what would be removed without removing anything.
USAGE
}

KEEP_HF_CACHE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --keep-hf-cache) KEEP_HF_CACHE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

run() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $*"
    else
        eval "$@"
    fi
}

echo "=== Killing running VoiceFlow processes ==="
run "pkill -f 'VoiceFlow\\.app/Contents/MacOS/VoiceFlow' 2>/dev/null || true"
run "pkill -f 'parakeet-daemon' 2>/dev/null || true"
run "pkill -f 'VoiceFlow\\.app/Contents/MacOS/llama-server' 2>/dev/null || true"
# Last-resort: anything still listening on the bundled llama-server port.
if [ "$DRY_RUN" = "0" ]; then
    PORT_PID=$(lsof -ti :8080 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "  killing PID $PORT_PID still on :8080"
        kill -9 "$PORT_PID" 2>/dev/null || true
    fi
fi
sleep 1

echo ""
echo "=== Bootout launchd records ==="
for label in $(launchctl list 2>/dev/null | awk -v id="$BUNDLE_ID" '$3 ~ id {print $3}'); do
    echo "  bootout $label"
    run "launchctl bootout gui/$(id -u)/$label 2>/dev/null || true"
done

echo ""
echo "=== Removing app bundle + user data ==="
for path in \
    "$APP_PATH" \
    "$HOME/Library/Application Support/$BUNDLE_ID" \
    "$HOME/Library/Application Support/VoiceFlow" \
    "$HOME/Library/Preferences/$BUNDLE_ID.plist" \
    "$HOME/Library/Caches/$BUNDLE_ID" \
    "$HOME/Library/HTTPStorages/$BUNDLE_ID"
do
    if [ -e "$path" ]; then
        echo "  rm -rf '$path'"
        run "rm -rf '$path'"
    fi
done

# Saved Application State uses a wildcard.
for path in "$HOME/Library/Saved Application State/$BUNDLE_ID"*; do
    [ -e "$path" ] || continue
    echo "  rm -rf '$path'"
    run "rm -rf '$path'"
done

if [ "$KEEP_HF_CACHE" = "0" ] && [ -e "$HF_PARAKEET" ]; then
    SIZE=$(du -sh "$HF_PARAKEET" 2>/dev/null | awk '{print $1}')
    echo "  rm -rf '$HF_PARAKEET' ($SIZE)"
    run "rm -rf '$HF_PARAKEET'"
elif [ "$KEEP_HF_CACHE" = "1" ]; then
    echo "  keeping HF Parakeet cache (--keep-hf-cache)"
fi

# Force cfprefsd to drop its in-memory copy of the plist, otherwise it can
# rewrite the file we just deleted with stale defaults.
echo ""
echo "=== Killing cfprefsd (so the plist re-reads on next launch) ==="
run "killall cfprefsd 2>/dev/null || true"

echo ""
echo "=== Resetting TCC permissions ==="
for service in Accessibility Microphone ListenEvent ScreenCapture PostEvent; do
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] tccutil reset $service $BUNDLE_ID"
    else
        if tccutil reset "$service" "$BUNDLE_ID" >/dev/null 2>&1; then
            echo "  $service ✓"
        else
            echo "  $service (no prior grant)"
        fi
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete. Re-run without --dry-run to actually wipe."
else
    echo "Wipe complete. The machine is now in a fresh-install state."
    echo "Next steps:"
    echo "  1. Build the DMG:  cd VoiceFlowApp && NOTARIZE_KEYCHAIN_PROFILE=notary ./create-dmg.sh"
    echo "  2. Install:        open VoiceFlowApp/dist/VoiceFlow.dmg  →  drag to /Applications"
    echo "  3. Walk onboarding and verify dictation produces clipboard output."
fi
