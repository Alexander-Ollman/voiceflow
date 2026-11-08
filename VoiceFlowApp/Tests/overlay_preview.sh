#!/bin/bash
# Live preview of the voice overlay without building the full app (no Rust).
# Usage: bash VoiceFlowApp/Tests/overlay_preview.sh [--mic]
set -euo pipefail
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$APP_ROOT/Sources/VoiceFlowApp"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/voiceflow-overlay.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
swiftc -O -parse-as-library -target arm64-apple-macos15.0 \
    "$SRC/VFDesign.swift" "$SRC/RecordingWaveformView.swift" \
    "$SRC/LiquidWaveformView.swift" "$SRC/RecordingOverlay.swift" "$SRC/IslandOverlay.swift" \
    "$APP_ROOT/Tests/overlay_preview.swift" -o "$OUT/overlay-preview"
echo "Overlay preview running — Ctrl-C to quit."
"$OUT/overlay-preview" "$@"
