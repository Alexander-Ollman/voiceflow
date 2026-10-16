#!/bin/bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/voiceflow-pr-tests.XXXXXX")"
trap 'rm -rf "$TEST_DIR"' EXIT
swiftc -parse-as-library \
    "$APP_ROOT/Sources/VoiceFlowApp/GlobeKey.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/HotkeySettings.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/AppBundleInstaller.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/RecordingWaveformView.swift" \
    "$APP_ROOT/Tests/test_pr_update.swift" \
    -o "$TEST_DIR/pr-update-tests"
"$TEST_DIR/pr-update-tests"
