#!/bin/bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/voiceflow-troubleshooting.XXXXXX")"
trap 'rm -rf "$TEST_DIR"' EXIT
swiftc -parse-as-library \
    "$APP_ROOT/Sources/VoiceFlowApp/DictationDiagnostics.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/ManagedProcessShutdown.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/ParakeetASREngine.swift" \
    "$APP_ROOT/Sources/VoiceFlowApp/ChildProcessWatchdog.swift" \
    "$APP_ROOT/Tests/test_troubleshooting.swift" -o "$TEST_DIR/checks"
"$TEST_DIR/checks"
