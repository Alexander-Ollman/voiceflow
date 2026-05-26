#!/bin/bash
# Build a PyInstaller-frozen Parakeet ASR daemon, cached for reuse.
#
# Produces an unsigned --onedir bundle at:
#   VoiceFlowApp/build/parakeet-cache/dist/parakeet-daemon/
#
# The main VoiceFlowApp/build.sh copies the contents into
# VoiceFlow.app/Contents/MacOS/parakeet-daemon/ and codesigns every Mach-O
# inside it. We do *not* sign here so this script works in dev environments
# without Era's Developer ID in the keychain.
#
# Cache invalidation: the cache key is the SHA-256 of
# scripts/parakeet_asr_daemon.py plus the pinned dep versions below. If any
# change, the venv is rebuilt and PyInstaller re-runs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DAEMON_SCRIPT="$SCRIPT_DIR/parakeet_asr_daemon.py"
CACHE_DIR="$PROJECT_ROOT/VoiceFlowApp/build/parakeet-cache"
VENV_DIR="$CACHE_DIR/venv"
DIST_DIR="$CACHE_DIR/dist"
WORK_DIR="$CACHE_DIR/build"
SPEC_PATH="$CACHE_DIR/parakeet-daemon.spec"
CACHE_KEY_FILE="$CACHE_DIR/cache-key.txt"

# Pinned dep versions — bump these when validated against a new MLX/parakeet release.
PIN_PYINSTALLER="6.20.0"
PIN_PARAKEET_MLX="0.5.1"
# hf_xet is HuggingFace's Xet content-addressed transfer protocol. Without it,
# parakeet-mlx falls back to plain HTTPS and the first-launch Parakeet weights
# download is ~5-10× slower. Adds ~15 MB to the daemon bundle.
PIN_HF_XET="hf_xet"

PYTHON_BIN="${VOICEFLOW_PARAKEET_PYTHON:-python3.13}"

# Compute cache key from script SHA + pinned versions.
SCRIPT_SHA=$(shasum -a 256 "$DAEMON_SCRIPT" | awk '{print $1}')
CURRENT_KEY="$SCRIPT_SHA|pyinstaller=$PIN_PYINSTALLER|parakeet-mlx=$PIN_PARAKEET_MLX|hf_xet=$PIN_HF_XET"

CACHED_KEY=""
if [ -f "$CACHE_KEY_FILE" ]; then
    CACHED_KEY=$(cat "$CACHE_KEY_FILE")
fi

if [ "$CURRENT_KEY" = "$CACHED_KEY" ] && [ -x "$DIST_DIR/parakeet-daemon/parakeet-daemon" ]; then
    echo "  Parakeet daemon: cache hit ($SCRIPT_SHA), skipping rebuild."
    exit 0
fi

echo "  Parakeet daemon: cache miss, rebuilding..."

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "  ERROR: $PYTHON_BIN not found in PATH." >&2
    echo "  Install Python 3.13 (e.g. \`brew install python@3.13\` or pyenv)" >&2
    echo "  or set VOICEFLOW_PARAKEET_PYTHON to a working python3.13 path." >&2
    exit 1
fi

# Fresh venv (cheaper than diffing pip state).
rm -rf "$VENV_DIR" "$DIST_DIR" "$WORK_DIR" "$SPEC_PATH"
mkdir -p "$CACHE_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet \
    "pyinstaller==$PIN_PYINSTALLER" \
    "parakeet-mlx==$PIN_PARAKEET_MLX" \
    "$PIN_HF_XET"

# PyInstaller in --onedir mode produces a self-contained directory the main
# build.sh can drop into the .app bundle.
# --collect-all flags are required because parakeet_asr_daemon.py imports
# mlx, parakeet_mlx, and (via huggingface_hub) hf_xet inside functions — not
# at module top — so PyInstaller's static analysis would otherwise miss them
# and ship a daemon that crashes on first model load.
pyinstaller \
    --noconfirm \
    --onedir \
    --name parakeet-daemon \
    --collect-all mlx \
    --collect-all parakeet_mlx \
    --collect-all hf_xet \
    --distpath "$DIST_DIR" \
    --workpath "$WORK_DIR" \
    --specpath "$CACHE_DIR" \
    "$DAEMON_SCRIPT"

deactivate

# Stamp the cache so future invocations short-circuit.
echo "$CURRENT_KEY" > "$CACHE_KEY_FILE"

DAEMON_SIZE=$(du -sh "$DIST_DIR/parakeet-daemon" | awk '{print $1}')
echo "  Parakeet daemon: built at $DIST_DIR/parakeet-daemon ($DAEMON_SIZE)"
