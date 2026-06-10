#!/bin/bash
# Build script for VoiceFlow macOS app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="VoiceFlow"
ONNX_VERSION="1.22.0"
ONNX_CACHE_DIR="$BUILD_DIR/onnxruntime-cache"

# PrismML llama-server release artifact. Era hosts a pre-built tarball
# (binary + dylibs) pinned to a PrismML SHA, since the PrismML repo is
# third-party and we can't publish releases there.
LLAMA_SERVER_SHA="d104cf1"
LLAMA_SERVER_CACHE_DIR="$BUILD_DIR/llama-server-cache"
LLAMA_SERVER_DIR="$LLAMA_SERVER_CACHE_DIR/llama-server-$LLAMA_SERVER_SHA"

echo "Building VoiceFlow macOS App..."

# Step 0: Download ONNX Runtime if not cached
ONNX_DYLIB="$ONNX_CACHE_DIR/libonnxruntime.$ONNX_VERSION.dylib"
if [ ! -f "$ONNX_DYLIB" ]; then
    echo "Step 0: Downloading ONNX Runtime $ONNX_VERSION..."
    mkdir -p "$ONNX_CACHE_DIR"

    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        ONNX_ARCH="arm64"
    else
        ONNX_ARCH="x86_64"
    fi

    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/onnxruntime-osx-$ONNX_ARCH-$ONNX_VERSION.tgz"
    curl -sL "$ONNX_URL" -o "$ONNX_CACHE_DIR/onnxruntime.tgz"
    tar -xzf "$ONNX_CACHE_DIR/onnxruntime.tgz" -C "$ONNX_CACHE_DIR"
    cp "$ONNX_CACHE_DIR/onnxruntime-osx-$ONNX_ARCH-$ONNX_VERSION/lib/libonnxruntime.$ONNX_VERSION.dylib" "$ONNX_DYLIB"
    rm -rf "$ONNX_CACHE_DIR/onnxruntime.tgz" "$ONNX_CACHE_DIR/onnxruntime-osx-$ONNX_ARCH-$ONNX_VERSION"
    echo "  ONNX Runtime downloaded and cached."
else
    echo "Step 0: Using cached ONNX Runtime $ONNX_VERSION"
fi

# Step 0b: Download bundled llama-server (PrismML fork, required for Bonsai Q1_0)
if [ ! -x "$LLAMA_SERVER_DIR/bin/llama-server" ]; then
    echo "Step 0b: Downloading llama-server ($LLAMA_SERVER_SHA)..."
    mkdir -p "$LLAMA_SERVER_DIR"
    LLAMA_URL="https://github.com/Alexander-Ollman/voiceflow/releases/download/llama-server-$LLAMA_SERVER_SHA/llama-server-macos-arm64-prismml-$LLAMA_SERVER_SHA.tgz"
    curl -fsSL "$LLAMA_URL" -o "$LLAMA_SERVER_CACHE_DIR/llama-server.tgz"
    tar -xzf "$LLAMA_SERVER_CACHE_DIR/llama-server.tgz" -C "$LLAMA_SERVER_DIR"
    rm -f "$LLAMA_SERVER_CACHE_DIR/llama-server.tgz"
    echo "  llama-server downloaded and cached."
else
    echo "Step 0b: Using cached llama-server $LLAMA_SERVER_SHA"
fi

# Step 0c: Build PyInstaller-frozen Parakeet daemon (cached; skips if up-to-date)
echo "Step 0c: Building Parakeet daemon..."
"$PROJECT_ROOT/scripts/build-parakeet-daemon.sh"
PARAKEET_DAEMON_DIR="$BUILD_DIR/parakeet-cache/dist/parakeet-daemon"

# Step 1: Build Rust FFI library
echo "Step 1: Building Rust FFI library..."
cd "$PROJECT_ROOT"
source ~/.cargo/env 2>/dev/null || true
cargo build --release -p voiceflow-ffi --features mistralrs

# Step 2: Create build directory
echo "Step 2: Creating build directory..."
mkdir -p "$BUILD_DIR"

# Step 3: Resolve Swift package dependencies
echo "Step 3: Resolving Swift dependencies..."
cd "$SCRIPT_DIR"
swift package resolve

# Step 4: Build Swift app
echo "Step 4: Building Swift app..."
# moonshine-swift declares libmoonshine.a as a binaryTarget — SwiftPM links it
# automatically via the Package.swift dependency. Don't pass an explicit
# `-Xlinker -lmoonshine`: the Moonshine wrapper target produces a stub
# `libmoonshine.a` (undefined refs only) in `.build/<triple>/release/` whose
# basename collides with the xcframework binary, and an explicit -lmoonshine
# resolves to whichever -L path comes first — usually the stub, producing a
# flood of "_moonshine_*" undefined-symbol errors at the final link step.
MOONSHINE_LIB_DIR="$SCRIPT_DIR/.build/artifacts/moonshine-swift/Moonshine/Moonshine.xcframework/macos-arm64_x86_64"
if [ ! -f "$MOONSHINE_LIB_DIR/libmoonshine.a" ]; then
    echo "  ERROR: moonshine-swift xcframework not present. Run 'swift package resolve' first." >&2
    exit 1
fi
swift build -c release \
    -Xlinker -L"$PROJECT_ROOT/target/release" \
    -Xlinker -lvoiceflow_ffi \
    -Xlinker -force_load -Xlinker "$MOONSHINE_LIB_DIR/libmoonshine.a" \
    -Xcc -I"$SCRIPT_DIR/Sources/VoiceFlowFFI"

# Step 5: Create app bundle
echo "Step 5: Creating app bundle..."
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"
mkdir -p "$APP_BUNDLE/Contents/Frameworks"

# Copy executable
cp "$(swift build -c release --show-bin-path)/VoiceFlowApp" "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# Copy Rust library
cp "$PROJECT_ROOT/target/release/libvoiceflow_ffi.dylib" "$APP_BUNDLE/Contents/Frameworks/"

# Copy ONNX Runtime library (required for Moonshine STT)
cp "$ONNX_DYLIB" "$APP_BUNDLE/Contents/Frameworks/libonnxruntime.dylib"

# Copy PrismML llama-server binary + its dylibs (Bonsai serving)
cp "$LLAMA_SERVER_DIR/bin/llama-server" "$APP_BUNDLE/Contents/MacOS/llama-server"
# cp -a preserves the versioned/symlink layout the dylibs need at runtime.
cp -a "$LLAMA_SERVER_DIR/lib/"lib*.dylib "$APP_BUNDLE/Contents/Frameworks/"

# Copy PyInstaller-frozen Parakeet daemon (whole --onedir tree) into Resources/.
# Resources/ is required (not MacOS/) because the PyInstaller layout contains
# subdirectories with Mach-O .so/.dylib files but no Info.plist. Under MacOS/
# codesign interprets those as nested bundles missing their signature and
# refuses to sign the outer .app ("code object is not signed at all" pointing
# at a .pyi sibling). Resources/ is treated as data and doesn't trigger that.
cp -R "$PARAKEET_DAEMON_DIR" "$APP_BUNDLE/Contents/Resources/parakeet-daemon"

# Copy navbar icon asset from img/navbar/
# White icon works for both light and dark menu bars
IMG_DIR="$PROJECT_ROOT/img"
NAVBAR_DIR="$IMG_DIR/navbar"

cp "$NAVBAR_DIR/navbar-light.png" "$APP_BUNDLE/Contents/Resources/MenuBarIcon.png"

# Copy app icon for Settings UI
cp "$IMG_DIR/app.png" "$APP_BUNDLE/Contents/Resources/AppLogo.png"

# Copy Python daemon script for Qwen3-ASR consolidated mode (developer
# fallback path — production users get the PyInstaller-frozen daemon).
if [ -f "$PROJECT_ROOT/scripts/qwen3_asr_daemon.py" ]; then
    cp "$PROJECT_ROOT/scripts/qwen3_asr_daemon.py" "$APP_BUNDLE/Contents/Resources/"
    echo "  Copied qwen3_asr_daemon.py to Resources"
fi

# Generate macOS app icon (.icns) from app.png
echo "Generating app icon..."
ICONSET_DIR="$BUILD_DIR/AppIcon.iconset"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

# Create all required icon sizes from app.png
sips -z 16 16     "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -z 32 32     "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -z 32 32     "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -z 64 64     "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -z 128 128   "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -z 256 256   "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -z 512 512   "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$IMG_DIR/app.png" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null

# Convert iconset to icns
iconutil -c icns "$ICONSET_DIR" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
rm -rf "$ICONSET_DIR"

# Fix library paths
OLD_PATH=$(otool -L "$APP_BUNDLE/Contents/MacOS/$APP_NAME" | grep voiceflow_ffi | awk '{print $1}')
if [ -n "$OLD_PATH" ]; then
    install_name_tool -change \
        "$OLD_PATH" \
        "@executable_path/../Frameworks/libvoiceflow_ffi.dylib" \
        "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
fi

# Add rpath to executable so it can find libraries in Frameworks
install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP_BUNDLE/Contents/MacOS/$APP_NAME" 2>/dev/null || true

# Add rpath to FFI library so it can find ONNX Runtime
install_name_tool -add_rpath "@loader_path" "$APP_BUNDLE/Contents/Frameworks/libvoiceflow_ffi.dylib" 2>/dev/null || true

# Fix ONNX Runtime install name to use @rpath
install_name_tool -id "@rpath/libonnxruntime.dylib" "$APP_BUNDLE/Contents/Frameworks/libonnxruntime.dylib"

# llama-server's build-time rpath points at the dev box that produced the
# tarball; redirect it into Contents/Frameworks/ so the bundled dylibs resolve.
install_name_tool -delete_rpath "/Users/alexanderollman/PrismML-llama.cpp/build-novoid/bin" \
    "$APP_BUNDLE/Contents/MacOS/llama-server" 2>/dev/null || true
install_name_tool -add_rpath "@executable_path/../Frameworks" \
    "$APP_BUNDLE/Contents/MacOS/llama-server" 2>/dev/null || true

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.era-laboratories.voiceflow</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>VoiceFlow</string>
    <key>CFBundleVersion</key>
    <string>2.3.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.3.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>VoiceFlow needs microphone access to transcribe your speech.</string>
    <key>NSScreenCaptureUsageDescription</key>
    <string>VoiceFlow uses screen capture for visual context to improve dictation. All processing is local.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>VoiceFlow reads the URL of your active browser tab to apply site-specific dictation personas (e.g. github.com → Software Engineer). No content is read.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Step 6: Code signing
SIGN_IDENTITY="Developer ID Application: Era Laboratories Inc. (JVSQ3LCY64)"
SIGNED=false
if security find-identity -v -p codesigning | grep -q "$SIGN_IDENTITY"; then
    echo "Step 6: Signing app bundle..."
    xattr -cr "$APP_BUNDLE"
    # Sign frameworks individually first (required for proper timestamps)
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_BUNDLE/Contents/Frameworks/libonnxruntime.dylib"
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_BUNDLE/Contents/Frameworks/libvoiceflow_ffi.dylib"
    # Sign each bundled llama.cpp dylib (skip the versioned/unversioned symlinks).
    for dylib in "$APP_BUNDLE/Contents/Frameworks/"libllama*.dylib \
                 "$APP_BUNDLE/Contents/Frameworks/"libggml*.dylib \
                 "$APP_BUNDLE/Contents/Frameworks/"libmtmd*.dylib; do
        [ -L "$dylib" ] && continue
        codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$dylib"
    done
    # Sign the llama-server helper executable
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_BUNDLE/Contents/MacOS/llama-server"
    # Sign every Mach-O inside the PyInstaller'd Parakeet daemon. PyInstaller
    # bundles ship hundreds of .so/.dylib files; each must be individually
    # signed under Hardened Runtime before the .app can pass deep verify.
    PARAKEET_BUNDLE="$APP_BUNDLE/Contents/Resources/parakeet-daemon"
    while IFS= read -r -d '' mach_o; do
        codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$mach_o" 2>/dev/null || true
    done < <(find "$PARAKEET_BUNDLE" -type f \( -name "*.so" -o -name "*.dylib" \) -print0)
    # Sign any other Mach-O executables inside the bundle (embedded Python interpreter, etc.)
    while IFS= read -r -d '' f; do
        if file "$f" 2>/dev/null | grep -q "Mach-O"; then
            codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$f" 2>/dev/null || true
        fi
    done < <(find "$PARAKEET_BUNDLE" -type f ! -name "*.so" ! -name "*.dylib" -perm +111 -print0)
    # Sign the top-level Parakeet daemon binary last, with its own entitlements
    # (allow-jit, disable-library-validation, etc — required by PyInstaller + MLX).
    codesign --force --options runtime --timestamp \
        --entitlements "$SCRIPT_DIR/Parakeet.entitlements" \
        --sign "$SIGN_IDENTITY" \
        "$PARAKEET_BUNDLE/parakeet-daemon"
    # Sign the main app bundle
    codesign --force --options runtime --timestamp --entitlements "$SCRIPT_DIR/VoiceFlow.entitlements" --sign "$SIGN_IDENTITY" "$APP_BUNDLE"
    codesign --verify --deep --strict "$APP_BUNDLE"
    echo "  Signed and verified."
    SIGNED=true
else
    echo "Step 6: Skipping code signing (Developer ID certificate not found in keychain)"
fi

# Step 7: Create distributable ZIP and notarize
DIST_DIR="$SCRIPT_DIR/dist"
mkdir -p "$DIST_DIR"
ZIP_PATH="$DIST_DIR/VoiceFlow.zip"

echo "Step 7: Creating distributable ZIP..."
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"

if [ "$SIGNED" = true ]; then
    # Notarize if Apple ID credentials are available
    APPLE_ID="${NOTARIZE_APPLE_ID:-}"
    TEAM_ID="JVSQ3LCY64"
    KEYCHAIN_PROFILE="${NOTARIZE_KEYCHAIN_PROFILE:-}"

    if [ -n "$KEYCHAIN_PROFILE" ]; then
        echo "  Notarizing with keychain profile '$KEYCHAIN_PROFILE'..."
        xcrun notarytool submit "$ZIP_PATH" \
            --keychain-profile "$KEYCHAIN_PROFILE" \
            --wait
        echo "  Stapling notarization ticket..."
        xcrun stapler staple "$APP_BUNDLE"

        # Re-create ZIP with stapled app
        rm -f "$ZIP_PATH"
        ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"
        echo "  Notarized and stapled."
    elif [ -n "$APPLE_ID" ]; then
        echo "  Notarizing with Apple ID..."
        xcrun notarytool submit "$ZIP_PATH" \
            --apple-id "$APPLE_ID" \
            --team-id "$TEAM_ID" \
            --wait
        echo "  Stapling notarization ticket..."
        xcrun stapler staple "$APP_BUNDLE"

        # Re-create ZIP with stapled app
        rm -f "$ZIP_PATH"
        ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"
        echo "  Notarized and stapled."
    else
        echo "  Skipping notarization (set NOTARIZE_KEYCHAIN_PROFILE or NOTARIZE_APPLE_ID to enable)"
    fi
fi

ZIP_SIZE=$(du -h "$ZIP_PATH" | awk '{print $1}')

echo ""
echo "Build complete!"
echo "App bundle: $APP_BUNDLE"
echo "Distributable: $ZIP_PATH ($ZIP_SIZE)"
echo ""
echo "To run: open $APP_BUNDLE"
echo "To distribute: upload $ZIP_PATH"
