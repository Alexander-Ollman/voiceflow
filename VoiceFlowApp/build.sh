#!/bin/bash
# Build script for VoiceFlow macOS app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="VoiceFlow"
ONNX_VERSION="1.22.0"
ONNX_CACHE_DIR="$BUILD_DIR/onnxruntime-cache"

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

# Step 1: Build Rust FFI library
echo "Step 1: Building Rust FFI library..."
cd "$PROJECT_ROOT"
source ~/.cargo/env 2>/dev/null || true
MISTRALRS_METAL_PRECOMPILE=0 cargo build --release -p voiceflow-ffi --features metal

# Step 2: Create build directory
echo "Step 2: Creating build directory..."
mkdir -p "$BUILD_DIR"

# Step 3: Resolve Swift package dependencies
echo "Step 3: Resolving Swift dependencies..."
cd "$SCRIPT_DIR"
swift package resolve

# Step 4: Build Swift app
echo "Step 4: Building Swift app..."
swift build -c release \
    -Xlinker -L"$PROJECT_ROOT/target/release" \
    -Xlinker -lvoiceflow_ffi \
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

# Copy navbar icon asset from img/navbar/
# White icon works for both light and dark menu bars
IMG_DIR="$PROJECT_ROOT/img"
NAVBAR_DIR="$IMG_DIR/navbar"

cp "$NAVBAR_DIR/navbar-light.png" "$APP_BUNDLE/Contents/Resources/MenuBarIcon.png"

# Copy app icon for Settings UI
cp "$IMG_DIR/app.png" "$APP_BUNDLE/Contents/Resources/AppLogo.png"

# Copy Python daemon script for Qwen3-ASR consolidated mode
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
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>VoiceFlow needs microphone access to transcribe your speech.</string>
    <key>NSScreenCaptureUsageDescription</key>
    <string>VoiceFlow uses screen capture for visual context to improve dictation. All processing is local.</string>
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
    codesign --deep --force --options runtime --entitlements "$SCRIPT_DIR/VoiceFlow.entitlements" --sign "$SIGN_IDENTITY" "$APP_BUNDLE"
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
