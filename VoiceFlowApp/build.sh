#!/bin/bash
# Build script for VoiceFlow macOS app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="VoiceFlow"

echo "Building VoiceFlow macOS App..."

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

# Copy navbar icon asset from img/navbar/
# White icon works for both light and dark menu bars
IMG_DIR="$PROJECT_ROOT/img"
NAVBAR_DIR="$IMG_DIR/navbar"

cp "$NAVBAR_DIR/navbar-light.png" "$APP_BUNDLE/Contents/Resources/MenuBarIcon.png"

# Copy app icon for Settings UI
cp "$IMG_DIR/app.png" "$APP_BUNDLE/Contents/Resources/AppLogo.png"

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

# Fix library path - find actual path in binary and replace it
OLD_PATH=$(otool -L "$APP_BUNDLE/Contents/MacOS/$APP_NAME" | grep voiceflow_ffi | awk '{print $1}')
if [ -n "$OLD_PATH" ]; then
    install_name_tool -change \
        "$OLD_PATH" \
        "@executable_path/../Frameworks/libvoiceflow_ffi.dylib" \
        "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
fi

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
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo ""
echo "Build complete!"
echo "App bundle: $APP_BUNDLE"
echo ""
echo "To run: open $APP_BUNDLE"
echo "To install: cp -r $APP_BUNDLE /Applications/"
