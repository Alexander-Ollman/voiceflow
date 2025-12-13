#!/bin/bash
# Create DMG installer for VoiceFlow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="VoiceFlow"
APP_BUNDLE="$SCRIPT_DIR/build/$APP_NAME.app"
DMG_NAME="VoiceFlow-Installer"
DMG_DIR="$SCRIPT_DIR/dist"
DMG_PATH="$DMG_DIR/$DMG_NAME.dmg"
VOLUME_NAME="VoiceFlow"
STAGING_DIR="$SCRIPT_DIR/build/dmg-staging"

echo "Creating VoiceFlow DMG installer..."

# Step 1: Build the app first
echo "Step 1: Building app..."
"$SCRIPT_DIR/build.sh"

# Verify app exists
if [ ! -d "$APP_BUNDLE" ]; then
    echo "Error: App bundle not found at $APP_BUNDLE"
    exit 1
fi

# Step 2: Create staging directory
echo "Step 2: Creating staging directory..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
mkdir -p "$DMG_DIR"

# Copy app to staging
cp -R "$APP_BUNDLE" "$STAGING_DIR/"

# Create Applications symlink
ln -s /Applications "$STAGING_DIR/Applications"

# Step 3: Create DMG
echo "Step 3: Creating DMG..."
rm -f "$DMG_PATH"

# Create temporary DMG
hdiutil create -volname "$VOLUME_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov -format UDRW \
    "$DMG_DIR/temp.dmg"

# Mount it
MOUNT_DIR=$(hdiutil attach "$DMG_DIR/temp.dmg" | grep "Volumes" | awk '{print $3}')
echo "Mounted at: $MOUNT_DIR"

# Set window properties using AppleScript
echo "Step 4: Configuring DMG appearance..."
osascript <<EOF
tell application "Finder"
    tell disk "$VOLUME_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {100, 100, 640, 400}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 100

        -- Position items
        set position of item "$APP_NAME.app" of container window to {150, 150}
        set position of item "Applications" of container window to {390, 150}

        update without registering applications
        close
    end tell
end tell
EOF

# Wait for Finder to finish
sync
sleep 2

# Unmount
hdiutil detach "$MOUNT_DIR"

# Convert to compressed DMG
echo "Step 5: Compressing DMG..."
hdiutil convert "$DMG_DIR/temp.dmg" -format UDZO -o "$DMG_PATH"
rm -f "$DMG_DIR/temp.dmg"

# Cleanup
rm -rf "$STAGING_DIR"

# Get file size
DMG_SIZE=$(du -h "$DMG_PATH" | awk '{print $1}')

echo ""
echo "========================================="
echo "DMG installer created successfully!"
echo "========================================="
echo ""
echo "Location: $DMG_PATH"
echo "Size: $DMG_SIZE"
echo ""
echo "To distribute:"
echo "1. Upload $DMG_PATH to your website"
echo "2. Users download, open DMG, drag VoiceFlow to Applications"
echo ""
echo "Note: For distribution outside the App Store, consider:"
echo "- Code signing with a Developer ID certificate"
echo "- Notarization with Apple"
echo ""
