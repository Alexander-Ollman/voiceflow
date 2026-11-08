#!/bin/bash
# Create a self-signed "VoiceFlow Dev" code-signing identity in the login
# keychain. build.sh uses it when the Developer ID isn't available, so local
# builds keep one signing identity and macOS privacy grants (Accessibility,
# Microphone) survive rebuilds. Not for distribution.
set -euo pipefail
NAME="${VOICEFLOW_DEV_SIGN_IDENTITY:-VoiceFlow Dev}"
if security find-identity -p codesigning | grep -q "\"$NAME\""; then
    echo "\"$NAME\" already exists."; exit 0
fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/cert.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
O = VoiceFlow local development
[ext]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
CNF
/usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -keyout "$TMP/key.pem" -out "$TMP/cert.pem" \
    -days 3650 -config "$TMP/cert.cnf" 2>/dev/null
PASS="$(/usr/bin/openssl rand -hex 12)"
/usr/bin/openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -name "$NAME" \
    -out "$TMP/dev.p12" -passout "pass:$PASS"
# Only codesign may use the private key.
security import "$TMP/dev.p12" -k "$HOME/Library/Keychains/login.keychain-db" -P "$PASS" -T /usr/bin/codesign
echo "Created \"$NAME\". It shows as not trusted; codesign can still use it."
