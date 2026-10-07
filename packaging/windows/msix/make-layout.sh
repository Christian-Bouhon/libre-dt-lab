#!/usr/bin/env bash
#
# Build a MSIX layout directory from a Libre DT-Lab Windows install tree.
#
# Usage:
#   make-layout.sh <install-prefix> <layout-dir> <icon-png> <manifest> \
#                  <msix-version> <arch> <identity-name> <publisher>
#
# <msix-version> must be a 4-part dotted version (e.g. 1.0.0.0).
# <arch> is x64 or arm64.
#
set -euo pipefail

INSTALL_PREFIX="$1"
LAYOUT="$2"
ICON="$3"
MANIFEST="$4"
MSIX_VERSION="$5"
ARCH="$6"
IDENTITY="$7"
PUBLISHER="$8"

if [ ! -f "$ICON" ]; then
  echo "icon not found: $ICON" >&2
  exit 1
fi

rm -rf "$LAYOUT"
mkdir -p "$LAYOUT/Assets"

# The whole install tree becomes the package root (bin/, lib/, share/).
cp -a "$INSTALL_PREFIX"/. "$LAYOUT"/

# drop build-only leftovers that may live in the install prefix
rm -f "$LAYOUT"/libre-dt-lab.iss "$LAYOUT"/dt_logo_multiresolution.ico

gen() { # <WxH> <output>
  magick "$ICON" -background none -resize "${1}^" -gravity center -extent "$1" "$2"
}

gen 50x50     "$LAYOUT/Assets/StoreLogo.png"
gen 44x44     "$LAYOUT/Assets/Square44x44Logo.png"
gen 150x150   "$LAYOUT/Assets/Square150x150Logo.png"
gen 71x71     "$LAYOUT/Assets/Square71x71Logo.png"
gen 310x150   "$LAYOUT/Assets/Wide310x150Logo.png"
gen 620x300   "$LAYOUT/Assets/SplashScreen.png"

sed -e "s|__IDENTITY_NAME__|${IDENTITY}|g" \
    -e "s|__PUBLISHER__|${PUBLISHER}|g" \
    -e "s|__VERSION__|${MSIX_VERSION}|g" \
    -e "s|__ARCH__|${ARCH}|g" \
    "$MANIFEST" > "$LAYOUT/AppxManifest.xml"

echo "MSIX layout ready: $LAYOUT"
