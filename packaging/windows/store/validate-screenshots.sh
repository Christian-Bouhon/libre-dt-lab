#!/usr/bin/env bash
#
# Validate Microsoft Store screenshots (format + dimensions).
#
# Usage: validate-screenshots.sh [directory]
#
set -euo pipefail

DIR="${1:-packaging/windows/store/screenshots}"
MIN_W=1366
MIN_H=768

if ! command -v magick >/dev/null 2>&1 && ! command -v identify >/dev/null 2>&1; then
  echo "ImageMagick (magick/identify) is required." >&2
  exit 1
fi

identify_size() {
  if command -v magick >/dev/null 2>&1; then
    magick identify -format '%w %h' "$1"
  else
    identify -format '%w %h' "$1"
  fi
}

shopt -s nullglob
found=0
rc=0

for f in "$DIR"/*.png "$DIR"/*.jpg "$DIR"/*.jpeg; do
  found=1
  size=$(identify_size "$f")
  w=${size%% *}
  h=${size##* }
  if [ "$w" -lt "$MIN_W" ] || [ "$h" -lt "$MIN_H" ]; then
    echo "FAIL $f (${w}x${h}) < ${MIN_W}x${MIN_H}"
    rc=1
  else
    echo "OK   $f (${w}x${h})"
  fi
done

if [ "$found" -eq 0 ]; then
  echo "No screenshot found in $DIR"
  echo "(add at least one PNG >= ${MIN_W}x${MIN_H})"
  exit 0
fi

exit "$rc"
