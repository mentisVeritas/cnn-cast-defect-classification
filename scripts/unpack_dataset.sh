#!/usr/bin/env bash
# Unpack data/data.zip after clone (Git LFS pulls the archive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/data/data.zip"
DEST="$ROOT/data"

if [[ -f "$DEST/label.csv" ]] && [[ -d "$DEST/raw_images" ]] && [[ -n "$(ls -A "$DEST/raw_images" 2>/dev/null)" ]]; then
  echo "Already unpacked: $DEST"
  exit 0
fi

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP — run: git lfs pull"
  exit 1
fi

unzip -q -o "$ZIP" -d "$DEST"
echo "Unpacked to $DEST (label.csv + raw_images/)"
