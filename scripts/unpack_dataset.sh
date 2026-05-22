#!/usr/bin/env bash
# Unpack data/raw_images.zip after clone (Git LFS pulls the zip).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/data/raw_images.zip"
DEST="$ROOT/data/raw_images"

if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "Already unpacked: $DEST"
  exit 0
fi

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP — run: git lfs pull"
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
unzip -q -o "$ZIP" -d "$ROOT/data"
echo "Unpacked to $DEST"
