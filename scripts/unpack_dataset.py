"""Unpack data/data.zip into data/ (label.csv + raw_images/). Works on Windows, macOS, Linux.

Usage: python scripts/unpack_dataset.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    zip_path = ROOT / "data" / "data.zip"
    dest = ROOT / "data"

    if (dest / "label.csv").is_file() and (dest / "raw_images").is_dir():
        if any((dest / "raw_images").iterdir()):
            print(f"Already unpacked: {dest}")
            return

    if not zip_path.is_file():
        print(f"Missing {zip_path}")
        print("Run: git lfs install && git lfs pull")
        sys.exit(1)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        print(f"data.zip is only {zip_path.stat().st_size} bytes — Git LFS file not downloaded.")
        print("Run: git lfs pull")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    print(f"Unpacked to {dest} ({size_mb:.1f} MB archive)")


if __name__ == "__main__":
    main()
