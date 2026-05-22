"""Split dataset from raw_images + label.csv: python scripts/split_dataset.py"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve_path, set_seed


def _load_labeled_images(
    labels_path: Path,
    raw_dir: Path,
    image_column: str,
    label_column: str,
) -> dict[str, list[Path]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw images directory not found: {raw_dir}")

    by_class: dict[str, list[Path]] = defaultdict(list)
    skipped_label = 0
    skipped_missing = 0

    with labels_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {labels_path}")
        if image_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns {image_column!r} and {label_column!r}, "
                f"got: {reader.fieldnames}"
            )

        for row in reader:
            filename = (row.get(image_column) or "").strip()
            label = (row.get(label_column) or "").strip().lower()
            if not filename or not label:
                skipped_label += 1
                continue

            src = raw_dir / filename
            if not src.is_file():
                skipped_missing += 1
                continue

            by_class[label].append(src)

    if not by_class:
        raise ValueError(f"No labeled images found in {labels_path}")

    for class_name, paths in by_class.items():
        unique = sorted(set(paths), key=lambda p: p.name)
        if len(unique) != len(paths):
            print(f"Warning: duplicate rows for class {class_name!r}, keeping unique files.")
        by_class[class_name] = unique

    if skipped_label:
        print(f"Skipped {skipped_label} rows with empty image or label.")
    if skipped_missing:
        print(f"Skipped {skipped_missing} rows: image not found in {raw_dir}")

    return dict(by_class)


def split_dataset() -> None:
    config = load_config()
    set_seed(config["random_seed"])

    data_cfg = config["data"]
    ratios = data_cfg["split_ratios"]
    processed_dir = resolve_path(data_cfg["processed_dir"])
    raw_dir = resolve_path(data_cfg["raw_images_dir"])
    labels_path = resolve_path(data_cfg["labels_file"])
    image_column = data_cfg.get("image_column", "image")
    label_column = data_cfg.get("label_column", "choice")

    by_class = _load_labeled_images(labels_path, raw_dir, image_column, label_column)

    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split in ("train", "val", "test"):
        for class_name in by_class:
            (processed_dir / split / class_name).mkdir(parents=True)

    for class_name, files in sorted(by_class.items()):
        shuffled = files.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        parts = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

        for split_name, split_files in parts.items():
            dest = processed_dir / split_name / class_name
            for src in split_files:
                shutil.copy2(src, dest / src.name)

        print(
            f"{class_name}: total={n} "
            f"train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}"
        )

    print(f"Done -> {processed_dir}")


if __name__ == "__main__":
    split_dataset()
