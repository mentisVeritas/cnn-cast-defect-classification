"""Append BTEC report sections to notebooks/experiments.ipynb (keeps existing cells)."""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [text],
        "outputs": [],
        "execution_count": None,
    }


NEW_SECTIONS = [
    md("## 1. Dataset Overview\n\nSee `docs/dataset_documentation.md` for source, size, and class distribution."),
    code(
        "# Dataset statistics from label.csv\n"
        "import csv\n"
        "from collections import Counter\n"
        "from pathlib import Path\n"
        "labels_path = ROOT / 'data/label.csv'\n"
        "counts = Counter()\n"
        "with labels_path.open(newline='', encoding='utf-8') as f:\n"
        "    for row in csv.DictReader(f):\n"
        "        counts[row['choice'].strip().lower()] += 1\n"
        "print('Labeled images:', sum(counts.values()))\n"
        "print('Class distribution:', dict(counts))\n"
    ),
    md(
        "## 2. Data Preprocessing\n\n"
        "- Resize to 224×224\n"
        "- ImageNet normalization\n"
        "- Train/val/test split 80/10/10 (stratified by class)\n"
        "- Augmentation (train): horizontal flip, rotation ±10°"
    ),
    md(
        "## 3. Custom CNN Architecture\n\n"
        "See `src/model.py` — 3× Conv blocks (32→64→96), BatchNorm, MaxPool, Dropout, Linear classifier."
    ),
    md("## 4. Training Results\n\nLoss/accuracy curves and test metrics are generated above and saved under `outputs/`."),
    md("## 5. Hyperparameter Tuning\n\nRun `python scripts/run_hyperparameter_tuning.py` then display results:"),
    code(
        "import pandas as pd\n"
        "tuning_csv = ROOT / 'outputs/tuning/tuning_results.csv'\n"
        "if tuning_csv.exists():\n"
        "    display(pd.read_csv(tuning_csv))\n"
        "    from IPython.display import Image, display\n"
        "    img = ROOT / 'outputs/tuning/tuning_comparison.png'\n"
        "    if img.exists():\n"
        "        display(Image(filename=str(img)))\n"
        "else:\n"
        "    print('Run: python scripts/run_hyperparameter_tuning.py')\n"
    ),
    md("## 6. Baseline Comparison (ResNet18 transfer learning)"),
    code(
        "import pandas as pd\n"
        "cmp = ROOT / 'outputs/baseline/model_comparison.csv'\n"
        "if cmp.exists():\n"
        "    display(pd.read_csv(cmp))\n"
        "    from IPython.display import Image, display\n"
        "    display(Image(filename=str(ROOT / 'outputs/baseline/model_comparison.png')))\n"
        "    display(Image(filename=str(ROOT / 'outputs/baseline/resnet18_confusion_matrix.png')))\n"
        "else:\n"
        "    print('Run: python scripts/train_baseline.py')\n"
    ),
    md(
        "## 7. Feature Extraction Analysis\n\n"
        "**Conv block 1** captures edges and local texture. "
        "**Conv block 2** combines patterns into larger defect-like regions. "
        "**Conv block 3** highlights semantically rich areas before classification."
    ),
    code(
        "from IPython.display import Image, display\n"
        "for label in ('normal', 'defect'):\n"
        "    for block in ('conv1', 'conv2', 'conv3'):\n"
        "        p = ROOT / f'outputs/feature_maps/{label}_{block}.png'\n"
        "        if p.exists():\n"
        "            print(label, block)\n"
        "            display(Image(filename=str(p), width=280))\n"
        "        else:\n"
        "            print('Missing', p, '— run scripts/generate_analysis_assets.py')\n"
    ),
    md(
        "## 8. Learned Filter Analysis\n\n"
        "First-layer filters show edge and blob detectors learned from cast surface images."
    ),
    code(
        "from IPython.display import Image, display\n"
        "p = ROOT / 'outputs/filters/conv1_filters.png'\n"
        "display(Image(filename=str(p))) if p.exists() else print('Run generate_analysis_assets.py')\n"
    ),
    md("## 9. Grad-CAM Analysis\n\nGrad-CAM shows which regions support the predicted class (defect vs normal)."),
    code(
        "from IPython.display import Image, display\n"
        "for p in [ROOT / 'outputs/gradcam/defect_gradcam.png', ROOT / 'outputs/gradcam/normal_gradcam.png']:\n"
        "    if p.exists():\n"
        "        display(Image(filename=str(p), width=500))\n"
        "    else:\n"
        "        print('Missing', p)\n"
    ),
    md("## 10. Deployment Evidence\n\nStreamlit app + static previews in `docs/screenshots/`."),
    code(
        "from IPython.display import Image, display\n"
        "shots = sorted((ROOT / 'docs/screenshots').glob('*.png'))\n"
        "for p in shots:\n"
        "    print(p.name)\n"
        "    display(Image(filename=str(p), width=520))\n"
    ),
]


def main() -> None:
    nb_path = Path(__file__).resolve().parent.parent / "notebooks/experiments.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    existing = "".join("".join(c.get("source", [])) for c in nb["cells"])
    if "BTEC report sections" in existing or "## 1. Dataset Overview" in existing:
        print("BTEC sections already present — skipping.")
        return
    nb["cells"].extend(NEW_SECTIONS)
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Appended {len(NEW_SECTIONS)} cells to {nb_path}")


if __name__ == "__main__":
    main()
