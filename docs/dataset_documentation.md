# Dataset documentation

Values below are taken from the project repository (`data/label.csv`, `data/raw_images/`, split outputs). No external URL is recorded in repository metadata.

## Dataset source

| Field | Value |
|-------|--------|
| **Distribution** | Git repository archive `data/data.zip` (Git LFS) |
| **Repository** | [github.com/mentisVeritas/cnn-cast-defect-classification](https://github.com/mentisVeritas/cnn-cast-defect-classification) |
| **Label file** | `data/label.csv` |
| **Image folder** | `data/raw_images/` (after `scripts/unpack_dataset.sh` or `.bat`) |
| **External dataset URL** | Not specified in project files |

Images are **cast surface photographs** collected for a manufacturing defect detection coursework project (binary quality inspection).

## Dataset size

| Metric | Count |
|--------|------:|
| Labeled rows in `label.csv` | **7,285** |
| Image files used after split (train+val+test) | **7,285** |
| Train split | **5,827** (80%) |
| Validation split | **727** (10%) |
| Test split | **731** (10%) |

Split produced by `scripts/split_dataset.py` with `random_seed: 42` and per-class stratified shuffle.

## Classes

| Class | Label in CSV | Meaning |
|-------|--------------|---------|
| 0 | `defect` | Cast surface shows a defect |
| 1 | `normal` | Cast surface acceptable |

## Class distribution (full labeled set)

From `data/label.csv` (`choice` column):

| Class | Images | Share |
|-------|-------:|------:|
| defect | 3,836 | 52.7% |
| normal | 3,449 | 47.3% |
| **Total** | **7,285** | 100% |

Train split class balance is visualized in the notebook (pie chart) and `outputs/` analysis.

## Sample images

Example filenames referenced in analysis outputs:

- **Defect:** `data/raw_images/img_00001.jpeg` (and images used in `outputs/feature_maps/defect_*.png`)
- **Normal:** first `normal` entry in `label.csv` (used in `outputs/feature_maps/normal_*.png`)

See generated figures:

- `outputs/feature_maps/`
- `docs/samples/` (if copied by `scripts/copy_dataset_samples.py`)

## Manufacturing relevance

- **Domain:** Metal casting / foundry surface inspection.
- **Task:** Automated **binary defect detection** on cast part images to support quality control.
- **Impact:** Reduces manual visual inspection load and enables consistent pass/fail decisions on the production line.
- **ML formulation:** Supervised image classification (`defect` vs `normal`) with a custom CNN and a ResNet18 baseline.

## File format

- Images: JPEG (`.jpeg`), 224×224 after preprocessing.
- Labels: CSV with columns `image`, `choice`.
