# Dataset folder

## What must be here before training

| Path | Required | Description |
|------|----------|-------------|
| `raw_images/` | **yes** | All cast surface images |
| `label.csv` | **yes** | Image filename → class (`defect` / `normal`) |
| `processed/` | no | Created by `python scripts/split_dataset.py` |

## Optional

| Path | Notes |
|------|--------|
| `data.zip` | Archive backup; only needed if `label.csv` / `raw_images/` are missing |

Unpack from zip (if you only have the archive):

```bash
python scripts/unpack_dataset.py
```

Then split for training:

```bash
python scripts/split_dataset.py
```
