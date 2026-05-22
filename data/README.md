# Dataset

| File / folder | In git | Notes |
|---------------|--------|--------|
| `raw_images.zip` | **Git LFS** (~68 MB) | All images in one archive |
| `label.csv` | git | Columns: `image`, `choice` |
| `raw_images/` | no | Run `bash scripts/unpack_dataset.sh` after clone |
| `processed/` | no | Run `python scripts/split_dataset.py` |

## After clone

```bash
git lfs install
git lfs pull
bash scripts/unpack_dataset.sh
python scripts/split_dataset.py
```
