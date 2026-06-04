# Dataset

| File / folder | In git | Notes |
|---------------|--------|--------|
| `data.zip` | **Git LFS** | Contains `label.csv` and `raw_images/` (~65–85 MB) |
| `label.csv` | no | Created after unpack |
| `raw_images/` | no | Created after unpack |
| `processed/` | no | Created by `python scripts/split_dataset.py` |

## After clone

```bash
git lfs install
git lfs pull
python scripts/unpack_dataset.py
python scripts/split_dataset.py
```

Check `data/data.zip` size: about **65–85 MB**. If it is ~130 bytes, only the LFS pointer was downloaded — run `git lfs pull` again.
