# Dataset

| File / folder | In git | Notes |
|---------------|--------|--------|
| `data.zip` | **Git LFS** | `label.csv` + `raw_images/` (~85 MB) |
| `label.csv` | no | Inside `data.zip`; unpack after clone |
| `raw_images/` | no | Inside `data.zip`; unpack after clone |
| `processed/` | no | Run `python scripts/split_dataset.py` |

## After clone

```bash
git lfs install
git lfs pull
bash scripts/unpack_dataset.sh
python scripts/split_dataset.py
```
