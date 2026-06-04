# Dataset

| File / folder | In git | Notes |
|---------------|--------|--------|
| `data.zip` | **Git LFS** | `label.csv` + `raw_images\` (~85 MB) |
| `label.csv` | no | Inside `data.zip`; unpack after clone |
| `raw_images\` | no | Inside `data.zip`; unpack after clone |
| `processed\` | no | Run `python scripts\split_dataset.py` |

## After clone (Windows)

```bat
git lfs install
scripts\lfs_pull_windows.bat
scripts\unpack_dataset.bat
python scripts\split_dataset.py
```

### LFS does not download?

1. Check file size:
   ```bat
   scripts\verify_lfs_files.bat
   ```
   `data\data.zip` must be **~65–85 MB**. If **~130 bytes** — only an LFS pointer, not the dataset.

2. Force download:
   ```bat
   scripts\lfs_pull_windows.bat
   ```

3. Fresh clone (recommended if project was copied without git):
   ```bat
   cd C:\Users\user\PyCharmMiscProject
   git clone https://github.com/mentisVeritas/cnn-cast-defect-classification.git
   cd cnn-cast-defect-classification
   git lfs install
   scripts\lfs_pull_windows.bat
   scripts\verify_lfs_files.bat
   ```

4. Private repo: sign in to GitHub in Git Credential Manager, then:
   ```bat
   git fetch origin
   scripts\lfs_pull_windows.bat
   ```
