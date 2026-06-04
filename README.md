# CNN Cast Defect Classification

Binary classification of cast images: **defect** vs **normal** (custom CNN, PyTorch).

This guide is for **Windows 10/11**.

---

## Requirements

| Tool | Notes |
|------|--------|
| [Python 3.11](https://www.python.org/downloads/) | During install enable **“Add python.exe to PATH”** |
| [Git for Windows](https://git-scm.com/download/win) | Includes Git Bash and `tar` for unpacking |
| [Git LFS](https://git-lfs.com/) | `git lfs install` once per machine |

**NVIDIA GPU** + актуальный драйвер (CUDA 12.x) — для обучения на видеокарте.

---

## 1. Clone and dataset (Git LFS)

```bat
git clone git@github.com:mentisVeritas/cnn-cast-defect-classification.git
cd cnn-cast-defect-classification

git lfs install
scripts\lfs_pull_windows.bat
scripts\verify_lfs_files.bat
scripts\unpack_dataset.bat
```

`git lfs pull` alone often prints nothing; **`lfs_pull_windows.bat`** runs `fetch` + `checkout` and checks sizes.

After unpack you should have:

- `data\label.csv`
- `data\raw_images\` (thousands of `.jpeg` files)
- `data\data.zip` (~65–85 MB, not ~130 bytes)

If download fails, see **Git LFS on Windows** in `data\README.md` or re-clone from GitHub (do not copy folder by hand).

---

## 2. Virtual environment

**Quick setup (recommended, GPU):**

```bat
scripts\setup_windows.bat
```

Скрипт ставит PyTorch с **CUDA 12.4** и в конце проверяет `torch.cuda.is_available()`.

**Manual setup (GPU):**

```bat
cd cnn-cast-defect-classification
py -3.11 -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

**Только CPU** (нет NVIDIA):

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Check that the venv is active — prompt shows `(.venv)` and:

```bat
where python
```

must point to:

```text
...\cnn-cast-defect-classification\.venv\Scripts\python.exe
```

Activate venv in every new terminal:

```bat
.venv\Scripts\activate.bat
```

---

## 3. Training pipeline

Run from the project root with venv active:

```bat
python scripts\split_dataset.py
python scripts\train.py
python scripts\evaluate.py
python scripts\inference.py --image data\raw_images\img_00002.jpeg
```

Outputs:

- checkpoints: `outputs\models\`
- plots: `outputs\plots\`
- metrics: `outputs\metrics\`

Settings: `configs\config.yaml` (batch size, epochs, learning rate, dropout).

---

## 4. Streamlit demo (browser UI)

```bat
.venv\Scripts\activate.bat
streamlit run app\streamlit_app.py
```

Open the URL from the terminal (usually `http://localhost:8501`).

Tabs in the app:

- **Prediction** — class + probability bar chart
- **Feature maps** — CNN activation heatmaps (conv1–conv3)
- **Grad-CAM** — where the model looked (overlay on image)
- **Model metrics** — test accuracy/F1, confusion matrix, training curves

**PyCharm:** Run configuration **Streamlit app**, or Run on `app\streamlit_app.py` (starts Streamlit automatically).  
Do **not** use plain “python streamlit_app.py” without Streamlit — the UI will not open.

**PyCharm interpreter:**  
`Settings → Project → Python Interpreter → Add → Existing` →  
`.venv\Scripts\python.exe`

---

## 5. Project layout

```text
configs\        — yaml settings
data\           — data.zip (LFS), label.csv, raw_images\ after unpack
scripts\        — train, evaluate, inference, split, unpack (.bat)
src\            — model, dataset, training loop, metrics
app\            — Streamlit demo
outputs\        — checkpoints, plots, logs (created when training)
```

---

## 6. Config notes (Windows)

- `num_workers: auto` → **0** on Windows (stable DataLoader, no multiprocessing errors).
- `device: auto` → CUDA if available, otherwise CPU.
- Retrain from scratch: set `checkpoint.start_from_scratch: true` in `configs\config.yaml`.
- Class folders: `defect`, `normal` (ImageFolder order).

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `'py' is not recognized` | Reinstall Python 3.11 with “Add to PATH”, or use `python` instead of `py` |
| `git lfs pull` does nothing / small `data.zip` (~130 B) | Run `scripts\lfs_pull_windows.bat` then `scripts\verify_lfs_files.bat` — zip must be **~65–85 MB** |
| Project opened without full clone (PyCharm copy) | Clone again: `git clone https://github.com/mentisVeritas/cnn-cast-defect-classification.git` |
| `git lfs pull` — file is a pointer | `git lfs install` → `scripts\lfs_pull_windows.bat` |
| `Missing data\data.zip` | `scripts\lfs_pull_windows.bat`, then `scripts\unpack_dataset.bat` |
| `train not found` / `processed` missing | Run `python scripts\split_dataset.py` first |
| Streamlit warnings / no browser | Use `streamlit run app\streamlit_app.py`, not `python app\streamlit_app.py` |
| PyCharm wrong interpreter | Point to `.venv\Scripts\python.exe`, not another project’s venv |
| `cuda available: False` | Обнови драйвер NVIDIA; переустанови torch с `cu124`; в `configs\config.yaml` можно поставить `device: cpu` |
| Slow training | Проверь, что в логе train видно `Device: cuda`, не `cpu` |

---

## What the model uses (beginner level)

- `CrossEntropyLoss` + `Adam`
- augmentations: horizontal flip + small rotation (train only)
- dropout in the CNN
- early stopping when validation loss stops improving

---

## BTEC assignment artifacts

```bash
python scripts/run_btec_assets.py
python scripts/run_btec_assets.py --with-training --epochs 12   # tuning + ResNet18
```

Outputs: `outputs/tuning/`, `outputs/baseline/`, `outputs/feature_maps/`, `docs/dataset_documentation.md`, `docs/screenshots/`.

---

## TODO

- [ ] More automated tests
- [ ] Export to ONNX / TorchScript
- [ ] Stricter `label.csv` validation before split
