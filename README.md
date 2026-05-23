# CNN Cast Defect Classification

Binary classification of cast images: **defect** vs **normal** (custom CNN, PyTorch).

## Setup (Windows)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

GPU (NVIDIA): replace the torch line with  
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

macOS/Linux: `pip install torch torchvision` then `pip install -r requirements.txt`

Dataset: `data/data.zip` in Git LFS (`label.csv` + `raw_images/`). After clone:

```bash
git lfs pull
bash scripts/unpack_dataset.sh
```

## Pipeline

```bash
python scripts/split_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/inference.py --image data/processed/test/defect/example.jpg
```

Optional demo UI:

```bash
streamlit run app/streamlit_app.py
```

Settings: `configs/config.yaml` (batch size, epochs, learning rate, dropout, simple augmentations).

**What the training code uses (beginner level):**
- `CrossEntropyLoss` + `Adam`
- augmentations: horizontal flip + small rotation (train only)
- dropout in the CNN
- early stopping if validation loss stops improving
- no mixup, label smoothing, or class weights (can add later)

## Project layout

```
configs/       — yaml settings
data/          — raw_images, label.csv, processed/ (after split)
scripts/       — train, evaluate, inference, split_dataset
src/           — model, dataset, training loop, metrics
notebooks/     — experiments + label cleanup draft
app/           — Streamlit upload demo (optional)
tests/         — smoke tests only
outputs/       — checkpoints, plots, logs (gitignored)
```

## TODO (not done yet)

- [ ] More tests (only smoke tests in `tests/`)
- [ ] Compare with a pretrained backbone (ResNet18) — only custom CNN for now
- [ ] Export model to ONNX / TorchScript
- [ ] Proper validation of `label.csv` before split (see `notebooks/label_cleanup.ipynb`)
- [ ] Try mixup / label smoothing / class weights (removed for simplicity)
- [ ] Hyperparameter search / cross-validation

## Notes

- Class order from ImageFolder: folder name order (usually `defect`, `normal`).
- `num_workers: auto` → 0 on Windows/macOS, 4 on Linux.
- Checkpoints: `outputs/models/`. Set `checkpoint.start_from_scratch: true` in config to retrain from zero.
