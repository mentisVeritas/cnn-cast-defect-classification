# CNN Cast Defect Classification

Binary classifier for cast surface images: **defect** vs **normal** (custom CNN, PyTorch).

Works on **Windows, macOS, and Linux**. Setup below assumes you **already have the project folder** (code + data), not a download from the internet.

---

## What is included

- Custom CNN training and evaluation
- Dataset ~7,285 images
- Streamlit demo (predictions, feature maps, metrics)
- ResNet18 baseline comparison and learning-rate tuning (0.001 vs 0.0005)
- Notebook `notebooks/experiments.ipynb` for report figures

---

## Requirements

- Python **3.11**
- ~2 GB free disk space  
- **CPU is fine** (GPU optional, faster training)

---

## 1. Project and data (check first)

Open the project folder in terminal or IDE. From the **project root** you should see at least:

```text
configs/
scripts/
src/
app/
data/
```

Inside `data/` you need:

| Path | Required |
|------|----------|
| `data/raw_images/` | yes — image files (`.jpeg`, etc.) |
| `data/label.csv` | yes — labels for each image |

If `label.csv` is missing but you have `data/data.zip`, unpack once:

```bash
python scripts/unpack_dataset.py
```

More detail: [data/README.md](data/README.md)

**Windows:** open Command Prompt or PowerShell, go to the project folder, for example:

```bat
cd C:\path\to\cnn-cast-defect-classification
```

---

## 2. Virtual environment

Create and activate a venv **inside the project** (once per machine).

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For **NVIDIA GPU**, use the CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) instead of `cpu`, and set `device: auto` in config.

Every new terminal session: activate again (`.venv\Scripts\activate` or `source .venv/bin/activate`).

Verify:

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## 3. Configuration

Edit `configs/config.yaml`:

```yaml
training:
  device: cpu    # laptops without GPU; or auto / cuda
  batch_size: 16
  epochs: 25
```

On Windows, `num_workers: auto` uses **0** (stable). Training on CPU can take a long time — lower `epochs` for a quick test.

---

## 4. Train and evaluate

With venv active, from the project root:

```bash
python scripts/split_dataset.py
python scripts/train.py
python scripts/evaluate.py
```

| Output | Description |
|--------|-------------|
| `outputs/models/best_model.pth` | Best model |
| `outputs/metrics/evaluation_metrics.json` | Test metrics |
| `outputs/plots/` | Loss, accuracy, confusion matrix |

Single image:

```bash
python scripts/inference.py --image data/raw_images/img_00002.jpeg
```

---

## 5. Experiments (report)

After `evaluate.py`:

```bash
python scripts/run_hyperparameter_tuning.py
python scripts/train_baseline.py
```

| File | Content |
|------|---------|
| `outputs/tuning/tuning_results.csv` | lr 0.001 vs 0.0005 |
| `outputs/baseline/model_comparison.csv` | CNN vs ResNet18 |

Skip re-training if results already exist:

```bash
python scripts/run_hyperparameter_tuning.py --skip-trained
```

---

## 6. Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` — use screenshots for your Word report.

---

## 7. Notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

Run after the pipeline above. Set `RUN_TRAINING = False` in the notebook if you already ran `scripts/train.py`.

---

## Project layout

```text
app/              Streamlit UI
configs/          config.yaml
data/
  label.csv       labels
  raw_images/     original images
  processed/      train/val/test (after split_dataset.py)
notebooks/        experiments.ipynb
scripts/          CLI entry points
src/              model, training, experiments, feature maps
outputs/          created when you train (on this PC)
.venv/            Python environment (create locally, not copied)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named ...` | Activate `.venv`, run `pip install -r requirements.txt` |
| Missing `data/raw_images` or `label.csv` | Copy full `data/` folder from the USB/archive you received |
| No `data/processed` | `python scripts/split_dataset.py` |
| No checkpoint / Streamlit error | `python scripts/train.py` |
| Very slow training | Normal on CPU; lower `epochs` or use GPU |
| Out of memory | Lower `batch_size` in config |

---

## Model (short)

- 224×224 RGB, ImageNet normalization  
- 3 conv blocks + FC head, Adam, cross-entropy  
- Early stopping on validation loss  
- Classes: `defect`, `normal`

---

## Dataset

| | |
|--|--:|
| Images | 7,285 |
| defect | ~52.7% |
| normal | ~47.3% |
| Split | 80% / 10% / 10% |
