# CNN Cast Defect Classification

Binary classifier for cast surface images: **defect** vs **normal** (custom CNN, PyTorch).

Works on **Windows, macOS, and Linux**. Below: standard **Git + Python venv** setup (no extra install scripts).

---

## What is included

- Custom CNN training and evaluation
- Dataset ~7,285 images (Git LFS)
- Streamlit demo (predictions, feature maps, metrics)
- ResNet18 baseline comparison and learning-rate tuning (0.001 vs 0.0005)
- Notebook `notebooks/experiments.ipynb` for report figures

---

## Requirements

- Python **3.11**
- Git + [Git LFS](https://git-lfs.com/)
- ~2 GB free disk space  
- **CPU is fine** (GPU optional, faster training)

---

## 1. Clone and download data

```bash
git clone https://github.com/mentisVeritas/cnn-cast-defect-classification.git
cd cnn-cast-defect-classification

git lfs install
git lfs pull
```

Unpack the dataset:

```bash
python scripts/unpack_dataset.py
```

You should get `data/label.csv` and `data/raw_images/`.  
Details: [data/README.md](data/README.md)

---

## 2. Virtual environment

**Windows (Command Prompt or PowerShell):**

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

Verify:

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## 3. Configuration

Edit `configs/config.yaml`:

```yaml
training:
  device: cpu    # use cpu on laptops without GPU; or auto / cuda
  batch_size: 16
  epochs: 25
```

On Windows, `num_workers: auto` uses **0** (stable). Training on CPU can take a long time — reduce `epochs` for a quick test.

---

## 4. Train and evaluate

Activate the venv, then from the project root:

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
data/             data.zip (LFS), raw_images after unpack
notebooks/        experiments.ipynb
scripts/          CLI entry points
src/              model, training, experiments, feature maps
outputs/          created when you train (local)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `data.zip` ~130 bytes | `git lfs install` then `git lfs pull` |
| Missing `label.csv` | `python scripts/unpack_dataset.py` |
| No `data/processed` | `python scripts/split_dataset.py` |
| No checkpoint / Streamlit error | Run `python scripts/train.py` |
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

Repository: [github.com/mentisVeritas/cnn-cast-defect-classification](https://github.com/mentisVeritas/cnn-cast-defect-classification)
