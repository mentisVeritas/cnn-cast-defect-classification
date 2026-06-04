# Deployment screenshots (BTEC evidence)

## Automated previews (reproducible)

Generated without a browser — same inference pipeline as Streamlit:

```bash
python scripts/generate_deployment_previews.py
```

Files created in this folder:

| File | Content |
|------|---------|
| `01_home.png` | Application home / empty state |
| `02_defect_upload.png` | Defect image selected |
| `03_defect_prediction.png` | Defect prediction result |
| `04_normal_upload.png` | Normal image selected |
| `05_normal_prediction.png` | Normal prediction result |

## Manual Streamlit screenshots (recommended for report)

1. Activate venv and run:
   ```bash
   streamlit run app/streamlit_app.py
   ```
2. Open `http://localhost:8501` in a browser.
3. Capture:
   - Home page (no upload)
   - **Prediction** tab with a defect image + bar chart
   - **Feature maps** tab
   - **Grad-CAM** tab
   - **Model metrics** tab
4. Save PNGs here as `streamlit_home.png`, `streamlit_defect.png`, etc.

Use the same test images as in `outputs/feature_maps/` for consistency.
