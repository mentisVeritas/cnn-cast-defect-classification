"""Streamlit demo: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_model_for_inference, predict_image
from src.utils import get_device, load_config

st.title("Cast Defect Classifier")
st.write("Upload image: **normal** or **defect**")

try:
    config = load_config(ROOT / "configs/config.yaml")
    device = get_device(config)
    model, class_names, device = load_model_for_inference(config, device=device)
except FileNotFoundError:
    st.error("Train the model first: python scripts/train.py")
    st.stop()

uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=300)

    tmp = ROOT / "outputs" / "temp.jpg"
    tmp.parent.mkdir(exist_ok=True)
    image.save(tmp)

    result = predict_image(tmp, model, class_names, config, device)
    st.write(f"**{result['predicted_class']}** — {result['confidence']*100:.1f}%")
    st.json(result["probabilities"])
