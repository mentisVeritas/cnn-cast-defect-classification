"""Streamlit demo.

Run:
    streamlit run app/streamlit_app.py

Or press Run in PyCharm on this file (starts streamlit automatically).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render() -> None:
    import streamlit as st
    from PIL import Image

    if str(ROOT) not in sys.path:
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


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_inside_streamlit():
    render()
elif __name__ == "__main__":
    app_path = Path(__file__).resolve()
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=ROOT,
        )
    )
