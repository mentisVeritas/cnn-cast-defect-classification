"""Streamlit demo with predictions, feature maps, Grad-CAM, and training metrics.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_metrics() -> dict | None:
    path = ROOT / "outputs" / "metrics" / "evaluation_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _plot_confusion_matrix(metrics: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(metrics["confusion_matrix"])
    names = metrics.get("class_names", ["defect", "normal"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), labels=names)
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def _plot_training_curves():
    import matplotlib.pyplot as plt

    path = ROOT / "outputs" / "metrics" / "training_history.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def render() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st
    from PIL import Image

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.explain import (
        CONV_LAYERS,
        extract_all_feature_maps,
        extract_feature_maps,
        grad_cam,
        image_to_tensor,
        overlay_heatmap,
        predict_tensor,
    )
    from src.inference import load_model_for_inference
    from src.utils import get_device, load_config, resolve_path

    st.set_page_config(
        page_title="Cast Defect CNN",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Cast Defect Classifier")
    st.caption("CNN demo: prediction · feature maps · Grad-CAM · training metrics")

    try:
        config = load_config(ROOT / "configs/config.yaml")
        device = get_device(config)
        model, class_names, device = load_model_for_inference(config, device=device)
    except FileNotFoundError:
        st.error("No checkpoint found. Train first: `python scripts/train.py`")
        st.stop()

    st.sidebar.header("Input")
    uploaded_files = st.sidebar.file_uploader(
        "Cast images (one or many)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    layer_key = st.sidebar.selectbox(
        "CNN layer (maps / Grad-CAM)",
        list(CONV_LAYERS.keys()),
        index=2,
    )
    gradcam_class = st.sidebar.selectbox(
        "Grad-CAM target class",
        class_names,
        index=0,
    )

    tab_pred, tab_maps, tab_cam, tab_metrics = st.tabs(
        ["Prediction", "Feature maps", "Grad-CAM", "Model metrics"]
    )

    # --- Metrics tab (no upload needed) ---
    with tab_metrics:
        metrics = _load_metrics()
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            c2.metric("Precision", f"{metrics['precision']:.1%}")
            c3.metric("Recall", f"{metrics['recall']:.1%}")
            c4.metric("F1", f"{metrics['f1_score']:.1%}")
            st.pyplot(_plot_confusion_matrix(metrics))
            plt.close()
        else:
            st.info("Run `python scripts/evaluate.py` to generate test metrics.")

        fig_hist = _plot_training_curves()
        if fig_hist:
            st.subheader("Training history")
            st.pyplot(fig_hist)
            plt.close()
        else:
            st.info("No `outputs/metrics/training_history.json` yet.")

        for name, rel in [
            ("Loss curve", "outputs/plots/loss_curve.png"),
            ("Accuracy curve", "outputs/plots/accuracy_curve.png"),
        ]:
            p = ROOT / rel
            if p.exists():
                st.image(str(p), caption=name, use_container_width=True)

    if not uploaded_files:
        with tab_pred:
            st.info("Upload one or more images in the sidebar.")
        return

    samples: list[dict] = []
    for f in uploaded_files:
        img = Image.open(f).convert("RGB")
        tensor = image_to_tensor(img, config, device)
        result = predict_tensor(model, tensor, class_names)
        samples.append(
            {
                "name": f.name,
                "image": img,
                "tensor": tensor,
                "result": result,
            }
        )

    target_idx = class_names.index(gradcam_class)
    n = len(samples)

    # --- Prediction tab ---
    with tab_pred:
        defect_n = sum(1 for s in samples if s["result"]["predicted_class"] == "defect")
        normal_n = sum(1 for s in samples if s["result"]["predicted_class"] == "normal")
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", n)
        c2.metric("Predicted defect", defect_n)
        c3.metric("Predicted normal", normal_n)

        rows = [
            {
                "file": s["name"],
                "prediction": s["result"]["predicted_class"],
                "confidence": f"{s['result']['confidence']:.1%}",
                "defect": f"{s['result']['probabilities'].get('defect', 0):.1%}",
                "normal": f"{s['result']['probabilities'].get('normal', 0):.1%}",
            }
            for s in samples
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Gallery")
        cols_per_row = min(4, n)
        for row_start in range(0, n, cols_per_row):
            cols = st.columns(cols_per_row)
            for col, sample in zip(cols, samples[row_start : row_start + cols_per_row]):
                r = sample["result"]
                with col:
                    st.image(
                        sample["image"],
                        caption=f"{sample['name']}: **{r['predicted_class']}** ({r['confidence']:.0%})",
                        use_container_width=True,
                    )

        if n == 1:
            st.subheader("Probabilities")
            st.bar_chart(
                {k: v for k, v in samples[0]["result"]["probabilities"].items()},
                horizontal=True,
            )

    # Pick one image for deep dive (maps / Grad-CAM)
    image_labels = [
        f"{s['name']} → {s['result']['predicted_class']} ({s['result']['confidence']:.0%})"
        for s in samples
    ]
    selected_label = st.sidebar.selectbox(
        "Image for maps / Grad-CAM",
        image_labels,
        index=0,
    )
    selected_idx = image_labels.index(selected_label)
    sample = samples[selected_idx]
    image = sample["image"]
    tensor = sample["tensor"]
    result = sample["result"]

    # --- Feature maps tab ---
    with tab_maps:
        st.write(
            f"**{sample['name']}** — activation heatmaps after **{layer_key}**."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original", use_container_width=True)
        with c2:
            fmap = extract_feature_maps(model, tensor, layer_key)
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Mean activation — {layer_key}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
            plt.close()

        st.subheader("All convolutional layers")
        all_maps = extract_all_feature_maps(model, tensor)
        cols = st.columns(len(all_maps))
        for col, (name, fmap) in zip(cols, all_maps.items()):
            with col:
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.imshow(fmap, cmap="magma")
                ax.set_title(name, fontsize=9)
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

    # --- Grad-CAM tab ---
    with tab_cam:
        st.write(
            f"**{sample['name']}** — Grad-CAM for class **{gradcam_class}**."
        )
        heatmap = grad_cam(model, tensor, target_idx, layer_key)
        overlay = overlay_heatmap(image, heatmap, alpha=0.5)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(image, caption="Original", use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(heatmap, cmap="jet")
            ax.set_title(f"Grad-CAM → {gradcam_class}")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        with c3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        st.caption(
            f"Layer: {layer_key}. "
            f"Prediction: **{result['predicted_class']}** ({result['confidence']:.1%})."
        )


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
