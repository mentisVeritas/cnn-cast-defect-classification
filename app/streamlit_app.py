"""Streamlit demo: image prediction and training metrics.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Theme
COLOR_DEFECT = "#e85d4c"
COLOR_NORMAL = "#14b8a6"
COLOR_PRIMARY = "#1e3a5f"
COLOR_ACCENT = "#3b82f6"


def _inject_styles() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] {{
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem;
        max-width: 1200px;
    }}
    .hero {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, #2d4a6f 55%, #1a2f4a 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.25);
    }}
    .hero h1 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    .hero p {{
        margin: 0;
        opacity: 0.88;
        font-size: 1rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }}
    .badge-defect {{
        background: rgba(232, 93, 76, 0.15);
        color: {COLOR_DEFECT};
        border: 1px solid rgba(232, 93, 76, 0.35);
    }}
    .badge-normal {{
        background: rgba(20, 184, 166, 0.15);
        color: {COLOR_NORMAL};
        border: 1px solid rgba(20, 184, 166, 0.35);
    }}
    .pred-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
    }}
    .pred-card h4 {{
        margin: 0.5rem 0 0.25rem 0;
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
    }}
    .conf-bar {{
        height: 8px;
        border-radius: 4px;
        background: #e2e8f0;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .conf-fill {{
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, {COLOR_ACCENT}, #60a5fa);
    }}
    div[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }}
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 0.5rem 0 1.25rem 0;
    }}
    .kpi-row.cols-3 {{
        grid-template-columns: repeat(3, 1fr);
    }}
    .kpi-card {{
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }}
    .kpi-label {{
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.15;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }}
    </style>
    """


def _kpi_row_html(items: list[tuple[str, str]], *, cols: int = 4) -> str:
    """Metric cards with explicit colors (works in Streamlit dark theme)."""
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in items
    )
    cls = "kpi-row cols-3" if cols == 3 else "kpi-row"
    return f'<div class="{cls}">{cards}</div>'


def _badge_html(label: str, confidence: float) -> str:
    cls = "badge-defect" if label.lower() == "defect" else "badge-normal"
    return (
        f'<span class="badge {cls}">{label.upper()}</span> '
        f'<span style="color:#64748b;font-size:0.9rem;">{confidence:.1%} confidence</span>'
    )


def _pred_card_html(filename: str, label: str, confidence: float) -> str:
    pct = int(confidence * 100)
    return f"""
    <div class="pred-card">
        <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.35rem;">{filename}</div>
        {_badge_html(label, confidence)}
        <div class="conf-bar"><div class="conf-fill" style="width:{pct}%;"></div></div>
    </div>
    """


def _load_metrics() -> dict | None:
    path = ROOT / "outputs" / "metrics" / "evaluation_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _style_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#475569",
            "axes.titlecolor": COLOR_PRIMARY,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )


def _plot_confusion_matrix(metrics: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    _style_matplotlib()
    cm = np.array(metrics["confusion_matrix"])
    names = metrics.get("class_names", ["defect", "normal"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(names)), labels=names, fontsize=11)
    ax.set_yticks(range(len(names)), labels=names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion matrix — test set", fontsize=12, pad=12)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else COLOR_PRIMARY
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def _plot_training_curves():
    import matplotlib.pyplot as plt

    path = ROOT / "outputs" / "metrics" / "training_history.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        history = json.load(f)
    if not history.get("train_loss"):
        return None

    _style_matplotlib()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    axes[0].plot(epochs, history["train_loss"], color=COLOR_ACCENT, lw=2, label="Train")
    axes[0].plot(epochs, history["val_loss"], color=COLOR_DEFECT, lw=2, ls="--", label="Val")
    axes[0].set_title("Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, history["train_accuracy"], color=COLOR_ACCENT, lw=2, label="Train")
    axes[1].plot(epochs, history["val_accuracy"], color=COLOR_NORMAL, lw=2, ls="--", label="Val")
    axes[1].set_title("Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def _feature_map_figure(activation, title: str):
    import matplotlib.pyplot as plt

    _style_matplotlib()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(activation, cmap="magma")
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def _prob_chart_figure(probabilities: dict):
    import matplotlib.pyplot as plt

    _style_matplotlib()
    labels = list(probabilities.keys())
    values = [probabilities[k] for k in labels]
    colors = [COLOR_DEFECT if lb == "defect" else COLOR_NORMAL for lb in labels]

    fig, ax = plt.subplots(figsize=(6, 2.2))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class probabilities", fontsize=11, pad=8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def render() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st
    from PIL import Image

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.explain import CONV_BLOCKS, extract_all_feature_maps, image_to_tensor
    from src.inference import load_model_for_inference, predict_pil_image
    from src.utils import get_device, load_config

    st.set_page_config(
        page_title="Cast Defect Inspector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_inject_styles(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>Cast Defect Inspector</h1>
            <p>Manufacturing surface quality · defect vs normal · powered by custom CNN</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        config = load_config(ROOT / "configs/config.yaml")
        device = get_device(config)
        model, class_names, device = load_model_for_inference(config, device=device)
    except FileNotFoundError:
        st.error("No trained model found. Run `python scripts/train.py` first.")
        st.stop()

    with st.sidebar:
        st.markdown("### 📤 Upload")
        uploaded_files = st.file_uploader(
            "Cast surface images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### ⚙️ System")
        device_label = str(device).replace("cuda", "GPU").replace("cpu", "CPU").replace("mps", "Apple GPU")
        st.success(f"Model loaded · {device_label}")
        st.caption(f"Classes: **{', '.join(class_names)}**")
        ckpt = ROOT / "outputs" / "models" / "best_model.pth"
        if ckpt.exists():
            st.caption(f"Checkpoint: `{ckpt.name}`")

    tab_pred, tab_maps, tab_metrics = st.tabs(
        ["🔮 Prediction", "🧠 Feature maps", "📊 Model metrics"]
    )

    results: list[dict] = []
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            result = predict_pil_image(img, model, class_names, config, device)
            results.append({"name": f.name, "image": img, "result": result})

    with tab_metrics:
        metrics = _load_metrics()
        if metrics:
            st.markdown("#### Test set performance")
            st.markdown(
                _kpi_row_html(
                    [
                        ("Accuracy", f"{metrics['accuracy']:.1%}"),
                        ("Precision", f"{metrics['precision']:.1%}"),
                        ("Recall", f"{metrics['recall']:.1%}"),
                        ("F1 score", f"{metrics['f1_score']:.1%}"),
                    ]
                ),
                unsafe_allow_html=True,
            )

            col_cm, col_hist = st.columns([1, 1])
            with col_cm:
                st.pyplot(_plot_confusion_matrix(metrics))
                plt.close()
            with col_hist:
                fig_hist = _plot_training_curves()
                if fig_hist:
                    st.pyplot(fig_hist)
                    plt.close()
                else:
                    st.info("No training history file yet.")
        else:
            st.warning("Run `python scripts/evaluate.py` to load test metrics.")

        st.markdown("---")
        st.markdown("#### Saved training plots")
        p_loss, p_acc = st.columns(2)
        for col, name, rel in [
            (p_loss, "Loss curve", "outputs/plots/loss_curve.png"),
            (p_acc, "Accuracy curve", "outputs/plots/accuracy_curve.png"),
        ]:
            p = ROOT / rel
            with col:
                if p.exists():
                    st.image(str(p), caption=name, use_container_width=True)

        exp_cols = st.columns(2)
        tables = [
            ("CNN vs ResNet18", "outputs/baseline/model_comparison.csv"),
            ("Learning rate tuning", "outputs/tuning/tuning_results.csv"),
        ]
        for col, (title, rel) in zip(exp_cols, tables):
            p = ROOT / rel
            with col:
                st.markdown(f"**{title}**")
                if p.exists():
                    st.dataframe(
                        pd.read_csv(p),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("_Not generated yet_")

    with tab_maps:
        if not results:
            st.info("Upload an image in the sidebar to view conv layer activations.")
        else:
            labels = [
                f"{s['name']} → {s['result']['predicted_class']} ({s['result']['confidence']:.0%})"
                for s in results
            ]
            pick = st.selectbox("Image for feature maps", labels, key="fm_image")
            sample = results[labels.index(pick)]
            tensor = image_to_tensor(sample["image"], config, device)
            maps = extract_all_feature_maps(model, tensor)

            st.caption(
                "Mean activation per conv block — brighter regions = stronger response. "
                "Block 1: edges/texture · Block 2: patterns · Block 3: higher-level regions."
            )
            c0, c1 = st.columns([1, 2])
            with c0:
                st.image(sample["image"], caption="Input image", use_container_width=True)
                st.markdown(
                    _pred_card_html(
                        sample["name"],
                        sample["result"]["predicted_class"],
                        sample["result"]["confidence"],
                    ),
                    unsafe_allow_html=True,
                )
            with c1:
                cols = st.columns(3)
                for col, (block, (_, block_title)) in zip(cols, CONV_BLOCKS.items()):
                    with col:
                        st.pyplot(_feature_map_figure(maps[block], block_title))
                        plt.close()

    with tab_pred:
        if not results:
            st.markdown(
                """
                <div style="text-align:center;padding:3rem 1rem;color:#64748b;
                background:#f8fafc;border-radius:14px;border:2px dashed #cbd5e1;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">📷</div>
                <strong>Upload images in the sidebar</strong><br>
                <span style="font-size:0.9rem;">JPEG or PNG · one or many cast photos</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            defect_n = sum(1 for s in results if s["result"]["predicted_class"] == "defect")
            normal_n = len(results) - defect_n
            st.markdown(
                _kpi_row_html(
                    [
                        ("Images", str(len(results))),
                        ("Defect", str(defect_n)),
                        ("Normal", str(normal_n)),
                    ],
                    cols=3,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("#### Results")
            cols_per_row = min(3, len(results))
            for row_start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, sample in zip(cols, results[row_start : row_start + cols_per_row]):
                    r = sample["result"]
                    with col:
                        st.image(sample["image"], use_container_width=True)
                        st.markdown(
                            _pred_card_html(sample["name"], r["predicted_class"], r["confidence"]),
                            unsafe_allow_html=True,
                        )

            st.markdown("#### Summary table")
            st.dataframe(
                [
                    {
                        "File": s["name"],
                        "Prediction": s["result"]["predicted_class"],
                        "Confidence": f"{s['result']['confidence']:.1%}",
                        "Defect %": f"{s['result']['probabilities'].get('defect', 0):.1%}",
                        "Normal %": f"{s['result']['probabilities'].get('normal', 0):.1%}",
                    }
                    for s in results
                ],
                use_container_width=True,
                hide_index=True,
            )

            if len(results) == 1:
                st.markdown("#### Probability breakdown")
                st.pyplot(_prob_chart_figure(results[0]["result"]["probabilities"]))
                plt.close()


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_inside_streamlit():
    render()
elif __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
            cwd=ROOT,
        )
    )
