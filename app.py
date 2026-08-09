"""
app.py
------
Streamlit deployment for the Pneumonia Detection system.

Features: drag-and-drop upload, prediction + confidence, Grad-CAM overlay,
downloadable PDF report, dark-mode-friendly theming, sidebar with model
info and in-session prediction history.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import CONFIG, ensure_directories
from src.inference import InferenceEngine
from src.preprocessing import to_rgb
from src.threshold_optimizer import load_recommended_threshold

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(127, 127, 127, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .pred-pneumonia { color: #e35d5d; font-weight: 700; }
    .pred-normal { color: #4caf82; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_engine(model_path: str) -> InferenceEngine:
    return InferenceEngine(
        model_path=model_path,
        class_names=CONFIG.data.class_names,
        image_size=CONFIG.data.image_size,
        use_clahe=CONFIG.data.use_clahe,
    )


def make_pdf_report(image: Image.Image, result: dict, filename: str) -> bytes:
    """Build a simple downloadable PDF report using fpdf2."""
    from fpdf import FPDF

    tmp_dir = Path("/tmp/pneumonia_reports")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / f"{filename}_input.png"
    overlay_path = tmp_dir / f"{filename}_overlay.png"
    image.save(img_path)
    Image.fromarray(result["overlay"]).save(overlay_path)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Pneumonia Detection Report", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Prediction: {result['prediction']}", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Confidence: {result['confidence'] * 100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Decision threshold used: {result['threshold_used']:.2f}", ln=True)
    pdf.ln(4)

    img_w = 85
    pdf.image(str(img_path), x=10, w=img_w)
    pdf.image(str(overlay_path), x=105, w=img_w)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This tool is a research/portfolio project and is NOT a "
        "certified medical device. It must not be used for real clinical "
        "diagnosis. Always consult a qualified radiologist or physician."
    )

    return bytes(pdf.output(dest="S"))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🫁 Model Info")
    model_path = st.text_input("Model path", value=str(CONFIG.paths.best_model_path))
    tta_enabled = st.toggle("Test-Time Augmentation (TTA)", value=False, help="Averages predictions over several augmented views for a more robust probability estimate.")

    st.markdown("---")
    st.subheader("Decision Threshold")
    default_threshold, threshold_provenance = load_recommended_threshold(CONFIG.paths.reports_dir)
    threshold = st.slider(
        "Threshold applied to the PNEUMONIA probability",
        min_value=0.01, max_value=0.99, value=float(default_threshold), step=0.01,
        help="Defaults to whatever tune_threshold.py last recommended on the validation set. "
             "Move the slider to explore other values, but the frozen recommendation is the "
             "scientifically defensible default — see the provenance note below.",
    )
    if threshold == default_threshold:
        st.caption(f"✓ {threshold_provenance}")
    else:
        st.caption(f"⚠ Manually overridden — validation-tuned default was {default_threshold:.2f}.")

    st.markdown("---")
    st.markdown(
        f"""
        **Architecture:** {CONFIG.model.backbone}
        **Input size:** {CONFIG.model.input_shape[0]}×{CONFIG.model.input_shape[1]}
        **Classes:** {", ".join(CONFIG.data.class_names)}
        """
    )

    st.markdown("---")
    st.subheader("Prediction History")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            st.write(f"**{h['prediction']}** — {h['confidence'] * 100:.1f}% ({h['filename']})")
    else:
        st.caption("No predictions yet this session.")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ Research/portfolio demo only. Not a certified diagnostic device. "
        "Do not use for real clinical decisions."
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("Pneumonia Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray and the model will classify it as **NORMAL** or "
    "**PNEUMONIA**, along with a Grad-CAM heatmap explaining which regions "
    "influenced the decision."
)

if not Path(model_path).exists():
    st.warning(
        f"No trained model found at `{model_path}`. Train one first with "
        "`python train.py`, then reload this app. The UI below will work "
        "once a model checkpoint exists."
    )

uploaded_file = st.file_uploader("Drag and drop a chest X-ray", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None and Path(model_path).exists():
    with st.spinner("Analyzing X-ray..."):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = to_rgb(np.array(image))

        engine = load_engine(model_path)
        result = engine.predict(image_np, tta=tta_enabled, threshold=threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Explanation")
        st.image(result["overlay"], use_container_width=True)

    pred_class_css = "pred-pneumonia" if result["prediction"] == "PNEUMONIA" else "pred-normal"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h4>Prediction</h4>'
            f'<p class="{pred_class_css}" style="font-size:1.6rem;">{result["prediction"]}</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h4>Confidence</h4>'
            f'<p style="font-size:1.6rem;">{result["confidence"] * 100:.2f}%</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h4>Raw Probability</h4>'
            f'<p style="font-size:1.6rem;">{result["raw_probability"]:.4f}</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h4>Threshold Used</h4>'
            f'<p style="font-size:1.6rem;">{result["threshold_used"]:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    st.progress(float(result["confidence"]))

    st.session_state.history.append(
        {"filename": uploaded_file.name, "prediction": result["prediction"], "confidence": result["confidence"]}
    )

    st.markdown("### Download Report")
    pdf_bytes = make_pdf_report(image, result, filename=Path(uploaded_file.name).stem)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"pneumonia_report_{Path(uploaded_file.name).stem}.pdf",
        mime="application/pdf",
    )

    with st.expander("What is the model looking at?"):
        st.write(
            "The Grad-CAM heatmap highlights regions of the X-ray that most "
            "strongly influenced the prediction — warmer colors (red/yellow) "
            "indicate higher influence. For pneumonia cases this typically "
            "corresponds to areas of consolidation or infiltrate."
        )

st.markdown("---")
st.caption("Built with TensorFlow/Keras, EfficientNet-B0, Grad-CAM, and Streamlit.")
