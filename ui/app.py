"""Streamlit UI: uptime, data visualizations, prediction, upload, and retrain.

Runs the model in-process (imports the same src modules the API uses) so the app
is self-contained on Streamlit Community Cloud, no separate API host needed.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

from src.prediction import load_class_names, predict_with_model
from src.retrain import retrain

ASSETS_DIR: Path = ROOT / "assets"
MODEL_PATH: Path = ROOT / "models" / "sound_model.h5"
UPLOADS_DIR: Path = ROOT / "data" / "uploads"


@st.cache_resource
def get_start_time() -> float:
    """App start time, stable across reruns (cached for the app lifetime)."""
    return time.time()


@st.cache_resource
def load_model_and_classes() -> Tuple[tf.keras.Model, List[str]]:
    """Load the model and labels once and reuse across reruns."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, load_class_names()


def load_insights() -> Optional[Dict[str, object]]:
    path = ASSETS_DIR / "insights.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render_status() -> None:
    st.sidebar.header("Model status")
    st.sidebar.metric("App uptime (seconds)", round(time.time() - get_start_time(), 1))
    if MODEL_PATH.exists():
        _, class_names = load_model_and_classes()
        st.sidebar.success("Model loaded")
        st.sidebar.caption("Classes: " + ", ".join(class_names))
    else:
        st.sidebar.warning("Model not found")


def tab_predict() -> None:
    st.subheader("Predict a single sound")
    audio_file = st.file_uploader("Upload a .wav clip", type=["wav"], key="predict")
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Predict", type="primary"):
            with st.spinner("Predicting ..."):
                model, class_names = load_model_and_classes()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name
                try:
                    result = predict_with_model(model, class_names, tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            st.success(f"Prediction: {result['label']} ({result['confidence']:.1%})")
            scores = pd.Series(result["scores"]).sort_values(ascending=False)
            st.bar_chart(scores)


def tab_insights() -> None:
    st.subheader("Data insights")
    insights = load_insights()
    if insights is None:
        st.info("Run scripts/generate_insights.py to create the insight charts.")
        return

    cols = st.columns(3)
    cols[0].metric("Clips", insights["num_clips"])
    cols[1].metric("Classes", insights["num_classes"])
    cols[2].metric("Clips per class", insights["clips_per_class"])

    for feature in insights["features"]:
        st.markdown(f"### {feature['name']}")
        chart_path = ASSETS_DIR / feature["chart"]
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        st.caption(feature["story"])

    spec_path = ASSETS_DIR / "melspectrograms.png"
    if spec_path.exists():
        st.markdown("### Mel-spectrograms")
        st.image(str(spec_path), use_container_width=True)


def save_uploads(label: str, uploads: List) -> int:
    dest = UPLOADS_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in uploads:
        if not f.name.endswith(".wav"):
            continue
        (dest / f.name).write_bytes(f.getvalue())
        saved += 1
    return saved


def tab_retrain() -> None:
    st.subheader("Upload data and retrain")
    _, class_names = load_model_and_classes()

    label = st.selectbox("Class label for the uploaded clips", class_names)
    uploads = st.file_uploader(
        "Upload multiple .wav clips", type=["wav"], accept_multiple_files=True, key="bulk"
    )

    if uploads and label and st.button("Upload clips"):
        saved = save_uploads(label, uploads)
        st.success(f"Saved {saved} clips for label '{label}'")

    st.divider()
    if st.button("Trigger retraining", type="primary"):
        with st.spinner("Retraining on uploaded data ..."):
            try:
                metrics = retrain()
            except ValueError as error:
                st.error(str(error))
                return
        load_model_and_classes.clear()
        st.success("Retraining complete")
        st.json(metrics)


def main() -> None:
    st.set_page_config(page_title="Sound Classifier", page_icon="🔊", layout="wide")
    st.title("🔊 Sound Classification MLOps Dashboard")
    render_status()

    predict, insights, retrain_tab = st.tabs(["Predict", "Data Insights", "Retrain"])
    with predict:
        tab_predict()
    with insights:
        tab_insights()
    with retrain_tab:
        tab_retrain()


if __name__ == "__main__":
    main()
