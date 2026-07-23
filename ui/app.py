"""Streamlit UI: uptime, data visualizations, prediction, upload, and retrain.

Talks to the FastAPI service over HTTP. API base URL comes from an env var so the
same UI works locally and when deployed.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = ROOT / "assets"


def get_health() -> Optional[Dict[str, object]]:
    """Fetch API health, or None if the service is unreachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def load_insights() -> Optional[Dict[str, object]]:
    path = ASSETS_DIR / "insights.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def render_status() -> None:
    st.sidebar.header("Model status")
    health = get_health()
    if health is None:
        st.sidebar.error("API unreachable")
        return
    st.sidebar.metric("Uptime (seconds)", health["uptime_seconds"])
    if health["model_loaded"]:
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.warning("Model not loaded")
    st.sidebar.caption("Classes: " + ", ".join(health.get("classes", [])))


def tab_predict() -> None:
    st.subheader("Predict a single sound")
    audio_file = st.file_uploader("Upload a .wav clip", type=["wav"], key="predict")
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Predict", type="primary"):
            files = {"file": (audio_file.name, audio_file.getvalue(), "audio/wav")}
            with st.spinner("Predicting ..."):
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=60)
            if response.ok:
                result = response.json()
                st.success(f"Prediction: {result['label']} ({result['confidence']:.1%})")
                scores = pd.Series(result["scores"]).sort_values(ascending=False)
                st.bar_chart(scores)
            else:
                st.error(f"Prediction failed: {response.text}")


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


def tab_retrain() -> None:
    st.subheader("Upload data and retrain")
    health = get_health()
    classes: List[str] = health.get("classes", []) if health else []

    label = st.selectbox("Class label for the uploaded clips", classes) if classes else st.text_input("Class label")
    uploads = st.file_uploader("Upload multiple .wav clips", type=["wav"], accept_multiple_files=True, key="bulk")

    if uploads and label and st.button("Upload clips"):
        files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in uploads]
        response = requests.post(f"{API_BASE_URL}/upload", data={"label": label}, files=files, timeout=120)
        if response.ok:
            st.success(f"Saved {response.json()['saved']} clips for label '{label}'")
        else:
            st.error(f"Upload failed: {response.text}")

    st.divider()
    if st.button("Trigger retraining", type="primary"):
        with st.spinner("Retraining on uploaded data ..."):
            response = requests.post(f"{API_BASE_URL}/retrain", timeout=1800)
        if response.ok:
            metrics = response.json()["metrics"]
            st.success("Retraining complete")
            st.json(metrics)
        else:
            st.error(f"Retraining failed: {response.text}")


def main() -> None:
    st.set_page_config(page_title="Sound Classifier", page_icon="🔊", layout="wide")
    st.title("🔊 Sound Classification MLOps Dashboard")
    render_status()

    predict, insights, retrain = st.tabs(["Predict", "Data Insights", "Retrain"])
    with predict:
        tab_predict()
    with insights:
        tab_insights()
    with retrain:
        tab_retrain()


if __name__ == "__main__":
    main()
