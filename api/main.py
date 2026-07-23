"""FastAPI service: prediction, bulk upload, and retrain trigger.

This is the image that gets dockerized, load-tested with Locust, and deployed.
The model is loaded once at startup and reused so predictions stay fast.
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.prediction import load_class_names, predict_with_model
from src.retrain import retrain

ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_PATH: Path = ROOT / "models" / "sound_model.h5"
UPLOADS_DIR: Path = ROOT / "data" / "uploads"

app: FastAPI = FastAPI(title="Sound Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME: float = time.time()
state: Dict[str, object] = {"model": None, "class_names": []}


def load_model_into_state() -> None:
    """Load the model and labels from disk into memory."""
    if MODEL_PATH.exists():
        state["model"] = tf.keras.models.load_model(MODEL_PATH)
        state["class_names"] = load_class_names()


@app.on_event("startup")
def startup() -> None:
    load_model_into_state()


@app.get("/health")
def health() -> Dict[str, object]:
    """Uptime and readiness check consumed by the UI."""
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "classes": state["class_names"],
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    """Predict the class of a single uploaded wav file."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path: str = tmp.name

    try:
        return predict_with_model(state["model"], state["class_names"], tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/upload")
def upload(label: str = Form(...), files: List[UploadFile] = File(...)) -> Dict[str, object]:
    """Save bulk wav files under data/uploads/<label> for later retraining."""
    dest_dir: Path = UPLOADS_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: int = 0
    for f in files:
        if not f.filename.endswith(".wav"):
            continue
        with open(dest_dir / f.filename, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved += 1

    return {"label": label, "saved": saved, "upload_dir": str(dest_dir)}


@app.post("/retrain")
def retrain_endpoint() -> Dict[str, object]:
    """Trigger retraining on the uploaded data, then reload the new model."""
    try:
        metrics: Dict[str, object] = retrain()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    load_model_into_state()
    return {"status": "retrained", "metrics": metrics}
