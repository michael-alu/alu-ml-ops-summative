"""FastAPI service: prediction, bulk upload, and retrain trigger.

This is the image that gets dockerized, load-tested with Locust, and deployed.
"""

from typing import Dict, List

from fastapi import FastAPI, UploadFile

app: FastAPI = FastAPI(title="Sound Classifier API")


@app.get("/health")
def health() -> Dict[str, str]:
    """Uptime check consumed by the UI."""
    raise NotImplementedError


@app.post("/predict")
def predict(file: UploadFile) -> Dict[str, float]:
    """Predict the class of a single uploaded wav file."""
    raise NotImplementedError


@app.post("/upload")
def upload(files: List[UploadFile]) -> Dict[str, int]:
    """Save bulk wav files for later retraining."""
    raise NotImplementedError


@app.post("/retrain")
def retrain_endpoint() -> Dict[str, float]:
    """Trigger retraining on the uploaded data and return new metrics."""
    raise NotImplementedError
