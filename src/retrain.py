"""Retraining pipeline triggered by user-uploaded wav files.

Steps: read uploaded files -> preprocess -> continue training from the existing
model (used as a pretrained base) -> evaluate -> save the new model version.
"""

from typing import Dict, List


def collect_uploaded_data(upload_dir: str) -> List[str]:
    """Gather the uploaded wav file paths saved for retraining."""
    raise NotImplementedError


def retrain(upload_dir: str, base_model_path: str, output_model_path: str) -> Dict[str, float]:
    """Run the full retraining flow and return the new evaluation metrics."""
    raise NotImplementedError
