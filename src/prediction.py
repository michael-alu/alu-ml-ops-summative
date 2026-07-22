"""Single-datapoint prediction for a wav file."""

from typing import Dict


def predict_file(file_path: str, model_path: str) -> Dict[str, float]:
    """Predict the class of one wav file.

    Returns the predicted label and per-class confidence scores.
    """
    raise NotImplementedError
