"""Single-datapoint prediction for a wav file."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_file

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH: str = str(MODELS_DIR / "sound_model.h5")
CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"


def load_class_names() -> List[str]:
    """Load the ordered class labels saved during training."""
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


def predict_with_model(
    model: tf.keras.Model, class_names: List[str], file_path: str
) -> Dict[str, object]:
    """Predict using an already-loaded model. Used by the API to avoid reloading."""
    spec: np.ndarray = preprocess_file(file_path)[np.newaxis, ...]
    probabilities: np.ndarray = model.predict(spec, verbose=0)[0]
    index: int = int(probabilities.argmax())

    scores: Dict[str, float] = {
        name: float(score) for name, score in zip(class_names, probabilities)
    }
    return {
        "label": class_names[index],
        "confidence": float(probabilities[index]),
        "scores": scores,
    }


def predict_file(file_path: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    """Predict the class of one wav file by loading the model from disk.

    Returns the predicted label, its confidence, and per-class scores.
    """
    model: tf.keras.Model = tf.keras.models.load_model(model_path)
    class_names: List[str] = load_class_names()
    return predict_with_model(model, class_names, file_path)
