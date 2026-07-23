"""Retraining pipeline triggered by user-uploaded wav files.

Steps: read uploaded files organized by class subfolder -> preprocess -> continue
training from the existing model (used as a pretrained base) -> evaluate -> save.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.preprocessing import augment, load_audio, to_melspectrogram

ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = ROOT / "models"
UPLOADS_DIR: Path = ROOT / "data" / "uploads"
MODEL_PATH: Path = MODELS_DIR / "sound_model.h5"
CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"


def load_class_names() -> List[str]:
    """Load the ordered class labels saved during training."""
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


def collect_uploaded_data(upload_dir: str) -> List[Tuple[str, str]]:
    """Return (wav_path, class_name) pairs from class subfolders under upload_dir."""
    pairs: List[Tuple[str, str]] = []
    base: Path = Path(upload_dir)
    if not base.exists():
        return pairs
    for class_dir in sorted(base.iterdir()):
        if class_dir.is_dir():
            for wav in sorted(class_dir.glob("*.wav")):
                pairs.append((str(wav), class_dir.name))
    return pairs


def build_arrays(
    pairs: List[Tuple[str, str]], class_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess uploaded wavs into model inputs, skipping unknown classes."""
    specs: List[np.ndarray] = []
    labels: List[int] = []
    for wav_path, class_name in pairs:
        if class_name not in class_names:
            continue
        waveform = load_audio(wav_path)
        for variant in [waveform] + augment(waveform):
            specs.append(to_melspectrogram(variant))
            labels.append(class_names.index(class_name))
    x = np.array(specs)[..., np.newaxis]
    y = np.array(labels)
    return x, y


def retrain(
    upload_dir: str = str(UPLOADS_DIR),
    base_model_path: str = str(MODEL_PATH),
    output_model_path: str = str(MODEL_PATH),
) -> Dict[str, object]:
    """Run the full retraining flow and return metrics for the new model."""
    class_names: List[str] = load_class_names()
    pairs: List[Tuple[str, str]] = collect_uploaded_data(upload_dir)
    if not pairs:
        raise ValueError("No uploaded wav files found for retraining")

    x, y = build_arrays(pairs, class_names)
    if len(y) == 0:
        raise ValueError("Uploaded classes do not match any known label")

    model: tf.keras.Model = tf.keras.models.load_model(base_model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
    ]
    history = model.fit(x, y, epochs=20, batch_size=16, callbacks=callbacks, verbose=2)
    model.save(output_model_path)

    loss, accuracy = model.evaluate(x, y, verbose=0)
    return {
        "samples": int(len(y)),
        "epochs_ran": int(len(history.history["loss"])),
        "final_loss": float(loss),
        "final_accuracy": float(accuracy),
    }
