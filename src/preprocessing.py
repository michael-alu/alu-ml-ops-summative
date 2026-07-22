"""Audio preprocessing: single source of truth for wav to mel-spectrogram.

Every part of the system (notebook, prediction, retraining) imports from here so
the transformation stays identical across train, predict, and retrain.
"""

from typing import Tuple

import numpy as np

SAMPLE_RATE: int = 22050
DURATION_SECONDS: float = 4.0
N_MELS: int = 128
N_FFT: int = 2048
HOP_LENGTH: int = 512


def load_wav(file_path: str) -> Tuple[np.ndarray, int]:
    """Load a wav file as a mono waveform at SAMPLE_RATE."""
    raise NotImplementedError


def wav_to_melspectrogram(waveform: np.ndarray) -> np.ndarray:
    """Convert a waveform into a fixed-size log-mel-spectrogram."""
    raise NotImplementedError


def preprocess_file(file_path: str) -> np.ndarray:
    """Full path -> model-ready input tensor. Used at predict and retrain time."""
    raise NotImplementedError
