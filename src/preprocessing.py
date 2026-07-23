"""Audio preprocessing: single source of truth for wav to mel-spectrogram.

Every part of the system (notebook, prediction, retraining) imports from here so
the transformation stays identical across train, predict, and retrain.
"""

from typing import List

import librosa
import numpy as np

SAMPLE_RATE: int = 22050
DURATION: float = 5.0
SAMPLES_PER_CLIP: int = int(SAMPLE_RATE * DURATION)
N_MELS: int = 128
N_FFT: int = 2048
HOP_LENGTH: int = 512


def load_audio(file_path: str) -> np.ndarray:
    """Load a wav file as a fixed-length mono waveform at SAMPLE_RATE."""
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(waveform) < SAMPLES_PER_CLIP:
        waveform = np.pad(waveform, (0, SAMPLES_PER_CLIP - len(waveform)))
    return waveform[:SAMPLES_PER_CLIP]


def to_melspectrogram(waveform: np.ndarray) -> np.ndarray:
    """Convert a waveform into a normalized log-mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    normalized = (log_mel + 80.0) / 80.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def augment(waveform: np.ndarray) -> List[np.ndarray]:
    """Return noise-added, time-shifted, and pitch-shifted variants for training."""
    noisy = waveform + 0.005 * np.random.randn(len(waveform)).astype(np.float32)
    shifted = np.roll(waveform, int(SAMPLE_RATE * 0.2))
    pitched = librosa.effects.pitch_shift(waveform, sr=SAMPLE_RATE, n_steps=2)
    return [noisy, shifted, pitched.astype(np.float32)]


def preprocess_file(file_path: str) -> np.ndarray:
    """Full path to a single model-ready input tensor with shape (N_MELS, frames, 1)."""
    waveform = load_audio(file_path)
    spec = to_melspectrogram(waveform)
    return spec[..., np.newaxis]
