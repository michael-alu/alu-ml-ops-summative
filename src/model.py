"""Model definition and training helpers for the sound classifier."""

from typing import List, Tuple

import numpy as np


def build_model(input_shape: Tuple[int, int, int], num_classes: int):
    """Build the optimized CNN (regularization, dropout, chosen optimizer)."""
    raise NotImplementedError


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
):
    """Train with early stopping and return the fitted model plus history."""
    raise NotImplementedError


def save_model(model, path: str) -> None:
    """Persist the trained model to disk (.h5)."""
    raise NotImplementedError


def load_saved_model(path: str):
    """Load a persisted model from disk."""
    raise NotImplementedError
