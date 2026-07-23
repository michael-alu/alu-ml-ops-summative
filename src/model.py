"""Model definition and training helpers for the sound classifier."""

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build the optimized CNN (BatchNorm, L2 regularization, dropout, Adam)."""
    def bn() -> layers.BatchNormalization:
        return layers.BatchNormalization(momentum=0.9)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        bn(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        bn(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        bn(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    """Train with early stopping and learning-rate reduction."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=6, factor=0.5, min_lr=1e-5),
    ]
    return model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )


def save_model(model: tf.keras.Model, path: str) -> None:
    """Persist the trained model to disk (.h5)."""
    model.save(path)


def load_saved_model(path: str) -> tf.keras.Model:
    """Load a persisted model from disk."""
    return tf.keras.models.load_model(path)
