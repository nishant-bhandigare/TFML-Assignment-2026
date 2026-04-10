"""TensorFlow model definitions for 64-X-3 assignment network."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(hidden_units: int = 32, dropout_rate: float = 0.1) -> tf.keras.Model:
    """Build a 64-X-3 MLP with regularization for robust validation performance."""
    model = models.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(hidden_units, activation="relu", name="hidden"),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
