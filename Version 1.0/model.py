"""
model.py
========
Part 2: Fully Connected Feedforward Neural Network — built with TensorFlow/Keras.

Architecture : 64 -> H -> 3  (default H = 3)
  - Input layer   : 64 units  (8×8 flattened pixels)
  - Hidden layer  : H units,  tanh activation + L2 regularisation, with bias
  - Output layer  : 3 units,  softmax activation, with bias

Training:
  - Loss      : Categorical cross-entropy
  - Optimizer : Adam with optional learning-rate decay
  - Callbacks : Early stopping, ReduceLROnPlateau

# AI-generated code — all logic produced with Claude assistance.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 0):
    tf.random.set_seed(seed)
    np.random.seed(seed)


# ─── Neural Network Class ─────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully connected 64-H-3 feedforward neural network backed by Keras.

    Matches the NumPy version's external interface so that train.py,
    visualisation.py, and the web app need no changes.

    Parameters
    ----------
    input_dim   : int   — number of input features (64 for 8×8 grid)
    hidden_dim  : int   — number of hidden units H
    output_dim  : int   — number of output classes (3)
    seed        : int   — random seed for weight initialisation
    l2_lambda   : float — L2 regularisation strength on hidden weights
    """

    def __init__(self,
                 input_dim: int  = 64,
                 hidden_dim: int = 3,
                 output_dim: int = 3,
                 seed: int       = 0,
                 l2_lambda: float = 1e-4):

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed       = seed
        self.l2_lambda  = l2_lambda

        set_seed(seed)
        self.model = self._build()

    # ── Model Construction ────────────────────────────────────────────────────

    def _build(self) -> keras.Model:
        """
        Construct and compile the Keras model.

        Improvements over the NumPy baseline
        -------------------------------------
        * L2 regularisation on hidden weights — penalises large weights and
          reduces overfitting on the small 300-sample dataset.
        * GlorotUniform initialiser (Xavier) on both layers — same principle
          as the NumPy version but guaranteed consistent across seeds.
        * Compiled with label_smoothing=0.05 — softens one-hot targets slightly
          (e.g. [0.975, 0.0125, 0.0125] instead of [1, 0, 0]), which prevents
          the model from driving softmax outputs to extremes and improves
          generalisation.
        """
        init = keras.initializers.GlorotUniform(seed=self.seed)

        model = keras.Sequential([
            # Hidden layer — tanh + L2
            layers.Dense(
                self.hidden_dim,
                activation="tanh",
                kernel_initializer=init,
                bias_initializer="zeros",
                kernel_regularizer=regularizers.l2(self.l2_lambda),
                input_shape=(self.input_dim,),
                name="hidden",
            ),
            # Output layer — softmax
            layers.Dense(
                self.output_dim,
                activation="softmax",
                kernel_initializer=init,
                bias_initializer="zeros",
                name="output",
            ),
        ], name=f"net_{self.input_dim}_{self.hidden_dim}_{self.output_dim}")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )
        return model

    # ── Forward Pass (NumPy-compatible interface) ──────────────────────────────

    def forward(self, X: np.ndarray) -> dict:
        """
        Run a full forward pass and return intermediate activations.

        Compatible with the NumPy version's cache dict so that
        visualisation code that inspects 'A1' and 'A2' still works.
        """
        X_t = tf.constant(X, dtype=tf.float32)

        hidden_layer = self.model.get_layer("hidden")
        output_layer = self.model.get_layer("output")

        # Hidden pre-activation and activation
        Z1 = X_t @ tf.cast(hidden_layer.kernel, tf.float32) \
             + tf.cast(hidden_layer.bias, tf.float32)
        A1 = tf.nn.tanh(Z1)

        # Output pre-activation and softmax
        Z2 = A1 @ tf.cast(output_layer.kernel, tf.float32) \
             + tf.cast(output_layer.bias, tf.float32)
        A2 = tf.nn.softmax(Z2)

        return {
            "X":  X,
            "Z1": Z1.numpy(),
            "A1": A1.numpy(),
            "Z2": Z2.numpy(),
            "A2": A2.numpy(),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities. Shape: (batch, 3)."""
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices. Shape: (batch,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Loss ──────────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray,
                           y_onehot: np.ndarray) -> float:
        """Mean cross-entropy loss (no label smoothing) for reporting."""
        eps = 1e-12
        return float(-np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1)))

    # ── Weight access (used by visualisation.py) ──────────────────────────────

    def get_weights(self) -> dict:
        """Return weight arrays in the same format as the NumPy version."""
        W1, b1 = self.model.get_layer("hidden").get_weights()
        W2, b2 = self.model.get_layer("output").get_weights()
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def set_weights(self, weights: dict):
        self.model.get_layer("hidden").set_weights(
            [weights["W1"], weights["b1"]])
        self.model.get_layer("output").set_weights(
            [weights["W2"], weights["b2"]])

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save weights to a .npz file (mirrors NumPy version's format)."""
        out = path if path.endswith(".npz") else f"{path}.npz"
        w = self.get_weights()
        np.savez(out, W1=w["W1"], b1=w["b1"],
                 W2=w["W2"], b2=w["b2"],
                 hidden_dim=np.array([self.hidden_dim]))

    @classmethod
    def load(cls, path: str, input_dim: int = 64, output_dim: int = 3):
        """Load weights from a .npz file saved by save()."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        hidden_dim = int(data["hidden_dim"][0])
        net = cls(input_dim=input_dim, hidden_dim=hidden_dim,
                  output_dim=output_dim)
        net.set_weights({"W1": data["W1"], "b1": data["b1"],
                         "W2": data["W2"], "b2": data["b2"]})
        return net

    def __repr__(self):
        n_params = (self.input_dim * self.hidden_dim + self.hidden_dim
                    + self.hidden_dim * self.output_dim + self.output_dim)
        return (f"NeuralNetwork(TF) {self.input_dim}->"
                f"{self.hidden_dim}->{self.output_dim}  params={n_params}")