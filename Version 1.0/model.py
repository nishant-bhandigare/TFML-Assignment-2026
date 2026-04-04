"""
model.py
========
Part 2: Fully Connected Feedforward Neural Network — built from scratch with NumPy.

Architecture : 64 -> 3 -> 3
  - Input layer   : 64 units  (8×8 flattened pixels)
  - Hidden layer  : H units   (default 3), tanh activation, with bias
  - Output layer  : 3 units   (one per class), softmax activation, with bias

Training:
  - Loss      : Cross-entropy
  - Optimizer : Adam  (adaptive moment estimation)
  - Gradients : Exact backpropagation (no autograd)
"""

import numpy as np


# ─── Activation Functions ─────────────────────────────────────────────────────

def tanh(z):
    """Hyperbolic tangent: maps ℝ -> (−1, +1)."""
    return np.tanh(z)


def tanh_deriv(z):
    """Derivative of tanh with respect to its pre-activation input z."""
    return 1.0 - np.tanh(z) ** 2


def softmax(z):
    """
    Numerically stable row-wise softmax.
    Subtracts the row-max before exponentiation to prevent overflow.
    """
    # z shape: (batch, n_classes)  or  (n_classes,)
    z = np.atleast_2d(z)
    z_shifted = z - z.max(axis=1, keepdims=True)   # numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ─── Neural Network Class ─────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully connected 64-H-3 feedforward neural network.

    Parameters
    ----------
    input_dim  : int   — number of input features (64 for 8×8 grid)
    hidden_dim : int   — number of hidden units H
    output_dim : int   — number of output classes (3)
    seed       : int   — random seed for weight initialisation
    """

    def __init__(self, input_dim: int = 64,
                 hidden_dim: int = 3,
                 output_dim: int = 3,
                 seed: int = 0):

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        rng = np.random.default_rng(seed)

        # ── Weight Initialisation (Xavier / Glorot uniform) ──────────────────
        # Keeps activations in a healthy range at the start of training.
        # W1: (input_dim,  hidden_dim),  b1: (hidden_dim,)
        # W2: (hidden_dim, output_dim),  b2: (output_dim,)

        lim1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-lim1, lim1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        lim2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = rng.uniform(-lim2, lim2, size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        # ── Adam optimiser state (one moment per weight matrix / bias) ───────
        self._init_adam()

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> dict:
        """
        Compute activations for a batch X of shape (batch, 64).

        Returns a cache dict with all intermediate values needed for backprop.
        """
        # Hidden layer
        Z1 = X @ self.W1 + self.b1          # (batch, H)
        A1 = tanh(Z1)                        # (batch, H)  — tanh activation

        # Output layer
        Z2 = A1 @ self.W2 + self.b2         # (batch, 3)
        A2 = softmax(Z2)                    # (batch, 3)  — softmax probabilities

        return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for input X. Shape: (batch, 3)."""
        return self.forward(X)["A2"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class index (argmax of softmax). Shape: (batch,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Loss ──────────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray,
                           y_onehot: np.ndarray) -> float:
        """
        Mean cross-entropy loss over a batch.

        L = − (1/N) Σ_i Σ_k  y_ik · log(p_ik)

        A small epsilon avoids log(0).
        """
        eps = 1e-12
        return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

    # ── Backpropagation ───────────────────────────────────────────────────────

    def backward(self, cache: dict, y_onehot: np.ndarray) -> dict:
        """
        Compute gradients via backpropagation.

        The derivative of (softmax + cross-entropy) w.r.t. Z2 simplifies to:
            dL/dZ2 = (A2 − y) / N

        Parameters
        ----------
        cache    : dict returned by forward()
        y_onehot : (batch, 3) one-hot targets

        Returns
        -------
        grads : dict with keys dW1, db1, dW2, db2
        """
        N  = y_onehot.shape[0]
        A1 = cache["A1"]        # (N, H)
        A2 = cache["A2"]        # (N, 3)
        Z1 = cache["Z1"]        # (N, H)
        X  = cache["X"]         # (N, 64)

        # ── Output layer gradient ─────────────────────────────────────────
        dZ2 = (A2 - y_onehot) / N           # (N, 3)
        dW2 = A1.T @ dZ2                    # (H, 3)
        db2 = dZ2.sum(axis=0)               # (3,)

        # ── Hidden layer gradient ─────────────────────────────────────────
        dA1 = dZ2 @ self.W2.T               # (N, H)
        dZ1 = dA1 * tanh_deriv(Z1)          # (N, H)  — chain rule through tanh
        dW1 = X.T @ dZ1                     # (64, H)
        db1 = dZ1.sum(axis=0)               # (H,)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ── Adam Optimiser ────────────────────────────────────────────────────────

    def _init_adam(self):
        """Initialise first- and second-moment vectors for Adam."""
        self.t  = 0   # time-step counter

        # First moments (mean of gradients)
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)

        # Second moments (uncentred variance of gradients)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def _adam_update(self, grads: dict,
                     lr: float = 0.001,
                     beta1: float = 0.9,
                     beta2: float = 0.999,
                     eps: float = 1e-8):
        """
        Apply one Adam parameter update step.

        Adam (Kingma & Ba 2015):
            m ← β1·m + (1−β1)·g
            v ← β2·v + (1−β2)·g²
            m̂ = m / (1−β1^t)
            v̂ = v / (1−β2^t)
            θ ← θ − lr · m̂ / (√v̂ + ε)
        """
        self.t += 1
        t = self.t
        bc1 = 1 - beta1 ** t      # bias-correction factor for first moment
        bc2 = 1 - beta2 ** t      # bias-correction factor for second moment

        def _update(param, m, v, grad):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / bc1
            v_hat = v / bc2
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            return param, m, v

        self.W1, self.mW1, self.vW1 = _update(self.W1, self.mW1, self.vW1, grads["dW1"])
        self.b1, self.mb1, self.vb1 = _update(self.b1, self.mb1, self.vb1, grads["db1"])
        self.W2, self.mW2, self.vW2 = _update(self.W2, self.mW2, self.vW2, grads["dW2"])
        self.b2, self.mb2, self.vb2 = _update(self.b2, self.mb2, self.vb2, grads["db2"])

    # ── Convenience: save / load weights ──────────────────────────────────────

    def get_weights(self) -> dict:
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2}

    def set_weights(self, weights: dict):
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]

    def save(self, path: str):
        out = path if str(path).endswith(".npz") else f"{path}.npz"
        np.savez(out, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 hidden_dim=np.array([self.hidden_dim]))
        print(f"Model weights saved -> {out}")

    @classmethod
    def load(cls, path: str, input_dim: int = 64, output_dim: int = 3):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        hidden_dim = int(data["hidden_dim"][0])
        net = cls(input_dim=input_dim, hidden_dim=hidden_dim,
                  output_dim=output_dim)
        net.set_weights({"W1": data["W1"], "b1": data["b1"],
                         "W2": data["W2"], "b2": data["b2"]})
        print(f"Model loaded from {path}  (hidden_dim={hidden_dim})")
        return net

    def __repr__(self):
        return (f"NeuralNetwork({self.input_dim}->{self.hidden_dim}->{self.output_dim})"
                f"  params={self.W1.size+self.b1.size+self.W2.size+self.b2.size}")
