"""
micro_mlp.py
------------
A tiny, dependency-free (numpy only) one-hidden-layer MLP with manual
backprop. This stands in for a "small neural net" (e.g. a mini DQN) so the
scripts don't require installing torch/tensorflow. Good enough for the
low-dimensional elevator state (8 features) used here.
"""

import numpy as np


class MicroMLP:
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int = 16, lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        # He-ish init
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_inputs), size=(n_inputs, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_outputs)).astype(np.float32)
        self.b2 = np.zeros(n_outputs, dtype=np.float32)

    def forward(self, x: np.ndarray):
        """x: shape (n_inputs,) or (batch, n_inputs). Returns output + cache for backward."""
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        out = z2
        cache = (x, z1, a1)
        if single:
            out = out[0]
        return out, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x)
        return out

    def train_step(self, x: np.ndarray, target: np.ndarray, action_mask: np.ndarray = None):
        """One SGD step of squared-error regression towards `target`.
        If action_mask is given (same shape as output), only those output
        units contribute to the loss/gradient (used for Q-learning where we
        only know the target for the action that was taken)."""
        single = x.ndim == 1
        if single:
            x = x[None, :]
            target = target[None, :]
            if action_mask is not None:
                action_mask = action_mask[None, :]

        out, (x_c, z1, a1) = self.forward(x)
        error = out - target  # dL/dout for 0.5*sum((out-target)^2)
        if action_mask is not None:
            error = error * action_mask

        batch = x_c.shape[0]
        dW2 = a1.T @ error / batch
        db2 = error.mean(axis=0)
        da1 = error @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = x_c.T @ dz1 / batch
        db1 = dz1.mean(axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return float((error ** 2).sum() / batch)
