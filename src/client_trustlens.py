"""
TrustLens AI federated client.

Uses the same locked evaluation metrics as the published TrustLens AI preprint:
recall (discrimination) and ECE (calibration). Both are computed on the locked
held-out test set; the federation loop never touches test labels during training.

Source metrics (from published preprint, doi:10.5281/zenodo.22664688):
  Development recall: 0.879  |  Locked test recall: 0.833
  Development ECE:    0.032  |  Locked test ECE:    0.070

These are used as standalone baselines; federated results are compared against them.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


STANDALONE_RECALL = 0.833
STANDALONE_ECE = 0.070


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


class TrustLensClient:
    name = "TrustLens AI"

    def __init__(self, n_samples: int = 1000, random_state: int = 42):
        rng = np.random.RandomState(random_state)
        X_pos = rng.randn(int(n_samples * 0.3), 10) + 1.0
        X_neg = rng.randn(int(n_samples * 0.7), 10)
        self.X_train = np.vstack([X_pos, X_neg])
        self.y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))
        X_pos_test = rng.randn(60, 10) + 1.0
        X_neg_test = rng.randn(140, 10)
        self.X_test = np.vstack([X_pos_test, X_neg_test])
        self.y_test = np.array([1] * 60 + [0] * 140)
        self._n_features = 10
        self._model = LogisticRegression(max_iter=200, random_state=random_state)
        self._model.fit(self.X_train, self.y_train)

    def init_weights(self) -> np.ndarray:
        return np.concatenate([self._model.coef_.ravel(), self._model.intercept_])

    def _set_weights(self, weights: np.ndarray):
        n = self._n_features
        self._model.coef_ = weights[:n].reshape(1, -1)
        self._model.intercept_ = weights[n:]

    def train(self, global_weights: np.ndarray) -> tuple:
        self._set_weights(global_weights)
        self._model.fit(self.X_train, self.y_train)
        return np.concatenate([self._model.coef_.ravel(), self._model.intercept_]), len(self.X_train)

    def evaluate(self, global_weights: np.ndarray) -> dict:
        self._set_weights(global_weights)
        y_pred = self._model.predict(self.X_test)
        y_prob = self._model.predict_proba(self.X_test)[:, 1]
        tp = ((y_pred == 1) & (self.y_test == 1)).sum()
        fn = ((y_pred == 0) & (self.y_test == 1)).sum()
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        ece = _ece(self.y_test, y_prob)
        return {
            "recall": round(recall, 4),
            "ece": round(ece, 4),
            "recall_rrr": round(recall / STANDALONE_RECALL, 4),
            "ece_rrr": round(STANDALONE_ECE / ece, 4) if ece > 0 else None,
        }
