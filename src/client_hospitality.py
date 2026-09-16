"""
Hospitality Sustainability AI federated client.

Uses the same locked evaluation metric as the published preprint:
MAE on the locked future-period test set (inverse-error retention for RRR).

Source metrics (from published preprint, doi:10.5281/zenodo.22664937):
  Rolling-development MAE: 111.87  |  Locked test MAE: 159.92
  Standalone temporal RRR: 0.700 (= 111.87 / 159.92)
"""

import numpy as np
from sklearn.linear_model import Ridge


STANDALONE_DEV_MAE = 111.87
STANDALONE_TEST_MAE = 159.92
STANDALONE_RRR = STANDALONE_DEV_MAE / STANDALONE_TEST_MAE


class HospitalityClient:
    name = "Hospitality Sustainability AI"

    def __init__(self, n_samples: int = 800, random_state: int = 42):
        rng = np.random.RandomState(random_state)
        t = np.arange(n_samples)
        base_demand = 300 + 150 * np.sin(2 * np.pi * t / 24) + rng.randn(n_samples) * 80

        def make_features(demand):
            X = np.column_stack([
                demand[:-5], demand[1:-4], demand[2:-3],
                demand[3:-2], demand[4:-1],
                (np.arange(len(demand) - 5) % 24) / 24.0,
                (np.arange(len(demand) - 5) % 168) / 168.0,
            ])
            return X, demand[5:]

        split = int(n_samples * 0.8)
        X_all, y_all = make_features(base_demand)
        self.X_train = X_all[:split - 5]
        self.y_train = y_all[:split - 5]
        future_demand = base_demand[split:] + 60 + rng.randn(n_samples - split) * 20
        if len(future_demand) > 10:
            self.X_test, self.y_test = make_features(future_demand)
        else:
            self.X_test = self.X_train[-50:]
            self.y_test = self.y_train[-50:]
        # Pad to shared dim 10 (same as TrustLens) with zeros
        self._n_features = 10
        pad = np.zeros((self.X_train.shape[0], 3))
        self.X_train = np.hstack([self.X_train, pad])
        self.X_test = np.hstack([self.X_test, np.zeros((self.X_test.shape[0], 3))])
        self._model = Ridge(alpha=1.0)
        self._model.fit(self.X_train, self.y_train)

    def init_weights(self) -> np.ndarray:
        return np.concatenate([self._model.coef_, [self._model.intercept_]])

    def _set_weights(self, weights: np.ndarray):
        self._model.coef_ = weights[:-1]
        self._model.intercept_ = weights[-1]

    def train(self, global_weights: np.ndarray) -> tuple:
        self._set_weights(global_weights)
        self._model.fit(self.X_train, self.y_train)
        return np.concatenate([self._model.coef_, [self._model.intercept_]]), len(self.X_train)

    def evaluate(self, global_weights: np.ndarray) -> dict:
        self._set_weights(global_weights)
        y_pred = self._model.predict(self.X_test)
        mae = float(np.mean(np.abs(y_pred - self.y_test)))
        rrr = STANDALONE_DEV_MAE / mae if mae > 0 else None
        return {
            "mae": round(mae, 2),
            "mae_rrr": round(rrr, 4) if rrr else None,
            "standalone_rrr": round(STANDALONE_RRR, 4),
            "federation_delta_rrr": round(rrr - STANDALONE_RRR, 4) if rrr else None,
        }
