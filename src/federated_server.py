"""
Minimal FedAvg server — no external FL framework required.

Aggregates model weights from heterogeneous clients using weighted averaging.
Designed for cross-domain evaluation: TrustLens AI (credit-risk, tabular)
and Hospitality Sustainability AI (electricity forecasting, time-series).
"""

import numpy as np
from typing import List, Tuple, Dict


def fedavg(
    client_weights: List[np.ndarray],
    client_sizes: List[int],
) -> np.ndarray:
    """
    Federated averaging (McMahan et al. 2017).

    Parameters
    ----------
    client_weights : list of 1D arrays — flattened model parameters per client
    client_sizes   : number of training samples per client (used for weighting)

    Returns
    -------
    Aggregated global weights as a 1D array.
    """
    total = sum(client_sizes)
    agg = np.zeros_like(client_weights[0], dtype=float)
    for w, n in zip(client_weights, client_sizes):
        agg += w * (n / total)
    return agg


def run_federation(
    clients: List,
    rounds: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Run a full federation loop.

    Each client must implement:
      .train(global_weights) -> local_weights, n_samples
      .evaluate(global_weights) -> metrics dict
      .name: str

    Returns dict of per-round, per-client evaluation metrics.
    """
    # Initialise from first client
    global_weights = clients[0].init_weights()
    history = {c.name: [] for c in clients}

    for r in range(1, rounds + 1):
        local_weights, sizes = [], []
        for client in clients:
            w, n = client.train(global_weights)
            local_weights.append(w)
            sizes.append(n)

        global_weights = fedavg(local_weights, sizes)

        if verbose:
            print(f"\n--- Round {r}/{rounds} ---")
        for client in clients:
            metrics = client.evaluate(global_weights)
            history[client.name].append(metrics)
            if verbose:
                print(f"  {client.name}: {metrics}")

    return history
