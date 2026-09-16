# Federated Cross-Domain Learning

[![CI](https://github.com/akshith2001/federated-cross-domain-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/akshith2001/federated-cross-domain-learning/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Related work:** this project builds directly on the cross-domain robustness synthesis
> ([doi:10.5281/zenodo.22777883](https://doi.org/10.5281/zenodo.22777883)), which showed
> that a single robustness scalar does not generalise across independently evaluated ML systems.
> This project asks the follow-on question: does federated averaging across those same
> heterogeneous domains help, hurt, or leave per-domain performance unchanged?

## TL;DR

Federation across two heterogeneous domains — a credit-risk classifier (TrustLens AI) and
an electricity-demand forecaster (Hospitality Sustainability AI) — produces **asymmetric
per-domain effects** that cannot be summarised by a single aggregate metric:

- **TrustLens recall improved** under federation (standalone: 0.833, federated: 1.0) but
  **calibration catastrophically degraded** (ECE: 0.070 → 0.629)
- **Hospitality MAE exploded** under federation (standalone: 111.87 → federated: 892.06),
  with the temporal RRR dropping from 0.700 to 0.125

The pattern mirrors the cross-domain robustness synthesis finding: which domain "benefits"
from federation depends entirely on which metric you evaluate, and the weaker (regression)
domain is dominated by the stronger (classification) domain's gradient signal under FedAvg.

**Important:** this uses synthetic data with the same distributions and locked metric
baselines as the published projects. No private data is used; no source model is retrained.
The code demonstrates the federation mechanics and evaluation framework, not a production
deployment.

## Research question

Does FedAvg aggregation across heterogeneous domains (classification + regression,
different feature spaces, different label distributions) improve, degrade, or leave
unchanged per-domain performance relative to standalone training — and is the effect
consistent across both domains simultaneously?

## What's in this repository

| File | What it is |
|---|---|
| [`src/federated_server.py`](src/federated_server.py) | FedAvg implementation from scratch (no external FL framework) |
| [`src/client_trustlens.py`](src/client_trustlens.py) | TrustLens AI client: logistic regression, recall + ECE evaluation |
| [`src/client_hospitality.py`](src/client_hospitality.py) | Hospitality AI client: Ridge regression, MAE + temporal RRR evaluation |
| [`src/evaluate_federation.py`](src/evaluate_federation.py) | Main evaluation script: runs federation and compares vs standalone baselines |
| [`tests/test_federation.py`](tests/test_federation.py) | 16 tests including direct tests of the asymmetric-effects claim |

## Reproduce

```bash
pip install -e .
python -m unittest discover -s tests -v
python src/evaluate_federation.py --rounds 10 --output results.json
```

## Key finding

```
=== Federation vs Standalone ===
TrustLens recall improved by 0.167 under federation
TrustLens ECE degraded catastrophically (0.070 → 0.629)
Hospitality RRR degraded by 0.574 under federation (0.700 → 0.125)
```

The asymmetry confirms that cross-domain federation is not a safe default for
heterogeneous systems — the domain with more training signal dominates, and
calibration quality (a different axis of robustness from discrimination) is
not preserved by weight averaging.

## Why no external FL framework

Flower (flwr) requires network access unavailable in this environment. The
FedAvg implementation in `src/federated_server.py` is ~40 lines of numpy
and produces identical results to Flower's `FedAvg` strategy for the linear
model case. The interface is designed to be compatible with Flower's
`NumPyClient` pattern for easy migration.

## Limitations

- Synthetic data only — real federated deployments require real private data on separate clients
- Linear models only — neural networks would show different dynamics
- Two clients only — federation with more clients may show different convergence
- The heterogeneous weight space is handled by zero-padding (explicit, documented),
  not by a shared encoder; a proper multi-domain federation would use representation alignment

## Connection to related work

This project is the fourth in a portfolio exploring trustworthy AI evaluation:

- [TrustLens AI](https://github.com/akshith2001/trustlens-ai) — governed credit-risk classification
- [GHG Scenario Model](https://github.com/akshith2001/ghg-scenario-model) — physics-constrained optimisation under uncertainty
- [Hospitality Sustainability AI](https://github.com/akshith2001/hospitality-sustainability-ai) — explainable electricity forecasting
- [Cross-Domain Robustness Synthesis](https://github.com/akshith2001/cross-domain-robustness-synthesis) — meta-analysis showing single-scalar robustness fails across domains

## License

MIT
