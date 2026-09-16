"""
Cross-domain federation evaluation.

Compares per-domain performance under federation vs standalone baselines,
using the same Robustness Retention Ratio (RRR) framework as the
cross-domain robustness synthesis (doi:10.5281/zenodo.22777883).
Research question: does federating across heterogeneous domains
(credit-risk + electricity forecasting) help, hurt, or leave
per-domain performance unchanged relative to standalone training?
"""

import json, argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from federated_server import run_federation
from client_trustlens import TrustLensClient, STANDALONE_RECALL, STANDALONE_ECE
from client_hospitality import HospitalityClient, STANDALONE_RRR


def main(rounds: int = 10, output: str = None, verbose: bool = True):
    clients = [
        TrustLensClient(n_samples=1000, random_state=42),
        HospitalityClient(n_samples=800, random_state=42),
    ]
    print("=== Cross-Domain Federation Evaluation ===")
    print(f"Clients: {[c.name for c in clients]}")
    print(f"Rounds: {rounds}")
    print(f"\nStandalone baselines:")
    print(f"  TrustLens AI - recall: {STANDALONE_RECALL}, ECE: {STANDALONE_ECE}")
    print(f"  Hospitality AI - temporal RRR: {STANDALONE_RRR:.4f}")
    history = run_federation(clients, rounds=rounds, verbose=verbose)
    print("\n=== Final round results ===")
    final = {name: rounds_list[-1] for name, rounds_list in history.items()}
    for name, metrics in final.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    tl = final["TrustLens AI"]
    hosp = final["Hospitality Sustainability AI"]
    print("\n=== Federation vs Standalone ===")
    if tl.get("recall_rrr") is not None:
        delta = tl["recall"] - STANDALONE_RECALL
        print(f"TrustLens recall {'improved' if delta > 0 else 'degraded'} by {abs(delta):.4f} under federation")
    if hosp.get("federation_delta_rrr") is not None:
        delta = hosp["federation_delta_rrr"]
        print(f"Hospitality RRR {'improved' if delta > 0 else 'degraded'} by {abs(delta):.4f} under federation")
    result = {"rounds": rounds, "history": history, "final": final}
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain federation evaluation")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(rounds=args.rounds, output=args.output, verbose=not args.quiet)
