"""Pre-run diagnostic for the GWPS CBO-U setup: computes each node's do-effect slope and SNR analytically (numpy + networkx
only, no GPy/jax) so you can pick a --weight_scale where the best lever's SNR ~ 0.5-2 before a full CBO-U job."""

import argparse
import os
import sys

import numpy as np

# make `graphs.gwps_build` importable whether run from repo root or scripts_gwps/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "algorithms")):
    if p not in sys.path:
        sys.path.append(p)

from graphs.gwps_build import build_gwps_dag


def analyse(W, target, noise_sigma):
    N = W.shape[0]
    M = np.linalg.inv(np.eye(N) - W)          # (I-W)^-1 = sum over paths
    slope = M[:, target]                       # do-effect of each X on target
    node_std = np.sqrt((M ** 2).sum(axis=0)) * noise_sigma
    std_y = node_std[target]
    snr = np.abs(slope) * node_std / max(std_y, 1e-12)
    snr[target] = 0.0                          # not a lever on itself
    return slope, node_std, std_y, snr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--top_k_parents", type=int, default=8)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--scales", type=str, default="1,2,3,5,10",
                   help="weight_scale values to compare")
    return p.parse_args()


def main():
    args = parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    # structure is scale-independent; build once at scale 1 for the report
    b = build_gwps_dag(target=args.target, max_nodes=args.max_nodes,
                       top_k_parents=args.top_k_parents, weight_scale=1.0)
    W0, i2e, tgt = b["W"], b["index_to_ensg"], b["target_index"]
    N = W0.shape[0]
    print(f"target: node {tgt} = {i2e[tgt]}   ({N} nodes)")

    # direct parents of the target and their raw G_hat
    parents = [(i, W0[i, tgt]) for i in range(N) if W0[i, tgt] != 0]
    parents.sort(key=lambda t: -abs(t[1]))
    print(f"\ndirect parents of target ({len(parents)}): node(ENSG) = G_hat")
    for i, w in parents:
        print(f"   {i:>3} ({i2e[i]}): {w:+.4f}")

    print(f"\n--- signal-to-noise vs --weight_scale (noise_sigma={args.noise_sigma}) ---")
    print(f"{'scale':>6} {'best lever':>28} {'|slope|':>8} {'SNR':>6}  {'#levers SNR>0.5':>16}")
    for s in scales:
        W = W0 * s
        slope, node_std, std_y, snr = analyse(W, tgt, args.noise_sigma)
        best = int(np.argmax(snr))
        nstrong = int((snr > 0.5).sum())
        print(f"{s:>6.1f} {best:>3} ({i2e[best]}) {abs(slope[best]):>8.3f} {snr[best]:>6.2f} {nstrong:>16}")

    print("\nGuidance: pick --weight_scale so the best lever's SNR is ~0.5-2 and a "
          "few levers exceed 0.5. Too low => CBO can't detect effects; too high => "
          "the linear SEM's variance blows up (watch node std).")


if __name__ == "__main__":
    main()
