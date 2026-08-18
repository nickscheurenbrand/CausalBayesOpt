"""Fit the PFN prior mean used by GEOMETRY_SCALE (appendix E.4).

The PFN is trained ONCE, offline, on synthetic draws from the prior in
utils.pfn_prior (random smooth functions on the unit sphere, a controlled share
of them monotone so both boundary and interior optima are represented). The
resulting checkpoint is then reused unchanged across runs -- that is the point
of an amortised prior.

    python scripts_geometry/train_pfn.py --steps 4000 --out checkpoints/pfn.pt
    python scripts_geometry/train_pfn.py --steps 200 --out /tmp/pfn_small.pt

--max_dim must be at least (largest intervention-set size) + 1, since the PFN
sees the PROJECTED input, which lives in R^{D+1}.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.pfn_prior import (ConstantPFN, SphericalPriorSampler, save_pfn,
                             train_pfn)


def evaluate(prior, n_tasks: int = 50, n_context: int = 24, n_query: int = 16,
             max_dim: int = 10, seed: int = 999):
    """Held-out MSE against the ConstantPFN baseline, on the normalised scale."""
    sampler = SphericalPriorSampler(max_dim=max_dim, seed=seed)
    baseline = ConstantPFN()
    errs, base_errs = [], []
    for _ in range(n_tasks):
        X, y = sampler.sample_task(n_context + n_query)
        y = (y - y.mean()) / (y.std() + 1e-8)
        Xc, yc, Xq, yq = X[:n_context], y[:n_context], X[n_context:], y[n_context:]
        errs.append(np.mean((prior.mean(Xc, yc, Xq) - yq) ** 2))
        base_errs.append(np.mean((baseline.mean(Xc, yc, Xq) - yq) ** 2))
    return float(np.mean(errs)), float(np.mean(base_errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="checkpoints/pfn.pt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--max_dim", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_context", type=int, default=24)
    p.add_argument("--n_query", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    pfn, history = train_pfn(
        max_dim=args.max_dim, steps=args.steps, batch_size=args.batch_size,
        n_context=args.n_context, n_query=args.n_query, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, lr=args.lr,
        seed=args.seed, device=args.device,
    )

    mse, base = evaluate(pfn, max_dim=args.max_dim, n_context=args.n_context,
                         n_query=args.n_query)
    print(f"held-out MSE: PFN {mse:.4f} vs constant-prior baseline {base:.4f} "
          f"({100 * (1 - mse / base):.1f}% better)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_pfn(pfn, args.out, n_heads=args.n_heads, steps=args.steps,
             final_loss=float(np.mean(history[-100:])), heldout_mse=mse,
             baseline_mse=base)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
