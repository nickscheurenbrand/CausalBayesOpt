"""
Thorough test of whether the CBO-U runs exhibit a *boundary bias* in the
intervention values they select -- i.e. do chosen intervention values pile up
near the edges of each variable's intervention range more than chance would
predict?

Runs entirely off the saved boundary-tracking pickles (no rerun). For each
graph size it computes, per selected intervention (all single-variable here):

  p = (value - lower) / (upper - lower)        normalized position in [0, 1]
  d = min(p, 1 - p)                            distance to the nearest edge

and evaluates several complementary lenses, because the pipeline's built-in
Boundary_Percentage uses a very strict eps (Epsilon_Fraction, default 0.01 =>
only the outer 1% of each side counts), which will undercount a real but
milder edge-seeking tendency:

  1. Consistency check: recompute the strict flag from value+range and confirm
     it matches the stored Boundary_Percentage.
  2. Null model: under uniform sampling in [lower, upper], E[p] deciles are
     flat, E[d] = 0.25, and P(outer-x% band) = x. Everything is compared to
     this null.
  3. Distribution lenses: decile histogram of p, fraction in the outer 10/20%,
     mean/median d, count of near-exact edge hits (box-constraint corners),
     count of values outside [0, 1] (extrapolation / clipping).
  4. Significance: binomial test of (# in outer 20%) vs the 20% uniform null.
  5. Cross-tab by whether the intervened variable is a TRUE PARENT -- the key
     split: edge-seeking is only expected on variables that actually drive the
     target, so this separates "no bias" from "bias present but masked by
     intervening on non-causal variables".
  6. Temporal: early (first third) vs late (last third) iterations.

Usage:
    python results_erdos/boundary_bias_analysis.py --run_num 1 --n_obs 200
"""

import argparse
import os
import pickle

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_TYPES = ["Erdos20", "Erdos50", "Erdos100"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=200)
    parser.add_argument("--n_int", type=int, default=2)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument(
        "--edge_tol",
        type=float,
        default=1e-6,
        help="fraction-of-range tolerance for counting a value as an exact box-corner hit",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="boundary_tracking",
        help="results/<subdir>/{graph_type}/... -- use boundary_tracking_oracle for the oracle runs",
    )
    return parser.parse_args()


def load_results(run_num, n_obs, n_int, nonlinear, results_subdir="boundary_tracking"):
    nonlinear_string = "_nonlinear" if nonlinear else ""
    results = {}
    for graph_type in GRAPH_TYPES:
        base = f"run{run_num}_cbo_unknown_dr2_boundary_{n_obs}_{n_int}{nonlinear_string}"
        path = f"{REPO_ROOT}/results/{results_subdir}/{graph_type}/{base}.pickle"
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {graph_type}")
            continue
        with open(path, "rb") as f:
            results[graph_type] = pickle.load(f)
    return results


def normalized_positions(result):
    """
    Returns arrays (p, d, is_true_parent, iter_index) over every intervened
    dimension across all trials. p is normalized position in [0,1], d is
    distance to nearest edge, is_true_parent flags whether that dimension's
    variable is an actual parent of the target.
    """
    ranges = result["Intervention_Ranges"]
    true_parents = set(result["True_Parents"])
    p_list, d_list, tp_list, it_list = [], [], [], []
    for it, (varset, valset) in enumerate(
        zip(result["Intervention_Set"], result["Intervention_Value"])
    ):
        for var, val in zip(varset, valset):
            lower, upper = ranges[var]
            width = upper - lower
            if width <= 0:
                continue
            p = (float(val) - lower) / width
            p_list.append(p)
            d_list.append(min(p, 1.0 - p))
            tp_list.append(var in true_parents)
            it_list.append(it)
    return (
        np.array(p_list),
        np.array(d_list),
        np.array(tp_list, dtype=bool),
        np.array(it_list, dtype=int),
    )


def consistency_check(result):
    """Recompute the strict boundary flag and compare to stored Boundary_Percentage."""
    ranges = result["Intervention_Ranges"]
    eps_frac = result["Epsilon_Fraction"]
    stored = np.array(result["Boundary_Percentage"], dtype=float)
    recomputed = []
    for varset, valset in zip(result["Intervention_Set"], result["Intervention_Value"]):
        n_on = 0
        for var, val in zip(varset, valset):
            lower, upper = ranges[var]
            eps = eps_frac * (upper - lower)
            if float(val) <= lower + eps or float(val) >= upper - eps:
                n_on += 1
        recomputed.append(n_on / len(varset))
    recomputed = np.array(recomputed)
    n = min(len(stored), len(recomputed))
    match = np.allclose(stored[:n], recomputed[:n])
    return match, stored[:n], recomputed[:n]


def summarize_band(p, outer_frac):
    """Fraction of positions within outer_frac of *either* edge; null = 2*outer_frac."""
    if len(p) == 0:
        return float("nan"), float("nan")
    in_band = np.mean((p <= outer_frac) | (p >= 1.0 - outer_frac))
    null = 2.0 * outer_frac
    return in_band, null


def binom_outer(p, outer_frac):
    """Binomial test: # in the outer (2*outer_frac) band vs uniform null."""
    if len(p) == 0:
        return float("nan"), 0, 0
    k = int(np.sum((p <= outer_frac) | (p >= 1.0 - outer_frac)))
    n = len(p)
    null = 2.0 * outer_frac
    res = stats.binomtest(k, n, null, alternative="greater")
    return res.pvalue, k, n


def describe(p, d):
    if len(p) == 0:
        return
    deciles = np.histogram(p, bins=10, range=(0, 1))[0]
    print(f"    n = {len(p)}")
    print(f"    p decile counts [0..1] (uniform-null ~ {len(p)/10:.1f} each): {list(deciles)}")
    print(f"    mean distance-to-edge d = {np.mean(d):.3f}  (uniform-null = 0.250)")
    print(f"    median distance-to-edge d = {np.median(d):.3f}  (uniform-null = 0.250)")
    for of in (0.10, 0.20):
        frac, null = summarize_band(p, of)
        pval, k, n = binom_outer(p, of)
        print(
            f"    outer {int(of*100)}% band: {frac*100:.1f}% of interventions "
            f"({k}/{n})  vs {null*100:.0f}% null   binom p(greater)={pval:.3g}"
        )


def verdict(p, d):
    """One-line direction call based on the outer-20% band and mean d."""
    if len(p) == 0:
        return "no data"
    frac, null = summarize_band(p, 0.20)
    pval, k, n = binom_outer(p, 0.20)
    mean_d = np.mean(d)
    if frac > null and pval < 0.05:
        return f"EDGE-BIASED (outer20%={frac*100:.0f}% vs {null*100:.0f}% null, p={pval:.3g}, mean_d={mean_d:.2f})"
    if frac > null:
        return f"weak edge-lean, not significant (outer20%={frac*100:.0f}% vs {null*100:.0f}%, p={pval:.2g}, mean_d={mean_d:.2f})"
    return f"NO edge bias (outer20%={frac*100:.0f}% vs {null*100:.0f}% null, mean_d={mean_d:.2f} >= 0.25 => center-leaning)"


def main():
    args = parse_args()
    results = load_results(
        args.run_num, args.n_obs, args.n_int, args.nonlinear, args.results_subdir
    )
    if not results:
        print("No results found, nothing to analyze")
        return

    pooled_p, pooled_d, pooled_tp = [], [], []

    for graph_type, result in results.items():
        true_parents = result["True_Parents"]
        p, d, tp, it = normalized_positions(result)
        pooled_p.append(p)
        pooled_d.append(d)
        pooled_tp.append(tp)

        print(f"\n{'='*70}")
        print(f"{graph_type}   true_parents={true_parents}   eps_frac={result['Epsilon_Fraction']}")
        print(f"{'='*70}")

        match, stored, recomp = consistency_check(result)
        print(f"[consistency] recomputed strict flag matches stored Boundary_Percentage: {match}")
        print(f"              stored mean boundary% = {np.mean(stored):.3f}, recomputed = {np.mean(recomp):.3f}")

        edge_hits = int(np.sum((p < args.edge_tol) | (p > 1.0 - args.edge_tol)))
        out_of_range = int(np.sum((p < 0) | (p > 1)))
        print(f"[range]       near-exact box-corner hits (|p|<{args.edge_tol}): {edge_hits}/{len(p)}")
        print(f"              values outside [lower,upper]: {out_of_range}/{len(p)}")

        print("[ALL interventions]")
        describe(p, d)
        print(f"    VERDICT: {verdict(p, d)}")

        # key split: only variables that actually drive the target
        print(f"[TRUE-PARENT interventions only] (n={int(np.sum(tp))})")
        describe(p[tp], d[tp])
        if np.sum(tp) > 0:
            print(f"    VERDICT: {verdict(p[tp], d[tp])}")
        print(f"[NON-PARENT interventions only] (n={int(np.sum(~tp))})")
        describe(p[~tp], d[~tp])
        if np.sum(~tp) > 0:
            print(f"    VERDICT: {verdict(p[~tp], d[~tp])}")

        # temporal: first third vs last third
        n_it = len(result["Intervention_Set"])
        third = max(1, n_it // 3)
        early = it < third
        late = it >= n_it - third
        fe, _ = summarize_band(p[early], 0.20)
        fl, _ = summarize_band(p[late], 0.20)
        print(f"[temporal]    outer20% early(first {third}) = {fe*100:.0f}%   late(last {third}) = {fl*100:.0f}%")

    # pooled
    P = np.concatenate(pooled_p)
    D = np.concatenate(pooled_d)
    TP = np.concatenate(pooled_tp)
    print(f"\n{'='*70}")
    print("POOLED across graph sizes")
    print(f"{'='*70}")
    print(f"[ALL] "); describe(P, D); print(f"    VERDICT: {verdict(P, D)}")
    print(f"[TRUE-PARENT only] (n={int(np.sum(TP))})"); describe(P[TP], D[TP])
    if np.sum(TP) > 0:
        print(f"    VERDICT: {verdict(P[TP], D[TP])}")
    print(f"[NON-PARENT only] (n={int(np.sum(~TP))})"); describe(P[~TP], D[~TP])
    if np.sum(~TP) > 0:
        print(f"    VERDICT: {verdict(P[~TP], D[~TP])}")


if __name__ == "__main__":
    main()
