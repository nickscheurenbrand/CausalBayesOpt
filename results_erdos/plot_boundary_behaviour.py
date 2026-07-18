"""
Presentation figures for the boundary-behaviour analysis.

Reads the real boundary-tracking pickles (results/boundary_tracking/) and the
oracle ones (results/boundary_tracking_oracle/) and produces three PNGs under
results/boundary_tracking/plots/:

  fig1_boundary_bias_by_parent.png
      Headline: pooled distribution of where interventions land in each
      variable's range, split by whether the intervened variable is a TRUE
      PARENT. True-parent interventions pile at the (lower) edge; non-parent
      interventions sit in the centre. Establishes "boundary bias exists, but
      only on true parents".

  fig2_confound_erdos100_real_vs_oracle.png
      The confound-closer: the SAME 100-node graph, real run (collapsed
      posterior -> intervenes on non-parents -> centre) vs oracle run (parents
      forced -> intervenes on true parents -> edge). Only the parent identity
      differs; everything else is identical.

  fig3_summary_outer_band.png
      Quantitative at-a-glance: fraction of interventions landing in the outer
      20% edge zones per run, against the 40% uniform-null reference.

Colours: Okabe-Ito colourblind-safe pair (blue = true parent / correct edge
behaviour, vermillion = non-parent / no bias). Identity is also carried by
panel position, so it is never colour-alone.

Usage:
    python results_erdos/plot_boundary_behaviour.py --run_num 1 --n_obs 200
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Okabe-Ito colourblind-safe palette
C_TRUE = "#0072B2"   # blue  -> true-parent interventions (correct edge-seeking)
C_NON = "#D55E00"    # vermillion -> non-parent interventions (no bias)
C_NULL = "#7A7A7A"   # neutral grey -> uniform-null reference
EDGE_FRAC = 0.20     # "outer band": within EDGE_FRAC of either edge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--n_obs", type=int, default=200)
    p.add_argument("--n_int", type=int, default=2)
    p.add_argument("--nonlinear", action="store_true")
    return p.parse_args()


def load_one(subdir, graph_type, run_num, n_obs, n_int, nonlinear):
    ns = "_nonlinear" if nonlinear else ""
    base = f"run{run_num}_cbo_unknown_dr2_boundary_{n_obs}_{n_int}{ns}"
    path = f"{REPO_ROOT}/results/{subdir}/{graph_type}/{base}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def positions(result):
    """(p, is_true_parent) over every intervened dimension; p = pos in [0,1]."""
    ranges = result["Intervention_Ranges"]
    tp = set(result["True_Parents"])
    ps, flags = [], []
    for varset, valset in zip(result["Intervention_Set"], result["Intervention_Value"]):
        for var, val in zip(varset, valset):
            lo, hi = ranges[var]
            if hi - lo <= 0:
                continue
            ps.append((float(val) - lo) / (hi - lo))
            flags.append(var in tp)
    return np.array(ps), np.array(flags, dtype=bool)


def outer_band_frac(p):
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p <= EDGE_FRAC) | (p >= 1 - EDGE_FRAC)))


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def hist_panel(ax, p, color, title, n_bins=20):
    """Fraction-normalised histogram of positions with edge zones shaded."""
    ax.axvspan(0, EDGE_FRAC, color=color, alpha=0.08)
    ax.axvspan(1 - EDGE_FRAC, 1, color=color, alpha=0.08)
    counts, edges = np.histogram(p, bins=n_bins, range=(0, 1))
    frac = counts / max(len(p), 1)
    ax.bar(
        edges[:-1], frac, width=np.diff(edges), align="edge",
        color=color, edgecolor="white", linewidth=0.5,
    )
    # uniform-null reference line (each bin would hold 1/n_bins under uniform)
    ax.axhline(1.0 / n_bins, color=C_NULL, linestyle="--", linewidth=1.2)
    mean_p = np.mean(p) if len(p) else float("nan")
    ax.axvline(mean_p, color="0.15", linewidth=1.6)
    band = outer_band_frac(p)
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(
        0.5, 0.92,
        f"n={len(p)}   mean pos={mean_p:.2f}\n{band*100:.0f}% in edge zones (null 40%)",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("position of intervention within variable's range\n(0 = lower edge, 1 = upper edge)", fontsize=10)
    style_axis(ax)


def fig1_by_parent(runs, out_dir):
    # pool across all real runs, split by parent status
    ps, flags = [], []
    for r in runs:
        p, f = positions(r)
        ps.append(p)
        flags.append(f)
    P = np.concatenate(ps)
    F = np.concatenate(flags)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    hist_panel(axes[0], P[F], C_TRUE, "Intervened on a TRUE PARENT")
    hist_panel(axes[1], P[~F], C_NON, "Intervened on a NON-PARENT")
    axes[0].set_ylabel("fraction of interventions", fontsize=10)
    fig.suptitle(
        "Boundary bias appears only when intervening on a true causal parent",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "Dashed grey = uniform-null per-bin height; black line = mean position; "
        "shaded = outer-20% edge zones.  Pooled across Erdos20/50/100 real runs.",
        ha="center", fontsize=9, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = f"{out_dir}/fig1_boundary_bias_by_parent.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_confound(real100, oracle100, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    pr, _ = positions(real100)
    po, _ = positions(oracle100)
    hist_panel(
        axes[0], pr, C_NON,
        "REAL run\nposterior collapsed -> intervenes on non-parents (90, 64)",
    )
    hist_panel(
        axes[1], po, C_TRUE,
        "ORACLE run\ntrue parents forced -> intervenes on 85, 98",
    )
    axes[0].set_ylabel("fraction of interventions", fontsize=10)
    fig.suptitle(
        "Same 100-node graph: fixing the parents flips centre → edge",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "Identical pipeline, acquisition and GP surrogates; the only difference "
        "is which variables the parent posterior selects.",
        ha="center", fontsize=9, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = f"{out_dir}/fig2_confound_erdos100_real_vs_oracle.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_summary(bars, out_dir):
    """bars: list of (label, band_fraction, on_true_parents_bool)."""
    labels = [b[0] for b in bars]
    vals = [b[1] * 100 for b in bars]
    colors = [C_TRUE if b[2] else C_NON for b in bars]
    x = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.axhline(40, color=C_NULL, linestyle="--", linewidth=1.4)
    ax.text(len(bars) - 0.5, 42, "uniform null = 40%", color=C_NULL,
            ha="right", va="bottom", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of interventions in outer-20% edge zones", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title("Edge-zone occupancy per run", fontsize=14, fontweight="bold", pad=10)
    # legend by parent status (identity not colour-alone: also stated in labels)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=C_TRUE, label="intervened on true parent(s)"),
            Patch(facecolor=C_NON, label="intervened on non-parent(s)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=10,
    )
    style_axis(ax)
    fig.tight_layout()
    path = f"{out_dir}/fig3_summary_outer_band.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    out_dir = f"{REPO_ROOT}/results/boundary_tracking/plots"
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(run_num=args.run_num, n_obs=args.n_obs, n_int=args.n_int, nonlinear=args.nonlinear)

    real = {}
    for g in ["Erdos20", "Erdos50", "Erdos100"]:
        r = load_one("boundary_tracking", g, **kw)
        if r is not None:
            real[g] = r
    oracle = {}
    for g in ["Erdos50", "Erdos100"]:
        r = load_one("boundary_tracking_oracle", g, **kw)
        if r is not None:
            oracle[g] = r

    if not real:
        print("No real boundary-tracking pickles found; nothing to plot.")
        return

    made = []
    made.append(fig1_by_parent(list(real.values()), out_dir))

    if "Erdos100" in real and "Erdos100" in oracle:
        made.append(fig2_confound(real["Erdos100"], oracle["Erdos100"], out_dir))
    else:
        print("Skipping fig2: need both real and oracle Erdos100 pickles.")

    # summary bars: real runs (parent status inferred: true if any intervention
    # hit a true parent AND none hit a non-parent -> but simplest: label by
    # whether the pooled interventions were majority on true parents)
    bars = []
    for g, r in real.items():
        p, f = positions(r)
        on_tp = bool(np.mean(f) >= 0.5)
        bars.append((f"{g}\n(real)", outer_band_frac(p), on_tp))
    for g, r in oracle.items():
        p, f = positions(r)
        bars.append((f"{g}\n(oracle)", outer_band_frac(p), True))
    made.append(fig3_summary(bars, out_dir))

    print("Saved figures:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
