"""
Boundary behaviour: the runs (baseline) vs the oracle, per graph size, one figure
per graph family x acquisition function. Just the result distributions -- no
interpretive captions.

Each figure is a 2x2 grid: rows = graph sizes, columns = {baseline, oracle}. Each
panel is the distribution of where interventions land within each variable's range
(0 = lower edge, 1 = upper edge), with the outer-20% edge zones shaded.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_BASE = "#D55E00"    # vermillion -> baseline runs
C_ORACLE = "#0072B2"  # blue -> oracle
EDGE_FRAC = 0.20

FAMILIES = {
    "erdos": dict(
        base_subdir="boundary_tracking",
        oracle_subdir="boundary_tracking_oracle",
        graphs=["Erdos50", "Erdos100"],
        n_int=2,
        nonlinear=False,
        label="Erdos",
    ),
    "dream": dict(
        base_subdir="boundary_tracking_dream",
        oracle_subdir="boundary_tracking_dream_oracle",
        graphs=["Size50-Ecoli1", "Size100-Ecoli1"],
        n_int=1,
        nonlinear=True,
        label="Ecoli (DREAM)",
    ),
}
ACQUISITIONS = ["EI", "UCB"]


def load(subdir, graph, acq, n_int, nonlinear, n_obs=200, run_num=1):
    ns = "_nonlinear" if nonlinear else ""
    base = f"run{run_num}_cbo_unknown_dr2_boundary_{acq}_{n_obs}_{n_int}{ns}"
    path = f"{REPO_ROOT}/results/{subdir}/{graph}/{base}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def positions(result):
    ranges = result["Intervention_Ranges"]
    ps = []
    for varset, valset in zip(result["Intervention_Set"], result["Intervention_Value"]):
        for var, val in zip(varset, valset):
            lo, hi = ranges[var]
            if hi > lo:
                ps.append((float(val) - lo) / (hi - lo))
    return np.array(ps)


def hist_panel(ax, p, color, title):
    for lo, hi in [(0, EDGE_FRAC), (1 - EDGE_FRAC, 1)]:
        ax.axvspan(lo, hi, color=color, alpha=0.08)
    if len(p) > 0:
        w = np.ones_like(p) / len(p)
        ax.hist(p, bins=20, range=(0, 1), weights=w, color=color,
                edgecolor="white", linewidth=0.5)
        ax.axvline(p.mean(), color="0.2", linewidth=1.4)
        edge = float(np.mean((p <= EDGE_FRAC) | (p >= 1 - EDGE_FRAC))) * 100
        ax.text(0.5, 0.9, f"{edge:.0f}% in edge zones", transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.75"))
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                color="0.6")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)


def make_figure(fam_key, fam, acq, out_dir):
    graphs = fam["graphs"]
    fig, axes = plt.subplots(len(graphs), 2, figsize=(10, 3.2 * len(graphs)),
                             squeeze=False)
    for row, g in enumerate(graphs):
        rb = load(fam["base_subdir"], g, acq, fam["n_int"], fam["nonlinear"])
        ro = load(fam["oracle_subdir"], g, acq, fam["n_int"], fam["nonlinear"])
        gname = g.replace("-Ecoli1", "")
        hist_panel(axes[row][0], positions(rb) if rb is not None else np.array([]),
                   C_BASE, f"{gname} — baseline (runs)")
        hist_panel(axes[row][1], positions(ro) if ro is not None else np.array([]),
                   C_ORACLE, f"{gname} — oracle")
        axes[row][0].set_ylabel("fraction of interventions", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("position within variable's range\n(0 = lower edge, 1 = upper edge)",
                      fontsize=9)
    fig.suptitle(f"{fam['label']}  —  {acq}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{out_dir}/boundary_vs_oracle_{fam_key}_{acq}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    out_dir = f"{REPO_ROOT}/results/boundary_tracking/plots"
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for fam_key, fam in FAMILIES.items():
        for acq in ACQUISITIONS:
            made.append(make_figure(fam_key, fam, acq, out_dir))
    print("Saved:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
