"""
Across all experiments: intervention position within the variable's range (x) vs
the experiment's parent-composition category (y lane):

  - ALL true parents      (100% of interventions on true parents)
  - PARTIAL               (a mix: some on true parents, some not)
  - ONLY non-parents      (0% on true parents)

Each dot is one intervention, placed in its experiment's lane, coloured by
whether that specific intervention was on a true parent (blue) or not (orange).
Edge zones (outer 20%) are shaded. A short vertical tick per lane marks the
median position.

This shows, aggregated over every experiment, whether edge behaviour goes with
intervening on real parents, and -- in the PARTIAL lane -- whether the parent
hits sit at the edge while the non-parent hits sit in the centre.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_PARENT = "#0072B2"
C_NONPARENT = "#D55E00"
EDGE_FRAC = 0.20

# every boundary experiment: (base_subdir, oracle_subdir, graph, n_int, nonlinear)
SPECS = [
    ("boundary_tracking", "boundary_tracking_oracle", "Erdos50", 2, False),
    ("boundary_tracking", "boundary_tracking_oracle", "Erdos100", 2, False),
    ("boundary_tracking_dream", "boundary_tracking_dream_oracle", "Size50-Ecoli1", 1, True),
    ("boundary_tracking_dream", "boundary_tracking_dream_oracle", "Size100-Ecoli1", 1, True),
]
ACQUISITIONS = ["EI", "UCB"]

# lanes, top -> bottom
LANES = ["all true parents", "partial", "only non-parents"]
LANE_Y = {"all true parents": 2, "partial": 1, "only non-parents": 0}


def load(subdir, graph, acq, n_int, nonlinear, n_obs=200, run_num=1):
    ns = "_nonlinear" if nonlinear else ""
    base = f"run{run_num}_cbo_unknown_dr2_boundary_{acq}_{n_obs}_{n_int}{ns}"
    path = f"{REPO_ROOT}/results/{subdir}/{graph}/{base}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def dots(result):
    ranges = result["Intervention_Ranges"]
    tp = set(result["True_Parents"])
    out = []
    for vs, valset in zip(result["Intervention_Set"], result["Intervention_Value"]):
        for var, val in zip(vs, valset):
            lo, hi = ranges[var]
            if hi > lo:
                out.append(((float(val) - lo) / (hi - lo), var in tp))
    return out


def category(frac_tp):
    if frac_tp >= 0.999:
        return "all true parents"
    if frac_tp <= 0.001:
        return "only non-parents"
    return "partial"


def collect():
    # lane -> list of (position, is_parent); and per-lane experiment counts
    by_lane = {L: [] for L in LANES}
    n_exp = {L: 0 for L in LANES}
    for base_sub, orac_sub, graph, n_int, nl in SPECS:
        for cond, sub in [("baseline", base_sub), ("oracle", orac_sub)]:
            for acq in ACQUISITIONS:
                r = load(sub, graph, acq, n_int, nl)
                if r is None:
                    continue
                d = dots(r)
                if not d:
                    continue
                frac = np.mean([p for _, p in d])
                lane = category(frac)
                by_lane[lane].extend(d)
                n_exp[lane] += 1
    return by_lane, n_exp


def main():
    by_lane, n_exp = collect()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for lo, hi in [(0, EDGE_FRAC), (1 - EDGE_FRAC, 1)]:
        ax.axvspan(lo, hi, color="0.88", alpha=0.6, zorder=0)
    for L in LANES:
        ax.axhline(LANE_Y[L], color="0.92", lw=8, zorder=0)

    for L in LANES:
        pts = by_lane[L]
        yc = LANE_Y[L]
        if not pts:
            continue
        xs = np.array([p for p, _ in pts])
        isp = np.array([q for _, q in pts])
        yj = yc + rng.uniform(-0.28, 0.28, size=len(xs))
        colors = np.where(isp, C_PARENT, C_NONPARENT)
        ax.scatter(xs, yj, s=34, c=colors, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
        # edge occupancy for the lane
        edge = float(np.mean((xs <= EDGE_FRAC) | (xs >= 1 - EDGE_FRAC))) * 100
        ax.text(1.03, yc, f"{n_exp[L]} exp\n{len(xs)} interv\n{edge:.0f}% edge",
                va="center", ha="left", fontsize=8, color="0.3")

    ax.set_yticks([LANE_Y[L] for L in LANES])
    ax.set_yticklabels(LANES, fontsize=11)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("position within variable's range  (0 = lower edge, 1 = upper edge;  shaded = outer-20% edge zones)")
    ax.set_title("Intervention position by parent-composition category (all experiments)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PARENT, markersize=10, label="intervention on a true parent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NONPARENT, markersize=10, label="intervention on a non-parent"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(right=0.86, bottom=0.22)

    out_dir = f"{REPO_ROOT}/results/boundary_tracking/plots"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/position_by_category.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    for L in LANES:
        print(f"  {L}: {n_exp[L]} experiments, {len(by_lane[L])} interventions")


if __name__ == "__main__":
    main()
