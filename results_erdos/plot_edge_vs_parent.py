"""
Per-experiment dotplot: for every intervention, its position within the
variable's range (x) and whether the intervened variable is an actual parent of
the target (y = 1 true parent / 0 non-parent, colour-coded).

The overview question: when the run showed edge behaviour (x near 0 or 1) vs
non-edge behaviour (x near the centre), were the intervened variables actual
parents? Each experiment gets its own panel; the annotation gives that
experiment's fraction of interventions on true parents.

Grid: rows = graph, columns = {baseline, oracle} x {EI, UCB}.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_PARENT = "#0072B2"     # true parent
C_NONPARENT = "#D55E00"  # non-parent
EDGE_FRAC = 0.20

# rows: (graph label, base_subdir, oracle_subdir, graph_key, n_int, nonlinear)
ROWS = [
    ("Erdos50", "boundary_tracking", "boundary_tracking_oracle", "Erdos50", 2, False),
    ("Erdos100", "boundary_tracking", "boundary_tracking_oracle", "Erdos100", 2, False),
    ("Ecoli50", "boundary_tracking_dream", "boundary_tracking_dream_oracle", "Size50-Ecoli1", 1, True),
    ("Ecoli100", "boundary_tracking_dream", "boundary_tracking_dream_oracle", "Size100-Ecoli1", 1, True),
]
# columns: (condition, acquisition, which subdir index) -- baseline uses base, oracle uses oracle
COLS = [("baseline", "EI"), ("oracle", "EI"), ("baseline", "UCB"), ("oracle", "UCB")]


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
    xs, ys = [], []
    for vs, valset in zip(result["Intervention_Set"], result["Intervention_Value"]):
        for var, val in zip(vs, valset):
            lo, hi = ranges[var]
            if hi > lo:
                xs.append((float(val) - lo) / (hi - lo))
                ys.append(1 if var in tp else 0)
    return np.array(xs), np.array(ys)


def panel(ax, result, rng):
    for lo, hi in [(0, EDGE_FRAC), (1 - EDGE_FRAC, 1)]:
        ax.axvspan(lo, hi, color="0.88", alpha=0.6, zorder=0)
    if result is None:
        ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", color="0.6")
    else:
        x, y = dots(result)
        if len(x):
            jit = y + rng.uniform(-0.10, 0.10, size=len(y))
            colors = [C_PARENT if v else C_NONPARENT for v in y]
            ax.scatter(x, jit, s=42, c=colors, edgecolor="white", linewidth=0.5, zorder=3)
            frac_tp = 100 * float(np.mean(y))
            ax.text(0.5, 0.5, f"{frac_tp:.0f}%\non true\nparents", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.35",
                    bbox=dict(boxstyle="round", fc="white", ec="0.85", alpha=0.85))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, 1.35)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-\nparent", "true\nparent"], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    rng = np.random.default_rng(0)
    nR, nC = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nR, nC, figsize=(3.1 * nC, 2.5 * nR), squeeze=False)

    for c, (cond, acq) in enumerate(COLS):
        axes[0][c].set_title(f"{cond}\n{acq}", fontsize=11, fontweight="bold")
    for r, (rlabel, base_sub, orac_sub, gkey, n_int, nl) in enumerate(ROWS):
        axes[r][0].annotate(rlabel, xy=(-0.42, 0.5), xycoords="axes fraction",
                            ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)
        for c, (cond, acq) in enumerate(COLS):
            sub = base_sub if cond == "baseline" else orac_sub
            panel(axes[r][c], load(sub, gkey, acq, n_int, nl), rng)

    for c in range(nC):
        axes[-1][c].set_xlabel("position in range\n(0=lower, 1=upper edge)", fontsize=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PARENT, markersize=10, label="true parent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NONPARENT, markersize=10, label="non-parent"),
    ], loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Intervention position vs. was the variable an actual parent",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0.03, 0, 1, 0.99])

    out_dir = f"{REPO_ROOT}/results/boundary_tracking/plots"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/edge_vs_parent.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
