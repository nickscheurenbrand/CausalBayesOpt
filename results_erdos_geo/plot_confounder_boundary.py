"""
Conf-vs-ctrl latent-confounder boundary comparison.

For each (graph, x_kind) the confounded variable X is oracle-FORCED so it is
intervened every trial. We compare:
  conf -> hidden confounder Z->X, Z->Y present (confounded observational data)
  ctrl -> control, no confounder, same forced X (clean observational data)

The conf-vs-ctrl contrast on the SAME forced X isolates the confounding's effect
on X's boundary behaviour:
  NON-PARENT: control X sits interior (non-causal, flat do-effect); confounding
    should push it to the edge  -> spurious boundary-seeking (false positive).
  TRUE PARENT: control X hugs its true optimal edge; confounding distorts the
    do-effect -> shifts where X lands (effect corruption).

Reads results/confounder_boundary_tracking/{graph}_{x_kind}_{conf,ctrl}/ and
writes results/confounder_boundary_tracking/plots/confounder_conf_vs_ctrl.png
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBDIR = "confounder_boundary_tracking"

C_CONF = "#D55E00"   # vermillion -> confounded
C_CTRL = "#7A7A7A"   # grey -> control (no confounder)
GRAPHS = ["Erdos50", "Erdos100", "Size50-Ecoli1", "Size100-Ecoli1"]
NULL = 40.0


def load(sub):
    for fn in ("run1_cbo_confounded_dr2_boundary_200_1.pickle",
               "run1_cbo_confounded_dr2_boundary_200_1_nonlinear.pickle"):
        p = f"{REPO_ROOT}/results/{SUBDIR}/{sub}/{fn}"
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
    return None


def xstats(r):
    conf = r["Confounders"][0]
    x = conf["x"]
    ranges = r["Intervention_Ranges"]
    ps = []
    for s, v in zip(r["Intervention_Set"], r["Intervention_Value"]):
        for var, val in zip(s, v):
            if var == x:
                lo, hi = ranges[var]
                if hi > lo:
                    ps.append((float(val) - lo) / (hi - lo))
    ps = np.array(ps)
    edge = float(np.mean((ps <= 0.2) | (ps >= 0.8))) * 100 if len(ps) else np.nan
    return x, (ps.mean() if len(ps) else np.nan), edge


def grouped_panel(ax, x_kind, title):
    xs = np.arange(len(GRAPHS))
    w = 0.38
    conf_e, ctrl_e, conf_m, ctrl_m, xlabels = [], [], [], [], []
    for g in GRAPHS:
        rc, rk = load(f"{g}_{x_kind}_conf"), load(f"{g}_{x_kind}_ctrl")
        if rc is None or rk is None:
            conf_e.append(np.nan); ctrl_e.append(np.nan); conf_m.append(np.nan); ctrl_m.append(np.nan)
            xlabels.append(g.replace("-Ecoli1", "")); continue
        xv, mc, ec = xstats(rc)
        _, mk, ek = xstats(rk)
        conf_e.append(ec); ctrl_e.append(ek); conf_m.append(mc); ctrl_m.append(mk)
        xlabels.append(g.replace("-Ecoli1", "") + f"\n(X={xv})")

    ax.bar(xs - w / 2, ctrl_e, w, color=C_CTRL, label="control (no confounder)", edgecolor="white")
    ax.bar(xs + w / 2, conf_e, w, color=C_CONF, label="confounded", edgecolor="white")
    ax.axhline(NULL, color="0.5", linestyle="--", linewidth=1.2)
    ax.text(len(GRAPHS) - 0.5, NULL + 1.5, "uniform null = 40%", color="0.4",
            ha="right", va="bottom", fontsize=8)
    # annotate mean-position under each bar pair
    for xi, (mk, mc) in enumerate(zip(ctrl_m, conf_m)):
        if not np.isnan(mk):
            ax.text(xi - w / 2, 3, f"{mk:.2f}", ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
        if not np.isnan(mc):
            ax.text(xi + w / 2, 3, f"{mc:.2f}", ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of X's interventions in outer-20% edge zones", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    grouped_panel(ax1, "non_parent",
                  "Confounded NON-parent\n(confounding -> spurious boundary-seeking)")
    grouped_panel(ax2, "true_parent",
                  "Confounded TRUE parent\n(confounding -> do-effect corruption)")
    fig.suptitle(
        "Latent confounder: forced X's boundary occupancy, confounded vs control "
        "(numbers on bars = mean position of X)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_dir = f"{REPO_ROOT}/results/{SUBDIR}/plots"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/confounder_conf_vs_ctrl.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
