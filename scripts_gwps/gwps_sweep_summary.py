"""Summarise the GWPS weight-scale sweep: for each --weight_scale, how much CBO-U improved the target and whether it
recovered the true parents. Prints a table and saves results/gwps/plots/gwps_sweep.png."""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--n_obs", type=int, default=200)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--scales", type=str, default="1,3,5")
    return p.parse_args()


def load(run_num, n_obs, n_int, max_nodes, scale):
    ws = str(float(scale)).replace(".", "p")
    base = f"run{run_num}_gwps_cbo_unknown_dr2_{n_obs}_{n_int}_n{max_nodes}_ws{ws}"
    path = f"{REPO_ROOT}/results/gwps/{base}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def recovered_in_map(result):
    tp = set(result["True_Parents"])
    hist = result["Posterior_History"]
    final = hist[-1] if hist else {}
    map_set = set(max(final.items(), key=lambda kv: kv[1])[0]) if final else set()
    ever = set()
    for snap in hist:
        if snap:
            ever |= set(max(snap.items(), key=lambda kv: kv[1])[0])
    return tp, tp & map_set, tp & ever


def main():
    args = parse_args()
    scales = [float(s) for s in args.scales.split(",")]
    results = {s: load(args.run_num, args.n_obs, args.n_int, args.max_nodes, s) for s in scales}

    print(f"\n{'scale':>6} {'obsYstd':>8} {'Best_Y start->end':>20} {'norm impr':>10} "
          f"{'#parents':>9} {'MAP rec':>8} {'ever rec':>9} {'#genes':>7}")
    print("-" * 90)
    for s in scales:
        r = results[s]
        if r is None:
            print(f"{s:>6.1f}   (missing)")
            continue
        best = np.asarray(r["Best_Y"], float)
        std = r.get("Obs_Y_Std", float("nan"))
        norm = (best[-1] - best[0]) / std if std else float("nan")
        tp, map_rec, ever_rec = recovered_in_map(r)
        ngenes = len(set(v for st in r["Intervention_Set"] for v in st))
        print(f"{s:>6.1f} {std:>8.2f} {best[0]:>8.2f} -> {best[-1]:>8.2f} {norm:>10.2f} "
              f"{len(tp):>9} {len(map_rec):>8} {len(ever_rec):>9} {ngenes:>7}")

    # plot
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(scales)))
    for s, c in zip(scales, colors):
        r = results[s]
        if r is None:
            continue
        best = np.asarray(r["Best_Y"], float)
        t = np.arange(len(best))
        axL.plot(t, best, color=c, marker="o", ms=3, label=f"scale {s:g}")
        std = r.get("Obs_Y_Std", 1.0) or 1.0
        axR.plot(t, (best - best[0]) / std, color=c, marker="o", ms=3, label=f"scale {s:g}")
    for ax in (axL, axR):
        ax.set_xlabel("trial")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="0.92")
        ax.legend(frameon=False, fontsize=9)
    axL.set_ylabel("Best_Y (absolute)")
    axL.set_title("Best_Y per trial (absolute; SEM scale differs)")
    axR.axhline(0, color="0.6", lw=1)
    axR.set_ylabel("(Best_Y - Best_Y[0]) / obs-Y-std")
    axR.set_title("Normalised improvement (comparable across scales)")
    fig.suptitle("GWPS CBO-U weight-scale sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_dir = f"{REPO_ROOT}/results/gwps/plots"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/gwps_sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
