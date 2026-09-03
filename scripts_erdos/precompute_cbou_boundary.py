"""Reconstruct each CBO-U run's intervention ranges (D_O min/max, exactly as
large_graph_script.py builds them) and compute the boundary fraction per
iteration from the stored Intervention_Value. Dumps a sidecar per (graph, kind)
that the plot notebook reads:  results/<graph>/cbou_boundary_200_2<ns>.pickle
containing a numpy array of shape (n_runs, n_iters), plus
results/<graph>/cbou_ranges_200_2<ns>.pickle mapping "run<k>" -> {var: [lo, hi]},
which the intervention-position dot plot needs to normalise CBO-U by its own box.

Run as a cluster job (needs the graph stack); do NOT run on the login node.
"""
import os
import sys
import pickle

REPO = os.path.expanduser("~/causal_bayes_opt")
os.chdir(REPO)
sys.path.insert(0, REPO)
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
if _cvd is None or _cvd.strip() == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "[]"
elif _cvd.startswith("GPU-"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.data_setup import setup_observational_interventional

EPS_FRAC = 0.01
N_OBS, N_INT = 200, 2
SEEDS = {1: 71, 2: 11, 3: 89}           # run_num -> CBO-U seed (as in run_erdos*_three.sh)
GRAPHS = {"Erdos50": (50, "23"), "Erdos100": (100, "80")}


def ranges_for(n, target, nonlinear, seed):
    # mirror large_graph_script.set_graph + setup (NO set_seed for erdos)
    g = ErdosRenyiGraph(num_nodes=n, nonlinear=nonlinear)
    g.set_target(target)
    D_O, _, _ = setup_observational_interventional(
        graph_type=None, noiseless=True, seed=seed, n_obs=N_OBS, n_int=N_INT, graph=g,
    )
    g.set_interventional_range_data(D_O)
    return {v: (float(lo), float(hi)) for v, (lo, hi) in g.interventional_range_data.items()}


def boundary_traj(res, ranges):
    traj, in_range, total = [], 0, 0
    for iv_set, iv_val in zip(res["Intervention_Set"], res["Intervention_Value"]):
        n_on = 0
        for var, val in zip(tuple(iv_set), tuple(iv_val)):
            lo, hi = ranges[var]
            eps = EPS_FRAC * (hi - lo)
            if val <= lo + eps or val >= hi - eps:
                n_on += 1
            in_range += (lo <= val <= hi)
            total += 1
        traj.append(n_on / len(tuple(iv_set)))
    return traj, in_range, total


def main():
    for graph, (n, target) in GRAPHS.items():
        for ns in ["", "_nonlinear"]:
            nonlinear = ns == "_nonlinear"
            base = f"results/{graph}"
            trajs, chk_in, chk_tot = [], 0, 0
            ranges_by_run = {}
            for run_num, seed in SEEDS.items():
                fn = f"{base}/run{run_num}_cbo_unknown_dr2_results_{N_OBS}_{N_INT}{ns}.pickle"
                if not os.path.exists(fn):
                    continue
                res = pickle.load(open(fn, "rb"))
                ranges = ranges_for(n, target, nonlinear, seed)
                traj, ir, tot = boundary_traj(res, ranges)
                ranges_by_run[f"run{run_num}"] = {v: list(b) for v, b in ranges.items()}
                trajs.append(traj)
                chk_in += ir
                chk_tot += tot
            if not trajs:
                print(f"{graph}{ns}: no CBO-U runs found, skipping", flush=True)
                continue
            arr = np.array(trajs, dtype=float)
            out = f"{base}/cbou_boundary_{N_OBS}_{N_INT}{ns}.pickle"
            with open(out, "wb") as f:
                pickle.dump(arr, f)
            out_r = f"{base}/cbou_ranges_{N_OBS}_{N_INT}{ns}.pickle"
            with open(out_r, "wb") as f:
                pickle.dump(ranges_by_run, f)

            frac_in = chk_in / chk_tot if chk_tot else float("nan")
            print(f"{graph}{ns}: {arr.shape[0]} runs, mean boundary {arr.mean():.3f}, "
                  f"values-within-range {frac_in:.3f} -> {out}, {out_r}", flush=True)


if __name__ == "__main__":
    main()
