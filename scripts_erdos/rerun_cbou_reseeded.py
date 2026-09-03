"""Rerun CBO-U (PARENT_SCALE, doubly-robust, dr2) on Erdos50/Erdos100, linear
and nonlinear, with the SAME per-replicate reseeding that CBO-U-Geo uses.

Why this exists: scripts_geometry/geometry_boundary_script.py calls
graph.set_seed(args.seeds_replicate) before sampling ("BASE" for CBO-U-Geo's
runs). scripts_erdos/large_graph_script.py's set_graph() never calls
graph.set_seed(...) at all, so every CBO-U/CBO/Random replicate reuses
ErdosRenyiGraph's constructor-default rng (seed=17) for the SEM noise --
np.random.seed(seed) in utils.sem_sampling.sample_model does NOT reach a
np.random.default_rng() Generator, so --seeds_replicate never actually moved
the data. All three baseline methods' "replicates" are the same draw.

This script fixes ONLY the CBO-U side (PARENT_SCALE, use_doubly_robust=True,
individual=True -- exactly what large_graph_script.py's run_cbo_unknown_dr_2
branch runs) so it can be compared against CBO-U-Geo run-for-run on identical
data. It intentionally does NOT touch CBO or Random: those are produced in the
same process as the old CBO-U (see run_erdos_50_100_dr2.sh), so fixing all
three properly means rerunning the whole baseline job, not patching one
method's script. See scripts_erdos/README or the geometry_boundary_bash.py
docstring for that larger rerun.

Output goes to results/<graph>_reseeded/run<k>_cbo_unknown_dr2_results_<n_obs>_
<n_int>[_nonlinear].pickle -- a NEW directory, so existing results/<graph>/...
pickles (and everything else that reads them) are untouched.

Usage (run from scripts_erdos/, matches run_erdos_50_100_dr2.sh's cwd
convention -- this script also os.chdir("..") itself, like large_graph_script.py):

  python3 rerun_cbou_reseeded.py                          # both graphs, lin+nonlin, runs 1-3
  python3 rerun_cbou_reseeded.py --graphs Erdos50          # one graph
  python3 rerun_cbou_reseeded.py --runs 5                  # more replicates
  python3 rerun_cbou_reseeded.py --dry_run                 # print the seed/run plan and exit
"""

import argparse
import os
import pickle
import sys
import time

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
if _cvd is not None and _cvd.startswith("GPU-"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif _cvd is None or _cvd.strip() == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "[]"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

GRAPHS = ["Erdos50", "Erdos100"]
N_OBS, N_INT, N_TRIALS = 200, 2, 30
BASE_SEED = 71          # matches geometry_boundary_bash.py's default --base_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", type=str, default=",".join(GRAPHS))
    p.add_argument("--runs", type=int, default=3,
                   help="number of replicates; run r uses seed base_seed+r-1, "
                        "same convention as geometry_boundary_bash.py")
    p.add_argument("--base_seed", type=int, default=BASE_SEED)
    p.add_argument("--variants", type=str, default="linear,nonlinear",
                   help="comma-separated subset of {linear, nonlinear}")
    p.add_argument("--n_observational", type=int, default=N_OBS)
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--n_int", type=int, default=N_INT)
    p.add_argument("--out_suffix", type=str, default="_reseeded",
                   help="results/<graph><out_suffix>/ is the output dir")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="rerun even if the output pickle already exists")
    return p.parse_args()


def run_one(graph_type: str, nonlinear: bool, run_num: int, seed: int, args) -> str:
    # deferred: these imports need the os.chdir("..") + sys.path fix above, and
    # transitively pull in the diffcbed/cdt stack (needs a parseable
    # CUDA_VISIBLE_DEVICES, set above) -- keep them out of module scope so
    # --dry_run doesn't pay for them.
    from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
    from graphs.data_setup import setup_observational_interventional
    from scripts_erdos.large_graph_script import set_graph

    nonlinear_string = "_nonlinear" if nonlinear else ""
    out_dir = f"results/{graph_type}{args.out_suffix}{nonlinear_string}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = (f"{out_dir}/run{run_num}_cbo_unknown_dr2_results_"
               f"{args.n_observational}_{args.n_int}{nonlinear_string}.pickle")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"  skip (exists): {out_path}", flush=True)
        return out_path

    graph = set_graph(graph_type, nonlinear=nonlinear)
    graph.set_seed(seed)            # <-- the fix: match geometry_boundary_script.py

    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=True,
        seed=seed,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )

    # exactly large_graph_script.py's run_cbo_unknown_dr_2 branch
    model = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
    )
    model.set_values(D_O, D_I, exploration_set)
    (best_y_array, current_y_array, cost_array,
     intervention_set, intervention_value, average_uncertainty
     ) = model.run_algorithm(T=args.n_trials, show_graphics=False)

    results_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
        "Intervention_Set": intervention_set,
        "Intervention_Value": intervention_value,
        "Uncertainty": average_uncertainty,
        # not produced by the original large_graph_script.py pickles; kept here
        # since this run already has them, for the intervention-position plot's
        # per-run normalisation (avoids the separate precompute_cbou_boundary.py
        # reconstruction step for THESE reseeded runs)
        "Intervention_Ranges": {v: list(b) for v, b in graph.interventional_range_data.items()},
        "True_Parents": tuple(graph.parents[graph.target]),   # matches geometry_boundary_script.py's dtype
        "Seed": seed,
    }
    with open(out_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main():
    args = parse_args()
    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - {"linear", "nonlinear"}
    if unknown:
        raise SystemExit(f"unknown variant(s) {sorted(unknown)}")

    jobs = [(g, v, r) for v in variants for g in graphs for r in range(1, args.runs + 1)]
    print(f"{len(jobs)} run(s): {graphs} x {variants} x runs 1..{args.runs}, "
          f"seeds {args.base_seed}..{args.base_seed + args.runs - 1}")
    for g, v, r in jobs:
        seed = args.base_seed + r - 1
        print(f"[{v}] {g} run{r} seed={seed}", flush=True)

    if args.dry_run:
        return

    t0 = time.time()
    for g, v, r in jobs:
        seed = args.base_seed + r - 1
        run_one(g, nonlinear=(v == "nonlinear"), run_num=r, seed=seed, args=args)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
