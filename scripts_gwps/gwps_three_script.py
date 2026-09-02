"""
Run the three-algorithm comparison (CBO true-parents + PARENT_SCALE dr2 +
RANDOM_SCALE) once on a GwpsGraph -- the gwps analogue of large_graph_script.py's
parent_method "three" (used by job_erdos_50_100.sh).

Mirrors gwps_cbo_script.py for the PARENT_SCALE (dr2) run, then adds the
true-parent CBO and RANDOM_SCALE runs exactly as run_script_unknown does for
Erdos. Output filenames match the Erdos "three" convention so the results_erdos
notebooks/regexes work unchanged, but land in results/gwps_three/.
"""

import argparse
import logging
import os
import pickle
import sys

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
_algorithms_path = os.path.join(os.getcwd(), "algorithms")
if _algorithms_path not in sys.path:
    sys.path.append(_algorithms_path)

import numpy as np

from algorithms.CBO_algorithm import CBO
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from algorithms.RANDOM_SCALE_algorithm import RANDOM_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph_gwps import GwpsGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default=None, help="target ENSG id (auto if omitted)")
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--top_k_parents", type=int, default=8)
    p.add_argument("--weight_scale", type=float, default=3.0)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    return p.parse_args()


def _save(results_dir, base, best_y, current_y, cost, iset, ival, unc):
    results_dict = {
        "Best_Y": best_y,
        "Per_trial_Y": current_y,
        "Cost": cost,
        "Intervention_Set": iset,
        "Intervention_Value": ival,
        "Uncertainty": unc,
    }
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved {results_dir}/{base}.pickle")


def build_graph(args):
    return GwpsGraph(
        target=args.target,
        max_nodes=args.max_nodes,
        top_k_parents=args.top_k_parents,
        noise_sigma=args.noise_sigma,
        weight_scale=args.weight_scale,
        seed=args.seeds_replicate,
    )


def run(args):
    results_dir = "results/gwps_three"
    os.makedirs(results_dir, exist_ok=True)
    n_obs, n_int, run_num = args.n_observational, args.n_int, args.run_num

    # ---- PARENT_SCALE (dr2 / CBO-U) on the clean graph ----
    graph = build_graph(args)
    true_parents = tuple(graph.parents[graph.target])
    logging.info(
        f"GWPS three [ws={args.weight_scale}]: {len(graph.variables)} nodes, "
        f"target {graph.target} ({graph.index_to_ensg[int(graph.target)]}), "
        f"true parents {true_parents}"
    )
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None, noiseless=args.noiseless, seed=args.seeds_replicate,
        n_obs=n_obs, n_int=n_int, graph=graph,
    )

    model = PARENT_SCALE(
        graph=graph, nonlinear=False, individual=True, use_doubly_robust=True
    )
    model.set_values(D_O, D_I, exploration_set)
    out = model.run_algorithm(T=args.n_trials, show_graphics=False)
    _save(results_dir, f"run{run_num}_cbo_unknown_dr2_results_{n_obs}_{n_int}", *out)

    # ---- CBO with the true parents forced (mutates the graph) ----
    parents = graph.parents[graph.target]
    edges = [(parent, graph.target) for parent in parents]
    graph.mispecify_graph(edges)
    graph.set_interventional_range_data(D_O)
    exploration_set = [(parent,) for parent in parents]
    model = CBO(graph=graph)
    model.set_values(D_O, D_I, exploration_set)
    out = model.run_algorithm(T=args.n_trials)
    _save(results_dir, f"run{run_num}_cbo_results_{n_obs}_{n_int}", *out)

    # ---- RANDOM_SCALE (same mutated graph + parent exploration set, as Erdos) ----
    model = RANDOM_SCALE(graph)
    model.set_values(D_O, D_I, exploration_set)
    out = model.run_algorithm(T=args.n_trials)
    _save(results_dir, f"run{run_num}_cbo_results_random_{n_obs}_{n_int}", *out)


if __name__ == "__main__":
    run(parse_args())