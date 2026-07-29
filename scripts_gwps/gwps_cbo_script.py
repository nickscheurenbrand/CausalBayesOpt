"""
Run CBO-U (PARENT_SCALE, dr2) once on a GwpsGraph -- the genome-wide Perturb-seq
gene network, as a DAG subgraph with a linear-Gaussian SEM using the real G_hat
direct-effect weights.

This is the "get CBO-U running first" step: a single end-to-end optimisation run,
structured exactly like base_script_dream.py's run_cbo_unknown_dr_2 but on a
GwpsGraph. Saves the results dict plus the integer-node -> ENSG gene map.
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

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
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
    p.add_argument("--weight_scale", type=float, default=1.0)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    return p.parse_args()


def run(args):
    graph = GwpsGraph(
        target=args.target,
        max_nodes=args.max_nodes,
        top_k_parents=args.top_k_parents,
        noise_sigma=args.noise_sigma,
        weight_scale=args.weight_scale,
        seed=args.seeds_replicate,
    )
    true_parents = tuple(graph.parents[graph.target])
    logging.info(
        f"GWPS: {len(graph.variables)} nodes, target {graph.target} "
        f"({graph.index_to_ensg[int(graph.target)]}), true parents {true_parents}"
    )

    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=args.noiseless,
        seed=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )

    model = PARENT_SCALE(
        graph=graph, nonlinear=False, individual=True, use_doubly_robust=True
    )
    model.set_values(D_O, D_I, exploration_set)
    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=args.n_trials, show_graphics=False)

    obs_y_std = float(np.asarray(D_O[graph.target]).std())
    results_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
        "Intervention_Set": intervention_set,
        "Intervention_Value": intervention_value,
        "Uncertainty": average_uncertainty,
        "Posterior_History": model.posterior_history,
        "True_Parents": true_parents,
        "Target": graph.target,
        "Variables": list(graph.variables),
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Index_To_ENSG": graph.index_to_ensg,
        "Target_ENSG": graph.index_to_ensg[int(graph.target)],
        "Weight_Scale": args.weight_scale,
        "Obs_Y_Std": obs_y_std,
    }

    results_dir = "results/gwps"
    os.makedirs(results_dir, exist_ok=True)
    ws = str(args.weight_scale).replace(".", "p")
    base = (
        f"run{args.run_num}_gwps_cbo_unknown_dr2_{args.n_observational}_"
        f"{args.n_int}_n{args.max_nodes}_ws{ws}"
    )
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved GWPS CBO-U results to {results_dir}/{base}.pickle")

    best = np.asarray(best_y_array, dtype=float)
    tp = set(true_parents)
    final = model.posterior_history[-1] if model.posterior_history else {}
    map_set = set(max(final.items(), key=lambda kv: kv[1])[0]) if final else set()
    logging.info(
        f"GWPS CBO-U done [weight_scale={args.weight_scale}]: "
        f"Best_Y {best[0]:.3f} -> {best[-1]:.3f} (obs Y std {obs_y_std:.3f}); "
        f"true parents recovered in final MAP: {sorted(tp & map_set)} / {sorted(tp)}; "
        f"intervened = {sorted(set(v for s in intervention_set for v in s))}"
    )


if __name__ == "__main__":
    run(parse_args())
