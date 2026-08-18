"""
Boundary-tracking experiment on the GWPS gene network (linear-G_hat SEM), the
GWPS analogue of the Erdos/DREAM boundary scripts.

Builds a GwpsGraph (DAG subgraph of the real Perturb-seq network, linear-Gaussian
SEM using the real G_hat weights scaled by --weight_scale), runs CBO-U
(PARENT_SCALE, dr2) while tracking the intervention boundary percentage and the
parent posterior, and saves the same schema as the Erdos/DREAM boundary scripts
so results_erdos/boundary_bias_analysis.py works unchanged.

--oracle forces the true parent set (probability 1.0), bypassing the bootstrap
(the confound-closing variant). GWPS is LINEAR, so PARENT_SCALE runs with
nonlinear=False and the boundary hypothesis (monotone effect -> edge optimum)
applies as for Erdos.

Output:
  results/boundary_tracking_gwps/{tag}/...            (baseline)
  results/boundary_tracking_gwps_oracle/{tag}/...     (--oracle)
  tag = gwps_n{max_nodes}_ws{weight_scale}
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
from utils.sem_sampling import draw_interventional_samples_sem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def compute_p_true_parents(posterior_history, true_parents):
    tp = set(true_parents)
    out = []
    for posterior in posterior_history:
        p = 0.0
        for parents, prob in posterior.items():
            if set(parents) == tp:
                p = prob
                break
        out.append(p)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--top_k_parents", type=int, default=8)
    p.add_argument("--weight_scale", type=float, default=3.0)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--n_non_ancestors", type=int, default=0,
                   help="reserve this many of --max_nodes for genes that are NOT "
                        "ancestors of the target; must match the random arm so "
                        "both run on the same graph (0 = original carve)")
    p.add_argument("--target", type=str, default=None, help="target ENSG (auto if omitted)")
    p.add_argument("--oracle", action="store_true", help="force the true parent set (prob 1.0)")
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--acquisition", type=str, default="EI", choices=["EI", "UCB"])
    p.add_argument(
        "--kernel", type=str, default="rbf",
        choices=["rbf", "spherical_linear", "spherical_rbf"],
    )
    return p.parse_args()


# results-subdir suffix for each surrogate kernel ("rbf" keeps the original path)
KERNEL_SUFFIX = {
    "rbf": "",
    "spherical_linear": "_spherical",
    "spherical_rbf": "_spherical_rbf",
}


def run(args):
    graph = GwpsGraph(
        target=args.target,
        max_nodes=args.max_nodes,
        top_k_parents=args.top_k_parents,
        noise_sigma=args.noise_sigma,
        weight_scale=args.weight_scale,
        seed=args.seeds_replicate,
        n_non_ancestors=args.n_non_ancestors,
    )
    true_parents = tuple(graph.parents[graph.target])
    logging.info(
        f"GWPS boundary [oracle={args.oracle}, acq={args.acquisition}]: "
        f"{len(graph.variables)} nodes, target {graph.target} "
        f"({graph.index_to_ensg[int(graph.target)]}), true parents {true_parents}"
    )
    if len(true_parents) == 0:
        raise ValueError(f"target {graph.target} has no parents")

    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=args.noiseless,
        seed=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )

    if args.oracle:
        # ---- JOINT ORACLE EXPLORATION SET ----
        # The oracle intervenes on ALL true parents simultaneously, so the
        # exploration set is the single joint set and the interventional data
        # must be drawn for it.
        exploration_set = [true_parents]
        D_I = draw_interventional_samples_sem(
            exploration_set, graph, n_int=args.n_int,
            seed=args.seeds_replicate, noiseless=args.noiseless,
        )
        logging.info(f"ORACLE: joint exploration set {exploration_set}")

    model = PARENT_SCALE(
        graph=graph,
        nonlinear=False,  # GWPS SEM is linear (real G_hat weights)
        individual=True,
        use_doubly_robust=True,
        acquisition=args.acquisition,
        kernel_type=args.kernel,
    )
    model.set_values(D_O, D_I, exploration_set)

    if args.oracle:
        # force the true parent set as the sole candidate, and pin the
        # exploration set to the JOINT parent set (otherwise
        # redefine_exploration_set flattens it into one singleton per parent)
        def _oracle_initial_probabilities():
            return {true_parents: 1.0}

        def _joint_exploration_set():
            model.exploration_set = [true_parents]

        model.determine_initial_probabilities = _oracle_initial_probabilities
        model.redefine_exploration_set = _joint_exploration_set

    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=args.n_trials, show_graphics=False)

    p_true_parents = compute_p_true_parents(model.posterior_history, true_parents)
    intervention_ranges = {
        var: list(bounds) for var, bounds in graph.interventional_range_data.items()
    }

    results_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
        "Intervention_Set": intervention_set,
        "Intervention_Value": intervention_value,
        "Uncertainty": average_uncertainty,
        "Boundary_Percentage": model.boundary_percentages,
        "Boundary_Counts": model.boundary_counts,
        "Posterior_History": model.posterior_history,
        "P_True_Parents": p_true_parents,
        "True_Parents": true_parents,
        "Intervention_Ranges": intervention_ranges,
        "Epsilon_Fraction": model.boundary_eps_frac,
        "Target": graph.target,
        "Target_ENSG": graph.index_to_ensg[int(graph.target)],
        "Index_To_ENSG": graph.index_to_ensg,
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Variables": list(graph.variables),
        "Weight_Scale": args.weight_scale,
        "Gwps": True,
        "Oracle": args.oracle,
        "Kernel_Type": args.kernel,
    }

    na = f"_na{args.n_non_ancestors}" if args.n_non_ancestors else ""
    tag = f"gwps_n{args.max_nodes}_ws{args.weight_scale:g}{na}"
    subdir = "boundary_tracking_gwps_oracle" if args.oracle else "boundary_tracking_gwps"
    results_dir = f"results/{subdir}{KERNEL_SUFFIX[args.kernel]}/{tag}"
    os.makedirs(results_dir, exist_ok=True)
    base_name = (
        f"run{args.run_num}_cbo_unknown_dr2_boundary_{args.acquisition}_"
        f"{args.n_observational}_{args.n_int}"
    )
    with open(f"{results_dir}/{base_name}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved GWPS boundary results to {results_dir}/{base_name}.pickle")

    boundary = np.array(model.boundary_percentages, dtype=float)
    logging.info(
        f"GWPS boundary [oracle={args.oracle}, {args.acquisition}]: mean strict "
        f"boundary% = {boundary.mean():.3f}; intervened variables = "
        f"{sorted(set(v for s in intervention_set for v in s))}"
    )


if __name__ == "__main__":
    run(parse_args())
