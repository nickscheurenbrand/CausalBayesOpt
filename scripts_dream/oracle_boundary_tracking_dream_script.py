"""ORACLE boundary-tracking on DREAM gene networks: same as boundary_tracking_dream_script.py but forces the parent
posterior to the TRUE parent set (prob 1.0). Output saved under results/boundary_tracking_dream_oracle/{graph_type}/."""

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
from graphs.graph import GraphStructure
from graphs.graph_dream import Dream4Graph
from utils.sem_sampling import draw_interventional_samples_sem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

GRAPH_TYPE_TO_YML = {
    "Size50-Ecoli1": "InSilicoSize50-Ecoli1",
    "Size100-Ecoli1": "InSilicoSize100-Ecoli1",
    "Size50-Ecoli2": "InSilicoSize50-Ecoli2",
    "Size100-Ecoli2": "InSilicoSize100-Ecoli2",
}


def pick_target(graph: GraphStructure) -> str:
    candidates = sorted(graph.variables, key=lambda v: (-len(graph.parents[v]), int(v)))
    best = candidates[0]
    if len(graph.parents[best]) == 0:
        raise ValueError("no variable in the graph has any parents")
    return best


def set_graph(graph_type: str, target: str, seed: int) -> GraphStructure:
    if graph_type not in GRAPH_TYPE_TO_YML:
        raise ValueError(f"unknown graph_type {graph_type}")
    graph = Dream4Graph(yml_name=GRAPH_TYPE_TO_YML[graph_type])
    if target is None:
        target = pick_target(graph)
        logging.info(f"auto-picked target {target} (most parents)")
    graph.set_target(target)
    graph.set_seed(seed)
    return graph


def compute_p_true_parents(posterior_history, true_parents):
    true_parents_set = set(true_parents)
    out = []
    for posterior in posterior_history:
        p = 0.0
        for parents, prob in posterior.items():
            if set(parents) == true_parents_set:
                p = prob
                break
        out.append(p)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_type", type=str, default="Size50-Ecoli1")
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--nonlinear", action="store_true")
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


def run_oracle_dream(args):
    nonlinear = True  # DREAM SEM is nonlinear
    nonlinear_string = "_nonlinear"

    graph = set_graph(args.graph_type, args.target, args.seeds_replicate)
    true_parents = tuple(graph.parents[graph.target])
    logging.info(
        f"ORACLE DREAM {args.graph_type}: target {graph.target}, "
        f"forcing true parents {true_parents}"
    )
    if len(true_parents) == 0:
        raise ValueError(f"target {graph.target} has no parents; oracle undefined")

    D_O, _, _ = setup_observational_interventional(
        graph_type=None,
        noiseless=args.noiseless,
        seed=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )

    # ---- JOINT ORACLE EXPLORATION SET ----
    # The oracle intervenes on ALL true parents simultaneously.
    exploration_set = [true_parents]
    D_I = draw_interventional_samples_sem(
        exploration_set, graph, n_int=args.n_int,
        seed=args.seeds_replicate, noiseless=args.noiseless,
    )
    logging.info(f"ORACLE: joint exploration set {exploration_set}")
    # --------------------------------------

    model = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
        acquisition=args.acquisition,
        kernel_type=args.kernel,
    )
    model.set_values(D_O, D_I, exploration_set)

    # ---- ORACLE INJECTION: force the true parent set as the sole candidate,
    # and pin the exploration set to the JOINT parent set (otherwise
    # redefine_exploration_set flattens it into one singleton per parent).
    def _oracle_initial_probabilities():
        return {true_parents: 1.0}

    def _joint_exploration_set():
        model.exploration_set = [true_parents]

    model.determine_initial_probabilities = _oracle_initial_probabilities
    model.redefine_exploration_set = _joint_exploration_set
    # ---------------------------------------------------------------------------

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
        "Dream": True,
        "Oracle": True,
        "Kernel_Type": args.kernel,
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Variables": list(graph.variables),
    }

    tag = f"{args.graph_type}_t{args.target}" if args.target else args.graph_type
    results_dir = (
        f"results/boundary_tracking_dream_oracle{KERNEL_SUFFIX[args.kernel]}/{tag}"
    )
    os.makedirs(results_dir, exist_ok=True)
    base_name = (
        f"run{args.run_num}_cbo_unknown_dr2_boundary_{args.acquisition}_"
        f"{args.n_observational}_{args.n_int}{nonlinear_string}"
    )
    with open(f"{results_dir}/{base_name}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved ORACLE DREAM results to {results_dir}/{base_name}.pickle")

    boundary = np.array(model.boundary_percentages, dtype=float)
    logging.info(
        f"ORACLE DREAM {args.graph_type} [{args.acquisition}]: mean strict "
        f"boundary% = {boundary.mean():.3f}; intervened variables = "
        f"{sorted(set(v for s in intervention_set for v in s))}"
    )


if __name__ == "__main__":
    run_oracle_dream(parse_args())
