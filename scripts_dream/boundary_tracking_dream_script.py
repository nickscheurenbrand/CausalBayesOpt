"""
Baseline boundary-tracking experiment on the DREAM (dream4) gene networks.

Structural copy of scripts_erdos/boundary_tracking_script.py, but builds a
Dream4Graph (InSilicoSize50 / InSilicoSize100 E. coli network) instead of an
Erdos-Renyi graph. Runs CBO-U (PARENT_SCALE, dr2 variant) while tracking, per
iteration:
  - the intervention boundary percentage (fraction of intervened dimensions
    within +- eps of the intervention-range boundary)
  - the posterior over parent sets of the target

NOTE (nonlinear only): Dream4Graph builds its environment with nonlinear=True
hardcoded, so the SEM is nonlinear. The boundary-bias hypothesis was derived
for LINEAR monotone effects, so this run is exploratory -- "does a boundary
bias appear at all under a nonlinear SEM?" -- not a direct replication of the
linear-Erdos result. PARENT_SCALE is therefore run with nonlinear=True.

Saves the raw tracking data as a pickle under
results/boundary_tracking_dream/{graph_type}/ using the same schema as the
Erdos boundary script, so results_erdos/boundary_bias_analysis.py works on it.
"""

import argparse
import logging
import os
import pickle
import sys

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# PBS sets CUDA_VISIBLE_DEVICES to a GPU UUID, which cdt cannot parse at
# import time; the allocated GPU is the only one visible inside the job's
# cgroup, so index 0 refers to the same device
if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
# the graphs package now lives under algorithms/, but modules import `graphs.*`
_algorithms_path = os.path.join(os.getcwd(), "algorithms")
if _algorithms_path not in sys.path:
    sys.path.append(_algorithms_path)

import numpy as np

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_dream import Dream4Graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# maps the CLI graph_type label -> the dream4 configuration XML basename
GRAPH_TYPE_TO_YML = {
    "Size50-Ecoli1": "InSilicoSize50-Ecoli1",
    "Size100-Ecoli1": "InSilicoSize100-Ecoli1",
    "Size50-Ecoli2": "InSilicoSize50-Ecoli2",
    "Size100-Ecoli2": "InSilicoSize100-Ecoli2",
}


def pick_target(graph: GraphStructure) -> str:
    """
    Deterministically choose a target: the variable with the most parents
    (ties broken by lowest integer index). Size50/100 DREAM nets have no
    hardcoded target, and a target with parents is required for the boundary /
    P_True_Parents signals to be meaningful.
    """
    candidates = sorted(graph.variables, key=lambda v: (-len(graph.parents[v]), int(v)))
    best = candidates[0]
    if len(graph.parents[best]) == 0:
        raise ValueError("no variable in the graph has any parents")
    return best


def set_graph(graph_type: str, target: str, seed: int) -> GraphStructure:
    if graph_type not in GRAPH_TYPE_TO_YML:
        raise ValueError(
            f"unknown graph_type {graph_type}; expected one of {list(GRAPH_TYPE_TO_YML)}"
        )
    graph = Dream4Graph(yml_name=GRAPH_TYPE_TO_YML[graph_type])
    if target is None:
        target = pick_target(graph)
        logging.info(f"auto-picked target {target} (most parents)")
    graph.set_target(target)
    graph.set_seed(seed)
    return graph


def compute_p_true_parents(posterior_history, true_parents):
    """
    Extracts the posterior probability assigned to the true parent set at
    each iteration (0.0 if the true parent set has been pruned / is absent)
    """
    true_parents_set = set(true_parents)
    p_true = []
    for posterior in posterior_history:
        p = 0.0
        for parents, prob in posterior.items():
            if set(parents) == true_parents_set:
                p = prob
                break
        p_true.append(p)
    return p_true


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_type", type=str, default="Size50-Ecoli1")
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="target node (string index); if omitted, the node with the most parents",
    )
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    # DREAM SEM is nonlinear; kept for the filename suffix / posterior model.
    p.add_argument("--nonlinear", action="store_true")
    return p.parse_args()


def run_boundary_tracking_dream(args):
    # DREAM's SEM is nonlinear regardless of the flag; force the posterior model
    # to match it.
    nonlinear = True
    nonlinear_string = "_nonlinear" if nonlinear else ""

    graph = set_graph(args.graph_type, args.target, args.seeds_replicate)
    true_parents = tuple(graph.parents[graph.target])
    logging.info(
        f"{args.graph_type}: target {graph.target}, true parents {true_parents}, "
        f"{len(graph.variables)} nodes"
    )
    if len(true_parents) == 0:
        raise ValueError(
            f"target {graph.target} has no parents; pick a different --target"
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
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
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
        # full structure so the confounder classifier works on these runs too
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Variables": list(graph.variables),
    }

    results_dir = f"results/boundary_tracking_dream/{args.graph_type}"
    os.makedirs(results_dir, exist_ok=True)
    base_name = (
        f"run{args.run_num}_cbo_unknown_dr2_boundary_"
        f"{args.n_observational}_{args.n_int}{nonlinear_string}"
    )
    filename_pickle = f"{results_dir}/{base_name}.pickle"
    with open(filename_pickle, "wb") as file:
        pickle.dump(results_dict, file)
    logging.info(f"Saved DREAM boundary results to {filename_pickle}")

    boundary = np.array(model.boundary_percentages, dtype=float)
    logging.info(
        f"DREAM {args.graph_type}: mean strict boundary% = {boundary.mean():.3f} "
        f"over {len(boundary)} trials; intervened variables = "
        f"{sorted(set(v for s in intervention_set for v in s))}"
    )


if __name__ == "__main__":
    run_boundary_tracking_dream(parse_args())
