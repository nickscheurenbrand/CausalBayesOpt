"""
Same as boundary_tracking_script.py (CBO-U, PARENT_SCALE dr2 on Erdos-Renyi
graphs, tracking the intervention-boundary percentage and the parent-set
posterior), but the surrogate GP uses a spherical (stereographic-projection)
kernel instead of the plain RBF core.

Choose the projected kernel with --kernel:
  * spherical_linear (default): linear/dot-product kernel on the projected
    features (results/boundary_tracking_spherical/)
  * spherical_rbf: RBF kernel on the projected features
    (results/boundary_tracking_spherical_rbf/)

Only the surrogate kernel changes -- everything else (causal prior, do-mean/
do-variance, acquisition, boundary tracking) is identical, and results are
written under a kernel-specific subdir so they do not collide with the RBF
baseline or with each other.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from scripts.base_script import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# results subdirectory for each supported projected kernel
RESULTS_SUBDIRS = {
    "spherical_linear": "boundary_tracking_spherical",
    "spherical_rbf": "boundary_tracking_spherical_rbf",
}


def set_graph(graph_type: str, nonlinear: bool = False) -> GraphStructure:
    assert graph_type in ["Erdos20", "Erdos50", "Erdos100"]
    if graph_type == "Erdos20":
        graph = ErdosRenyiGraph(num_nodes=20, nonlinear=nonlinear)
        graph.set_target("18")
    elif graph_type == "Erdos50":
        graph = ErdosRenyiGraph(num_nodes=50, nonlinear=nonlinear)
        graph.set_target("23")
    elif graph_type == "Erdos100":
        graph = ErdosRenyiGraph(num_nodes=100, nonlinear=nonlinear)
        graph.set_target("80")
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


def plot_boundary_percentage(boundary_percentages, graph_type, filename):
    iterations = np.arange(1, len(boundary_percentages) + 1)
    running_mean = np.cumsum(boundary_percentages) / iterations

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        iterations, boundary_percentages, alpha=0.4, label="Per-iteration boundary %"
    )
    ax.plot(iterations, running_mean, color="tab:red", label="Running mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Boundary percentage")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Intervention boundary percentage ({graph_type}, spherical-linear)")
    ax.legend()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_p_true_parents(p_true_parents, graph_type, filename):
    # iteration 0 is the initial posterior before any intervention
    iterations = np.arange(len(p_true_parents))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, p_true_parents, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P(true parents)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Posterior probability of true parent set ({graph_type}, spherical-linear)"
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def run_boundary_tracking(
    graph_type: str,
    run_num: int,
    noiseless: bool,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_trials: int,
    nonlinear: bool,
    acquisition: str = "EI",
    kernel_type: str = "spherical_linear",
):
    results_subdir = RESULTS_SUBDIRS[kernel_type]
    nonlinear_string = "_nonlinear" if nonlinear else ""
    graph = set_graph(graph_type, nonlinear=nonlinear)
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        graph=graph,
    )

    model = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
        acquisition=acquisition,
        kernel_type=kernel_type,
    )
    model.set_values(D_O, D_I, exploration_set)
    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=n_trials, show_graphics=False)

    true_parents = tuple(graph.parents[graph.target])
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
        "Kernel_Type": kernel_type,
        # full structure so the confounder classifier works on these runs too
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Variables": list(graph.variables),
        "Target": graph.target,
    }

    results_dir = f"results/{results_subdir}/{graph_type}"
    os.makedirs(results_dir, exist_ok=True)
    base_name = (
        f"run{run_num}_cbo_unknown_dr2_boundary_{acquisition}_"
        f"{n_obs}_{n_int}{nonlinear_string}"
    )

    filename_pickle = f"{results_dir}/{base_name}.pickle"
    with open(filename_pickle, "wb") as file:
        pickle.dump(results_dict, file)
    logging.info(f"Saved results to {filename_pickle}")

    plot_boundary_percentage(
        model.boundary_percentages,
        graph_type,
        f"{results_dir}/{base_name}_boundary_percentage.png",
    )
    plot_p_true_parents(
        p_true_parents,
        graph_type,
        f"{results_dir}/{base_name}_p_true_parents.png",
    )
    logging.info(f"Saved plots to {results_dir}")


if __name__ == "__main__":
    args = parse_args()
    # --kernel is parsed locally; parse_args uses parse_known_args so it is
    # ignored there and does not clash with the shared argument parser.
    kernel_parser = argparse.ArgumentParser(add_help=False)
    kernel_parser.add_argument(
        "--kernel",
        type=str,
        default="spherical_linear",
        choices=list(RESULTS_SUBDIRS.keys()),
        help="Projected surrogate kernel to use.",
    )
    kernel_args, _ = kernel_parser.parse_known_args()

    run_boundary_tracking(
        graph_type=args.graph_type,
        run_num=args.run_num,
        noiseless=args.noiseless,
        seeds_int_data=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=2,
        n_trials=args.n_trials,
        nonlinear=args.nonlinear,
        acquisition=args.acquisition,
        kernel_type=kernel_args.kernel,
    )
