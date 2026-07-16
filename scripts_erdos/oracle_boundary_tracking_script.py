"""
ORACLE variant of boundary_tracking_script.py: confound-closing experiment.

The non-oracle runs showed a strong boundary bias in the intervention values
ONLY on true-parent variables, and none on Erdos100 -- but in those 3 runs
"true parent" is perfectly confounded with "small/medium graph" (all
true-parent interventions came from Erdos20/50, all non-parent ones from
Erdos100). So we cannot yet tell whether Erdos100 lacks boundary bias because
it is large, or simply because its collapsed posterior made it intervene on
non-parents.

This script removes the confound by forcing the parent posterior to the TRUE
parent set (probability 1.0), bypassing the doubly-robust bootstrap entirely,
while keeping EVERYTHING ELSE identical -- same PARENT_SCALE acquisition, same
GP surrogates, same boundary tracking. The exploration set then becomes exactly
the true parents (as singletons), so Erdos100 is guaranteed to intervene on
its real parents. If boundary bias appears here, the boundary mechanism works
at scale and the earlier absence was entirely downstream of the posterior
failure.

The injection point is determine_initial_probabilities() -- the single place
the candidate parent-set posterior is created. define_all_possible_graphs()
then rebuilds the correct local structure around the target from it via
graph.mispecify_graph(edges). Since there is only one hypothesis at prob 1.0,
every subsequent per-trial posterior update leaves it at 1.0, so the
exploration set stays pinned to the true parents for the whole run.

Output pickle format is IDENTICAL to boundary_tracking_script.py (so
boundary_bias_analysis.py --results_subdir boundary_tracking_oracle works on
it), saved under results/boundary_tracking_oracle/{graph_type}/.
"""

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
_algorithms_path = os.path.join(os.getcwd(), "algorithms")
if _algorithms_path not in sys.path:
    sys.path.append(_algorithms_path)

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


def run_oracle_boundary_tracking(
    graph_type: str,
    run_num: int,
    noiseless: bool,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_trials: int,
    nonlinear: bool,
):
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

    true_parents = tuple(graph.parents[graph.target])
    logging.info(f"ORACLE: forcing parent posterior to true parents {true_parents}")
    if len(true_parents) == 0:
        raise ValueError(
            f"target {graph.target} has no parents; oracle run is undefined"
        )

    model = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
    )
    model.set_values(D_O, D_I, exploration_set)

    # ---- ORACLE INJECTION ----
    # Replace the bootstrap parent-identification with a point mass on the
    # true parent set. This is the ONLY behavioural difference from
    # boundary_tracking_script.py.
    def _oracle_initial_probabilities():
        return {true_parents: 1.0}

    model.determine_initial_probabilities = _oracle_initial_probabilities
    # --------------------------

    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=n_trials, show_graphics=False)

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
        "Oracle": True,
    }

    results_dir = f"results/boundary_tracking_oracle/{graph_type}"
    os.makedirs(results_dir, exist_ok=True)
    base_name = (
        f"run{run_num}_cbo_unknown_dr2_boundary_{n_obs}_{n_int}{nonlinear_string}"
    )
    filename_pickle = f"{results_dir}/{base_name}.pickle"
    with open(filename_pickle, "wb") as file:
        pickle.dump(results_dict, file)
    logging.info(f"Saved ORACLE results to {filename_pickle}")

    # quick inline summary so the log is self-contained
    boundary = np.array(model.boundary_percentages, dtype=float)
    logging.info(
        f"ORACLE {graph_type}: mean strict boundary% = {boundary.mean():.3f} "
        f"over {len(boundary)} trials; intervened variables = "
        f"{sorted(set(v for s in intervention_set for v in s))}"
    )


if __name__ == "__main__":
    args = parse_args()
    run_oracle_boundary_tracking(
        graph_type=args.graph_type,
        run_num=args.run_num,
        noiseless=args.noiseless,
        seeds_int_data=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=2,
        n_trials=args.n_trials,
        nonlinear=args.nonlinear,
    )
