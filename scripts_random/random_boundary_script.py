"""
Boundary tracking when intervening on RANDOM NON-PARENT nodes.

The counterfactual to the oracle: instead of forcing the true parent set, force a
randomly drawn set of NON-parents of the same cardinality, and intervene on them
jointly. Everything else (acquisition, GP surrogate, boundary tracking, joint
intervention) is identical to the oracle, so any difference is attributable to
*which* nodes are intervened on.

Why this is informative: PARENT_SCALE builds each candidate graph via
mispecify_graph([(x, target) for x in chosen_set]), which discards all other
edges -- so the surrogate's do-effect is essentially the OBSERVATIONAL regression
of Y on the chosen set, while realised outcomes come from the TRUE SEM. A random
non-parent set therefore tests whether boundary-seeking tracks observational
association rather than causal effect. The saved metadata includes each chosen
node's observational correlation with the target so this can be checked directly.

The random set is drawn with the replicate seed, so each seed gives a DIFFERENT
set -- the 5 replicates provide set-to-set variability, not just data noise.

Output:
  results/boundary_tracking_{erdos|dream|gwps}_random{kernel_suffix}/{tag}/
      run{run_num}_cbo_unknown_dr2_boundary_{ACQ}_{n_obs}_{n_int}[_nonlinear].pickle
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
from graphs.graph import GraphStructure
from graphs.graph_dream import Dream4Graph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.graph_gwps import GwpsGraph
from utils.sem_sampling import draw_interventional_samples_sem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# results-subdir suffix per surrogate kernel ("rbf" keeps the plain path)
KERNEL_SUFFIX = {"rbf": "", "spherical_linear": "_spherical",
                 "spherical_rbf": "_spherical_rbf"}

ERDOS = {"Erdos20": (20, "18"), "Erdos50": (50, "23"), "Erdos100": (100, "80")}
DREAM_YML = {
    "Size50-Ecoli1": "InSilicoSize50-Ecoli1",
    "Size100-Ecoli1": "InSilicoSize100-Ecoli1",
    "Size50-Ecoli2": "InSilicoSize50-Ecoli2",
    "Size100-Ecoli2": "InSilicoSize100-Ecoli2",
}


def build_graph(args):
    """Returns (graph, family, nonlinear, tag)."""
    gt = args.graph_type
    if gt in ERDOS:
        n, target = ERDOS[gt]
        graph = ErdosRenyiGraph(num_nodes=n, nonlinear=False)
        graph.set_target(target)
        # vary the observational draw across replicates (see oracle script)
        graph.set_seed(args.seeds_replicate)
        return graph, "erdos", False, gt
    if gt in DREAM_YML:
        graph = Dream4Graph(yml_name=DREAM_YML[gt])
        target = sorted(graph.variables,
                        key=lambda v: (-len(graph.parents[v]), int(v)))[0]
        graph.set_target(target)
        graph.set_seed(args.seeds_replicate)
        return graph, "dream", True, gt
    if gt == "gwps":
        graph = GwpsGraph(
            target=args.target, max_nodes=args.max_nodes,
            top_k_parents=args.top_k_parents, noise_sigma=args.noise_sigma,
            weight_scale=args.weight_scale, seed=args.seeds_replicate,
        )
        return graph, "gwps", False, f"gwps_n{args.max_nodes}_ws{args.weight_scale:g}"
    raise ValueError(f"unknown graph_type {gt!r}")


def pick_random_non_parents(graph: GraphStructure, k: int, seed: int):
    """k distinct non-parent, non-target nodes, drawn with `seed`."""
    target = graph.target
    parents = set(graph.parents[target])
    pool = [v for v in graph.variables if v != target and v not in parents]
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} non-parents available, need {k}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(pool, dtype=object), size=k, replace=False)
    # deterministic ordering (numeric where possible) for a stable tuple key
    try:
        return tuple(sorted(chosen, key=lambda v: int(v)))
    except ValueError:
        return tuple(sorted(chosen))


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
    p.add_argument("--graph_type", type=str, default="Erdos50",
                   help="Erdos20/50/100, Size50-Ecoli1, Size100-Ecoli1, or gwps")
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--acquisition", type=str, default="EI", choices=["EI", "UCB"])
    p.add_argument("--kernel", type=str, default="rbf",
                   choices=list(KERNEL_SUFFIX.keys()))
    # gwps-only knobs (ignored otherwise)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--top_k_parents", type=int, default=8)
    p.add_argument("--weight_scale", type=float, default=3.0)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    return p.parse_args()


def run(args):
    graph, family, nonlinear, tag = build_graph(args)
    target = graph.target
    true_parents = tuple(graph.parents[target])
    if len(true_parents) == 0:
        raise ValueError(f"target {target} has no parents; nothing to match in size")

    # random NON-PARENT set, same cardinality as the true parent set so the
    # boundary fraction has the same denominator as the oracle
    chosen = pick_random_non_parents(graph, len(true_parents), args.seeds_replicate)
    logging.info(
        f"RANDOM non-parents [{args.graph_type}, seed={args.seeds_replicate}]: "
        f"target {target}, true parents {true_parents}, intervening on {chosen}"
    )

    D_O, _, _ = setup_observational_interventional(
        graph_type=None, noiseless=args.noiseless, seed=args.seeds_replicate,
        n_obs=args.n_observational, n_int=args.n_int, graph=graph,
    )

    # joint intervention on the random set (same shape as the joint oracle)
    exploration_set = [chosen]
    D_I = draw_interventional_samples_sem(
        exploration_set, graph, n_int=args.n_int,
        seed=args.seeds_replicate, noiseless=args.noiseless,
    )

    # observational association of each chosen node with the target -- the
    # quantity the "boundary tracks association, not causation" test needs
    y = np.asarray(D_O[target]).reshape(-1)
    corr = {}
    for v in chosen:
        x = np.asarray(D_O[v]).reshape(-1)
        corr[v] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
    logging.info(f"observational corr(chosen, target): "
                 f"{ {k: round(c, 3) for k, c in corr.items()} }")

    model = PARENT_SCALE(
        graph=graph, nonlinear=nonlinear, individual=True,
        use_doubly_robust=True, acquisition=args.acquisition,
        kernel_type=args.kernel,
    )
    model.set_values(D_O, D_I, exploration_set)

    # ---- force the RANDOM set (mirrors the oracle injection) ----
    def _random_initial_probabilities():
        return {chosen: 1.0}

    def _joint_exploration_set():
        model.exploration_set = [chosen]

    model.determine_initial_probabilities = _random_initial_probabilities
    model.redefine_exploration_set = _joint_exploration_set
    # -------------------------------------------------------------

    (
        best_y_array, current_y_array, cost_array,
        intervention_set, intervention_value, average_uncertainty,
    ) = model.run_algorithm(T=args.n_trials, show_graphics=False)

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
        "P_True_Parents": compute_p_true_parents(model.posterior_history, true_parents),
        "True_Parents": true_parents,
        "Intervention_Ranges": {v: list(b) for v, b in
                                graph.interventional_range_data.items()},
        "Epsilon_Fraction": model.boundary_eps_frac,
        "Target": target,
        "Variables": list(graph.variables),
        "Edges": list(graph.edges),
        "Parents": {v: list(graph.parents[v]) for v in graph.variables},
        "Kernel_Type": args.kernel,
        # random-set metadata
        "Random_Set": chosen,
        "Node_Category": "any_non_parent",
        "Chosen_Corr_With_Target": corr,
        "Random_Seed": args.seeds_replicate,
        "Oracle": False,
    }
    if family == "gwps":
        results_dict["Target_ENSG"] = graph.index_to_ensg[int(target)]
        results_dict["Index_To_ENSG"] = graph.index_to_ensg
        results_dict["Weight_Scale"] = args.weight_scale

    results_dir = (f"results/boundary_tracking_{family}_random"
                   f"{KERNEL_SUFFIX[args.kernel]}/{tag}")
    os.makedirs(results_dir, exist_ok=True)
    ns = "_nonlinear" if nonlinear else ""
    base = (f"run{args.run_num}_cbo_unknown_dr2_boundary_{args.acquisition}_"
            f"{args.n_observational}_{args.n_int}{ns}")
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved RANDOM results to {results_dir}/{base}.pickle")

    boundary = np.array(model.boundary_percentages, dtype=float)
    best = np.asarray(best_y_array, dtype=float)
    logging.info(
        f"RANDOM {args.graph_type} [{args.acquisition}, seed={args.seeds_replicate}]: "
        f"mean boundary% = {boundary.mean():.3f}; "
        f"Best_Y {best[0]:.3f} -> {best[-1]:.3f} (improvement {best[-1]-best[0]:+.3f})"
    )


if __name__ == "__main__":
    run(parse_args())