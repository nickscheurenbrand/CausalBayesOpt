"""
Boundary tracking under a LATENT (hidden) confounder, for Erdos (linear) and
DREAM (nonlinear) graphs.

A hidden common cause Z is injected on (X, target): Z->X, Z->Y. Z is present in
the data-generating SEM (so it correlates X and Y in the observational data) but
is stripped from the observed data and never given to the algorithm. The model
runs on the ORIGINAL base graph (no Z); under do(X) the backdoor is cut so the
true interventional effect is flat -- the base SEM already yields correct
interventional outcomes, so no PARENT_SCALE surgery is needed.

Two configurations (--x_kind):
  non_parent   -> X is a manipulable non-parent of the target (false positive:
                  does the algorithm chase a non-causal variable's boundary?)
  true_parent  -> X is a true parent of the target (effect corruption: does the
                  confounding flip/bias a real parent's do-effect?)

X is seeded into the candidate posterior at small mass (like the seeded
experiment) so its boundary behaviour is observable. Saves the boundary pickle
schema plus graph structure + confounder metadata under
results/boundary_tracking_confounded/{graph_type}_{x_kind}/.
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
from graphs.latent_confounder import (
    generate_confounded_observational_data,
    pick_confounded_x,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

DREAM_YML = {
    "Size50-Ecoli1": "InSilicoSize50-Ecoli1",
    "Size100-Ecoli1": "InSilicoSize100-Ecoli1",
}
ERDOS_NODES = {"Erdos20": 20, "Erdos50": 50, "Erdos100": 100}
ERDOS_TARGET = {"Erdos20": "18", "Erdos50": "23", "Erdos100": "80"}


def build_base_graph(graph_type, target, seed):
    """Returns (base_graph, is_dream). Dream => nonlinear SEM."""
    if graph_type in ERDOS_NODES:
        graph = ErdosRenyiGraph(num_nodes=ERDOS_NODES[graph_type], nonlinear=False)
        graph.set_target(target or ERDOS_TARGET[graph_type])
        return graph, False
    if graph_type in DREAM_YML:
        graph = Dream4Graph(yml_name=DREAM_YML[graph_type])
        if target is None:
            # node with most parents (as in the DREAM boundary script)
            target = sorted(
                graph.variables, key=lambda v: (-len(graph.parents[v]), int(v))
            )[0]
        graph.set_target(target)
        graph.set_seed(seed)
        return graph, True
    raise ValueError(f"unknown graph_type {graph_type}")


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
    p.add_argument("--graph_type", type=str, default="Erdos50")
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--x_kind", choices=["non_parent", "true_parent"], default="non_parent")
    p.add_argument("--w_zx", type=float, default=2.0)
    p.add_argument("--w_zy", type=float, default=2.0)
    p.add_argument("--sigma_z", type=float, default=1.0)
    p.add_argument("--seed_x", action="store_true", default=True,
                   help="seed X into the candidate posterior so its behaviour is observable")
    p.add_argument("--inject_prob", type=float, default=0.1)
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    return p.parse_args()


def run(args):
    base_graph, is_dream = build_base_graph(
        args.graph_type, args.target, args.seeds_replicate
    )
    nonlinear = is_dream
    target = base_graph.target
    true_parents = tuple(base_graph.parents[target])
    x_var = pick_confounded_x(base_graph, args.x_kind)
    logging.info(
        f"{args.graph_type} [{args.x_kind}]: target {target}, true parents "
        f"{true_parents}; confounding X={x_var} (is_true_parent="
        f"{x_var in set(true_parents)})"
    )

    confounders = [
        {"x": x_var, "w_zx": args.w_zx, "w_zy": args.w_zy, "sigma_z": args.sigma_z}
    ]

    # confounded observational data (hidden Z stripped); interventional data +
    # exploration set from the base graph (interventions are unconfounded)
    D_O, conf_meta = generate_confounded_observational_data(
        base_graph, confounders, n_obs=args.n_observational, seed=args.seeds_replicate
    )
    _, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=args.noiseless,
        seed=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=base_graph,
    )

    # quick sanity: X-Y correlation is present in the confounded observational data
    try:
        corr = float(np.corrcoef(D_O[x_var].reshape(-1), D_O[target].reshape(-1))[0, 1])
        logging.info(f"confounded observational corr(X={x_var}, Y={target}) = {corr:.3f}")
    except Exception:
        pass

    model = PARENT_SCALE(
        graph=base_graph, nonlinear=nonlinear, individual=True, use_doubly_robust=True
    )
    model.set_values(D_O, D_I, exploration_set)

    if args.seed_x:
        original_init = model.determine_initial_probabilities

        def seeded_initial_probabilities():
            probs = dict(original_init())
            probs[(x_var,)] = probs.get((x_var,), 0.0) + args.inject_prob
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            logging.info(f"SEED: injected ({x_var},) at {args.inject_prob}; candidates={probs}")
            return probs

        model.determine_initial_probabilities = seeded_initial_probabilities

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
        var: list(bounds) for var, bounds in base_graph.interventional_range_data.items()
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
        # structure + confounder metadata for the analysis classifier
        "Edges": list(base_graph.edges),
        "Parents": {v: list(base_graph.parents[v]) for v in base_graph.variables},
        "Variables": list(base_graph.variables),
        "Target": target,
        "Confounders": conf_meta,
        "X_Kind": args.x_kind,
        "Dream": is_dream,
    }

    results_dir = f"results/boundary_tracking_confounded/{args.graph_type}_{args.x_kind}"
    os.makedirs(results_dir, exist_ok=True)
    ns = "_nonlinear" if nonlinear else ""
    base = f"run{args.run_num}_cbo_confounded_dr2_boundary_{args.n_observational}_{args.n_int}{ns}"
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved confounded boundary results to {results_dir}/{base}.pickle")

    boundary = np.array(model.boundary_percentages, dtype=float)
    x_interventions = sum(1 for s in intervention_set for v in s if v == x_var)
    x_in_posterior = any(
        x_var in set(k) for snap in model.posterior_history for k in snap
    )
    logging.info(
        f"CONFOUNDED {args.graph_type} [{args.x_kind}]: mean boundary% = "
        f"{boundary.mean():.3f}; interventions on confounded X={x_var}: "
        f"{x_interventions}/{args.n_trials}; X ever in posterior: {x_in_posterior}"
    )


if __name__ == "__main__":
    run(parse_args())
