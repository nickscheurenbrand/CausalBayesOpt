"""Seed-and-track experiment: mixes true-parent hypotheses into the real doubly-robust bootstrap posterior at small prior
mass, then checks whether the 30-trial CBO update learns/retains or prunes them. Output: results/boundary_tracking_seeded/{graph_type}/."""

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

import argparse

import numpy as np

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_erdos_renyi import ErdosRenyiGraph

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_type", type=str, default="Erdos100")
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=2)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument(
        "--inject_prob",
        type=float,
        default=0.1,
        help="prior mass added per injected hypothesis",
    )
    return p.parse_args()


def true_parent_hypotheses(true_parents):
    """The exact true set plus each singleton true parent."""
    hyps = [tuple(true_parents)]
    for v in true_parents:
        hyps.append((v,))
    # dedup while preserving order (exact set may equal a singleton if 1 parent)
    seen, out = set(), []
    for h in hyps:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def p_mass_true(snapshot, true_parents):
    tp = set(true_parents)
    exact = sum(p for k, p in snapshot.items() if set(k) == tp)
    incl = sum(p for k, p in snapshot.items() if set(k) & tp)
    return exact, incl


def run_seeded(args):
    graph = set_graph(args.graph_type, nonlinear=args.nonlinear)
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=args.noiseless,
        seed=args.seeds_replicate,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )
    true_parents = tuple(graph.parents[graph.target])
    if len(true_parents) == 0:
        raise ValueError(f"target {graph.target} has no parents; experiment undefined")

    model = PARENT_SCALE(
        graph=graph, nonlinear=args.nonlinear, individual=True, use_doubly_robust=True
    )
    model.set_values(D_O, D_I, exploration_set)

    # ---- SEED INJECTION: real bootstrap + true-parent hypotheses at small mass ----
    original_init = model.determine_initial_probabilities
    injected = true_parent_hypotheses(true_parents)

    def seeded_initial_probabilities():
        probs = dict(original_init())  # the real doubly-robust bootstrap output
        logging.info(f"SEED: raw bootstrap candidates = {probs}")
        for hyp in injected:
            probs[hyp] = probs.get(hyp, 0.0) + args.inject_prob
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        logging.info(
            f"SEED: injected {injected} at {args.inject_prob} each; "
            f"post-mix candidates = {probs}"
        )
        return probs

    model.determine_initial_probabilities = seeded_initial_probabilities
    # -------------------------------------------------------------------------------

    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=args.n_trials, show_graphics=False)

    # per-iteration trajectory of true-parent mass
    exact_traj, incl_traj = [], []
    for snap in model.posterior_history:
        e, i = p_mass_true(snap, true_parents)
        exact_traj.append(e)
        incl_traj.append(i)

    intervened_true = sorted(
        set(v for s in intervention_set for v in s if v in set(true_parents))
    )
    n_int_on_true = sum(1 for s in intervention_set for v in s if v in set(true_parents))

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
        "P_True_Parents": exact_traj,
        "P_Incl_True_Parents": incl_traj,
        "True_Parents": true_parents,
        "Intervention_Ranges": {
            var: list(b) for var, b in graph.interventional_range_data.items()
        },
        "Epsilon_Fraction": model.boundary_eps_frac,
        "Seeded": True,
        "Inject_Prob": args.inject_prob,
        "Injected_Hypotheses": injected,
    }

    results_dir = f"results/boundary_tracking_seeded/{args.graph_type}"
    os.makedirs(results_dir, exist_ok=True)
    ns = "_nonlinear" if args.nonlinear else ""
    base = f"run{args.run_num}_cbo_seeded_dr2_boundary_{args.n_observational}_{args.n_int}{ns}"
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved seeded results to {results_dir}/{base}.pickle")

    # ---- human-readable trajectory + verdict ----
    print(f"\n{'='*72}")
    print(
        f"SEED-AND-TRACK  {args.graph_type}  true_parents={true_parents}  "
        f"inject_prob={args.inject_prob}"
    )
    print(f"{'='*72}")
    print(f"{'iter':>4} {'#cand':>6} {'P(true exact)':>14} {'P(incl true)':>13}")
    for i, snap in enumerate(model.posterior_history):
        print(f"{i:>4} {len(snap):>6} {exact_traj[i]:>14.4f} {incl_traj[i]:>13.4f}")
    print(
        f"\ninterventions landing on a true parent: {n_int_on_true} "
        f"(variables {intervened_true})"
    )
    start, end = incl_traj[0], incl_traj[-1]
    if end >= 0.5 and end > start:
        verdict = "RECOVERED -- true-parent mass climbed; dynamics can learn given inclusion"
    elif end <= 1e-4:
        verdict = "PRUNED/FAILED -- true-parent mass driven to ~0 despite inclusion"
    else:
        verdict = f"PARTIAL -- true-parent mass {start:.3f} -> {end:.3f}"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    run_seeded(parse_args())
