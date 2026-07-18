"""
Standalone check for the boundary-tracking cold-start finding: does the
individual doubly-robust bootstrap parent-set selector (used to build the
iteration-0 candidate posterior in PARENT_SCALE.determine_initial_probabilities,
see algorithms/PARENT_SCALE_algorithm.py:145-187) ever propose a parent-set
with more than one variable, or does it structurally cap out at a single
variable per bootstrap draw?

Only needs the observational data (D_O) -- determine_initial_probabilities()
never looks at interventional data or runs any CBO trial, so this is cheap:
no GP surrogate fitting, no 30-trial acquisition loop. It replicates exactly
what that method does, but keeps a handle on the DoublyRobustModel instance
so we can inspect robust_model.markov_dags, the raw list of per-bootstrap
parent-set estimates (only the aggregated proportions normally survive into
PARENT_SCALE.posterior_history[0]).

Run with the same venv as boundary_tracking_script.py:
    python cold_start_bootstrap_dump.py --graph_type Erdos100 --num_bootstraps 10
    python cold_start_bootstrap_dump.py --graph_type Erdos100 --num_bootstraps 200
    python cold_start_bootstrap_dump.py --graph_type Erdos50 --num_bootstraps 10
"""

import argparse
import os
import sys
from collections import Counter

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.chdir("..")
if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
_algorithms_path = os.path.join(os.getcwd(), "algorithms")
if _algorithms_path not in sys.path:
    sys.path.append(_algorithms_path)

import numpy as np

from diffcbed.replay_buffer import ReplayBuffer
from graphs.data_setup import setup_observational_interventional
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from posterior_model.model import DoublyRobustModel
from utils.sem_sampling import change_obs_data_format_to_mi


def set_graph(graph_type: str):
    assert graph_type in ["Erdos20", "Erdos50", "Erdos100"]
    if graph_type == "Erdos20":
        graph = ErdosRenyiGraph(num_nodes=20)
        graph.set_target("18")
    elif graph_type == "Erdos50":
        graph = ErdosRenyiGraph(num_nodes=50)
        graph.set_target("23")
    elif graph_type == "Erdos100":
        graph = ErdosRenyiGraph(num_nodes=100)
        graph.set_target("80")
    return graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", type=str, default="Erdos100")
    parser.add_argument("--n_observational", type=int, default=200)
    parser.add_argument("--n_int", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated data seeds (overrides --seed); each reseeds the "
        "observational sample, so inclusion is tested across DIFFERENT datasets "
        "as well as across bootstrap draws",
    )
    parser.add_argument("--num_bootstraps", type=int, default=10)
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Use the joint (non-individual) selector instead of the per-variable one",
    )
    return parser.parse_args()


def run_one_seed(args, seed):
    """Runs the doubly-robust bootstrap once and returns (true_parents,
    markov_dags, prob_estimate) for the given data seed."""
    graph = set_graph(args.graph_type)
    D_O, _D_I, _es = setup_observational_interventional(
        graph_type=None,
        noiseless=True,
        seed=seed,
        n_obs=args.n_observational,
        n_int=args.n_int,
        graph=graph,
    )
    topological_order = list(D_O.keys())
    D_O_mi = change_obs_data_format_to_mi(
        D_O,
        graph_variables=graph.variables,
        intervention_node=np.zeros(shape=len(graph.variables)),
    )
    robust_model = DoublyRobustModel(
        graph=graph,
        topological_order=topological_order,
        target=graph.target,
        indivdual=not args.joint,
        num_bootstraps=args.num_bootstraps,
    )
    buffer = ReplayBuffer(binary=True)
    buffer.update(D_O_mi)
    robust_model.run_method(buffer.data())
    true_parents = tuple(graph.parents[graph.target])
    return true_parents, list(robust_model.markov_dags), dict(robust_model.prob_estimate)


def main():
    args = parse_args()
    seeds = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    )

    all_draws = []          # (seed, parents_estimate) across every bootstrap draw
    true_parents = None
    aggregated_has_parent = Counter()  # per true parent: in how many seeds it is a candidate

    for seed in seeds:
        tp, markov_dags, prob_estimate = run_one_seed(args, seed)
        true_parents = tp
        print(
            f"\n================ seed={seed}  graph={args.graph_type}  "
            f"target-parents={tp}  num_bootstraps={args.num_bootstraps} ================"
        )
        print("--- raw per-bootstrap parent-set estimates ---")
        sizes = Counter()
        for i, est in enumerate(markov_dags):
            all_draws.append((seed, tuple(est)))
            overlap = set(est) & set(tp)
            flag = f"  <- includes true parent(s) {sorted(overlap)}" if overlap else ""
            print(f"  bootstrap {i}: {est}{flag}")
            sizes[len(est)] += 1
        print(f"parent-set size distribution: {dict(sizes)}")

        print("--- aggregated candidate posterior (feeds determine_initial_probabilities) ---")
        agg_vars = set()
        for parents, prob in sorted(prob_estimate.items(), key=lambda kv: -kv[1]):
            agg_vars.update(parents)
            overlap = set(parents) & set(tp)
            flag = f"  <- includes true parent(s) {sorted(overlap)}" if overlap else ""
            print(f"  {parents}: {prob:.4f}{flag}")
        for v in tp:
            if v in agg_vars:
                aggregated_has_parent[v] += 1

    # ---- marginal inclusion summary: the "in more than one case" answer ----
    n_draws = len(all_draws)
    print(f"\n{'='*70}")
    print(f"MARGINAL INCLUSION SUMMARY  ({args.graph_type}, true parents {true_parents})")
    print(f"{len(seeds)} seed(s) x {args.num_bootstraps} bootstraps = {n_draws} draws")
    print(f"{'='*70}")
    for v in true_parents:
        cnt = sum(1 for _s, est in all_draws if v in est)
        print(
            f"  parent {v}: appears in {cnt}/{n_draws} bootstrap draws "
            f"({100*cnt/max(n_draws,1):.0f}%); "
            f"is a candidate in {aggregated_has_parent[v]}/{len(seeds)} seed posteriors"
        )
    tp_set = set(true_parents)
    n_all = sum(1 for _s, est in all_draws if tp_set.issubset(est))
    n_any = sum(1 for _s, est in all_draws if tp_set & set(est))
    n_exact = sum(1 for _s, est in all_draws if set(est) == tp_set)
    print(
        f"  draws containing ALL true parents: {n_all}/{n_draws}; "
        f"ANY: {n_any}/{n_draws}; EXACTLY the true set: {n_exact}/{n_draws}"
    )


if __name__ == "__main__":
    main()
