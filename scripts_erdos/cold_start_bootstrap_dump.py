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
    parser.add_argument("--num_bootstraps", type=int, default=10)
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Use the joint (non-individual) selector instead of the per-variable one",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = set_graph(args.graph_type)
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=True,
        seed=args.seed,
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
    print(
        f"\ngraph_type={args.graph_type}  target={graph.target}  "
        f"true_parents={true_parents}  individual={not args.joint}"
    )
    print(f"num_bootstraps={args.num_bootstraps}  n_obs={args.n_observational}\n")

    print("--- raw per-bootstrap parent-set estimates ---")
    sizes = Counter()
    for i, parents_estimate in enumerate(robust_model.markov_dags):
        overlap = set(parents_estimate) & set(true_parents)
        flag = f"  <- overlaps true parents at {sorted(overlap)}" if overlap else ""
        print(f"  bootstrap {i}: {parents_estimate}{flag}")
        sizes[len(parents_estimate)] += 1

    print(
        f"\nparent-set size distribution across {args.num_bootstraps} "
        f"bootstraps: {dict(sizes)}"
    )
    print("\n--- aggregated proportions (what determine_initial_probabilities returns) ---")
    for parents, prob in sorted(robust_model.prob_estimate.items(), key=lambda kv: -kv[1]):
        overlap = set(parents) & set(true_parents)
        flag = f"  <- overlaps true parents at {sorted(overlap)}" if overlap else ""
        print(f"  {parents}: {prob:.4f}{flag}")


if __name__ == "__main__":
    main()
