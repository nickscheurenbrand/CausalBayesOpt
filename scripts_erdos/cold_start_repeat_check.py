"""Repeats cold-start candidate-parent-set construction N times on the same data to measure iteration-0 posterior variance
from the doubly-robust bootstrap's unseeded RNG (posterior_model/model.py:75). No GP/CBO; reports raw + final posteriors."""

import argparse
import os
import sys

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

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph_erdos_renyi import ErdosRenyiGraph


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
    parser.add_argument("--num_repeats", type=int, default=5)
    return parser.parse_args()


def run_one_cold_start(graph, D_O, D_I, exploration_set, true_parents):
    model = PARENT_SCALE(
        graph=graph,
        nonlinear=False,
        individual=True,
        use_doubly_robust=True,
    )
    model.set_values(D_O, D_I, exploration_set)

    captured = {}
    original_method = model.determine_initial_probabilities

    def wrapped():
        result = original_method()
        captured["raw"] = dict(result)
        return result

    model.determine_initial_probabilities = wrapped
    model.data_and_prior_setup()
    model.define_all_possible_graphs()

    raw = captured["raw"]
    final = dict(zip(model.graphs.keys(), model.posterior))

    true_parents_set = set(true_parents)
    raw_exact = raw.get(tuple(true_parents), 0.0)
    final_exact = final.get(tuple(true_parents), 0.0)
    raw_vars = set()
    for parents in raw:
        raw_vars.update(parents)
    final_vars = set()
    for parents in final:
        final_vars.update(parents)
    raw_missing = true_parents_set - raw_vars
    final_missing = true_parents_set - final_vars

    return {
        "raw_n_candidates": len(raw),
        "raw_exact_prob": raw_exact,
        "raw_missing_true_parents": sorted(raw_missing),
        "final_n_candidates": len(final),
        "final_exact_prob": final_exact,
        "final_missing_true_parents": sorted(final_missing),
        "final_top": sorted(final.items(), key=lambda kv: -kv[1])[:3],
    }


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
    true_parents = tuple(graph.parents[graph.target])
    print(
        f"\ngraph_type={args.graph_type}  target={graph.target}  "
        f"true_parents={true_parents}  num_repeats={args.num_repeats}\n"
    )

    exact_survives_count = 0
    any_missing_count = 0
    for r in range(args.num_repeats):
        result = run_one_cold_start(graph, D_O, D_I, exploration_set, true_parents)
        exact_survives = result["final_exact_prob"] > 0
        any_missing = len(result["final_missing_true_parents"]) > 0
        exact_survives_count += int(exact_survives)
        any_missing_count += int(any_missing)

        print(f"--- repeat {r} ---")
        print(
            f"  raw bootstrap:   {result['raw_n_candidates']} candidates, "
            f"exact_prob={result['raw_exact_prob']:.3f}, "
            f"missing={result['raw_missing_true_parents']}"
        )
        print(
            f"  final (iter 0):  {result['final_n_candidates']} candidates, "
            f"exact_prob={result['final_exact_prob']:.3f}, "
            f"missing={result['final_missing_true_parents']}"
        )
        print(f"  final top-3: {result['final_top']}")

    print(
        f"\n=== summary over {args.num_repeats} repeats ({args.graph_type}, "
        f"true_parents={true_parents}) ==="
    )
    print(
        f"  exact true parent set survived to iter-0 posterior: "
        f"{exact_survives_count}/{args.num_repeats}"
    )
    print(
        f"  at least one true parent NEVER a candidate at iter-0: "
        f"{any_missing_count}/{args.num_repeats}"
    )


if __name__ == "__main__":
    main()
