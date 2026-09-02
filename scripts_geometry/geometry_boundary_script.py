"""Boundary tracking with the GEOMETRY_SCALE loop (appendix E).

Same protocol and output schema as the existing boundary scripts, so
results_erdos/boundary_bias_analysis.py and the notebooks work unchanged -- the
only difference is the surrogate: adaptive geometry + PFN prior mean + spherical
Bayesian linear regression, with EI untouched. The extra appendix-E diagnostics
(Delta_t per iteration, the pi trajectory, per-surrogate hyperparameters) are
saved alongside.

Supports the same three families as scripts_random/random_boundary_script.py.

  python scripts_geometry/geometry_boundary_script.py --graph_type Erdos50 \
      --prior tabpfn --acquisition EI --noiseless --n_int 2

--prior zero runs the identical loop with m_PFN = 0, which is the ablation for
E.4 (classical prior mean on the same adaptive geometry); --no_adapt_geometry
fixes pi = 1 and ablates E.3.

Output:
  results/boundary_tracking_{family}_geometry{_oracle}/{tag}/
      run{run}_cbo_unknown_dr2_boundary_{ACQ}_{n_obs}_{n_int}[_nonlinear].pickle
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

from algorithms.GEOMETRY_SCALE_algorithm import GEOMETRY_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph_dream import Dream4Graph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.graph_gwps import GwpsGraph
from utils.sem_sampling import draw_interventional_samples_sem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

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
        graph = ErdosRenyiGraph(num_nodes=n, nonlinear=args.nonlinear)
        graph.set_target(target)
        graph.set_seed(args.seeds_replicate)
        tag = f"{gt}_nonlinear" if args.nonlinear else gt
        return graph, "erdos", args.nonlinear, tag
    if gt in DREAM_YML:
        graph = Dream4Graph(yml_name=DREAM_YML[gt])
        if args.target is not None:
            target = args.target
            tag = f"{gt}_t{target}"
        else:
            target = sorted(graph.variables,
                            key=lambda v: (-len(graph.parents[v]), int(v)))[0]
            tag = gt
        graph.set_target(target)
        graph.set_seed(args.seeds_replicate)
        return graph, "dream", True, tag
    if gt == "gwps":
        graph = GwpsGraph(
            target=args.target, max_nodes=args.max_nodes,
            top_k_parents=args.top_k_parents, noise_sigma=args.noise_sigma,
            weight_scale=args.weight_scale, seed=args.seeds_replicate,
        )
        return graph, "gwps", False, f"gwps_n{args.max_nodes}_ws{args.weight_scale:g}"
    raise ValueError(f"unknown graph_type {gt!r}")


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
    p.add_argument("--seeds_replicate", type=int, default=71)
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=1)
    p.add_argument("--run_num", type=int, default=1)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--nonlinear", action="store_true",
                   help="nonlinear SEM (erdos only); saves to <graph>_nonlinear")
    p.add_argument("--acquisition", type=str, default="EI", choices=["EI", "UCB"])
    p.add_argument("--oracle", action="store_true",
                   help="force the true parent set (prob 1.0), intervened jointly")
    # appendix-E knobs
    p.add_argument("--prior", type=str, default="zero",
                   choices=["tabpfn", "pfn", "zero", "constant"],
                   help="'tabpfn' (the E.4 prior; 'pfn' is an alias) or "
                        "'zero'/'constant' (E.4 ablations)")
    p.add_argument("--prior_mean", type=str, default="pfn",
                   choices=["pfn", "pfn+do", "do", "zero"])
    p.add_argument("--no_adapt_geometry", action="store_true",
                   help="fix pi = 1, ablating the adaptive geometry of E.3")
    p.add_argument("--no_do_variance", action="store_true",
                   help="drop the causal do-variance from the predictive variance")
    p.add_argument("--pi_floor", type=float, default=1e-2)
    p.add_argument("--shift_bins", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
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
    logging.info(
        f"GEOMETRY_SCALE [{args.graph_type}, seed={args.seeds_replicate}, "
        f"oracle={args.oracle}, prior={args.prior}]: target {target}, "
        f"true parents {true_parents}"
    )

    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None, noiseless=args.noiseless, seed=args.seeds_replicate,
        n_obs=args.n_observational, n_int=args.n_int, graph=graph,
    )

    if args.oracle:
        if len(true_parents) == 0:
            raise ValueError(f"target {target} has no parents; oracle undefined")
        exploration_set = [true_parents]
        D_I = draw_interventional_samples_sem(
            exploration_set, graph, n_int=args.n_int,
            seed=args.seeds_replicate, noiseless=args.noiseless,
        )
        logging.info(f"ORACLE: joint exploration set {exploration_set}")

    model = GEOMETRY_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        individual=True,
        use_doubly_robust=True,
        acquisition=args.acquisition,
        prior=args.prior,
        prior_mean=args.prior_mean,
        adapt_geometry=not args.no_adapt_geometry,
        use_do_variance=not args.no_do_variance,
        pi_floor=args.pi_floor,
        shift_bins=args.shift_bins,
        device=args.device,
    )
    model.set_values(D_O, D_I, exploration_set)

    if args.oracle:
        def _oracle_initial_probabilities():
            return {true_parents: 1.0}

        def _joint_exploration_set():
            model.exploration_set = [true_parents]

        model.determine_initial_probabilities = _oracle_initial_probabilities
        model.redefine_exploration_set = _joint_exploration_set

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
        "Oracle": args.oracle,
        "Algorithm": "GEOMETRY_SCALE",
    }
    results_dict.update(model.geometry_results())
    if family == "gwps":
        results_dict["Target_ENSG"] = graph.index_to_ensg[int(target)]
        results_dict["Index_To_ENSG"] = graph.index_to_ensg
        results_dict["Weight_Scale"] = args.weight_scale

    oracle_suffix = "_oracle" if args.oracle else ""
    results_dir = f"results/boundary_tracking_{family}_geometry{oracle_suffix}/{tag}"
    os.makedirs(results_dir, exist_ok=True)
    ns = "_nonlinear" if nonlinear else ""
    base = (f"run{args.run_num}_cbo_unknown_dr2_boundary_{args.acquisition}_"
            f"{args.n_observational}_{args.n_int}{ns}")
    with open(f"{results_dir}/{base}.pickle", "wb") as f:
        pickle.dump(results_dict, f)
    logging.info(f"Saved GEOMETRY results to {results_dir}/{base}.pickle")

    boundary = np.array(model.boundary_percentages, dtype=float)
    best = np.asarray(best_y_array, dtype=float)
    shift = model.shift_history
    delta = (f"; Delta_t(JS) {shift[0]['js']:.4f} -> {shift[-1]['js']:.4f}"
             if shift else "")
    logging.info(
        f"GEOMETRY {args.graph_type} [{args.acquisition}]: mean boundary% = "
        f"{boundary.mean():.3f}; Best_Y {best[0]:.3f} -> {best[-1]:.3f} "
        f"(improvement {best[-1] - best[0]:+.3f}){delta}"
    )


if __name__ == "__main__":
    run(parse_args())
