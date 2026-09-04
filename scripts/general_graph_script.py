import argparse
import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import List, Tuple

os.chdir("..")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from algorithms.CBO_algorithm import CBO
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_chain import ChainGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.toy_graph import ToyGraph
from scripts.base_script import parse_args

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)

# set some global variables here
RUN_CBO_UNKNOWN = True
RUN_CBO_PARENTS = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable Multi-Target Causal Bayesian Experimental Design"
    )
    parser.add_argument("--graph_type", type=str, default="Erdos10")
    parser.add_argument("--seeds_replicate", type=int)
    parser.add_argument("--n_observational", type=int)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--n_anchor_points", type=int)
    parser.add_argument("--run_num", type=int)
    parser.add_argument("--noiseless", action="store_true")
    parser.add_argument("--nonlinear", action="store_true")

    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_nodes", type=int, default=5, help="Number of nodes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dag_bootstrap",
        help="Posterior model {dag_bootstrap}",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Acquisition strategy {abcd, random}",
    )
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--sparsity_factor",
        type=float,
        default=0.0,
        help="Sparsity regularizer weight",
    )
    parser.add_argument(
        "--exp_edges",
        type=float,
        default=0.1,
        help="Expected edge probability",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of synthetic samples",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=30,
        help="Number of posterior samples",
    )
    parser.add_argument(
        "--num_starting_samples",
        type=int,
        default=100,
        help="Initial number of samples",
    )
    parser.add_argument(
        "--temperature_type",
        type=str,
        default="anneal",
        help="Anneal relaxed-distribution temperature",
    )
    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=1,
        help="Exploration steps in GP-UCB",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Noise type",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1
    )
    parser.add_argument(
        "--scm_bias",
        type=float,
        default=0.0,
        help="Additive noise bias",
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0
    )

    # TODO: improve names
    parser.add_argument(
        "--num_intervention_values",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--intervention_values",
        type=float,
        nargs="+",
        help="Values for `grid` strategy",
    )
    parser.add_argument(
        "--intervention_value",
        type=float,
        default=0.0,
        help="Value for `fixed` strategy",
    )

    parser.add_argument(
        "--noise_bias",
        type=float,
        default=0.0,
        # help="Interventional value to set in `fixed` value_strategy, else ingored.",
    )

    parser.add_argument("--group_interventions", action="store_true")
    parser.add_argument("--plot_graphs", action="store_true")
    parser.add_argument("--save_models", action="store_true", default=False)
    parser.set_defaults(group_interventions=False)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--old_er_logic", action="store_true")
    parser.set_defaults(nonlinear=False)
    parser.add_argument("--weighted_posterior", action="store_true")
    parser.set_defaults(weighted_posterior=False)
    parser.add_argument("--reuse_posterior_samples", action="store_true")
    parser.set_defaults(reuse_posterior_samples=False)
    parser.add_argument("--include_gt_mec", action="store_true")
    parser.set_defaults(include_gt_mec=False)
    parser.add_argument(
        "--value_strategy",
        type=str,
        default="fixed",
        help="Strategy {gp-ucb, grid, fixed, sample-dist}",
    )
    parser.add_argument(
        "--dream4_path",
        type=str,
        default="envs/dream4/configurations/",
    )
    parser.add_argument(
        "--dream4_name",
        type=str,
        default="insilico_size10_1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_graphs", action="store_true")

    # optimizers
    parser.add_argument(
        "--opt_lr",
        type=float,
        default=0.5,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--opt_epochs",
        type=int,
        default=100,
        help="Optimizer epochs",
    )

    parser.add_argument(
        "--node_range", default="-10:10", help="Node value range"
    )

    parser.set_defaults(nonlinear=False)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    args.node_range = [float(item) for item in args.node_range.split(":")]

    if args.env == "sf":
        args.dibs_graph_prior = "sf"

    # if args.nonlinear == False:
    #     args.group_interventions = True

    if args.include_gt_mec:
        print("Setting weighted_posterior to True because include_gt_mec is set")
        args.weighted_posterior = True

    if args.reuse_posterior_samples:
        print(
            "Setting weighted_posterior to True because reuse_posterior_samples is set"
        )
        args.weighted_posterior = True

    return args


def set_graph(graph_type: str) -> GraphStructure:
    assert graph_type in [
        "Toy",
        "Graph4",
        "Graph5",
        "Graph6",
        "Graph10",
        "Erdos10",
        "Erdos15",
        "Erdos20",
    ]
    if graph_type == "Toy":
        graph = ToyGraph()
    elif graph_type == "Graph4":
        graph = Graph4Nodes()
    elif graph_type == "Graph5":
        graph = Graph5Nodes()
    elif graph_type == "Graph6":
        graph = Graph6Nodes()
    elif graph_type == "Graph10":
        graph = ChainGraph(num_nodes=10)
    elif graph_type == "Erdos10":
        graph = ErdosRenyiGraph(num_nodes=10)
        graph.set_target("1")
    elif graph_type == "Erdos15":
        graph = ErdosRenyiGraph(num_nodes=15)
        graph.set_target("8")
    elif graph_type == "Erdos20":
        graph = ErdosRenyiGraph(num_nodes=20)
        graph.set_target("9")
    return graph


def run_script_unknown(
    graph_type: str,
    run_num: int,
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_trials: int,
    nonlinear: bool,
    filename: str,
):

    graph = set_graph(graph_type)
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        graph=graph,
    )

    if RUN_CBO_UNKNOWN:
        model = PARENT_SCALE(graph=graph, nonlinear=nonlinear)
        model.set_values(D_O, D_I, exploration_set)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials, show_graphics=False)

        cbo_unknown_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }
        filename_cbo_unknown = f"results/{filename}/run{run_num}_cbo_unknown_results_{n_obs}_{n_int}{noisy_string}.pickle"
        with open(filename_cbo_unknown, "wb") as file:
            pickle.dump(cbo_unknown_results_dict, file)

    if RUN_CBO_PARENTS:
        parents = graph.parents[graph.target]
        edges = [(parent, graph.target) for parent in parents]
        graph.mispecify_graph(edges)
        graph.set_interventional_range_data(D_O)
        exploration_set = [tuple(parent) for parent in parents]
        model = CBO(graph=graph)
        model.set_values(D_O, D_I, exploration_set)
        model.run_algorithm(T=n_trials)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_results_{n_obs}_{n_int}{noisy_string}.pickle"
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }
        with open(filename_cbo, "wb") as file:
            pickle.dump(cbo_results_dict, file)


args = parse_args()
n_int = 2
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
graph_type = args.graph_type
noisy_string = "" if noiseless else "_noisy"

print(f"THE GRAPH TYPE IS {graph_type}")
run_script_unknown(
    graph_type=graph_type,
    run_num=run_num,
    noiseless=noiseless,
    noisy_string=noisy_string,
    seeds_int_data=seeds_int_data,
    n_obs=n_obs,
    n_int=n_int,
    n_trials=n_trials,
    nonlinear=False,
    filename=graph_type,
)
