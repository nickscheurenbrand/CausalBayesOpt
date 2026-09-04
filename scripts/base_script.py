import argparse
import logging
import os
import pickle
from copy import deepcopy
from typing import List, Tuple

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

from algorithms.BO_algorithm import BO
from algorithms.BOED_algorithm import BOED
from algorithms.CBO_algorithm import CBO
from algorithms.CEO_algorithm import CEO
from config import (
    RUN_BO,
    RUN_BOED_POLICY_OPT,
    RUN_BOED_POLICY_OPT_FIXED,
    RUN_BOED_RANDOM,
    RUN_CBO,
    RUN_CEO,
    SAVE_RUN,
)
from diffcbed.envs import Chain, ErdosRenyi, OnlyDAGDream4Environment, ScaleFree
from diffcbed.envs.graph_to_env import GraphStructureEnv
from diffcbed.models import DagBootstrap
from diffcbed.strategies import (
    ABCDStrategy,
    BALDStrategy,
    BatchBALDStrategy,
    FScoreBatchStrategy,
    GreedyABCDStrategy,
    GridOptPCE,
    PCEBatchStrategy,
    PolicyOptCovEig,
    PolicyOptNMC,
    PolicyOptNMCFixedValue,
    PolicyOptPCE,
    RandomAcquisitionStrategy,
    RandOptPCE_BO,
    RandOptPCE_GD,
    ReplayStrategy,
    SoftBALDStrategy,
    SoftFScoreStrategy,
    SoftPCE_BO,
    SoftPCE_GD,
    SSFinite,
)
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph

# the only ones that work are the ones used in the diffcbed code
STRATEGIES = {
    "greedyabcd": GreedyABCDStrategy,  # does not work
    "abcd": ABCDStrategy,  # does not work
    "softbald": SoftBALDStrategy,  # does not work
    "batchbald": BatchBALDStrategy,
    "bald": BALDStrategy,
    "random": RandomAcquisitionStrategy,  # works
    "replay": ReplayStrategy,  # does not work
    "f-score": FScoreBatchStrategy,  # does not work
    "softf-score": SoftFScoreStrategy,
    "pce": PCEBatchStrategy,
    "softpce_bo": SoftPCE_BO,  # does not work
    "randoptpce_bo": RandOptPCE_BO,
    "softpce_gd": SoftPCE_GD,
    "randoptpce_gd": RandOptPCE_GD,
    "policyoptpce": PolicyOptPCE,
    "ss_finite": SSFinite,
    "policyoptnmc": PolicyOptNMC,  # works
    "policyoptnmc_fixed_value": PolicyOptNMCFixedValue,  # works
    "gridpce": GridOptPCE,  # does not work
    "policyoptcoveig": PolicyOptCovEig,  # does not work
}

MODELS = {
    "dag_bootstrap": DagBootstrap,
}

ENVS = {
    "erdos": ErdosRenyi,
    "chain": Chain,
    "sf": ScaleFree,
    "semidream4": OnlyDAGDream4Environment,
    "graph": GraphStructureEnv,
}

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def set_graph(graph_type: str) -> GraphStructure:
    assert graph_type in ["Toy", "Graph4", "Graph5", "Graph6"]
    if graph_type == "Toy":
        graph = ToyGraph()
    elif graph_type == "Graph4":
        graph = Graph4Nodes()
    elif graph_type == "Graph5":
        graph = Graph5Nodes()
    elif graph_type == "Graph6":
        graph = Graph6Nodes()
    return graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable Multi-Target Causal Bayesian Experimental Design"
    )
    parser.add_argument("--seeds_replicate", type=int)
    parser.add_argument("--n_observational", type=int)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--n_anchor_points", type=int)
    parser.add_argument("--run_num", type=int)
    parser.add_argument(
        "--doubly_robust", action="store_true", help="Doubly robust methodology"
    )
    parser.add_argument("--noiseless", action="store_true")
    parser.add_argument(
        "--parent_method",
        type=str,
        default="dr1",
        help="Parent-learning method",
    )
    parser.add_argument("--graph_type", type=str, default="Erdos10")
    parser.add_argument(
        "--reseed_sem",
        action="store_true",
        help="Reseed SEM noise per replicate",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default=None,
        help="Results output subdirectory",
    )
    parser.add_argument("--acquisition", type=str, default="EI")

    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--id", type=str, default=None, help="Run ID")
    parser.add_argument("--data_seed", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
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
    parser.add_argument("--num_targets", type=int, default=1)
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
        help="Initial synthetic sample count",
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
        help="GP-UCB exploration steps",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Soft-BALD sampling temperature",
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Noise std"
    )
    parser.add_argument(
        "--scm_bias",
        type=float,
        default=0.0,
        help="Additive noise bias",
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0, help="Parameter mean"
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0, help="Parameter std"
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0, help="Gibbs factor temperature"
    )

    # TODO: improve names
    parser.add_argument(
        "--num_intervention_values",
        type=int,
        default=10,
        help="Number of intervention values",
    )
    parser.add_argument(
        "--intervention_values",
        type=float,
        nargs="+",
        help="Values for grid value_strategy",
    )
    parser.add_argument(
        "--intervention_value",
        type=float,
        default=0.0,
        help="Value for fixed value_strategy",
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
        help="DREAM4 experiment name",
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
        "--node_range", default="-10:10", help="Node value bounds"
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


def run_script(
    graph_type: str,
    run_num: int,
    all_graph_edges: List[List[Tuple]],
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_anchor_points: int,
    n_trials: int,
    filename: str,
    acquisition: str = "EI",
):

    # using this as the interventional and observational data
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=graph_type,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
    )

    if RUN_CEO:
        model: CEO = CEO(
            graph_type=graph_type,
            all_graph_edges=all_graph_edges,
            n_obs=n_obs,
            n_int=n_int,
            n_anchor_points=n_anchor_points,
            seed=seeds_int_data,
            noiseless=noiseless,
        )

        model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials, safe_optimization=True)

        ceo_result_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }

        filename_ceo = f"results/{filename}/run_ceo_{run_num}_results_{n_obs}_{n_int}_{noisy_string}.pickle"

        if SAVE_RUN:
            with open(filename_ceo, "wb") as file:
                pickle.dump(ceo_result_dict, file)

    if RUN_CBO:
        print(f"The acquisition is {acquisition}")
        # now for the CBO algorithm
        for i, edges in enumerate(all_graph_edges):
            graph = set_graph(graph_type)
            graph.mispecify_graph(edges)
            cbo_model = CBO(
                graph=graph,
                noiseless=noiseless,
                acquisition=acquisition,
                n_anchor_points=n_anchor_points,
            )
            cbo_model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)

            (
                best_y_array,
                current_y_array,
                cost_array,
                intervention_set,
                intervention_value,
                average_uncertainty,
            ) = cbo_model.run_algorithm(T=n_trials)

            cbo_results_dict = {
                "Best_Y": best_y_array,
                "Per_trial_Y": current_y_array,
                "Cost": cost_array,
                "Intervention_Set": intervention_set,
                "Intervention_Value": intervention_value,
                "Uncertainty": average_uncertainty,
            }

            filename_cbo = f"results/{filename}/run{run_num}_cbo_results_{n_obs}_{n_int}_graph_{i}{noisy_string}.pickle"

            if SAVE_RUN:
                with open(filename_cbo, "wb") as file:
                    pickle.dump(cbo_results_dict, file)

    # now for the BO implementation

    if RUN_BO:
        graph = set_graph(graph_type)
        graph.break_dependency_structure()
        bo_model = BO(graph=graph, noiseless=noiseless)
        bo_model.set_values(deepcopy(D_O))
        best_y_array, current_y_array, cost_array, average_uncertainty = (
            bo_model.run_algorithm(T=n_trials)
        )
        bo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Uncertainty": average_uncertainty,
        }

        filename_bo = f"results/{filename}/run{run_num}_bo_results_new_{n_obs}_{n_int}_{noisy_string}.pickle"
        if SAVE_RUN:
            with open(filename_bo, "wb") as file:
                pickle.dump(bo_results_dict, file)

    # strategy_name = "random"
    model_name = "dag_bootstrap"
    args = parse_args()
    graph = set_graph(graph_type)
    args.num_nodes = len(graph.variables)
    graph_env = GraphStructureEnv(graph, args)
    graph_variables = graph_env.nodes
    node_ranges = []
    interventional_ranges = graph.get_interventional_range()
    for node in graph.variables:
        if node in interventional_ranges.keys():
            node_ranges.append(tuple(interventional_ranges[node]))
        else:
            node_ranges.append((0, 0))
    args.node_range = node_ranges

    if RUN_BOED_RANDOM:
        strategy_name = "random"
        model = MODELS[model_name](graph_env, args)
        strategy = STRATEGIES[strategy_name](model, graph_env, args)
        boed = BOED(graph_env, model, strategy, args, graph_variables)
        boed.set_values(D_O, D_I)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
        ) = boed.run_algorithm()

        boed_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
        }

        filename_boed = f"results/{filename}/run{run_num}_boed_results_{n_obs}_{n_int}_{strategy_name}.pickle"
        if SAVE_RUN:
            with open(filename_boed, "wb") as file:
                pickle.dump(boed_results_dict, file)

    if RUN_BOED_POLICY_OPT:
        strategy_name = "policyoptnmc"
        model = MODELS[model_name](graph_env, args)
        strategy = STRATEGIES[strategy_name](model, graph_env, args)
        boed = BOED(graph_env, model, strategy, args, graph_variables)
        boed.set_values(D_O, D_I)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
        ) = boed.run_algorithm()

        boed_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
        }

        filename_boed = f"results/{filename}/run{run_num}_boed_results_{n_obs}_{n_int}_{strategy_name}.pickle"
        if SAVE_RUN:
            with open(filename_boed, "wb") as file:
                pickle.dump(boed_results_dict, file)

    if RUN_BOED_POLICY_OPT_FIXED:
        strategy_name = "policyoptnmc_fixed_value"
        model = MODELS[model_name](graph_env, args)
        strategy = STRATEGIES[strategy_name](model, graph_env, args)
        boed = BOED(graph_env, model, strategy, args, graph_variables)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
        ) = boed.run_algorithm()

        boed_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
        }

        filename_boed = f"results/{filename}/run{run_num}_boed_results_{n_obs}_{n_int}_{strategy_name}.pickle"
        if SAVE_RUN:
            with open(filename_boed, "wb") as file:
                pickle.dump(boed_results_dict, file)


def run_script_uncertainty(
    graph_type: str,
    run_num: int,
    all_graph_edges: List[List[Tuple]],
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_anchor_points: int,
    n_trials: int,
    filename: str,
):
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=graph_type,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
    )
    if RUN_CBO:
        for i, edges in enumerate(all_graph_edges):
            graph = set_graph(graph_type)
            graph.mispecify_graph(edges)
            graph.fit_samples_to_graph(D_O)
            uncertainties = graph.decompose_all_variance(D_O)
            cbo_uncertainty_dict = {
                "total": uncertainties["aleatoric"] + uncertainties["epistemic"],
                "epistemic": uncertainties["epistemic"],
                "aleatoric": uncertainties["aleatoric"],
            }
            filename_cbo = f"results/{filename}/run{run_num}_cbo_uncertainties_{n_obs}_{n_int}_graph_{i}{noisy_string}.pickle"
            with open(filename_cbo, "wb") as file:
                pickle.dump(cbo_uncertainty_dict, file)

    if RUN_BO:
        graph = set_graph(graph_type)
        graph.break_dependency_structure()
        graph.fit_samples_to_graph(D_O)
        uncertainties = graph.decompose_all_variance(D_O)

        bo_uncertainty_dict = {
            "total": uncertainties["aleatoric"] + uncertainties["epistemic"],
            "epistemic": uncertainties["epistemic"],
            "aleatoric": uncertainties["aleatoric"],
        }
        filename_bo = f"results/{filename}/run{run_num}_bo_uncertainties_{n_obs}_{n_int}_{noisy_string}.pickle"
        with open(filename_bo, "wb") as file:
            pickle.dump(bo_uncertainty_dict, file)
