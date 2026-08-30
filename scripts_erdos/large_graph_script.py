import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import List, Tuple

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from algorithms.CBO_algorithm import CBO
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from algorithms.RANDOM_SCALE_algorithm import RANDOM_SCALE
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


def set_graph(graph_type: str, nonlinear: bool = False) -> GraphStructure:
    assert graph_type in [
        "Toy",
        "Graph4",
        "Graph5",
        "Graph6",
        "Graph10",
        "Erdos10",
        "Erdos15",
        "Erdos20",
        "Erdos50",
        "Erdos100",
        "Erdos100_2",
        "Erdos100_3",
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
        graph = ChainGraph(num_nodes=10, nonlinear=nonlinear)
    elif graph_type == "Erdos10":
        graph = ErdosRenyiGraph(num_nodes=10, nonlinear=nonlinear)
        graph.set_target("1")
    elif graph_type == "Erdos15":
        graph = ErdosRenyiGraph(num_nodes=15, nonlinear=nonlinear)
        graph.set_target("8")
    elif graph_type == "Erdos20":
        graph = ErdosRenyiGraph(num_nodes=20, nonlinear=nonlinear)
        graph.set_target("18")
    elif graph_type == "Erdos50":
        graph = ErdosRenyiGraph(num_nodes=50, nonlinear=nonlinear)
        graph.set_target("23")
    elif graph_type == "Erdos100":
        graph = ErdosRenyiGraph(num_nodes=100, nonlinear=nonlinear)
        graph.set_target("80")
    elif graph_type == "Erdos100_2":
        graph = ErdosRenyiGraph(num_nodes=100, nonlinear=nonlinear)
        graph.set_target("2")
    elif graph_type == "Erdos100_3":
        graph = ErdosRenyiGraph(num_nodes=100, nonlinear=nonlinear)
        graph.set_target("16")
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
    run_cbo_unknown_dr_1: bool = False,
    run_cbo_unknown_dr_2: bool = False,
    run_cbo_unknown_all: bool = False,
    run_cbo_parents: bool = False,
    run_misspecified: bool = False,
    run_random: bool = False,
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

    if run_cbo_unknown_dr_1:
        model = PARENT_SCALE(
            graph=graph,
            nonlinear=nonlinear,
            individual=False,
            use_doubly_robust=True,
        )
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
        filename_cbo_unknown = f"results/{filename}/run{run_num}_cbo_unknown_dr1_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        with open(filename_cbo_unknown, "wb") as file:
            pickle.dump(cbo_unknown_results_dict, file)

    if run_cbo_unknown_dr_2:
        model = PARENT_SCALE(
            graph=graph,
            nonlinear=nonlinear,
            individual=True,
            use_doubly_robust=True,
        )
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
        filename_cbo_unknown = f"results/{filename}/run{run_num}_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        with open(filename_cbo_unknown, "wb") as file:
            pickle.dump(cbo_unknown_results_dict, file)

    if run_cbo_unknown_all:
        model = PARENT_SCALE(
            graph=graph,
            nonlinear=nonlinear,
            use_doubly_robust=False,
        )
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
        filename_cbo_unknown = f"results/{filename}/run{run_num}_cbo_unknown_results_all_{n_obs}_{n_int}{nonlinear_string}.pickle"
        with open(filename_cbo_unknown, "wb") as file:
            pickle.dump(cbo_unknown_results_dict, file)

    if run_cbo_parents:
        parents = graph.parents[graph.target]
        edges = [(parent, graph.target) for parent in parents]
        graph.mispecify_graph(edges)
        graph.set_interventional_range_data(D_O)
        exploration_set = [(parent,) for parent in parents]
        model = CBO(graph=graph)
        model.set_values(D_O, D_I, exploration_set)
        # model.run_algorithm(T=n_trials)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
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

    if run_random:
        model = RANDOM_SCALE(graph)
        model.set_values(D_O, D_I, exploration_set)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
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

    if run_misspecified:
        graph.misspecify_graph_random(seed=14)
        parents = graph.parents[graph.target]
        edges = [(parent, graph.target) for parent in parents]
        graph.mispecify_graph(edges)
        graph.set_interventional_range_data(D_O)
        exploration_set = [(parent,) for parent in parents]
        model = CBO(graph=graph)
        model.set_values(D_O, D_I, exploration_set)
        # model.run_algorithm(T=n_trials)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_misspecified_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
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
nonlinear = args.nonlinear
noisy_string = "" if noiseless else "_noisy"

parent_method = args.parent_method
graph_type = args.graph_type

print(graph_type, parent_method, nonlinear)
print(f"THE parent_method IS {parent_method}")
# if parent_method == "dr1":
#     run_script_unknown(
#         graph_type=graph_type,
#         run_num=run_num,
#         noiseless=noiseless,
#         noisy_string=noisy_string,
#         seeds_int_data=seeds_int_data,
#         n_obs=n_obs,
#         n_int=n_int,
#         n_trials=n_trials,
#         nonlinear=nonlinear,
#         filename=graph_type,
#         run_cbo_unknown_dr_1=True,
#         run_random=True,
#         run_cbo_parents=True,
#     )
# elif parent_method == "dr2":
#     run_script_unknown(
#         graph_type=graph_type,
#         run_num=run_num,
#         noiseless=noiseless,
#         noisy_string=noisy_string,
#         seeds_int_data=seeds_int_data,
#         n_obs=n_obs,
#         n_int=n_int,
#         n_trials=n_trials,
#         nonlinear=nonlinear,
#         filename=graph_type,
#         run_cbo_unknown_dr_2=True,
#         run_cbo_parents=True,
#         run_random=True,
#     )
# elif parent_method == "all":
#     run_script_unknown(
#         graph_type=graph_type,
#         run_num=run_num,
#         noiseless=noiseless,
#         noisy_string=noisy_string,
#         seeds_int_data=seeds_int_data,
#         n_obs=n_obs,
#         n_int=n_int,
#         n_trials=n_trials,
#         nonlinear=nonlinear,
#         filename=graph_type,
#         run_cbo_unknown_all=True,
#     )
# elif parent_method == "random":
#     run_script_unknown(
#         graph_type=graph_type,
#         run_num=run_num,
#         noiseless=noiseless,
#         noisy_string=noisy_string,
#         seeds_int_data=seeds_int_data,
#         n_obs=n_obs,
#         n_int=n_int,
#         n_trials=n_trials,
#         nonlinear=nonlinear,
#         filename=graph_type,
#         run_cbo_parents=True,
#         run_random=False,
#     )

# elif parent_method == "misspecified":
#     run_script_unknown(
#         graph_type=graph_type,
#         run_num=run_num,
#         noiseless=noiseless,
#         noisy_string=noisy_string,
#         seeds_int_data=seeds_int_data,
#         n_obs=n_obs,
#         n_int=n_int,
#         n_trials=n_trials,
#         nonlinear=nonlinear,
#         filename=graph_type,
#         run_misspecified=True,
#     )

if parent_method == "dr2":
    run_script_unknown(
        graph_type=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=True,
        run_cbo_unknown_dr_2=True,
        run_misspecified=True,
        run_random=True,
    )

elif parent_method == "three":
    run_script_unknown(
        graph_type=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=True,
        run_cbo_unknown_dr_2=True,
        run_random=True,
    )

elif parent_method == "misspecified":
    run_script_unknown(
        graph_type=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_misspecified=True,
    )
elif parent_method == "random":
    run_script_unknown(
        graph_type=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=True,
        run_random=True,
    )

elif parent_method == "random_2":
    run_script_unknown(
        graph_type=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=False,
        run_random=True,
    )
