import logging
from itertools import combinations
from typing import List, OrderedDict, Tuple

import numpy as np

from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.synthetics_graph import SyntheticGraph
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import (
    create_grid_interventions,
    draw_interventional_samples,
    sample_model,
)


def graph_setup(
    graph_type: str,
    seed: int = None,
    exploration_set: List[List[str]] = None,
    n_obs: int = 100,
    n_int: int = 2,
) -> Tuple[GraphStructure]:
    """
    The function does the setup for the toy_graph which inherits from GraphStructure
    It returns many important variables that will be necessary for both the CBO and
    the BO algorithm
    """
    if seed is not None:
        np.random.seed(seed)

    assert graph_type in ["Toy", "Synthetic", "Graph4", "Graph5", "Graph6"]
    if graph_type == "Toy":
        logging.info("Setting up the toy graph")
        graph = ToyGraph()
    elif graph_type == "Synthetic":
        logging.info("Setting up the synthetic graph")
        graph = SyntheticGraph()
    elif graph_type == "Graph4":
        logging.info("Setting up the 4 nodes graph")
        graph = Graph4Nodes()
    elif graph_type == "Graph5":
        logging.info("Setting up the 5 nodes graph")
        graph = Graph5Nodes()
    elif graph_type == "Graph6":
        logging.info("Setting up the 6 nodes graph")
        graph = Graph6Nodes()

    sem_model = graph.SEM
    variables = graph.variables
    target = graph.target
    _, _, manipulative_variables = graph.get_sets()
    exploration_set = graph.get_exploration_set()

    # first define observational samples
    samples = sample_model(sem_model, graph=graph, sample_count=n_obs)
    observational_samples = np.hstack(([samples[var] for var in variables]))

    # now define interventional samples
    interventional_ranges = graph.get_interventional_range()
    interventions = create_grid_interventions(interventional_ranges)
    interventional_samples = draw_interventional_samples(
        interventions, exploration_set, graph
    )

    return (
        graph,
        exploration_set,
        manipulative_variables,
        target,
        samples,
        observational_samples,
        interventional_samples,
    )


def get_all_intervention_sets(variables):
    # Remove 'Y' from the list of variables to intervene upon
    intervention_variables = [var for var in variables if var != "Y"]

    # Generate all possible combinations of intervention variables
    all_interventions = []
    for r in range(len(intervention_variables) + 1):
        all_interventions.extend(combinations(intervention_variables, r))

    return all_interventions
