import argparse
import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import networkx as nx
import numpy as np
from GPy.models.gp_regression import GPRegression

from config import NOISE_TYPE_INDEX, NOISE_TYPES
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.envs.erdos_renyi import ErdosRenyi
from diffcbed.envs.samplers import D
from graphs.graph import GraphStructure
from graphs.graph_chain import (
    define_SEM_causalenv_linear,
    define_SEM_causalenv_nonlinear,
)


class ErdosRenyiGraph(GraphStructure):

    def __init__(
        self,
        num_nodes: int,
        seed: int = 17,
        nonlinear: bool = False,
        noise_sigma=1.0,
        exp_edges: int = 1,
    ):
        self.noise_sigma = 0.1 if nonlinear else noise_sigma
        args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)
        self.nonlinear = nonlinear
        self.num_nodes = num_nodes

        self.causal_env: CausalEnvironment = ErdosRenyi(
            args=args,
            num_nodes=num_nodes,
            binary_nodes=True,
            nonlinear=nonlinear,
            seed=seed,
            exp_edges=exp_edges,
        )
        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(num_nodes)]

        self._edges = [
            (str(edge[0]), str(edge[1])) for edge in self.causal_env.graph.edges
        ]

        self._nodes = sorted(set(self.variables))
        self._parents, self._children = self.build_relationships()
        self._target = f"{num_nodes - 1}"
        self._functions: Optional[Dict[str, Callable]] = None
        self._G = self.make_graphical_model()

        self.rng = np.random.default_rng(seed)
        self._standardised = False
        self.use_intervention_range_data = False
        self.population_mean_variance = {
            var: {"mean": 0, "std": 1} for var in self.variables
        }

    def set_target(self, target: str):
        # choose the variable that is the best one to optimize for the ErdosRenyi graph
        self._target = target

    def misspecify_graph_random(self, seed=14):
        # Find a way to misspecify these graphs
        target = int(self.target)  # Get the target index
        num_nodes = self.num_nodes
        adj_matrix = self.causal_env.adjacency_matrix  # Get the adjacency matrix

        print(adj_matrix)
        # Shuffle rows except the target row
        indices = list(range(num_nodes))  # Create a list of all indices (rows)
        # indices.remove(
        #     target
        # )  # Remove the target index from the list of rows to shuffle

        # Shuffle the remaining indices
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(indices)

        # Create a new adjacency matrix where rows are shuffled but the target row remains in place
        shuffled_matrix = adj_matrix.copy()

        for i, new_row_index in enumerate(shuffled_indices):
            shuffled_matrix[i] = adj_matrix[new_row_index]  # Replace with shuffled rows

        # Keep the target row unchanged
        shuffled_matrix[target] = adj_matrix[target]
        np.fill_diagonal(shuffled_matrix, 0)
        print(shuffled_matrix)

        num_nodes = self.num_nodes
        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(num_nodes)]

        # Redefine the edges based on the new adjacency matrix
        self._edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if shuffled_matrix[i][j] == 1:
                    # Add edge from node i to node j based on the new shuffled matrix
                    self._edges.append((str(i), str(j)))

        # Create a directed graph using the edges
        G = nx.DiGraph(self._edges)

        # Check if the graph contains a cycle
        try:
            # Raises NetworkXUnfeasible if the graph is not a DAG
            cycle = nx.find_cycle(G, orientation="original")
            print("Cycle detected:", cycle)

            # If there is a cycle, you could remove edges to resolve it
            for edge in cycle:
                G.remove_edge(edge[0], edge[1])  # Remove one edge in the cycle
                print(f"Removed edge {edge} to prevent cycle.")
                break  # Removing one edge should break the cycle
        except nx.NetworkXNoCycle:
            print("No cycle detected.")

        # Update self._edges with the new edges from the acyclic graph
        self._edges = [(str(u), str(v)) for u, v in G.edges()]
        self._nodes = sorted(set(self.variables))
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def define_SEM(self):
        if self.nonlinear:
            sem_functions = define_SEM_causalenv_nonlinear(
                self.causal_env, self.causal_env.conditionals
            )
        else:
            sem_functions = define_SEM_causalenv_linear(
                self.causal_env.graph, self.causal_env.weighted_adjacency_matrix
            )
        return sem_functions

    def get_error_distribution(self, noiseless=False):
        err_dist = {}

        noise_type = "isotropic-gaussian"
        if noise_type.endswith("gaussian"):
            # Identifiable
            if noise_type == "isotropic-gaussian":
                self._noise_std = [self.noise_sigma] * len(self.nodes)
            elif noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, len(self.nodes))
            for i in range(len(self.nodes)):
                err_dist[self.nodes[i]] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                ).sample(1)
        elif noise_type == "exponential":
            noise_std = [self.noise_sigma] * len(self.nodes)
            for i in range(len(self.nodes)):
                err_dist[self.nodes[i]] = D(
                    self.rng.exponential, scale=noise_std[i]
                ).sample(1)

        return err_dist

    def set_noise(self, noise):
        self.noise_sigma = noise

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables
