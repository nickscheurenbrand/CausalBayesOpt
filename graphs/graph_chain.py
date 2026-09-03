import argparse
import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # NumPy is API-compatible for the ops used in nn_forward
import networkx as nx
import numpy as np
from GPy.models.gp_regression import GPRegression
from scipy.special import expit

from config import NOISE_TYPE_INDEX, NOISE_TYPES
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.envs.chain import Chain
from diffcbed.envs.samplers import D
try:
    from diffcbed.models.dibs.models.nonlinearGaussian import DenseNonlinearGaussianJAX
except Exception:
    from diffcbed.envs.numpy_nonlinear_sem import (
        DenseNonlinearGaussianNumpy as DenseNonlinearGaussianJAX,
    )

from graphs.graph import GraphStructure


def define_SEM_causalenv_linear(
    graph: nx.MultiDiGraph, weighted_adjacency_matrix: np.ndarray
) -> OrderedDict[str, Callable]:

    sem_functions = OrderedDict()
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        if not parents:
            sem_functions[str(node)] = lambda epsilon, sample, node=node: epsilon
        else:
            sem_functions[str(node)] = (
                lambda epsilon, sample, node=node, parents=parents: sum(
                    weighted_adjacency_matrix[parent, node] * sample[str(parent)]
                    for parent in parents
                )
                + epsilon
            )

    return sem_functions


def define_SEM_causalenv_nonlinear(
    causal_env: CausalEnvironment, conditionals: DenseNonlinearGaussianJAX
) -> OrderedDict[str, Callable]:

    graph = causal_env.graph
    topological_list = list(nx.topological_sort(graph))
    num_variables = len(topological_list)
    theta = causal_env.weights
    sem_functions = OrderedDict()

    def nn_forward(node, parents, sample, theta, epsilon):
        N = 1 if isinstance(epsilon, float) else len(epsilon)
        # NumPy-native (was jnp with .at[].set); parents laid into leading cols
        parent_values = np.zeros((N, num_variables))
        for i, parent in enumerate(parents):
            parent_values[:, i] = np.asarray(sample[str(parent)]).reshape(-1)
        return conditionals.eltwise_nn_forward(theta, parent_values)[:, node] + epsilon

    for node in topological_list:
        parents = list(graph.predecessors(node))
        if not parents:
            sem_functions[str(node)] = lambda epsilon, sample, node=node: epsilon
        else:
            sem_functions[str(node)] = (
                lambda epsilon, sample, node=node, parents=parents: nn_forward(
                    node, parents, sample, theta, epsilon
                )
            )

    return sem_functions


class ChainGraph(GraphStructure):

    def __init__(
        self,
        num_nodes: int,
        noise_sigma: float = 1.0,
        seed: int = None,
        nonlinear: bool = False,
    ):
        args = argparse.Namespace(scm_bias=1.0, noise_bias=1.0, old_er_logic=True)
        self.num_nodes = num_nodes

        self.causal_env: CausalEnvironment = Chain(
            args=args, num_nodes=num_nodes, nonlinear=nonlinear
        )
        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(num_nodes)]

        self._edges = [
            (str(edge[0]), str(edge[1])) for edge in self.causal_env.graph.edges
        ]
        self.nonlinear = nonlinear
        self._nodes = sorted(set(chain(*self.edges)))
        self._parents, self._children = self.build_relationships()
        self._target = f"{num_nodes - 1}"
        self._functions: Optional[Dict[str, Callable]] = None
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        self._G = self.make_graphical_model()
        self._standardised = False
        self.use_intervention_range_data = False

    def define_SEM(self):
        if self.nonlinear:
            pass
        else:
            sem_functions = define_SEM_causalenv_linear(
                self.causal_env.graph, self.causal_env.weighted_adjacency_matrix
            )
        return sem_functions

    def get_error_distribution(self, noiseless=False):
        err_dist = {}

        noise_type = NOISE_TYPES[NOISE_TYPE_INDEX]
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

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables
