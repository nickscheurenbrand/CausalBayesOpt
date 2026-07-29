"""
GwpsGraph: a GraphStructure over the GWPS (genome-wide Perturb-seq) gene network.

The source model is the linear autoregressive SEM Y = YG + Xbeta + gamma, with
G the direct-effect matrix (G[parent, child]) -- our extracted G_hat. CBO-U needs
a DAG, so we carve a bounded acyclic subgraph around a target gene and build a
linear-Gaussian SEM with the real G_hat weights (reusing
define_SEM_causalenv_linear). A DAG's weighted adjacency is nilpotent, so
r(G) = 0 and I - G is invertible by construction; (I-G)^-1 = sum over paths.

Mirrors Dream4Graph but reads the pre-extracted data/gwps_direct_edges.csv
(Exposure, Outcome, G_hat) instead of an XML topology, and needs no causal_env.
"""

import logging
from typing import Callable, Dict, Optional

import numpy as np

from graphs.graph import GraphStructure
from graphs.graph_chain import define_SEM_causalenv_linear
from graphs.gwps_build import DEFAULT_EDGE_CSV, build_gwps_dag


class GwpsGraph(GraphStructure):
    def __init__(
        self,
        edge_csv: str = DEFAULT_EDGE_CSV,
        target: Optional[str] = None,
        max_nodes: int = 60,
        top_k_parents: int = 8,
        noise_sigma: float = 1.0,
        weight_scale: float = 1.0,
        seed: int = 17,
    ):
        self.noise_sigma = noise_sigma
        self.nonlinear = False

        built = build_gwps_dag(
            edge_csv=edge_csv,
            target=target,
            max_nodes=max_nodes,
            top_k_parents=top_k_parents,
            weight_scale=weight_scale,
        )
        int_graph = built["int_graph"]
        W = built["W"]
        self.index_to_ensg = built["index_to_ensg"]
        self.ensg_to_index = {g: i for i, g in self.index_to_ensg.items()}
        target_index = built["target_index"]
        N = int_graph.number_of_nodes()
        self.num_nodes = N
        self._int_graph = int_graph
        self.weighted_adjacency_matrix = W

        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(N)]
        self._edges = [(str(u), str(v)) for u, v in int_graph.edges()]
        self._nodes = sorted(set(self._variables))
        self._parents, self._children = self.build_relationships()
        self._target = str(target_index)
        self._functions: Optional[Dict[str, Callable]] = None
        self._G = self.make_graphical_model()

        self.rng = np.random.default_rng(seed)
        self._standardised = False
        self.use_intervention_range_data = False
        self.population_mean_variance = {
            var: {"mean": 0, "std": 1} for var in self._variables
        }
        logging.info(
            f"GwpsGraph: {N} nodes, {int_graph.number_of_edges()} edges, "
            f"target={self._target} ({self.index_to_ensg[target_index]}), "
            f"target in-degree={len(self._parents[self._target])}"
        )

    # ---- GraphStructure overrides (mirror Dream4Graph) ----
    def define_SEM(self):
        return define_SEM_causalenv_linear(self._int_graph, self.weighted_adjacency_matrix)

    def set_target(self, target: str):
        self._target = target

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_noise(self, noise):
        self.noise_sigma = noise

    def get_error_distribution(self, noiseless=False):
        self._noise_std = [self.noise_sigma] * len(self.nodes)
        err_dist = {}
        for i in range(len(self.nodes)):
            err_dist[self.nodes[i]] = self.rng.normal(
                loc=0.0, scale=self._noise_std[i], size=1
            )
        return err_dist

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [v for v in self.variables if v != self.target]
        return mis, pomis, manipulative_variables
