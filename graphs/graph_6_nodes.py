import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np

from graphs.graph import GraphStructure


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Graph6Nodes(GraphStructure):
    """
    This is the graph for the healthcare experiment
    """

    def __init__(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        C: np.ndarray = None,
        S: np.ndarray = None,
        As: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.S = S
        self.As = As
        self.Y = Y
        self._SEM = self.define_SEM()
        self._edges = [
            ("A", "B"),
            ("A", "C"),
            ("A", "S"),
            ("A", "a"),
            ("A", "Y"),
            ("B", "a"),
            ("B", "C"),
            ("B", "S"),
            ("B", "Y"),
            ("a", "C"),
            ("a", "Y"),
            ("S", "C"),
            ("S", "Y"),
            ("C", "Y"),
        ]
        self._nodes = set(chain(*self.edges))
        self._parents, self._children = self.build_relationships()
        self._variables = ["A", "B", "a", "S", "C", "Y"]
        self._G = self.make_graphical_model()
        self._target = "Y"
        self._functions: Optional[Dict[str, Callable]] = None
        self._standardised = False
        self.use_intervention_range_data = False

    def define_SEM(self):
        fa = lambda epsilon, sample: epsilon
        fb = lambda epsilon, sample: 27 - 0.01 * sample["A"] + epsilon
        fas = lambda epsilon, sample: sigmoid(
            -8 + 0.1 * sample["A"] + 0.03 * sample["B"]
        )
        fs = lambda epsilon, sample: sigmoid(
            -13 + 0.1 * sample["A"] + 0.2 * sample["B"]
        )
        fc = lambda epsilon, sample: sigmoid(
            2.2
            - 0.05 * sample["A"]
            + 0.01 * sample["B"]
            - 0.04 * sample["S"]
            + 0.02 * sample["a"]
        )
        # this is different for the text and the code
        fy = (
            lambda epsilon, sample: 6.8
            + 0.04 * sample["A"]
            + 0.15 * sample["B"]
            - 0.6 * sample["S"]
            + 0.55 * sample["a"]
            + sample["C"]
            + epsilon
        )
        graph = OrderedDict(
            [("A", fa), ("B", fb), ("a", fas), ("S", fs), ("C", fc), ("Y", fy)]
        )
        return graph

    def fit_all_models(self):
        samples = {
            "A": self.A,
            "B": self.B,
            "a": self.As,
            "S": self.S,
            "C": self.C,
            "Y": self.Y,
        }
        self.fit_samples_to_graph(samples)

    def get_all_do(self):
        do_dict = {}
        do_dict["compute_do_a"] = self.compute_do_a
        do_dict["compute_do_S"] = self.compute_do_S
        do_dict["compute_do_Sa"] = self.compute_do_a_S
        return do_dict

    def get_interventional_range(self):

        min_intervention_As = 0.0
        max_intervention_As = 1.0

        min_intervention_S = 0.0
        max_intervention_S = 1.0

        if self.standardised:
            min_intervention_As = (min_intervention_As - self.means["a"]) / self.stds[
                "a"
            ]
            max_intervention_As = (max_intervention_As - self.means["a"]) / self.stds[
                "a"
            ]
            min_intervention_S = (min_intervention_S - self.means["S"]) / self.stds["S"]
            max_intervention_S = (max_intervention_S - self.means["S"]) / self.stds["S"]

        dict_ranges = OrderedDict(
            [
                ("S", [min_intervention_S, max_intervention_S]),
                ("a", [min_intervention_As, max_intervention_As]),
            ]
        )
        return dict_ranges

    def get_sets(self):
        mis = [["a"], ["S"], ["S", "a"]]
        pomis = [["S", "a"]]
        manipulative_variables = ["S", "a"]
        return mis, pomis, manipulative_variables

    def get_variable_equal_costs(self):
        logging.info("Using the variable equal cost structure")
        cost_variable_As_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_S_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [
                ("a", cost_variable_As_equal),
                ("S", cost_variable_S_equal),
            ]
        )
        return costs

    def get_fixed_equal_costs(self):
        logging.info("Using the fixed equal cost structure")
        cost_fix_As_equal = lambda intervention_value: 1.0
        cost_fix_S_equal = lambda intervention_value: 1.0
        costs = OrderedDict([("a", cost_fix_As_equal), ("S", cost_fix_S_equal)])
        return costs

    def get_fixed_different_costs(self):
        logging.info("Using the fixed different cost structure")
        cost_fix_As_different = lambda intervention_value: 1.0
        cost_fix_S_different = lambda intervention_value: 3.0
        costs = OrderedDict(
            [
                ("a", cost_fix_As_different),
                ("S", cost_fix_S_different),
            ]
        )
        return costs

    def get_variable_different_costs(self):
        logging.info("Using the variable different cost structure")
        cost_variable_As_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_S_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 3.0
        )
        costs = OrderedDict(
            [
                ("a", cost_variable_As_different),
                ("S", cost_variable_S_different),
            ]
        )
        return costs

    def compute_do_a(self, observational_samples, value):
        interventions_nodes = ["a"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_S(self, observational_samples, value):
        interventions_nodes = ["S"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_a_S(self, observational_samples, value):
        interventions_nodes = ["S", "a"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def get_exploration_set(self):
        return [("a",), ("S",), ("S", "a")]

    def get_error_distribution(self, noiseless: bool = False):
        err_dist = {}
        err_dist["A"] = np.random.uniform(55, 75)
        err_dist["B"] = np.random.normal(scale=np.sqrt(0.7))
        err_dist["a"] = 0
        err_dist["S"] = 0
        err_dist["C"] = 0
        err_dist["Y"] = np.random.normal(scale=np.sqrt(0.4))
        return err_dist
