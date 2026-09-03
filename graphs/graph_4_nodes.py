import logging
from collections import OrderedDict
from itertools import chain
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from GPy.models.gp_regression import GPRegression
from scipy.special import expit

from graphs.graph import GraphStructure


class Graph4Nodes(GraphStructure):
    """
    This is my own synthetic toy graph
    """

    def __init__(
        self,
        X: np.ndarray = None,
        T: np.ndarray = None,
        Z: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.X = X
        self.A = T
        self.Z = Z
        self.Y = Y
        self._SEM = self.define_SEM()
        self._edges = [("X", "T"), ("X", "Z"), ("T", "Y"), ("Z", "Y")]
        self._target = "Y"
        self._functions: Optional[Dict[str, Callable]] = None
        self._nodes = set(chain(*self.edges))
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()
        self._variables = ["X", "Z", "T", "Y"]
        self._standardised = False
        self.use_intervention_range_data = False

    def define_SEM(self):
        fx = lambda epsilon, sample: epsilon
        fz = lambda epsilon, sample: 1.5 * np.tanh(sample["X"]) + epsilon
        ft = (
            lambda epsilon, sample: np.cos(sample["X"]) + np.exp(-sample["X"]) + epsilon
        )
        fy = lambda epsilon, sample: sample["Z"] ** 2 + np.sin(sample["T"]) + epsilon
        graph = OrderedDict([("X", fx), ("Z", fz), ("T", ft), ("Y", fy)])
        return graph

    def fit_all_models(self):
        return super().fit_all_models()

    def refit_models(self, observational_samples):
        return super().refit_models(observational_samples)

    def get_exploration_set(self) -> List[Tuple[str]]:
        return [("T",)]

    def get_all_do(self):
        do_dict = {}
        # do_dict["compute_do_X"] = self.compute_do_X
        # do_dict["compute_do_Z"] = self.compute_do_Z
        do_dict["compute_do_T"] = self.compute_do_T
        # do_dict["compute_do_XZ"] = self.compute_do_XZ
        # do_dict["compute_do_TX"] = self.compute_do_TX
        # do_dict["compute_do_TZ"] = self.compute_do_TZ
        # do_dict["compute_do_TXZ"] = self.compute_do_TXZ
        return do_dict

    def get_interventional_range(self):
        # min_intervention_x = -3
        # max_intervention_x = 2

        # min_intervention_z = -3
        # max_intervention_z = 3

        min_intervention_t = -2
        max_intervention_t = 8

        dict_ranges = OrderedDict(
            [
                ("T", [min_intervention_t, max_intervention_t]),
                # ("X", [min_intervention_x, max_intervention_x]),
                # ("Z", [min_intervention_z, max_intervention_z]),
            ]
        )
        return dict_ranges

    def get_sets(self):
        mis = [["T"]]
        pomis = [["T"]]
        manipulative_variables = ["T"]
        return mis, pomis, manipulative_variables

    def get_fixed_equal_costs(self) -> OrderedDict:
        logging.info("Using the fixed equal cost structure")
        cost_fix_X_equal = lambda intervention_value: 1.0
        cost_fix_Z_equal = lambda intervention_value: 1.0
        cost_fix_T_equal = lambda intervention_value: 1.0
        costs = OrderedDict(
            [
                ("X", cost_fix_X_equal),
                ("Z", cost_fix_Z_equal),
                ("T", cost_fix_T_equal),
            ]
        )
        return costs

    def get_fixed_different_costs(self) -> OrderedDict:
        logging.info("Using the fixed different cost structure")
        cost_fix_X_different = lambda intervention_value: 1.0
        cost_fix_Z_different = lambda intervention_value: 3.0
        cost_fix_T_different = lambda intervention_value: 5.0
        costs = OrderedDict(
            [
                ("X", cost_fix_X_different),
                ("Z", cost_fix_Z_different),
                ("T", cost_fix_T_different),
            ]
        )
        return costs

    def get_variable_equal_costs(self) -> OrderedDict:
        logging.info("Using the variable equal cost structure")
        cost_variable_X_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_T_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [
                ("X", cost_variable_X_equal),
                ("Z", cost_variable_Z_equal),
                ("T", cost_variable_T_equal),
            ]
        )
        return costs

    def get_variable_different_costs(self) -> OrderedDict:
        logging.info("Using the variable different cost structure")
        cost_variable_X_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 3.0
        )
        cost_variable_T_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 5.0
        )
        costs = OrderedDict(
            [
                ("X", cost_variable_X_different),
                ("Z", cost_variable_Z_different),
                ("T", cost_variable_T_different),
            ]
        )
        return costs

    def compute_do_X(self, observational_samples, value):
        interventions_nodes = ["X"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_Z(self, observational_samples, value):

        interventions_nodes = ["Z"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_T(self, observational_samples, value):

        interventions_nodes = ["T"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_XZ(self, observational_samples, value):

        interventions_nodes = ["X", "Z"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_TX(self, observational_samples, value):

        interventions_nodes = ["T", "X"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_TZ(self, observational_samples, value):

        interventions_nodes = ["T", "Z"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_TXZ(self, observational_samples, value):

        interventions_nodes = ["T", "X", "Z"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def get_error_distribution(self, noiseless: bool = True) -> Dict:
        error_distr = {}
        error_distr["X"] = np.random.uniform(-2, 2)
        error_distr["T"] = 0
        error_distr["Z"] = 0
        error_distr["Y"] = 0 if noiseless else np.random.normal(scale=0.3)
        return error_distr
