import logging
import sys
from collections import OrderedDict
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from graphs.graph import GraphStructure

sys.path.append("..")


class ToyGraph(GraphStructure):
    """
    This is the class for the TOY graph structure used in the CBO
    paper
    """

    def __init__(
        self,
        X: np.ndarray = None,
        Z: np.ndarray = None,
        Y: np.ndarray = None,
        noiseless: bool = False,
    ):
        logging.info("Initializing the Toy Graph Structures")
        self.X = X
        self.Z = Z
        self.Y = Y
        self.noiseless = noiseless
        self._SEM = self.define_SEM()
        self._edges = [("X", "Z"), ("Z", "Y")]
        self._nodes = set(chain(*self.edges))
        self._parents, self._children = self.build_relationships()
        self._variables = ["X", "Z", "Y"]
        self._G = self.make_graphical_model()
        self._target = "Y"
        self._functions: Optional[Dict[str, GPRegression]] = None
        self._standardised = False

    def define_SEM(self) -> OrderedDict:
        # Define named functions within the method
        def fx(epsilon, sample):
            return epsilon

        def fz(epsilon, sample):
            return 4 + np.exp(-sample["X"]) + epsilon

        def fy(epsilon, sample):
            return np.cos(sample["Z"]) - np.exp(-sample["Z"] / 20) + epsilon

        # Create the graph using the named functions
        graph = OrderedDict([("X", fx), ("Z", fz), ("Y", fy)])
        return graph

    def set_data(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray):
        """
        setting the data for the first time
        """
        self.X = X
        self.Z = Z
        self.Y = Y

    def fit_all_models(self) -> None:
        """
        Fit the model based on the original data
        """
        logging.info("Fitting the Gaussian Processes based on the original data")
        assert self.X is not None

        # regress Y on Z
        num_features = self.Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Y = GPRegression(X=self.Z, Y=self.Y, kernel=kernel, noise_var=1.0)
        gp_Y.optimize()

        # regress Z on Y
        num_features = self.X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Z = GPRegression(X=self.X, Y=self.Z, kernel=kernel)
        gp_Z.optimize()

        # there is no variable to regress X on
        self._functions = OrderedDict([("Y", gp_Y), ("Z", gp_Z), ("X", [])])

    def refit_models(self, observational_samples: dict) -> None:
        """
        Refit the gaussian processes based on the observed samples
        """
        logging.info("Fitting the Gaussian Processes based on the new data")
        X = np.asarray(observational_samples["X"])
        Z = np.asarray(observational_samples["Z"])
        Y = np.asarray(observational_samples["Y"])

        num_features = Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Y = GPRegression(X=Z, Y=Y, kernel=kernel, noise_var=1.0)
        gp_Y.optimize()

        # should it not still be x on z
        num_features = X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Z = GPRegression(X=X, Y=Z, kernel=kernel)
        gp_Z.optimize()

        self._functions = OrderedDict([("Y", gp_Y), ("Z", gp_Z), ("X", [])])

    def get_interventional_range(self) -> OrderedDict:
        min_intervention_x = -5
        max_intervention_x = 5

        min_intervention_z = -5.5
        max_intervention_z = 13

        if self.standardised:
            min_intervention_x = (min_intervention_x - self.means["X"]) / self.stds["X"]
            max_intervention_x = (max_intervention_x - self.means["X"]) / self.stds["X"]
            min_intervention_z = (min_intervention_z - self.means["Z"]) / self.stds["Z"]
            max_intervention_z = (max_intervention_z - self.means["Z"]) / self.stds["Z"]

        dict_ranges = OrderedDict(
            [
                ("X", [min_intervention_x, max_intervention_x]),
                ("Z", [min_intervention_z, max_intervention_z]),
            ]
        )
        return dict_ranges

    def get_original_interventional_range(self, D_O=None) -> OrderedDict:
        min_intervention_x = -5
        max_intervention_x = 5

        min_intervention_z = -5.5
        max_intervention_z = 13

        dict_ranges = OrderedDict(
            [
                ("X", [min_intervention_x, max_intervention_x]),
                ("Z", [min_intervention_z, max_intervention_z]),
            ]
        )
        return dict_ranges

    def get_exploration_set(self) -> List[Tuple[str]]:
        return [("X",), ("Z",), ("X", "Z")]

    def get_parameter_space(self, exploration_set: List) -> ParameterSpace:
        interventional_range = self.get_interventional_range()
        space = {
            "X": ContinuousParameter(
                "X", interventional_range["X"][0], interventional_range["X"][1]
            ),
            "Z": ContinuousParameter(
                "Z", interventional_range["Z"][0], interventional_range["Z"][1]
            ),
        }

        es_space = []
        for var in exploration_set:
            es_space.append(space[var])
        return ParameterSpace(es_space)

    def get_fixed_equal_costs(self) -> OrderedDict:
        logging.info("Using the fixed equal cost structure")
        cost_fix_X_equal = lambda intervention_value: 1.0
        cost_fix_Z_equal = lambda intervention_value: 1.0
        costs = OrderedDict([("X", cost_fix_X_equal), ("Z", cost_fix_Z_equal)])
        return costs

    def get_fixed_different_costs(self) -> OrderedDict:
        logging.info("Using the fixed different cost structure")
        cost_fix_X_different = lambda intervention_value: 1.0
        cost_fix_Z_different = lambda intervention_value: 10.0
        costs = OrderedDict([("X", cost_fix_X_different), ("Z", cost_fix_Z_different)])
        return costs

    def get_variable_equal_costs(self) -> OrderedDict:
        logging.info("Using the variable equal cost structure")
        cost_variable_X_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [("X", cost_variable_X_equal), ("Z", cost_variable_Z_equal)]
        )
        return costs

    def get_variable_different_costs(self) -> OrderedDict:
        logging.info("Using the variable different cost structure")
        cost_variable_X_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 10.0
        )
        costs = OrderedDict(
            [("X", cost_variable_X_different), ("Z", cost_variable_Z_different)]
        )
        return costs

    def get_set_BO(self) -> List:
        """
        This is just for the regular BayesOpt algorithm
        """
        logging.info("Getting the variables for the BO algorithm")
        manipulative_variables = ["X", "Z"]
        return manipulative_variables

    def get_sets(self) -> Tuple[List, List, List]:
        """
        This is for the CBO algorithm
        """
        logging.info("Getting the variables (mis and pomis) for the CBO algorithm")
        mis = [["X"], ["Z"]]
        pomis = [["Z"]]
        manipulative_variables = ["X", "Z"]
        return mis, pomis, manipulative_variables

    def get_all_do(self):
        do_dict = {}
        do_dict["compute_do_X"] = self.compute_do_X
        do_dict["compute_do_Z"] = self.compute_do_Z
        do_dict["compute_do_XZ"] = self.compute_do_XZ
        return do_dict

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

    def compute_do_XZ(self, observational_samples, value):

        interventions_nodes = ["X", "Z"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def get_error_distribution(self, noiseless: bool = None):
        if noiseless is not None:
            use_noise = noiseless
        else:
            use_noise = self.noiseless
        rng = np.random.default_rng()
        error_distr = {}
        # error_distr["X"] = np.random.uniform(-5, 5)
        error_distr["X"] = rng.normal()
        error_distr["Z"] = rng.normal()
        error_distr["Y"] = rng.normal()
        return error_distr
