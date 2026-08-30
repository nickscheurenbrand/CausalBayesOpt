import logging
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork

from graphs.graph import GraphStructure


class SyntheticGraph(GraphStructure):
    """
    This is a graph data structure that do have unobserved confounders
    """

    def __init__(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        C: np.ndarray = None,
        D: np.ndarray = None,
        E: np.ndarray = None,
        F: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.Y = Y
        self._SEM = self.define_SEM()
        self._edges = [
            ("F", "A"),
            ("U1", "A"),
            ("U2", "B"),
            ("B", "C"),
            ("C", "D"),
            ("A", "E"),
            ("C", "E"),
            ("D", "Y"),
            ("E", "Y"),
            ("U1", "Y"),
            ("U2", "Y"),
        ]
        self._G = self.make_graphical_model()
        self._target = "Y"
        self.functions: Dict[str, GPRegression] = None
        self._standardised = False

    def define_SEM(self):
        logging.info("Setting up the structural equation model for the Synthetic Graph")
        fu1 = lambda epsilon, sample: epsilon
        fu2 = lambda epsilon, sample: epsilon
        ff = lambda epsilon, sample: epsilon
        fa = lambda epsilon, sample: sample["F"] ** 2 + sample["U1"] + epsilon
        fb = lambda epsilon, sample: sample["U2"] + epsilon
        fc = lambda epsilon, sample: np.exp(-sample["B"]) + epsilon
        fd = lambda epsilon, sample: np.exp(-sample["C"]) / 10 + epsilon
        fe = lambda epsilon, sample: np.cos(sample["A"]) + sample["C"] / 10 + epsilon
        fy = (
            lambda epsilon, sample: np.cos(sample["D"])
            + np.sin(sample["E"])
            + sample["U1"]
            + sample["U2"] * epsilon
        )

        graph = OrderedDict(
            [
                ("U1", fu1),
                ("U2", fu2),
                ("F", ff),
                ("A", fa),
                ("B", fb),
                ("C", fc),
                ("D", fd),
                ("E", fe),
                ("Y", fy),
            ]
        )
        return graph

    def SEM(self):
        return self._SEM

    def get_interventional_range(self):
        min_intervention_e = -6
        max_intervention_e = 3

        min_intervention_b = -5
        max_intervention_b = 4

        min_intervention_d = -5
        max_intervention_d = 5

        dict_ranges = OrderedDict(
            [
                ("E", [min_intervention_e, max_intervention_e]),
                ("B", [min_intervention_b, max_intervention_b]),
                ("D", [min_intervention_d, max_intervention_d]),
            ]
        )
        return dict_ranges

    def get_parameter_space(self, exploration_set) -> ParameterSpace:
        interventional_range = self.get_interventional_range()
        space = {}

        # Create ContinuousParameter objects dynamically for all variables in the interventional range
        for var in interventional_range:
            bounds = interventional_range[var]
            space[var] = ContinuousParameter(var, bounds[0], bounds[1])

        # Generate a list of parameters for only the variables in the exploration set
        es_space = [space[var] for var in exploration_set if var in space]

        return ParameterSpace(es_space)

    def fit_all_models(self) -> OrderedDict:
        logging.info("Fitting the Gaussian Processes based on the original data")
        assert self.X is not None

        self._fit_all_models(self.F, self.A, self.B, self.C, self.D, self.E, self.Y)

    def _fit_all_models(
        self,
        F: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        Y: np.ndarray,
    ):
        name_data_list = {
            "gp_C_B": (np.hstack((C, B)), Y, [1, 1, 1]),
            "gp_C": (B, C, [1, 1, 0.0001]),
            "gp_C_D": (np.hstack((C, D)), Y, [1, 1, 1]),
            "gp_A_C_E": (np.hstack((A, C, E)), Y, [1, 1, 10]),
            "gp_B_C_D": (np.hstack((B, C, D)), Y, [1, 1, 1]),
            "gp_A_B_C_E": (np.hstack((A, B, C, E)), Y, [1, 1, 10]),
        }

        functions = {}

        for name, (X, Y, parameter_list) in name_data_list.items():
            kernel = RBF(
                X.shape[1],
                ARD=False,
                lengthscale=parameter_list[0],
                variance=parameter_list[1],
            )
            gp = GPRegression(X=X, Y=Y, kernel=kernel, noise_var=parameter_list[2])
            gp.likelihood.variance.fix(1e-2)
            gp.optimize()
            functions[name] = gp

        self.functions = functions

    def refit_models(self, observational_samples):
        F = observational_samples["F"]
        A = observational_samples["A"]
        B = observational_samples["B"]
        C = observational_samples["C"]
        D = observational_samples["D"]
        E = observational_samples["E"]
        Y = observational_samples["Y"]
        self._fit_all_models(F, A, B, C, D, E, Y)

    def make_graphical_model(self):
        model = BayesianNetwork(self.edges)

        # Manually creating a networkx graph from the BayesianModel
        G = nx.MultiDiGraph()
        G.add_edges_from(model.edges())
        return G

    def show_graphical_model(self):
        pos = nx.spring_layout(self.G)  # positions for all nodes
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_size=700,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
        )
        plt.title("Bayesian Network")
        plt.show()

    def get_variables(self) -> List:
        variables = ["F", "A", "B", "C", "D", "E", "Y"]
        return variables

    def get_sets(self) -> Tuple[List[List[str]], List[List[str]], List[str]]:
        mis = [["B"], ["D"], ["E"], ["B", "D"], ["B", "E"], ["D", "E"]]
        pomis = [["B"], ["D"], ["E"], ["B", "D"], ["D", "E"]]
        manipulative_variables = ["B", "D", "E"]
        return mis, pomis, manipulative_variables

    def get_all_do(self):
        """
        The calculation of these do functions follow from the do_calculus of
        the graph -> look at the appendix of the CBO paper for the
        derivations
        """
        logging.info("Getting the do-functions for the ToyGraph")
        do_dict = {}
        do_dict["compute_do_E"] = self.compute_do_E
        do_dict["compute_do_D"] = self.compute_do_D
        do_dict["compute_do_B"] = self.compute_do_B
        do_dict["compute_do_BD"] = self.compute_do_BD
        do_dict["compute_do_BDE"] = self.compute_do_BDE
        do_dict["compute_do_DE"] = self.compute_do_DE
        do_dict["compute_do_BE"] = self.compute_do_BE
        return do_dict

    def compute_do_E(self, observational_samples, value) -> Tuple[float, float]:
        gp_A_C_E = self.functions["gp_A_C_E"]
        # getting the input for the specific do function
        length = observational_samples["A"].shape[0]
        repeated_value = np.repeat(value, length).reshape(length, 1)
        X = np.hstack(
            (observational_samples["A"], observational_samples["C"], repeated_value)
        )

        mean_do, var_do = gp_A_C_E.predict(X)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_D(self, observational_samples, value) -> Tuple[float, float]:
        gp_C_D = self.functions["gp_C_D"]
        length = observational_samples["C"].shape[0]
        repeated_value = np.repeat(value, length).reshape(-1, 1)
        X = np.hstack((observational_samples["C"], repeated_value))
        mean_do, var_do = gp_C_D.predict(X)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_B(self, observational_samples, value) -> Tuple[float, float]:

        gp_C_B = self.functions["gp_C_B"]
        gp_C = self.functions["gp_C"]

        # predicting the new values for C
        length = observational_samples["B"].shape[0]
        intervened_value = np.repeat(value, length).reshape(-1, 1)
        new_c, _ = gp_C.predict(intervened_value)
        new_c = np.mean(new_c)

        # repeating the C for insertion into the predicting GP
        repeated_C = np.repeat(new_c, length).reshape(-1, 1)
        X1 = np.hstack(
            (
                observational_samples["B"],
                repeated_C,
                observational_samples["D"],
            )
        )

        mean_do, var_do = gp_C_B.predict(X1)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_BD(self, observational_samples, value) -> Tuple[float, float]:
        gp_B_C_D = self.functions["gp_B_C_D"]
        gp_C = self.functions["gp_C"]

        length = observational_samples["B"].shape[0]
        intervened_B = np.repeat(value[0], length).reshape(-1, 1)

        c_new = np.mean(gp_C.predict(intervened_B)[0])

        # now make the c an intervention
        c_intervened = np.repeat(c_new, length).reshape(-1, 1)
        d_intervened = np.repeat(value[1], length).reshape(-1, 1)

        X = np.hstack((observational_samples["B"], c_intervened, d_intervened))
        mean_do, var_do = gp_B_C_D.predict(X)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_DE(self, observational_samples, value) -> Tuple[float, float]:
        gp_A_B_C_E = self.functions["gp_A_B_C_E"]
        gp_C = self.functions["gp_C"]

        length = observational_samples["B"].shape[0]
        intervened_B = np.repeat(value[0], length).reshape(-1, 1)
        c_new = np.mean(gp_C.predict(intervened_B)[0])

        c_intervened = np.repeat(c_new, length).reshape(-1, 1)
        e_intervened = np.repeat(value[1], length).reshape(-1, 1)

        X = np.hstack(
            (
                observational_samples["A"],
                observational_samples["B"],
                c_intervened,
                e_intervened,
            )
        )

        mean_do, var_do = gp_A_B_C_E.predict(X)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_BE(self, observational_samples, value) -> Tuple[float, float]:
        gp_A_B_C_E = self.functions["gp_A_B_C_E"]
        length = observational_samples["B"].shape[0]
        intervened_B = np.repeat(value[0], length).reshape(-1, 1)
        intervened_E = np.repeat(value[1], length).reshape(-1, 1)
        X = np.hstack(
            (
                observational_samples["A"],
                intervened_B,
                observational_samples["C"],
                intervened_E,
            )
        )

        mean_do, var_do = gp_A_B_C_E.predict(X)
        return np.mean(mean_do), np.mean(var_do)

    def compute_do_BDE(self, observational_samples, value) -> Tuple[float, float]:
        return self.compute_do_DE(observational_samples, [value[0], value[1]])

    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        return super().get_cost_structure(type_cost)

    def get_variable_equal_costs(self):
        logging.info("Using the variable equal cost structure")
        cost_variable_B_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_D_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_E_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [
                ("B", cost_variable_B_equal),
                ("D", cost_variable_D_equal),
                ("E", cost_variable_E_equal),
            ]
        )
        return costs

    def get_fixed_equal_costs(self):
        logging.info("Using the fixed equal cost structure")
        cost_fix_B_equal = lambda intervention_value: 1.0
        cost_fix_D_equal = lambda intervention_value: 1.0
        cost_fix_E_equal = lambda intervention_value: 1.0
        costs = OrderedDict(
            [("B", cost_fix_B_equal), ("D", cost_fix_D_equal), ("E", cost_fix_E_equal)]
        )
        return costs

    def get_fixed_different_costs(self):
        logging.info("Using the fixed different cost structure")
        cost_fix_B_different = lambda intervention_value: 1.0
        cost_fix_D_different = lambda intervention_value: 3.0
        cost_fix_E_different = lambda intervention_value: 5.0
        costs = OrderedDict(
            [
                ("B", cost_fix_B_different),
                ("D", cost_fix_D_different),
                ("E", cost_fix_E_different),
            ]
        )
        return costs

    def get_variable_different_costs(self):
        logging.info("Using the variable different cost structure")
        cost_variable_B_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_D_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 3.0
        )
        cost_variable_E_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 5.0
        )
        costs = OrderedDict(
            [
                ("B", cost_variable_B_different),
                ("D", cost_variable_D_different),
                ("E", cost_variable_E_different),
            ]
        )
        return costs
