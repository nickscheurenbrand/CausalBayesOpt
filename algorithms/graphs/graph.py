import abc
import logging
from collections import OrderedDict, defaultdict, deque
from functools import partial
from typing import Callable, Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers import GPyModelWrapper
from GPy.core.parameterization import priors
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

MESSAGE = "Subclass should implement this."


class MyKDE(KernelDensity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None

    def fit_and_update(self, X):
        self.X = X
        return super().fit(X)

    def predict(self):
        return np.mean(super().sample(n_samples=500)), np.var(
            super().sample(n_samples=500)
        )


def safe_optimization(
    gpy_model: GPRegression,
    lower_bound_var: float = 1e-05,
    upper_bound_var: float = 2.0,
    bound_len: int = 20,
) -> GPyModelWrapper:
    if gpy_model.kern.variance[0] < lower_bound_var:
        logging.info("SAFE OPTIMIZATION: Resetting the kernel variance to lower bound")
        gpy_model.kern.variance[0] = lower_bound_var

    if gpy_model.kern.lengthscale[0] > bound_len:
        logging.info("SAFE OPTIMZATION: Resetting kernel lenghtscale")
        gpy_model.kern.lengthscale[0] = 1.0

    if gpy_model.likelihood.variance[0] > upper_bound_var:
        logging.info("SAFE OPTIMIZATION: restting likelihood var to upper bound")
        gpy_model.likelihood.variance[0] = upper_bound_var

    if gpy_model.likelihood.variance[0] < lower_bound_var:
        logging.info("SAFE OPTIMIZATION: resetting likelihood var to lower bound")
        gpy_model.likelihood.variance[0] = lower_bound_var
    return gpy_model


# this is for calculating the causal effect in a more clear way
# these functions are just outside of the class as it is not needed in the class
def set_intervention_values(
    variables: Dict, intervention: str, value: float, num_observations: int
):
    """
    Changing the variables that was intervened upon when computing
    the outcome of the new graph
    """
    # for variable, value in interventions.items():
    #     variables[variable] = value * np.ones((num_observations, 1))
    variables[intervention] = value * np.ones((num_observations, 1))


def update_children_values_og(
    functions: OrderedDict,
    variables: OrderedDict,
    children: OrderedDict,
    parents: OrderedDict,
    observational_samples,
):
    """
    Propogates the intervention effects down in the graph. There are many cases
    to consider, that is why the function is so messy
    """
    # trying to update the children correctly in the DAG
    for var, children_list in children.items():
        if var in variables:
            parent_values = {}
            for child in children_list:
                if child in functions and child not in variables:
                    # this makes sure child was not previously intervened upon
                    parents_of_child = parents[child]
                    for current_parent in parents_of_child:
                        # makes sure that you are considering other variables that
                        # are not the current variable we are considering
                        if current_parent in variables:
                            parent_values[current_parent] = variables[current_parent]
                        else:
                            parent_values[current_parent] = observational_samples[
                                current_parent
                            ]

                    variables[child] = predict_child(
                        functions[child], parent_values, parents_of_child
                    )


def update_children_values_new(
    functions: OrderedDict,
    variables: OrderedDict,
    children: OrderedDict,
    parents: OrderedDict,
    observational_samples: OrderedDict,
):
    keys = variables.keys()
    for var in keys:
        propogate_effects(
            var, functions, variables, children, parents, observational_samples
        )


def propogate_effects(
    node: str,
    functions: OrderedDict,
    variables: Dict,
    children: Dict,
    parents: Dict,
    observational_samples: Dict,
):
    if node in children:
        for child in children[node]:
            if child not in variables:  # Child has not been intervened upon
                # Get parent values for the child
                parent_values = {
                    p: variables[p] if p in variables else observational_samples[p]
                    for p in parents[child]
                }
                # Calculate new value for the child
                variables[child] = predict_child(
                    functions[child], parent_values, parents[child]
                )
                # Recursively update the child's children
                propogate_effects(
                    child,
                    functions,
                    variables,
                    children,
                    parents,
                    observational_samples,
                )


def predict_child(
    function: GPRegression, parent_values: Dict[str, np.ndarray], parents: List
):
    """
    Makes sure that the parent variables are in the correct order for the GP model
    """
    parent_values_cols = np.hstack([parent_values[val] for val in parents])
    return function.predict(parent_values_cols)[0]


def maintain_independent_and_parents(variables, nodes):
    """
    Makes sure the independent nodes do not change, adds a check if the variable
    has not been changed in one of the previous interventions
    """
    for _, independent_nodes in nodes.items():
        for var, value in independent_nodes.items():
            if var not in variables:
                variables[var] = value


def predict_causal_effect(
    functions: Dict[str, GPRegression],
    variables: Dict[str, np.ndarray],
    parents_Y: List[str],
    num_observations: int,
    target: str,
):
    """
    Predicts the causal effect after all the variables has been intervened upon
    Returns the mean and the variance of the observation
    """
    input_Y = np.hstack(
        [variables[parent].reshape(num_observations, -1) for parent in parents_Y]
    )
    gp_Y = functions[target]
    predictions = gp_Y.predict(input_Y)
    return np.mean(predictions[0]), np.mean(predictions[1])


class GraphStructure:
    """
    This is the generic graph structure that all the simulated examples will follow
    """

    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def SEM(self):
        """
        Specifies the type of graph (e.g., 'directed', 'undirected')
        """
        return self._SEM

    @property
    @abc.abstractmethod
    def iscm_paramters(self):
        return self.population_mean_variance

    @property
    @abc.abstractmethod
    def target(self) -> str:
        return self._target

    @property
    @abc.abstractmethod
    def standardised(self) -> bool:
        return self._standardised

    @property
    @abc.abstractmethod
    def edges(self) -> List[Tuple[str, str]]:
        return self._edges

    @property
    @abc.abstractmethod
    def functions(self):
        return self._functions

    @property
    @abc.abstractmethod
    def parents(self):
        return self._parents

    @property
    @abc.abstractmethod
    def children(self):
        return self._children

    @property
    @abc.abstractmethod
    def nodes(self) -> List[str]:
        return self._nodes

    @property
    @abc.abstractmethod
    def target(self):
        return self._target

    @property
    @abc.abstractmethod
    def variables(self):
        return self._variables

    @property
    def G(self):
        return self._G

    @abc.abstractmethod
    def define_SEM():
        """
        This method defines the structural equation model (SEM) for
        each of the simulated graph structures
        """
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def fit_all_models(self):
        """
        Fit the models based on the initial data
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_exploration_set(self):
        variables = self.variables
        return [(var,) for var in variables if var != self.target]

    @abc.abstractmethod
    def refit_models(self, observational_samples):
        """
        Refit the GP models based on the new observational samples
        """
        self.fit_samples_to_graph(observational_samples)

    @abc.abstractmethod
    def get_all_do(self):
        """
        This assigns assigns the calculation of the interventional distribution
        to each of the variables in the SCM
        """
        do_dict = {}
        exploration_set = self.get_exploration_set()
        for es in exploration_set:
            key = "_".join(es)
            do_function_name = f"compute_do_{key}"
            do_dict[do_function_name] = partial(
                self.compute_do_generic, intervention_nodes=es
            )

        return do_dict

    @abc.abstractmethod
    def set_interventional_range_data(self, D_O):
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [D_O[var].min(), D_O[var].max()]
        self.use_intervention_range_data = True
        self.interventional_range_data = interventional_range

    @abc.abstractmethod
    def get_interventional_range(self, D_O: Dict = None):
        """
        Sets the range of the variables we can intervene upon
        """
        logging.warning(MESSAGE)
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [-5, 5]
        return interventional_range

    @abc.abstractmethod
    def get_original_interventional_range(self):
        return self.get_interventional_range()

    @abc.abstractmethod
    def get_set_BO(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_sets(self) -> Tuple[List, List, List]:
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        if type_cost == 1:
            costs = self.get_fixed_equal_costs()
        elif type_cost == 2:
            costs = self.get_fixed_different_costs()
        elif type_cost == 3:
            costs = self.get_variable_equal_costs()
        elif type_cost == 4:
            costs = self.get_variable_different_costs()
        else:
            logging.warning("Undefined cost structure")

        assert isinstance(costs, OrderedDict)
        return costs

    @abc.abstractmethod
    def get_fixed_equal_costs(self) -> OrderedDict[str, Callable]:
        manipulative_variables = self.get_sets()[2]
        costs = OrderedDict()
        for var in manipulative_variables:
            func = lambda intervention_value: 1.0
            costs[var] = func

        return costs

    @abc.abstractmethod
    def get_fixed_different_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_variable_equal_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_variable_different_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_parameter_space(self, exploration_set) -> ParameterSpace:
        if self.use_intervention_range_data:
            interventional_range = self.interventional_range_data
        else:
            interventional_range = self.get_interventional_range()
        space = {}

        # Create ContinuousParameter objects dynamically for all variables in the interventional range
        for var in interventional_range:
            bounds = interventional_range[var]
            space[var] = ContinuousParameter(var, bounds[0], bounds[1])

        # Generate a list of parameters for only the variables in the exploration set
        es_space = [space[var] for var in exploration_set if var in space]

        return ParameterSpace(es_space)

    @abc.abstractmethod
    def make_graphical_model(self) -> nx.MultiDiGraph:
        model = BayesianNetwork(self.edges)

        # Create a new MultiDiGraph
        G = nx.MultiDiGraph()

        # Add all nodes from the BayesianNetwork
        G.add_nodes_from(self.variables)

        # Add all edges from the BayesianNetwork
        G.add_edges_from(model.edges())

        return G

    @abc.abstractmethod
    def show_graphical_model(self) -> None:
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

    @abc.abstractmethod
    def get_error_distribution(self) -> Dict[str, Callable]:
        err_dist = {var: np.random.normal() for var in self.variables}
        return err_dist

    @abc.abstractmethod
    def fit_samples_to_graph(
        self,
        samples: OrderedDict,
        parameters: OrderedDict = None,
        set_priors: bool = False,
    ) -> None:

        # first step is get the set of all the child nodes
        children_parents = self.parents

        self._functions = OrderedDict()
        for child, parents in children_parents.items():
            if parents:
                logging.debug(
                    f"Fitting child: {child} to parents: {parents} for {self.edges}"
                )
                Y = samples[child]
                X = np.hstack([samples[parent] for parent in parents])
                kernel = RBF(
                    input_dim=len(parents), variance=1.0, ARD=False, lengthscale=1.0
                )
                gp = GPRegression(X=X, Y=Y, kernel=kernel)

                gp.optimize()
                gp = safe_optimization(gp)
                self._functions[child] = gp
            else:
                logging.debug(f"Fitting marginal distribution for child {child}")
                Y = samples[child]
                self._functions[child] = MyKDE(kernel="gaussian").fit_and_update(Y)

            # this can also be a flag - not sure why it is added
            # if set_priors:
            #     prior_len = priors.InverseGamma.from_EV(1.0, 1.0)
            #     prior_sigma_f = priors.InverseGamma.from_EV(4.0, 0.5)
            #     prior_lik = priors.InverseGamma.from_EV(3, 1)

            #     gp.kern.variance.set_prior(prior_sigma_f)
            #     gp.kern.lengthscale.set_prior(prior_len)
            #     gp.likelihood.variance.set_prior(prior_lik)

    @abc.abstractmethod
    def build_relationships(self) -> Tuple[dict, dict]:
        parents = {}
        children = {}
        for node in self.nodes:
            parents[node] = []
            children[node] = []

        for parent, child in self.edges:
            children[parent].append(child)
            parents[child].append(parent)

        return dict(parents), dict(children)

    @abc.abstractmethod
    def find_independent_nodes(self, target: str) -> List[str]:
        """
        Find nodes that are neither ancestors nor descendants of the target.
        """

        # Function to find all descendants of a given node
        def find_descendants(node):
            visited = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                for child in self.children.get(current, []):
                    if child not in visited:
                        queue.append(child)
            return visited

        # Function to find all ancestors of a given node
        def find_ancestors(node):
            visited = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                for parent in self.parents.get(current, []):
                    if parent not in visited:
                        queue.append(parent)
            return visited

        descendants = find_descendants(target)
        ancestors = find_ancestors(target)

        # Independent nodes are those not in descendants or ancestors of the target node
        return [
            node
            for node in self.nodes
            if node not in descendants and node not in ancestors
        ]

    @abc.abstractmethod
    def causal_effect_DO(
        self,
        interventions,
        functions,
        # parents_Y,
        children,
        parents,
        independent_nodes,
        observational_samples,
    ):
        """
        Computes the causal effect based on the interventions in the system
        """
        final_variables = OrderedDict()
        num_observations = observational_samples[self.target].shape[0]
        update_check_dict = {var: False for var in self.variables}

        topological_order = list(nx.topological_sort(self.G))
        # Process interventions
        for intervention in interventions:
            set_intervention_values(
                final_variables,
                intervention,
                interventions[intervention],
                num_observations,
            )
            # this one has been updated
            update_check_dict[intervention] = True

        for var in topological_order:
            var_parents = self.parents[var]
            # this condition checks if any of the parents of this
            if update_check_dict[var]:
                # this one has already been updated
                continue
            # variable has been updated through an intervention
            if var_parents:
                parents_updated = any(
                    [update_check_dict[parent] for parent in var_parents]
                )
            else:
                parents_updated = False

            # if none of them have been updated use the observational samples
            if not parents_updated:

                final_variables[var] = observational_samples[var]
                # continue to the next variable
                continue

            parents_dict = {var: final_variables[var] for var in var_parents}
            final_variables[var] = predict_child(
                functions[var], parents_dict, var_parents
            )
            update_check_dict[var] = True

        parents_Y = self.parents[self.target]
        mean_effect, variance_effect = predict_causal_effect(
            functions,
            final_variables,
            parents_Y,
            num_observations,
            self.target,
        )

        return mean_effect, variance_effect

    @abc.abstractmethod
    def get_parents_children_independent(
        self, observational_samples: dict, interventions_nodes: List[str]
    ):
        """
        Get the parents, children and independent nodes of a specific intervention
        node
        """
        # getting the children nodes for these interventions
        children = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (child, observational_samples[child])
                            for child in self.children[var]
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        # getting the parent nodes for these interventions
        parents_nodes = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (child, observational_samples[child])
                            for child in self.parents[var]
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        # getting the independent nodes for these interventions
        independent_nodes = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (node, observational_samples[node])
                            for node in self.find_independent_nodes(var)
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        return children, parents_nodes, independent_nodes

    @abc.abstractmethod
    def compute_do(
        self,
        observational_samples: dict,
        value: np.ndarray,
        interventions_nodes: List[str],
    ):
        """
        Computes the interventional outcome on the system
        """
        # this may cause issues if multiple interventions are considered at once
        value = value.reshape(1, -1)
        # parents_Y = OrderedDict(
        #     [(var, observational_samples[var]) for var in self.parents[self.target]]
        # )

        functions = self.functions
        children, parents, independent = self.get_parents_children_independent(
            observational_samples, interventions_nodes
        )
        num_interventions = value.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        var_do = np.zeros((num_interventions, 1))

        for i in range(num_interventions):
            interventions = {
                var: value[i, j] for j, var in enumerate(interventions_nodes)
            }
            mean_do[i], var_do[i] = self.causal_effect_DO(
                interventions,
                functions=functions,
                # parents_Y=parents_Y,
                children=children,
                parents=parents,
                independent_nodes=independent,
                observational_samples=observational_samples,
            )

        return mean_do, var_do

    @abc.abstractmethod
    def break_dependency_structure(self):
        """
        Changing the edges for the BO method
        """
        target = self.target
        nodes = self.nodes
        self._edges = []
        for node in nodes:
            if node != target:
                self._edges.append((node, target))

        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    @abc.abstractmethod
    def mispecify_graph(self, edges):
        """
        Flip some of the edges of the graph for further experimentation
        """
        self._edges = edges
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    @abc.abstractmethod
    def entropy_decomposition(self, observational_samples: Dict, num_points: int = 100):
        # decompose the variance into epistemic uncertainty and aleatoric uncertainty
        _, _, manipulative_variables = self.get_sets()
        total_entropy = 0
        for var in self.variables:
            parents = self.parents[var]
            function = self.functions[var]
            if parents:
                dataset = np.hstack(
                    [observational_samples[parent] for parent in parents]
                )
                variance = function.predict(dataset)[1]
                # print(variance)
                entropy = 1 / 2 * np.log(2 * np.pi * np.exp(1) * variance)
                # print(entropy)
                total_entropy += np.mean(entropy)
                print(f"{var} taking average {np.mean(entropy)}")
            else:
                # since we assume gaussian noise and use gausian kernel use gausian again
                variance = function.predict()[1]
                entropy = 1 / 2 * np.log(2 * np.pi * np.exp(1) * variance)
                total_entropy += entropy
                print(f"{var} taking marginal {np.mean(entropy)}")

        entropy_dict = {"entropy": total_entropy}
        return entropy_dict

    @abc.abstractmethod
    def decompose_variance(
        self, variable: str, observational_samples: Dict, num_points: int = 100
    ):
        parents = self.parents[variable]

        _, _, manipulative_variables = self.get_sets()
        n_obs = len(observational_samples[self.target])

        aleatoric_uncertainty = 0
        epistemic_uncertainty = 0

        count_m = 0
        count_nm = 0

        if parents:
            target_function: GPRegression = self.functions[variable]
            for i, var in enumerate(parents):
                dataset = np.hstack(
                    [observational_samples[parent] for parent in parents]
                )
                min_var = np.min(observational_samples[var])
                max_var = np.max(observational_samples[var])
                vals = np.linspace(start=min_var, stop=max_var, num=num_points)

                variance = np.zeros(shape=num_points)
                for j, val in enumerate(vals):
                    dataset[:, i] = np.repeat(val, n_obs)
                    variance[j] = np.mean(target_function.predict(dataset)[1])

                if var in manipulative_variables:
                    # this means we are in the subset X
                    count_m += 1
                    epistemic_uncertainty += np.mean(variance)
                else:
                    # this means we are in the subset C
                    count_nm += 1
                    aleatoric_uncertainty += np.mean(variance)
        if count_m > 0:
            # XXX can divide by count_m here but it overall
            epistemic_uncertainty = epistemic_uncertainty

        if count_nm > 0:
            # XXX can divide by count_nm here but want it overall
            aleatoric_uncertainty = aleatoric_uncertainty
        uncertanties = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
        }
        return uncertanties

    @abc.abstractmethod
    def decompose_target_variance(
        self, observational_samples: Dict, num_points: int = 100
    ):
        return self.decompose_variance(
            variable=self.target,
            observational_samples=observational_samples,
            num_points=num_points,
        )

    @abc.abstractmethod
    def decompose_all_variance(
        self, observational_samples: Dict, num_points: int = 100
    ):
        aleatoric_uncertainty = 0
        epistemic_uncertainty = 0
        for var in self.variables:
            uncertanties = self.decompose_variance(
                variable=var,
                observational_samples=observational_samples,
                num_points=num_points,
            )
            aleatoric_uncertainty += uncertanties["aleatoric"]
            epistemic_uncertainty += uncertanties["epistemic"]

        uncertanties = {
            "aleatoric": aleatoric_uncertainty,
            "epistemic": epistemic_uncertainty,
        }
        return uncertanties

    @abc.abstractmethod
    def compute_do_generic(self, intervention_nodes, observational_samples, value):
        mean_do, var_do = self.compute_do(
            observational_samples, value, intervention_nodes
        )
        return mean_do, var_do

    @abc.abstractmethod
    def set_data_standardised_flag(
        self,
        standardised: bool = True,
        means: Dict[str, np.ndarray] = None,
        stds: Dict[str, np.ndarray] = None,
    ):
        if standardised:
            self._standardised = standardised
            self.means = means
            self.stds = stds
        else:
            self._standardised = standardised

    @abc.abstractmethod
    def set_params_iscm(self, var: str, mean: float, std: float):
        self.population_mean_variance[var]["mean"] = mean
        self.population_mean_variance[var]["std"] = std
