# second version which manages the exploration set in a more flexible way
# maybe use this if the manipulative variables are greater than 4
import itertools
import logging
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression

import utils.cbo_functions as cbo_functions
import utils.ceo_utils as ceo_utils
from algorithms.BASE_algorithm import BASE
from diffcbed.replay_buffer import ReplayBuffer
from graphs.graph import GraphStructure
from posterior_model.model import (
    DoublyRobustModel,
    LinearSCMModel,
    NonLinearSCMModel,
    SCMModel,
)
from utils.cbo_classes import DoFunctions, TargetClass
from utils.sem_sampling import (
    change_int_data_format_to_mi,
    change_intervention_list_format,
    change_obs_data_format_to_mi,
    sample_model,
)


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


class PARENT_SCALE(BASE):
    """
    This is the class of my developed methodology
    """

    def __init__(
        self,
        graph: GraphStructure,
        nonlinear: bool = True,
        causal_prior: bool = True,
        noiseless: bool = True,
        cost_num: int = 1,
        scale_data: bool = True,
        individual: bool = False,
        use_doubly_robust: bool = True,
        use_iscm: bool = False,
        boundary_eps_frac: float = 0.01,
        acquisition: str = "EI",
        ucb_beta: float = 2.0,
        kernel_type: str = "rbf",
    ):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        self.nonlinear = nonlinear
        # acquisition function for get_new_x_y_list: "EI" or "UCB"
        self.acquisition = acquisition
        self.ucb_beta = ucb_beta
        # surrogate GP kernel: "rbf" (default) or "spherical_linear"
        self.kernel_type = kernel_type

        # setting up some more variables
        # self.graph_env = graph_env
        self.buffer = ReplayBuffer(binary=True)

        self.manipulative_variables = self.graph.get_sets()[2]
        self.causal_prior = causal_prior
        self.noiseless = noiseless
        self.cost_num = cost_num
        self.scale_data = scale_data
        self.individual = individual
        self.use_doubly_robust = use_doubly_robust
        self.use_iscm = use_iscm
        # fraction of each variable's intervention-range width used as the
        # epsilon when checking whether an intervention lies on the boundary
        self.boundary_eps_frac = boundary_eps_frac

    def set_values(self, D_O, D_I, exploration_set):
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        self.graph.set_interventional_range_data(self.D_O)
        self.topological_order = list(self.D_O.keys())
        # this is too much now, so we are continuously going to update it
        self.exploration_set = exploration_set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

    def do_function_graph(self, es: Tuple, size: int = 100, edge_num: int = 0):

        # setting up the plotting stuff
        true_vals = np.zeros(shape=size)
        predictions = np.zeros(shape=size)
        var = np.zeros(shape=size)
        es_num = self.es_to_n_mapping[es]
        interventions = {}

        # getting the number of entries in the exploration set
        intervention_domain = self.graph.get_interventional_range()
        min_intervention, max_intervention = intervention_domain[es[0]]
        intervention_vals = np.linspace(
            start=min_intervention, stop=max_intervention, num=100
        )

        for i in range(1, len(es)):
            min_i, max_i = intervention_domain[es[i]]
            interventions[es[i]] = (min_i + max_i) / 2

        for i, intervention_val in enumerate(intervention_vals):
            interventions[es[0]] = intervention_val
            true_vals[i] = np.mean(
                sample_model(
                    self.graph.SEM,
                    interventions=interventions,
                    sample_count=500,
                    graph=self.graph,
                    use_iscm=self.use_iscm,
                )["Y"]
            )

            value = np.array([interventions[var] for var in es]).reshape(1, -1)
            predictions[i] = self.do_effects_functions[edge_num][
                es_num
            ].mean_function_do(value)
            var[i] = self.do_effects_functions[edge_num][es_num].var_function_do(value)

        plt.plot(intervention_vals, true_vals, label="True")
        plt.plot(intervention_vals, predictions, label="Do 1")
        plt.fill_between(
            intervention_vals,
            [p - 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            [p + 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            color="gray",
            alpha=0.5,
        )
        plt.legend()
        plt.show()

    def determine_initial_probabilities(self) -> Dict[Tuple, float]:
        if self.use_doubly_robust:
            topological_order = list(self.D_O.keys())
            D_O_mi = change_obs_data_format_to_mi(
                self.D_O,
                graph_variables=self.graph.variables,
                intervention_node=np.zeros(shape=len(self.graph.variables)),
            )
            robust_model = DoublyRobustModel(
                graph=self.graph,
                topological_order=topological_order,
                target=self.graph.target,
                indivdual=self.individual,
                num_bootstraps=10,
            )
            buffer = ReplayBuffer(binary=True)
            buffer.update(D_O_mi)
            robust_model.run_method(buffer.data())
            probabilities = robust_model.prob_estimate
            if (
                isinstance(probabilities, dict)
                and len(probabilities) == 1
                and () in probabilities
            ):
                probabilities = {
                    (var,): 1 / len(self.graph.variables)
                    for var in self.graph.variables
                    if var != self.graph.target
                }
            if () in probabilities:
                del probabilities[()]
        else:
            variables = [
                var for var in self.graph.variables if var != self.graph.target
            ]

            combinations = []
            max_vars = max(10, len(variables) + 1)
            for r in range(1, max_vars):
                combinations.extend(itertools.combinations(variables, r))

            probabilities = {combo: 1 / len(combinations) for combo in combinations}
        return probabilities

    def standardize_all_data(self):
        """
        This one just standardises the dataset
        """
        input_keys = [key for key in self.D_O.keys() if key != self.graph.target]
        self.means = {key: np.mean(self.D_O[key]) for key in input_keys}
        self.stds = {key: np.std(self.D_O[key]) for key in input_keys}

        D_O_scaled = {}
        for key in self.D_O:
            if key in input_keys:
                D_O_scaled[key] = standardize(
                    self.D_O[key], self.means[key], self.stds[key]
                )
            else:
                D_O_scaled[key] = self.D_O[key]

        interventions = self.D_I.keys()
        D_I_scaled = {intervention: {} for intervention in interventions}
        for intervention in interventions:
            for key in self.D_I[intervention]:
                if key in input_keys:
                    D_I_scaled[intervention][key] = standardize(
                        self.D_I[intervention][key], self.means[key], self.stds[key]
                    )
                else:
                    D_I_scaled[intervention][key] = self.D_I[intervention][key]

        self.D_O_scaled = D_O_scaled
        self.D_I_scaled = D_I_scaled

        # setting data up to use for CBO algorithm
        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in self.variables])
        )

        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set, target=self.graph.target
        )

    def define_all_possible_graphs(self, error_tol=1e-5):
        self.graphs: Dict[Tuple, GraphStructure] = {}
        self.posterior: List[float] = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                continue

            graph: GraphStructure = deepcopy(self.graph)
            edges = [(parent, graph.target) for parent in parents]
            graph.mispecify_graph(edges)
            self.graphs[parents] = graph
            self.posterior.append(self.prior_probabilities[parents])

    def redefine_all_possible_graphs(self, error_tol=1e-4):
        self.posterior: List[float] = []
        parents_to_remove = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                # remove from self.graphs[parents] from self.graphs
                parents_to_remove.append(parents)
                continue
            self.posterior.append(self.prior_probabilities[parents])

        for parents in parents_to_remove:
            if parents in self.graphs:
                del self.graphs[parents]

    def redefine_exploration_set(self):
        # start with individual interventions
        print(f"The flattened list {list(self.graphs.keys())}")
        # flattened_list = [tuple(item) for sublist in self.graphs for item in sublist]
        flattened_list = []
        for sublist in self.graphs:
            for item in sublist:
                flattened_list.append((item,))
        unique_set = set(flattened_list)
        unique_list = list(unique_set)
        self.exploration_set = unique_list

    def calculate_do_statistics(self):
        do_effects_functions: List[List[DoFunctions]] = []
        for i, graph_parents in enumerate(self.graphs):
            # this is the mean and variance for each graph for each element in the exploration set

            graph = self.graphs[graph_parents]
            do_effects_functions.append(
                cbo_functions.update_all_do_functions(
                    graph, self.observational_samples, self.exploration_set
                )
            )
        self.do_effects_functions: List[List[DoFunctions]] = do_effects_functions

    def fit_samples_to_graphs(self):
        sem_emit_fncs: List[OrderedDict[str, GPRegression]] = []
        for key, graph in self.graphs.items():
            graph.fit_samples_to_graph(self.D_O, set_priors=False)
            sem_emit_fncs.append(graph.functions)
        return sem_emit_fncs

    def data_and_prior_setup(self):

        # normalize the datasets for the posterior probability calculations
        self.standardize_all_data()

        self.prior_probabilities = self.determine_initial_probabilities()
        if self.nonlinear:
            self.posterior_model: SCMModel = NonLinearSCMModel(
                self.prior_probabilities, self.graph
            )
        else:
            self.posterior_model: SCMModel = LinearSCMModel(
                self.prior_probabilities, self.graph
            )
        self.posterior_model.set_data(self.D_O_scaled)

        # update the posterior probabilities now with the interventional data
        for intervention in self.D_I_scaled:

            D_I_sample = self.D_I_scaled[intervention]
            num_samples = len(D_I_sample[self.graph.target])
            for n in range(num_samples):
                x_dict = {
                    obs_key: D_I_sample[obs_key][n]
                    for obs_key in D_I_sample
                    if obs_key != self.graph.target
                }
                y = D_I_sample[self.graph.target][n]
                self.posterior_model.update_all(x_dict, y)
                D_I = {key: np.array([D_I_sample[key][n]]) for key in D_I_sample}
                self.posterior_model.add_data(D_I)

        self.prior_probabilities = self.posterior_model.prior_probabilities.copy()

    def return_elements_for_new_exploration_set(
        self, data_x_list: Dict, data_y_list: Dict
    ):
        """
        In the algorithm, the exploration set becomes smaller due to less
        variables being possible parents of the target variable, many of the variables
        are set up for the initial exploration set. Thus, a lot of the variables
        will start to change as the exploration set decreases
        """
        data_x_list_new = []
        data_y_list_new = []
        for i, es in enumerate(self.exploration_set):
            # this is still the previous mapping
            j = self.es_to_n_mapping[es]
            data_x_list_new.append(data_x_list[j])
            data_y_list_new.append(data_y_list[j])

        # only remap everything now
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

        parameter_spaces = [None] * len(self.exploration_set)
        target_classes: List[TargetClass] = [None] * len(self.exploration_set)
        # redefining the model class as well due to the increase here
        self.model_list_overall: List[GPyModelWrapper] = [None] * len(
            self.exploration_set
        )

        for i in range(len(self.exploration_set)):
            parameter_spaces[i] = self.graph.get_parameter_space(
                self.exploration_set[i]
            )
            target_classes[i] = TargetClass(
                sem_model=self.graph.SEM,
                interventions=self.exploration_set[i],
                variables=self.graph.variables,
                graph=self.graph,
                noiseless=self.noiseless,
                use_iscm=self.use_iscm,
            )

        return data_x_list_new, data_y_list_new, parameter_spaces, target_classes

    def compute_boundary_percentage(
        self, var_to_intervene: Tuple[str], intervention_values: np.ndarray
    ) -> Tuple[float, int, int]:
        """
        Computes the fraction of intervened dimensions whose value lies within
        +- eps of the intervention-range boundary, where eps is
        boundary_eps_frac of the range width of each variable
        """
        intervention_ranges = self.graph.interventional_range_data
        n_dims = len(var_to_intervene)
        n_on_boundary = 0
        for j, var in enumerate(var_to_intervene):
            lower, upper = intervention_ranges[var]
            eps = self.boundary_eps_frac * (upper - lower)
            value = intervention_values[j]
            if value <= lower + eps or value >= upper - eps:
                n_on_boundary += 1
        return n_on_boundary / n_dims, n_on_boundary, n_dims

    def build_surrogates(
        self,
        trial_observed: bool,
        data_x_list,
        data_y_list,
        best_variable: int,
        input_space,
        iteration: int = 0,
    ):
        """Build/refresh one surrogate per exploration set.

        Seam for alternative surrogates: subclasses override this to change the
        model without touching the algorithm loop. The default is the standard
        causal-prior GP (RBF or spherical core, per ``self.kernel_type``).
        """
        return ceo_utils.update_posterior_model_aggregate_2(
            self.exploration_set,
            trial_observed,
            self.model_list_overall,
            data_x_list,
            data_y_list,
            self.causal_prior,
            best_variable,
            input_space,
            self.do_effects_functions,
            self.posterior,
            kernel_type=self.kernel_type,
        )

    def run_algorithm(self, T: int = 30, show_graphics: bool = False, file: str = None):

        self.data_and_prior_setup()
        self.define_all_possible_graphs()
        self.fit_samples_to_graphs()

        # starting the setup for the CBO algorithm
        (
            data_x_list,
            data_y_list,
            best_intervention_value,
            current_global_min,
            best_variable,
        ) = cbo_functions.define_initial_data_CBO(
            self.interventional_samples,
            self.exploration_set,
            self.manipulative_variables,
            self.target,
        )

        input_space = [len(vars) for vars in self.exploration_set]
        current_best_x = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        current_best_y = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        parameter_spaces = [None] * len(self.exploration_set)
        target_classes: List[TargetClass] = [None] * len(self.exploration_set)
        self.model_list_overall: List[GPyModelWrapper] = [None] * len(
            self.exploration_set
        )

        for i in range(len(self.exploration_set)):
            parameter_spaces[i] = self.graph.get_parameter_space(
                self.exploration_set[i]
            )
            target_classes[i] = TargetClass(
                sem_model=self.graph.SEM,
                interventions=self.exploration_set[i],
                variables=self.graph.variables,
                graph=self.graph,
                noiseless=self.noiseless,
            )

        # STARTING THE ALGORITHM
        current_cost: List[int] = []
        global_opt: List[float] = []
        current_y: List[float] = []
        average_uncertainty: List[float] = []
        intervention_set: List[Tuple[str]] = []
        intervention_values: List[Tuple[float]] = []
        # per-iteration tracking of boundary interventions and parent posterior
        self.boundary_percentages: List[float] = []
        self.boundary_counts: List[Tuple[int, int]] = []
        self.posterior_history: List[Dict[Tuple, float]] = []
        # global_opt.append(current_global_min)
        current_cost.append(0.0)
        cost_functions = self.graph.get_cost_structure(self.cost_num)

        # define the initial surrogate models
        self.redefine_exploration_set()
        logging.info(f"The current exploration set is {self.exploration_set}")
        data_x_list, data_y_list, parameter_spaces, target_classes = (
            self.return_elements_for_new_exploration_set(data_x_list, data_y_list)
        )
        self.calculate_do_statistics()
        self.model_list_overall = self.build_surrogates(
            True, data_x_list, data_y_list, best_variable, input_space, iteration=0
        )
        # iteration-0 snapshot of the posterior over parent sets
        self.posterior_history.append(dict(zip(self.graphs.keys(), self.posterior)))

        for i in range(T):
            logging.info(f"----------------------ITERATION {i}----------------------")
            logging.info(f"Updated posterior distribution {self.posterior}")
            logging.info(f"The corresponding parents are {list(self.graphs.keys())}")

            # redefine the exploration set

            # get the next sample
            y_acquisition_list, x_new_list = cbo_functions.get_new_x_y_list(
                self.exploration_set,
                self.graph,
                current_global_min,
                self.model_list_overall,
                cost_functions,
                acquisition=self.acquisition,
                ucb_beta=self.ucb_beta,
            )

            # find the optimal intervention, which maximises the acquisition function
            target_index = np.argmax(y_acquisition_list)
            var_to_intervene = tuple(self.exploration_set[target_index])
            intervention_set.append(var_to_intervene)

            # Setting the data after the new point was found
            intervention_values.append(tuple(x_new_list[target_index][0]))

            x_new_list_intervention = np.array(
                [
                    x_new_list[target_index][0, j]
                    for j, var in enumerate(var_to_intervene)
                ]
            ).reshape(1, -1)
            print(f"Back to the original range {x_new_list_intervention}")

            # track how many intervened dimensions lie on the boundary
            boundary_pct, n_on_boundary, n_dims = self.compute_boundary_percentage(
                var_to_intervene, x_new_list_intervention[0]
            )
            self.boundary_percentages.append(boundary_pct)
            self.boundary_counts.append((n_on_boundary, n_dims))
            logging.info(
                f"Boundary percentage {boundary_pct:.2f} "
                f"({n_on_boundary}/{n_dims} dimensions on boundary)"
            )
            y_new = target_classes[target_index].compute_target(x_new_list_intervention)
            print(f"The outcome is {y_new}")
            data_x_list[target_index] = np.vstack(
                (data_x_list[target_index], x_new_list[target_index])
            )

            data_y_list[target_index] = np.concatenate(
                (data_y_list[target_index], y_new.reshape(-1))
            )

            # set the new best variable
            best_variable = target_index

            ## Update the dict storing the current optimal solution

            intervention = {
                var: x_new_list_intervention[0][j]
                for j, var in enumerate(var_to_intervene)
            }

            sample = sample_model(
                static_sem=self.graph.SEM,
                interventions=intervention,
                sample_count=1,
                graph=self.graph,
            )

            # sample[self.graph.target] = y_new
            print(f"The interventional sample {sample}")
            current_best_x[var_to_intervene].append(x_new_list_intervention[0])
            current_best_y[var_to_intervene].append(y_new[0][0])

            current_y.append(y_new[0][0])
            best_y = global_opt[i - 1] if global_opt else y_new[0][0]
            all_values = [
                value for values in current_best_y.values() for value in values
            ]
            min_y = np.min(all_values)
            global_opt.append(min_y if min_y < best_y else best_y)

            logging.info(
                f"Selected intervention {var_to_intervene} at {x_new_list[target_index]} with y = {y_new}"
            )
            logging.info(f"Current global optimum {global_opt[i]}")

            total_cost = 0
            for j, val in enumerate(self.exploration_set[target_index]):
                total_cost += cost_functions[val](x_new_list[target_index][0, j])
            current_cost.append(current_cost[i] + total_cost)

            # now update the probabilities

            for key in sample:
                if key != self.graph.target:
                    sample[key] = standardize(
                        sample[key], self.means[key], self.stds[key]
                    )
            x_dict = {
                key: val[0][0]
                for key, val in sample.items()
                if key != self.graph.target
            }
            y = sample[self.graph.target][0][0]
            self.posterior_model.update_all(x_dict, y)
            print(f"The x vector for update is {x_dict} with y {y}")
            print(f"The standardized sample {sample}")
            D_I = {key: val.reshape(-1) for key, val in sample.items()}
            self.posterior_model.add_data(D_I)
            self.prior_probabilities = self.posterior_model.prior_probabilities

            self.redefine_all_possible_graphs()
            non_null_parents = list(self.graphs.keys())
            self.posterior_model.update_probabilities(non_null_parents)

            self.calculate_do_statistics()
            self.posterior = [
                self.prior_probabilities[parents] for parents in self.graphs
            ]
            # snapshot of the posterior over parent sets after this iteration
            self.posterior_history.append(
                dict(zip(self.graphs.keys(), self.posterior))
            )

            self.redefine_exploration_set()
            logging.info(f"The current exploration set is {self.exploration_set}")
            data_x_list, data_y_list, parameter_spaces, target_classes = (
                self.return_elements_for_new_exploration_set(data_x_list, data_y_list)
            )
            self.model_list_overall = self.build_surrogates(
                False, data_x_list, data_y_list, best_variable, input_space,
                iteration=i + 1,
            )
            current_global_min = global_opt[i]

            if show_graphics:
                for es in self.exploration_set:
                    fig, ax = self.plot_model_list(self.model_list_overall, es)
                    for j, intervention in enumerate(intervention_set):
                        if intervention == es:
                            ax.scatter(
                                intervention_values[j],
                                current_y[j],
                                marker="x",
                                color="black",
                                s=100,
                                label="Intervention Points",
                            )
                    if file:
                        filename = f"{file}_{es[0]}_iter_{i+1}"
                        plt.savefig(filename, bbox_inches="tight")
                    else:
                        plt.show()

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
            average_uncertainty,
        )
