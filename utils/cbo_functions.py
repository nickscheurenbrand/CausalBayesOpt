import itertools
import logging
from typing import Callable, List, OrderedDict, Tuple

import numpy as np
import scipy
import scipy.spatial
from emukit.bayesian_optimization.acquisitions import MaxValueEntropySearch
from emukit.core import ParameterSpace
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core import Mapping
from GPy.kern.src.rbf import RBF
from GPy.models.gp_regression import GPRegression

from graphs.graph import GraphStructure
from utils.cbo_classes import (
    CausalExpectedImprovement,
    CausalGradientAcquisitionOptimizer,
    CausalRBF,
    CausalSphericalLinear,
    CausalSphericalRBF,
    CausalUpperConfidenceBound,
    Cost,
    DoFunctions,
)


def _data_bounds(X: np.ndarray) -> np.ndarray:
    """Per-dimension (min, max) bounds from the training inputs, used to centre
    and scale the stereographic projection. Degenerate (constant) dimensions are
    padded to keep a non-zero span."""
    mins = X.min(axis=0).astype(float)
    maxs = X.max(axis=0).astype(float)
    degenerate = (maxs - mins) < 1e-9
    mins[degenerate] -= 0.5
    maxs[degenerate] += 0.5
    return np.stack([mins, maxs], axis=1)


def set_up_GP(
    causal_prior: bool,
    input_space: int,
    mean_function_do: Callable,
    var_function_do: Callable,
    X: np.ndarray,
    Y: np.ndarray,
    kernel_type: str = "rbf",
) -> GPyModelWrapper:
    """
    Setting up the Gaussian Process based on the previous computed interventional
    mean function and interventional variance function.

    kernel_type: "rbf" (default, unchanged behaviour) or "spherical_linear" to
    use the stereographic-projection linear kernel in place of the RBF core.
    """
    if causal_prior:
        logging.info("Using the Causal Gaussian Prior")
        # define the model for the causal prior here
        # this one uses the computed mean and variance with Gaussian Processes
        mf = Mapping(input_space, 1)
        mf.f = mean_function_do
        mf.update_gradients = lambda a, b: None
        if kernel_type == "spherical_linear":
            kernel = CausalSphericalLinear(
                input_space,
                variance_adjustment=var_function_do,
                variance=1.0,
                lengthscale=1.0,
                bounds=_data_bounds(X),
            )
        elif kernel_type == "spherical_rbf":
            kernel = CausalSphericalRBF(
                input_space,
                variance_adjustment=var_function_do,
                variance=1.0,
                lengthscale=1.0,
                proj_lengthscale=1.0,
                bounds=_data_bounds(X),
            )
        else:
            kernel = CausalRBF(
                input_space,
                variance_adjustment=var_function_do,
                lenghtscale=1.0,
                variance=1.0,
                rescale_variance=1.0,
                ARD=False,
            )

        gpy_model = GPRegression(
            X=X, Y=Y, kernel=kernel, noise_var=1e-10, mean_function=mf
        )
        # gpy_model.optimize_restarts(num_restarts=5)
        gpy_model.optimize()
        emukit_model = GPyModelWrapper(gpy_model)
    else:
        logging.info("Setting up the gaussian prior")
        # this one just uses the data
        if kernel_type == "spherical_linear":
            kernel = CausalSphericalLinear(
                input_space,
                variance_adjustment=lambda x: np.zeros(np.atleast_2d(x).shape[0]),
                variance=1.0,
                lengthscale=1.0,
                bounds=_data_bounds(X),
            )
        elif kernel_type == "spherical_rbf":
            kernel = CausalSphericalRBF(
                input_space,
                variance_adjustment=lambda x: np.zeros(np.atleast_2d(x).shape[0]),
                variance=1.0,
                lengthscale=1.0,
                proj_lengthscale=1.0,
                bounds=_data_bounds(X),
            )
        else:
            kernel = RBF(input_space, lengthscale=1.0, variance=1.0)
        gpy_model = GPRegression(
            X=X,
            Y=Y,
            kernel=kernel,
            noise_var=1e-10,
        )
        # gpy_model.optimize_restarts(num_restarts=5)
        gpy_model.optimize()
        emukit_model = GPyModelWrapper(gpy_model)

    gpy_model = safe_optimization(emukit_model.model)
    emukit_model = GPyModelWrapper(gpy_model)
    return emukit_model


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


def update_all_do_functions(
    graph: GraphStructure,
    samples: np.ndarray,
    exploration_set: List,
) -> List[DoFunctions]:
    """
    This is for CBO algorithm when the variables in the exploration set changes. This changes
    based on what is in the intervention set, as well as what was newly observed. Each of
    these classes have a mean_function_do and var_function_do instance contained within them
    """
    variables = graph.variables
    samples_dict = {
        var: samples[:, i].reshape(-1, 1) for i, var in enumerate(variables)
    }
    # to make it an instance of the class rather than the individual functions
    do_functions_list = [None] * len(exploration_set)
    for i, intervention_set in enumerate(exploration_set):
        do_functions_list[i] = DoFunctions(
            graph.get_all_do(),
            samples_dict,
            intervention_set,
        )

    return do_functions_list


def update_hull(
    observational_samples: np.ndarray, manipulative_variables: List
) -> float:
    """
    Calculates the coverage of the observational space, for the computation of epsilon
    """
    list_variables = [
        list(observational_samples[:, i]) for i in range(len(manipulative_variables))
    ]

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    return coverage_obs


def compute_coverage(
    D_O: dict, manipulative_variables: List, dict_ranges: dict
) -> Tuple[float, float, float]:
    """
    Calculates the different quantities of the observation intervention tradeoff
    """
    list_variables = [list(D_O[var]) for var in manipulative_variables]
    print(len(list_variables))

    list_ranges = [list(dict_ranges[var]) for var in manipulative_variables]

    # calculate the total possible volume of the set
    vertices = list(
        itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))])
    )
    coverage_total = scipy.spatial.ConvexHull(vertices).volume

    # now do the calculation for the observational samples
    stack_variables = np.transpose(np.vstack(list_variables))
    # stack_variables = np.hstack(list_variables)
    # print(vertices)
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    hull_obs = scipy.spatial.ConvexHull(stack_variables)
    alpha_coverage = coverage_obs / coverage_total
    return alpha_coverage, hull_obs, coverage_total


def update_posterior_model(
    exploration_set: List,
    trial_observed: bool,
    model_list: List[GPyModelWrapper],
    data_x_list: dict,
    data_y_list: dict,
    causal_prior: bool,
    best_variable: int,
    input_space: List,
    do_function_list: List[DoFunctions],
) -> List:
    """
    Update the posterior of the gaussian process if it was intervened on in the previous timestep
    """
    # update the Gaussian Processes
    if trial_observed:
        # update all the models if we observed in the previous trial
        # this one uses the computed do functions by fitting the causal graph
        for j in range(len(exploration_set)):
            X = data_x_list[j]
            Y = data_y_list[j].reshape(-1, 1)
            model_list[j] = set_up_GP(
                causal_prior,
                input_space[j],
                do_function_list[j].mean_function_do,
                do_function_list[j].var_function_do,
                X,
                Y,
            )
    else:
        # only update the model of the set that was intervened upon
        Y = data_y_list[best_variable].reshape(-1, 1)
        X = data_x_list[best_variable]
        model_list[best_variable] = set_up_GP(
            causal_prior,
            input_space[best_variable],
            do_function_list[best_variable].mean_function_do,
            do_function_list[best_variable].var_function_do,
            X,
            Y,
        )

    return model_list


def get_new_x_y_list(
    exploration_set: List[List[str]],
    graph: GraphStructure,
    current_global_min: float,
    model_list: List,
    cost_functions: OrderedDict,
    task: str = "min",
    acquisition: str = "EI",
    ucb_beta: float = 2.0,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Get the new acquisitions for the all the elements in the exploration set.
    acquisition: "EI" (causal expected improvement) or "UCB" (causal confidence
    bound).
    """
    y_acquisition_list = [None] * len(exploration_set)
    x_new_list = [None] * len(exploration_set)
    for j, vars in enumerate(exploration_set):
        space = graph.get_parameter_space(vars)
        cost = Cost(cost_functions, vars)
        optimizer = CausalGradientAcquisitionOptimizer(space)
        # acquisition = (
        #     CausalExpectedImprovement(current_global_min, task, model_list[j]) / cost
        # )
        if acquisition.upper() == "UCB":
            acq = CausalUpperConfidenceBound(
                current_global_min, task, model_list[j], beta=ucb_beta
            )
        else:
            acq = CausalExpectedImprovement(current_global_min, task, model_list[j])
        x_new, _ = optimizer.optimize(acq)
        y_acquisition = acq.evaluate(x_new).flatten()
        y_acquisition_list[j] = y_acquisition
        x_new_list[j] = x_new

    return y_acquisition_list, x_new_list


def get_new_x_y_list_entropy(
    exploration_set: List[List[str]],
    graph: GraphStructure,
    current_global_min: float,
    model_list: List,
    cost_functions: OrderedDict,
    task: str = "min",
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Get the new acquisitions for all the elements in the exploration set using Predictive Entropy Search (PES).
    """
    y_acquisition_list = [None] * len(exploration_set)
    x_new_list = [None] * len(exploration_set)

    for j, vars in enumerate(exploration_set):
        space = graph.get_parameter_space(vars)
        # cost = Cost(cost_functions, vars)
        optimizer = GradientAcquisitionOptimizer(space, num_samples=100)
        acquisition = MaxValueEntropySearch(model_list[j], space=space)

        x_new, _ = optimizer.optimize(acquisition)
        y_acquisition = acquisition.evaluate(x_new).flatten()
        y_acquisition_list[j] = np.max(y_acquisition)
        x_new_list[j] = x_new

    return y_acquisition_list, x_new_list


def define_initial_data_CBO(
    interventional_data: dict,
    exploration_set: List[List[str]],
    manipulative_variables: List[str],
    outcome_variable: str,
    task: str = "min",
):
    """
    Processes interventional data to identify optimal interventions based on a specified criterion (min/max).

    Parameters:
        interventional_data (dict): Dictionary of datasets containing interventions,
                                    indexed by type and containing numpy arrays for each variable.
        num_interventions (int): Number of interventions to consider in the optimization.
        exploration_set (list): List of potential variables or configurations used in the interventions.
        name_index (int): Index used for setting a seed for reproducibility.
        manipulative_variables (list): List of variables that are manipulated.
        outcome_variable (str): The variable used as the outcome for optimization.
        task (str): Task to perform, either 'min' for minimization or 'max' for maximization.

    Returns:
        tuple: Tuple containing lists of X data, Y data, best intervention values, optimal Y value, and best variables.
    """
    assert task in ["min", "max"]
    data_x_list = []
    data_y_list = []
    opt_list = []

    # Process each dataset in the provided interventional data
    for idx, (index, data) in enumerate(interventional_data.items()):
        # Combine data for manipulative variables
        num_interventions = len(data) - 1
        data_x = np.column_stack(
            [data[var] for var in manipulative_variables if var in data]
        )
        data_y = np.array(data[outcome_variable])

        # Combine and shuffle data using a reproducible random seed
        all_data = np.column_stack((data_x, data_y))

        # Select the subset of data for optimization
        subset_all_data = all_data

        data_x_list.append(subset_all_data[:, :-1])
        data_y_list.append(subset_all_data[:, -1])

        current_opt_val = (
            np.min(subset_all_data[:, -1])
            if task == "min"
            else np.max(subset_all_data[:, -1])
        )
        opt_list.append(current_opt_val)

    # Identify the best overall intervention and corresponding values
    best_idx = np.argmin(opt_list) if task == "min" else np.argmax(opt_list)
    best_variable = exploration_set[best_idx]
    best_data = data_x_list[best_idx]
    opt_y = opt_list[best_idx]
    best_intervention_value = best_data[
        (
            np.argmin(data_y_list[best_idx])
            if task == "min"
            else np.argmax(data_y_list[best_idx])
        )
    ]

    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable
