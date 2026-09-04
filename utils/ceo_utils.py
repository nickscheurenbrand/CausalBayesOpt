import logging
import random
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, OrderedDict, Tuple, Union

import numpy as np
import statsmodels.api as sm
from emukit.core.acquisition import Acquisition
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression
from scipy.special import logsumexp, softmax

from graphs.graph import GraphStructure
from utils.cbo_classes import DoFunctions
from utils.cbo_functions import set_up_GP
from utils.sem_sampling import sample_from_SEM_hat, sample_model


class MyKDENew(sm.nonparametric.KDEUnivariate):
    """
    This is the class to fit the kernel density estimates of the data
    """

    def __init__(self, *args):
        super().__init__(*args)

    def sample(self, n_samples=1, random_state=None):
        u = np.random.uniform(0, 1, size=n_samples)
        i = (u * self.endog.shape[0]).astype(np.int64)

        # if self.kernel == 'gaussian':
        return (
            np.atleast_2d(np.random.normal(self.endog[i], self.kernel.h)),
            self.endog[i],
        )


def inverse_cdf(su, W):
    """
    The inverse CDF for a finite distribution
    """
    # ensure W sums to one here
    cumulative_sums = np.cumsum(W)
    return np.searchsorted(cumulative_sums, su)


def update_posterior_interventional(
    graphs: List[GraphStructure],
    posterior: List[float],
    intervened_var: Tuple,
    all_emission_fncs: List[OrderedDict[str, Callable]],
    interventional_samples: Dict[Tuple, Dict[str, np.ndarray]],
    lr=0.05,
):
    """
    Updating the posterior probabilities of each graph after intervening on the system
    """
    current_interventional_sample = interventional_samples[intervened_var]
    for graph_idx, emission_fncs in enumerate(
        all_emission_fncs
    ):  # as many emission_fncs dicts as graphs
        graph = graphs[graph_idx]
        # logging.info(f"{emission_fncs}")
        # i = 0
        for var in emission_fncs:
            parents = graph.parents[var]
            if parents:
                xx = np.hstack(
                    [
                        current_interventional_sample[parent].reshape(-1, 1)
                        for parent in parents
                    ]
                )
                yy = current_interventional_sample[var].reshape(-1, 1)
                if var in intervened_var:
                    # Here the truncated assumption comes in. Dont compute posterior
                    continue
                # else:
                #     i += 1
                posterior[graph_idx] += lr * log_likelihood(
                    emission_fncs[var], xx, yy  # the model
                )
            else:
                yy = current_interventional_sample[var].reshape(-1, 1)
                if var in intervened_var:
                    # Here the truncated assumption comes in. Dont compute posterior
                    continue

                posterior[graph_idx] += lr * log_likelihood(
                    emission_fncs[var], None, yy  # the model
                )

    return posterior


def log_likelihood(
    model: GPRegression, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Computes the log-likelihood for the observations based on the fitted functions
    """
    if X_test is not None:
        log_lik = model.log_predictive_density(x_test=X_test, y_test=y_test)
        res = log_lik.flatten().sum()
    else:
        # Marginal
        log_lik = model.score_samples(y_test)
        res = log_lik.flatten().sum()

    return res


def stratified(W, M):
    # Generate stratified random samples
    su = (np.random.rand(M) + np.arange(M)) / M

    # Ensure su is sorted because inverse CDF requires sorted inputs
    su = np.sort(su)

    # Call inverse_cdf which should be defined to handle this sorted uniform data
    return inverse_cdf(su, W)


def aggregate_mean_function(
    i: int,
    do_functions: List[List[DoFunctions]],
    posterior: np.ndarray,
    x: np.ndarray,
):
    """
    The overall mean of the interventional distribution based on all the graphs, uses the latent
    variable G and the posterior distribution to calculate it
    """
    # calculates E[Y | do(x)] = E[E[Y|(do(x), G]], i.e. it calculates the mean over all the graphs
    unweighted_means = np.hstack(
        [do_functions_es[i].mean_function_do(x) for do_functions_es in do_functions]
    )
    mean = unweighted_means @ posterior
    mean = mean.reshape(-1, 1)
    return mean


def aggregate_var_function(
    i: int,
    do_functions: List[List[DoFunctions]],
    posterior: np.ndarray,
    x: np.ndarray,
):
    """
    The overall variance function for the interventional distribution based on all the graphs,
    introduces the latent variable G in the graph
    """
    # calculates V[Y | do(x)] = E[V[Y|(do(x), G]] + V[E[Y | do(x), G]], i.e., it calculates the variance over all the graphs
    unweighted_means = np.hstack(
        [do_functions_es[i].mean_function_do(x) for do_functions_es in do_functions]
    )

    unweighted_second_moments = np.hstack(
        [do_function_es[i].mean_function_do(x) ** 2 for do_function_es in do_functions]
    )

    unweighted_vars = np.hstack(
        [do_functions_es[i].var_function_do(x) for do_functions_es in do_functions]
    )

    var = (
        unweighted_vars @ posterior
        + unweighted_second_moments @ posterior
        - (unweighted_means @ posterior) ** 2
    )
    var = var.reshape(-1, 1)
    return var


# def get_new_x_y_list(
#     exploration_set: List[List[str]],
#     graph: GraphStructure,
#     current_global_min: float,
#     model_list: List,
#     cost_functions: OrderedDict,
#     task: str = "min",
# ) -> Tuple[np.ndarray, List[List[float]]]:
#     """
#     Get the new acquisitions for the all the elements in the exploration set
#     """
#     y_acquisition_list = [None] * len(exploration_set)
#     x_new_list = [None] * len(exploration_set)
#     for j, vars in enumerate(exploration_set):
#         space = graph.get_parameter_space(vars)
#         cost = Cost(cost_functions, vars)
#         optimizer = CausalGradientAcquisitionOptimizer(space)
#         acquisition = (
#             CausalExpectedImprovement(current_global_min, task, model_list[j]) / cost
#         )
#         x_new, _ = optimizer.optimize(acquisition)
#         y_acquisition = acquisition.evaluate(x_new)
#         y_acquisition_list[j] = y_acquisition
#         x_new_list[j] = x_new

#     return np.array(y_acquisition_list), x_new_list


def create_n_dimensional_intervention_grid(
    limits: list, size_intervention_grid: int = 100
):
    """
    Usage: combine_n_dimensional_intervention_grid([[-2,2],[-5,10]],10)
    """
    if any(isinstance(el, list) for el in limits) is False:
        # We are just passing a single list
        return np.linspace(limits[0], limits[1], size_intervention_grid)[:, None]
    else:
        extrema = np.vstack(limits)
        inputs = [
            np.linspace(i, j, size_intervention_grid)
            for i, j in zip(extrema[:, 0], extrema[:, 1])
        ]
        return np.dstack(np.meshgrid(*inputs)).ravel("F").reshape(len(inputs), -1).T


def to_prob(arm_values: np.ndarray, task: str = "min") -> np.ndarray:
    """
    Returns the probability form of the arm distribution
    """
    return (
        softmax(-(1) * np.array(arm_values))
        if task == "min"
        else softmax(np.array(arm_values))
    )


def update_posterior_model_aggregate(
    exploration_set: List,
    trial_observed: bool,
    model_list: List[GPyModelWrapper],
    data_x_list: dict,
    data_y_list: dict,
    causal_prior: bool,
    best_variable: int,
    input_space: List,
    do_function_list: List[List[DoFunctions]],
    posterior: np.ndarray,
) -> List[GPyModelWrapper]:
    """
    Updates the GP posterior for the set(s) intervened on in the previous timestep, using do_functions aggregated over all graphs.
    """
    # update the Gaussian Processes
    if trial_observed:
        # update all the models if we observed in the previous trial
        # this one uses the computed do functions by fitting the causal graph
        for j in range(len(exploration_set)):
            logging.info(f"Updating posterior for {exploration_set[j]}")
            X = data_x_list[j]
            Y = data_y_list[j].reshape(-1, 1)
            mean_function = partial(
                aggregate_mean_function, j, do_function_list, posterior
            )
            var_function = partial(
                aggregate_var_function, j, do_function_list, posterior
            )
            model_list[j] = set_up_GP(
                causal_prior, input_space[j], mean_function, var_function, X, Y
            )
    else:
        # only update the model of the set that was intervened upon
        logging.info(f"Updating posterior for {exploration_set[best_variable]}")
        X = data_x_list[best_variable]
        Y = data_y_list[best_variable].reshape(-1, 1)
        mean_function = partial(
            aggregate_mean_function, best_variable, do_function_list, posterior
        )
        var_function = partial(
            aggregate_var_function, best_variable, do_function_list, posterior
        )
        model_list[best_variable] = set_up_GP(
            causal_prior, input_space[best_variable], mean_function, var_function, X, Y
        )

    return model_list


def update_posterior_model_aggregate_2(
    exploration_set: List,
    trial_observed: bool,
    model_list: List[GPyModelWrapper],
    data_x_list: dict,
    data_y_list: dict,
    causal_prior: bool,
    best_variable: int,
    input_space: List,
    do_function_list: List[List[DoFunctions]],
    posterior: np.ndarray,
    kernel_type: str = "rbf",
) -> List[GPyModelWrapper]:
    """
    Updates the GP posterior for the set(s) intervened on in the previous timestep, using do_functions aggregated over all graphs.
    """
    # update the Gaussian Processes
    # update all the models if we observed in the previous trial
    # this one uses the computed do functions by fitting the causal graph
    for j in range(len(exploration_set)):
        logging.info(f"Updating posterior for {exploration_set[j]}")
        X = data_x_list[j]
        Y = data_y_list[j].reshape(-1, 1)
        mean_function = partial(aggregate_mean_function, j, do_function_list, posterior)
        var_function = partial(aggregate_var_function, j, do_function_list, posterior)
        model_list[j] = set_up_GP(
            causal_prior,
            input_space[j],
            mean_function,
            var_function,
            X,
            Y,
            kernel_type=kernel_type,
        )

    return model_list


def update_arm_distribution(
    arm_distribution: np.ndarray,
    bo_models: List[GPyModelWrapper],
    inputs: List,
    arm_mapping_n_es: Dict[Tuple, int],
    beta=0.1,
) -> np.ndarray:
    """
    The arm distribution is the probability that the current exploration set is the optimal
    exploration set
    """
    for i in range(len(bo_models)):
        es = arm_mapping_n_es[i]
        inp = inputs[i]
        preds_mean, preds_var = bo_models[i].predict(inp)
        # assuming that the task is to take the minimum
        min_val = np.argmin(preds_mean)
        arm_distribution[i] = preds_mean[min_val] - beta * preds_var[min_val]

    return arm_distribution


def update_arm_dist_single_model(
    arm_distribution,
    es,
    single_updated_bo_model,
    inputs,
    arm_mapping_es_to_n,
    beta=0.1,
):
    corresponding_n = arm_mapping_es_to_n[es]
    inps = np.array(inputs)
    preds_mean, preds_var = single_updated_bo_model.predict(inps)  # Predictive mean
    arm_distribution[corresponding_n] = np.min(preds_mean) - beta * np.sqrt(
        preds_var[np.argmin(preds_mean)]
    )

    return arm_distribution


def build_p_y_star(
    exploration_set: List[str],
    bo_models: List[GPyModelWrapper],
    int_grids,
    parameter_int_domain,
    n_samples: int = 200,
) -> Tuple[np.ndarray, List]:
    """
    Building an optimal value for every element in the exploration set
    """
    # sets = bo_models.keys()
    parameter_int_domain = {
        tuple(sorted(key)): value for key, value in parameter_int_domain.items()
    }
    all_ystar = np.empty(shape=(len(exploration_set), n_samples))
    all_xstar = [[] for _ in range(len(exploration_set))]

    for i, es in enumerate(exploration_set):
        emukit_model: GPyModelWrapper = bo_models[i]
        gpy_model: GPRegression = emukit_model.model

        if len(es) > 1:
            # can change this to sample uniformly
            inps = parameter_int_domain[tuple(es)]
            inps = np.array(inps).reshape(-1, len(es))
        else:
            inps = parameter_int_domain[tuple(es)]
            inps = np.array(inps).reshape(-1, 1)

        # this is different from the one used in the github code
        samples = gpy_model.posterior_samples(inps, size=n_samples).squeeze()
        all_ystar[i, :] = np.min(samples, axis=0).squeeze()
        all_xstar[i] = inps[np.argmin(samples, axis=0), :].squeeze()

    return all_ystar, all_xstar


def update_pystar_single_model(
    arm_mapping: dict,
    es: Tuple,
    bo_model: GPyModelWrapper,
    inputs,
    all_ystar,
    all_xstar,
):
    """
    Update the samples for the optimal values based on the current elements in the
    exploration set (only changes one of the posterior probabilities)
    """
    corresponding_idx = arm_mapping[es]
    n_samples = all_ystar.shape[1]  # samples to build local p(y*, x*)
    gpy_model: GPRegression = bo_model.model
    inps = np.array(inputs)
    samples = gpy_model.posterior_samples_f(
        inps, size=n_samples
    )  # less samples to speed up
    samples = samples.squeeze()

    all_ystar[corresponding_idx, :] = np.min(
        samples, axis=0
    )  # NOTE: it ispod  really important all_ystar is the previouss one ! This is an UPDATE move

    return all_ystar, all_xstar  # used only for plotting so not tracking x for now


def sample_global_xystar(
    n_samples_mixture: int, all_ystar: np.ndarray, arm_dist: np.ndarray
):
    """
    This calculate p(y*|D) = sum P(X_I = X*) P(yI*|D)
    It does this by sampling from the conditional distribution
    """
    # Select indexes of mixture components to sample from first
    mixture_idxs = stratified(
        W=to_prob(arm_dist), M=n_samples_mixture
    )  # lower variance sampling

    local_pystars: List[MyKDENew] = []
    # This fits a KDE to each local p(Y*) i.e. p(Y*_(Z) | D), p(Y*_(X) | D)
    # It looks at all the elements in the exploration set
    for mixt_idx in range(all_ystar.shape[0]):
        temp = all_ystar[mixt_idx, :].reshape(-1, 1)
        kde = MyKDENew(temp)
        try:
            kde.fit()
        except RuntimeError:
            kde.fit(bw=0.5)

        local_pystars.append(kde)

    resy = np.empty(mixture_idxs.shape[0])

    unique_mixture_idxs, counts = np.unique(mixture_idxs, return_counts=True)
    running_cumsum = 0

    corresponding_x = []  # TODO

    for j, (mix_id, count) in enumerate(zip(unique_mixture_idxs, counts)):
        if j == 0:
            resy[:count], _ = local_pystars[mix_id].sample(n_samples=count)
        else:
            resy[running_cumsum : running_cumsum + count], _ = local_pystars[
                mix_id
            ].sample(n_samples=count)
        running_cumsum += count

    return resy, corresponding_x


def normalize_log(l):
    return np.exp(l - logsumexp(l))


def get_monte_carlo_expectation(intervention_samples: dict):
    """
    Computes the interventional expectation based on the data
    """
    assert isinstance(intervention_samples, dict)
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = intervention_samples[es].mean(axis=0)

    # Returns the expected value of the intervention via MC sampling
    return new


def optimal_sequence_of_interventions(
    exploration_sets: List,
    interventional_grids: Dict,
    graph: GraphStructure,
    task: str = "min",
) -> Tuple:
    """
    This is to start off the CEO algorithm, don't think it is used, rather used the one
    in the CBO algorithm
    """

    # setting up the noise model
    model_variables = graph.variables
    # static_noise_model = {k: 0 for k in model_variables}
    static_noise_model = graph.get_error_distribution()

    target_variable = graph.target
    SEM = graph.SEM
    assert target_variable is not None
    assert target_variable in model_variables

    best_s_sequence = []
    best_s_values = []
    best_objective_values = []

    optimal_interventions = {setx: [] for setx in exploration_sets}

    y_stars = deepcopy(optimal_interventions)
    all_CE = []
    blank_intervention_blanket = {node: [] for node in graph.nodes}

    CE = {es: [] for es in exploration_sets}

    for s in exploration_sets:

        # Reset blanket so as to not carry over levels from previous exploration set
        intervention_blanket = {}
        for level in interventional_grids[s]:

            # Univariate intervention
            if len(s) == 1:
                intervention_blanket[s[0]] = float(level[0])

            # Multivariate intervention
            else:
                for var, val in zip(s, level):
                    intervention_blanket[var] = val

            intervention_samples = sample_model(
                SEM,
                interventions=intervention_blanket,
                sample_count=10,
                epsilon=static_noise_model,
            )

            out = get_monte_carlo_expectation(intervention_samples)
            CE[s].append((out[target_variable][0]))

    local_target_values = []
    for s in exploration_sets:
        if task == "min":
            idx = np.array(CE[s]).argmin()
        else:
            idx = np.array(CE[s]).argmax()
        local_target_values.append((s, idx, CE[s][idx]))
        y_stars[s] = CE[s][idx]
        optimal_interventions[s] = interventional_grids[s][idx]

    # Find best intervention at time t
    best_s, best_idx, best_objective_value = min(
        local_target_values, key=lambda t: t[2]
    )
    best_s_value = interventional_grids[best_s][best_idx]

    best_s_sequence.append(best_s)
    best_s_values.append(best_s_value)
    best_objective_values.append(best_objective_value)
    all_CE.append(CE)

    return (
        best_s_values,
        best_s_sequence,
        best_objective_values,
        y_stars,
        optimal_interventions,
        all_CE,
    )


def numerical_optimization(
    acquisition: Acquisition,
    inputs: np.ndarray,
    exploration_set: List[str],
    task: str = "min",
):
    """
    Does the optimization for the CEO acquisition function
    """
    logging.info("Starting the optimization for CES")
    # Finds the new best point by evaluating the function in a set of given inputs
    _, D = inputs.shape
    improvements = acquisition.evaluate(inputs)
    # Is this correct ?
    # if task == "min":
    #     idx = np.argmax(improvements)
    # else:
    #     idx = np.argmin(improvements)
    # i think it should always be argmax
    idx = np.argmax(improvements)

    # Get point with best improvement, the x new should be taken from the inputs
    x_new = inputs[idx]
    y_new = improvements[idx]
    # Reshape point
    if len(x_new.shape) == 1 and len(exploration_set) == 1:
        x_new = x_new.reshape(-1, 1)
    elif len(exploration_set) > 1 and len(x_new.shape) == 1:
        x_new = x_new.reshape(1, -1)
    else:
        raise ValueError(
            "The new point is not an array. Or something else fishy is going on."
        )

    # TODO: consider removing
    if x_new.shape[0] == D:
        # The function make_column_shape_2D might convert a (D, ) array in a (D,1) array that needs to be reshaped
        x_new = np.transpose(x_new)

    assert x_new.shape[1] == inputs.shape[1], "New point has a wrong dimension"

    return x_new, y_new, inputs, improvements


def fake_do_x(
    x: np.ndarray,
    graphs: List[GraphStructure],
    log_graph_post: np.ndarray,
    intervened_vars: Tuple[str],
    all_sem: OrderedDict,
    # all_emission_fncs,
):
    """
    Get interventional samples based on the estimated functions, these functions are
    estimated based on the graph structure of the variables
    """
    # Get a set of all variables

    # This will hold the fake intervention
    intervention_blanket = {k: np.array([None]).reshape(-1, 1) for k in intervened_vars}

    for i, intervened_var in enumerate(intervened_vars):
        intervention_blanket[intervened_var] = np.array(x.reshape(1, -1)[0, i]).reshape(
            -1, 1
        )
    # Better than  MAP
    posterior_to_avg = []
    for idx_graph in range(len(all_sem)):
        sem_hat = all_sem[idx_graph]
        # print(intervention_blanket)
        interv_sample = sample_from_SEM_hat(
            static_sem=sem_hat,
            graph=graphs[idx_graph],
            interventions=intervention_blanket,
            epsilon=graphs[idx_graph].get_error_distribution(),
        )

        # In theory could/should replace Y with sample from surrogate model
        for var, val in interv_sample.items():
            interv_sample[var] = val.reshape(-1, 1)

        interv_sample = {intervened_vars: interv_sample}
        # P(G | D, (x,y) )  . avg over V_y  =  V \ (x,y)

        # maybe check this again when there is a different number of interventions at one timestep
        posterior_to_avg.append(
            normalize_log(
                update_posterior_interventional(
                    graphs=graphs,
                    posterior=deepcopy(log_graph_post),
                    intervened_var=intervened_vars,
                    all_emission_fncs=all_sem,
                    interventional_samples=interv_sample,
                )
            )
        )

    posterior_to_avg = np.vstack(posterior_to_avg)
    # Average over intervention outcomes
    return np.average(posterior_to_avg, axis=0, weights=log_graph_post)
