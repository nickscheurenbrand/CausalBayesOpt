import logging
import random
from collections import OrderedDict, namedtuple
from itertools import combinations, islice, product
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
from GPy.models.gp_regression import GPRegression

from graphs.graph import GraphStructure

Data = namedtuple("Data", ["samples", "intervention_node"])


# removed the dynamic component as I am only considering static SEMs
# I also don't think I need this timestep component to it
def sample_from_SEM(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon: dict = None,
    seed: int = None,
    std: float = 0.1,
) -> OrderedDict:
    """Draw one sample by walking `static_sem` in insertion order, which must be topological.
    Each variable is set from `interventions`/`initial_values`, else its SEM function."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Initialize noise model
    if epsilon is None:
        epsilon = {k: rng.normal(scale=std) for k in static_sem.keys()}
    assert isinstance(epsilon, dict)

    # Initialize sample
    sample = OrderedDict()

    for var, function in static_sem.items():
        # Apply interventions or initial values if specified
        if interventions and var in interventions:
            sample[var] = interventions[var]
        elif initial_values and var in initial_values:
            sample[var] = initial_values[var]
        else:
            # Otherwise, sample from the model using the specified noise
            sample[var] = function(epsilon[var], sample)

    return sample


def sample_from_SEM_iscm(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon: dict = None,
    seed: int = None,
    std: float = 0.1,
    graph: GraphStructure = None,
) -> OrderedDict:
    """Draw one sample by walking `static_sem` in insertion order, which must be topological.
    Each variable is set from `interventions`/`initial_values`, else its SEM function."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Initialize noise model
    if epsilon is None:
        epsilon = {k: rng.normal(scale=std) for k in static_sem.keys()}
    assert isinstance(epsilon, dict)

    # Initialize sample
    sample = OrderedDict()

    for var, function in static_sem.items():
        # Apply interventions or initial values if specified
        if interventions and var in interventions:
            sample[var] = interventions[var]
        elif initial_values and var in initial_values:
            sample[var] = initial_values[var]
        else:
            # Otherwise, sample from the model using the specified noise
            sample[var] = (
                function(epsilon[var], sample) - graph.iscm_paramters[var]["mean"]
            ) / graph.iscm_paramters[var]["std"]

    return sample


def sample_from_SEM_hat(
    static_sem: OrderedDict[str, GPRegression],
    graph: GraphStructure,
    initial_values: dict = None,
    interventions: dict = None,
    seed: int = None,
    epsilon: Dict[str, float] = None,
) -> OrderedDict:
    """Like `sample_from_SEM` but `static_sem` holds fitted GPs (estimated SEM), sampled in
    `graph`'s topological order and predicted from parent values using the GP posterior mean."""
    # if seed:
    #     np.random.seed(seed)
    assert epsilon is not None
    sample = OrderedDict()
    topological_order = list(nx.topological_sort(graph.G))

    for var in topological_order:
        # Apply interventions or initial values if specified
        if interventions and var in interventions:
            value = interventions[var]
        elif initial_values and var in initial_values:
            value = initial_values[var]
        else:
            # Find parents and sample from the model
            parents = graph.parents[var]
            if parents:
                gp_function = static_sem[var]
                # Create parent matrix for GP
                parent_values = np.array([[sample[parent] for parent in parents]])
                # Predict from GP model, mean of distribution used as sample
                mean = gp_function.predict(parent_values)[0][0]
                value = mean.squeeze()
            else:
                # For source nodes, you might need a default sampling strategy
                # or handle as no-parent scenario typically with some prior
                function = static_sem[var]
                mean = function.predict()[0]
                value = mean.squeeze()

            # Convert arrays to scalar if necessary
            # Ensure value is a scalar
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()  # Convert one-element array to scalar

        sample[var] = np.array(value)

    return sample


def sample_model(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    node_parents: Callable = None,
    sample_count: int = 100,
    epsilon: dict = None,
    use_sem_estimate: bool = False,
    seed: int = None,
    graph: GraphStructure = None,
    noiseless: bool = False,
    use_iscm: bool = False,
) -> dict:
    """Draws `sample_count` samples from the SEM, one call to sample_from_SEM(_iscm/_hat) each.
    Returns {var: (n_samples, timesteps) array}."""

    if seed:
        np.random.seed(seed)

    new_samples = {k: [] for k in static_sem.keys()}
    for _ in range(sample_count):
        # This option uses the estimates of the SEMs, estimates found through use of GPs.
        if use_sem_estimate:
            tmp = sample_from_SEM_hat(
                static_sem=static_sem,
                node_parents=node_parents,
                initial_values=initial_values,
                interventions=interventions,
            )
        # This option uses the true SEMs.
        else:
            if graph is not None:
                epsilon_term = graph.get_error_distribution(noiseless=noiseless)
            else:
                epsilon_term = epsilon

            if use_iscm:
                tmp = sample_from_SEM_iscm(
                    static_sem=static_sem,
                    initial_values=initial_values,
                    interventions=interventions,
                    epsilon=epsilon_term,
                    seed=seed,
                    graph=graph,
                )
            else:
                tmp = sample_from_SEM(
                    static_sem=static_sem,
                    initial_values=initial_values,
                    interventions=interventions,
                    epsilon=epsilon_term,
                    seed=seed,
                )
        for var in static_sem.keys():
            new_samples[var].append(tmp[var])

    for var in static_sem.keys():
        new_samples[var] = np.vstack(new_samples[var])

    return new_samples


def sample_model_set_icm_params(
    static_sem: OrderedDict,
    initial_values: dict = None,
    interventions: dict = None,
    node_parents: Callable = None,
    sample_count: int = 500,
    epsilon: dict = None,
    use_sem_estimate: bool = False,
    seed: int = None,
    graph: GraphStructure = None,
    noiseless: bool = False,
) -> dict:
    """Draws `sample_count` samples from the SEM, one call to sample_from_SEM(_iscm/_hat) each.
    Returns {var: (n_samples, timesteps) array}."""
    if seed:
        np.random.seed(seed)

    topological_order = list(nx.topological_sort(graph.G))

    for set_var in topological_order:
        new_samples = {k: [] for k in static_sem.keys()}
        for _ in range(sample_count):
            # This option uses the estimates of the SEMs, estimates found through use of GPs.
            if use_sem_estimate:
                tmp = sample_from_SEM_hat(
                    static_sem=static_sem,
                    node_parents=node_parents,
                    initial_values=initial_values,
                    interventions=interventions,
                )
            # This option uses the true SEMs.
            else:

                if graph is not None:
                    epsilon_term = graph.get_error_distribution(noiseless=noiseless)
                else:
                    epsilon_term = epsilon

                tmp = sample_from_SEM_iscm(
                    static_sem=static_sem,
                    initial_values=initial_values,
                    interventions=interventions,
                    epsilon=epsilon_term,
                    seed=seed,
                    graph=graph,
                )
            for var in static_sem.keys():
                new_samples[var].append(tmp[var])

        mean = np.mean(new_samples[set_var])
        std = np.std(new_samples[set_var])
        print(set_var, mean, std)
        graph.set_params_iscm(set_var, mean, std)
        for var in static_sem.keys():
            new_samples[var] = np.vstack(new_samples[var])

    return new_samples


def create_grid_interventions(
    ranges: OrderedDict,
    num_points: int = 20,
    include_full_combination: bool = True,
    get_list_format: bool = False,
) -> List:
    """Create individual and combined (up to `num_points` each) grid interventions per variable."""
    max_combinations = num_points
    grids = {
        var: np.linspace(min_val, max_val, num_points)
        for var, (min_val, max_val) in ranges.items()
    }

    interventions = []

    # Individual and combination interventions
    for size in range(1, len(ranges) + 1):
        for variable_subset in combinations(ranges.keys(), size):
            full_product = list(product(*(grids[var] for var in variable_subset)))

            if len(full_product) > max_combinations:
                sampled_product = random.sample(full_product, max_combinations)
            else:
                sampled_product = full_product

            for product_values in sampled_product:
                interventions.append(dict(zip(variable_subset, product_values)))

    # Optionally remove the full combination if not desired
    if not include_full_combination and len(ranges) > 1:
        interventions = [intv for intv in interventions if len(intv) < len(ranges)]

    new_grid = {}

    # Loop through each dictionary in the original grid
    for entry in interventions:
        keys = tuple(
            sorted(entry.keys())
        )  # Sort keys to ensure consistent order for tuples
        values = tuple(
            entry[key] for key in keys
        )  # Get the corresponding values in tuple form

        if keys in new_grid:
            new_grid[keys].append(values)
        else:
            new_grid[keys] = [values]

    if get_list_format:
        interventions = new_grid.copy()

    return interventions


def change_intervention_list_format(
    interventions: Dict, exploration_set: List[List[str]], target: str = "Y"
) -> Dict:
    """
    This is for the CEO so that the intervention list can be in
    two different formats
    """
    new_grid = {i: {} for i in range(len(exploration_set))}

    for i, es in enumerate(exploration_set):
        for val in es:
            new_grid[i][val] = interventions[es][val]
        new_grid[i][target] = interventions[es][target]
    return new_grid


def draw_interventional_samples(
    interventions: List[Dict[str, float]],
    exploration_set: List[List[str]],
    graph: GraphStructure,
    n_int: int = 2,
) -> dict:
    """Draw interventional samples per exploration-set intervention, returning only the target
    and intervened values -- effectively computing E[Y|do(X=x)]."""
    np.random.shuffle(interventions)

    interventional_data = {
        i: {var: [] for var in intervention}
        for i, intervention in enumerate(exploration_set)
    }
    target = graph.target

    # Add a 'target' key for each intervention type
    for i in interventional_data:
        interventional_data[i][target] = []

    for intervention in interventions:
        intervention_keys_sorted = sorted(intervention.keys())
        index = next(
            (
                i
                for i, sublist in enumerate(exploration_set)
                if sorted(sublist) == intervention_keys_sorted
            ),
            None,
        )
        if index is not None:
            sample = sample_model(graph.SEM, sample_count=1, interventions=intervention)
            for var in intervention:
                if len(interventional_data[index][var]) >= n_int:
                    continue
                interventional_data[index][var].append(sample[var][0, 0])

            if len(interventional_data[index][target]) >= n_int:
                continue
            interventional_data[index][target].append(sample[target][0, 0])

    # Convert lists to numpy arrays for easier manipulation and consistency
    for idx in interventional_data:
        for var in interventional_data[idx]:
            interventional_data[idx][var] = np.array(interventional_data[idx][var])

    return interventional_data


def draw_interventional_samples_sem(
    exploration_set: List[List[str]],
    graph: GraphStructure,
    n_int: int = 2,
    seed: int = None,
    noiseless: bool = True,
    use_iscm: bool = False,
) -> dict:
    """Like `draw_interventional_samples` but returns samples for every SEM variable, not just
    the target, after each intervention."""
    if seed is not None:
        np.random.seed(seed=seed)

    interventional_data = {
        tuple(es): {var: [] for var in graph.variables} for es in exploration_set
    }

    interventional_range = graph.get_interventional_range()
    for _ in range(n_int):
        for es in exploration_set:
            intervention = {}
            for var in es:
                intervention_sample = np.random.uniform(
                    interventional_range[var][0], interventional_range[var][1]
                )
                intervention[var] = intervention_sample

            # Drawing the samples
            sample = sample_model(
                graph.SEM,
                sample_count=1,
                interventions=intervention,
                graph=graph,
                noiseless=noiseless,
                use_iscm=use_iscm,
            )
            for var in sample:
                interventional_data[tuple(intervention)][var].append(sample[var][0, 0])

    for idx in interventional_data:
        for var in interventional_data[idx]:
            interventional_data[idx][var] = np.array(interventional_data[idx][var])
    return interventional_data


def change_obs_data_format_to_mi(
    D_O: Dict, graph_variables: List, intervention_node
) -> Data:
    """
    Change the data format from the format for the BO methods to the format needed for the
    the MI methods
    """
    D_O_mi = np.hstack([D_O[key].reshape(-1, 1) for key in D_O])
    # if intervention_node == -1:
    #     intervention_node = np.zeros(shape=len(graph_variables))
    # else:
    #     intervention_node = np.zeros(shape=len(graph_variables))
    #     for i, var in enumerate(graph_variables):
    #         if var in intervention_node:
    #             intervention_node[i] = 1
    return Data(samples=D_O_mi, intervention_node=intervention_node)


def change_obs_data_format_to_bo(D_O: Data, graph_variables: List) -> Dict:
    """
    Change the data format from the format for the MI methods to the format needed for the
    the BO methods
    """
    data = D_O.samples
    data_bo = {}
    for i, node_name in enumerate(graph_variables):
        data_bo[node_name] = data[:, i]
    return data_bo


def change_obs_data_format_to_bo_sergio(D_O: np.array, graph_variables: List) -> Dict:
    """
    Change the data format from the format for the MI methods to the format needed for the
    the BO methods
    """
    data = D_O
    data_bo = {}
    for i, node_name in enumerate(graph_variables):
        data_bo[node_name] = data[:, i, 0]
    return data_bo


def change_int_data_format_to_mi(D_I: Dict, graph_variables: List) -> List[Data]:
    # this only takes into account one intervention
    D_I_mi = []
    for key in D_I:
        if len(key) == 1:
            intervention_node = np.zeros(shape=len(graph_variables))
            for i, var in enumerate(graph_variables):
                if var in key:
                    intervention_node[i] = 1
            D_I_mi.append(
                change_obs_data_format_to_mi(
                    D_I[key], graph_variables, intervention_node=intervention_node
                )
            )

    return D_I_mi


def change_int_data_format_to_bo(D_I: Data, graph_variables: List) -> Dict:
    D_I_bo = {}
    intervention = D_I.intervention_node
    D_I_bo[intervention] = change_obs_data_format_to_bo(D_I, graph_variables)
    return D_I_bo
