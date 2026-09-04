import logging
import random
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, OrderedDict, Tuple, Union

import numpy as np
import statsmodels.api as sm
from emukit.bayesian_optimization.interfaces import IEntropySearchModel
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression
from scipy.stats import entropy
from tqdm import tqdm

import utils.ceo_utils as ceo_utils
from graphs.graph import GraphStructure
from graphs.graph_functions import create_grid_interventions
from utils.cbo_classes import Cost, DoFunctions
from utils.cbo_functions import set_up_GP


def evaluate_acquisition_ceo(
    # these are the ones i defined
    graphs: List[GraphStructure],
    bo_model: GPyModelWrapper,
    exploration_set: List[List[str]],
    cost_functions: OrderedDict,
    posterior: np.ndarray,
    # these are taken from the code
    num_anchor_points: int = 30,
    sample_anchor_points: bool = False,
    seed_anchor_points=None,
    # NEW CEO STUFF. TODO: PASS A DICT AND MAKE IT INTO KWARGS
    all_sem_hat=None,
    # all_emit_fncs=None, # i think we can do it without this
    # Local and global posterior over y* stuff
    kde_globalystar: ceo_utils.MyKDENew = None,
    pxstar_samples: List = None,
    pystar_samples: np.ndarray = None,
    samples_global_ystar: np.ndarray = None,
    samples_global_xstar: List = None,
    interventional_grid=None,
    # Arm stuff
    arm_distribution: np.ndarray = None,
    arm_mapping_es_to_num: Dict = None,
    arm_mapping_num_to_es: Dict = None,
):
    """
    This function does the optimization of the acquisition function for the entropy search
    All the parameters are calculated and are necessary for the algorithm
    """
    # get the parameter space from the graph data structure
    parameter_intervention_domain: ParameterSpace = graphs[0].get_parameter_space(
        exploration_set
    )
    dim = parameter_intervention_domain.dimensionality
    cost_of_acquisition = Cost(cost_functions, exploration_set)
    acquisition = (
        CausalEntropySearch(
            all_sem_hat=all_sem_hat,
            space=parameter_intervention_domain,
            # all_emit_fncs=all_emit_fncs,
            graphs=graphs,
            current_posterior=posterior,
            es=exploration_set,
            model=bo_model,
            # space=parameter_intervention_domain,
            kde=kde_globalystar,
            interventional_grid=interventional_grid,
            es_num_arm_mapping=arm_mapping_es_to_num,
            num_es_arm_mapping=arm_mapping_num_to_es,
            arm_distr=arm_distribution,
            seed=seed_anchor_points,
            # task=task,
            all_xstar=pxstar_samples,
            all_ystar=pystar_samples,
            samples_global_ystar=samples_global_ystar,
            samples_global_xstar=samples_global_xstar,
        )
        / cost_of_acquisition
    )

    if dim > 1:
        num_anchor_points = int(np.sqrt(num_anchor_points))

    if sample_anchor_points:
        # This is to ensure the points are different every time we call the function
        if seed_anchor_points is not None:
            np.random.seed(seed_anchor_points)
        else:
            np.random.seed()

        sampled_points = parameter_intervention_domain.sample_uniform(
            point_count=num_anchor_points
        )
    else:
        limits = [list(tup) for tup in parameter_intervention_domain.get_bounds()]
        sampled_points = ceo_utils.create_n_dimensional_intervention_grid(
            limits=limits, size_intervention_grid=num_anchor_points
        )

    x_new, y_acquisition, inputs, improvements = ceo_utils.numerical_optimization(
        acquisition, sampled_points, exploration_set
    )
    y_acquisition = np.asarray([y_acquisition]).reshape(-1, 1)

    logging.debug(f"Found {x_new} with acquisition value of {y_acquisition}")
    return y_acquisition, x_new, inputs, improvements


def get_new_x_y_list_ceo(
    graphs: List[GraphStructure],
    model_list: List[GPyModelWrapper],
    exploration_set: List[List[str]],
    arm_distribution: np.ndarray,
    cost_functions,
    posteriors,
    data_x_list,
    data_y_list,
    all_sem_hat: List[OrderedDict[str, GPRegression]],
    n_anchor_points: int,
) -> Tuple[np.ndarray, List[List[float]], np.ndarray]:
    """
    Use the acquisition of the CEO method
    """
    arm_n_es_mapping = {i: es for i, es in enumerate(exploration_set)}
    arm_es_n_mapping = {es: i for i, es in enumerate(exploration_set)}
    graph = graphs[0]  # just to get some properties from the graph
    interventional_range = graph.get_interventional_range()
    intervention_grid = create_grid_interventions(
        interventional_range, get_list_format=True, num_points=n_anchor_points
    )

    arm_distribution = ceo_utils.update_arm_distribution(
        arm_distribution, model_list, data_x_list, arm_n_es_mapping
    )

    print(intervention_grid)
    py_star_samples, p_x_star_samples = ceo_utils.build_p_y_star(
        exploration_set,
        model_list,
        interventional_range,
        intervention_grid,
    )

    samples_global_ystar, samples_global_xstar = ceo_utils.sample_global_xystar(
        n_samples_mixture=1000,
        all_ystar=py_star_samples,
        arm_dist=arm_distribution,
    )

    logging.info("Fitting the global KDE estimate")
    kde_global = ceo_utils.MyKDENew(samples_global_ystar)
    try:
        kde_global.fit()
    except RuntimeError:
        kde_global.fit(bw=0.5)

    y_acquisition_list = [None] * len(exploration_set)
    x_new_list = [None] * len(exploration_set)
    inputs = [None] * len(exploration_set)
    improvements = [None] * len(exploration_set)
    logging.info("Starting with the entropy search")
    for s, es in enumerate(exploration_set):
        # figure out the sem_hat and sem_ems_fncs
        # not sure what to do with inputs and improvements
        y_acquisition_list[s], x_new_list[s], inputs[s], improvements[s] = (
            evaluate_acquisition_ceo(
                graphs=graphs,
                bo_model=model_list[s],
                exploration_set=es,
                cost_functions=cost_functions,
                posterior=posteriors,
                arm_distribution=arm_distribution,
                pystar_samples=py_star_samples,
                pxstar_samples=p_x_star_samples,
                samples_global_ystar=samples_global_ystar,
                samples_global_xstar=samples_global_xstar,
                kde_globalystar=kde_global,
                arm_mapping_es_to_num=arm_es_n_mapping,
                arm_mapping_num_to_es=arm_n_es_mapping,
                interventional_grid=intervention_grid,
                all_sem_hat=all_sem_hat,
            )
        )

    return y_acquisition_list, x_new_list, arm_distribution


class CausalEntropySearch(Acquisition):
    def __init__(
        self,
        all_sem_hat: Dict[str, Callable],
        # all_emit_fncs,
        graphs: List[GraphStructure],
        # node_parents,
        current_posterior: np.ndarray,
        es: List[str],
        model: Union[IModel, IEntropySearchModel],
        space: ParameterSpace,
        interventional_grid,
        kde: ceo_utils.MyKDENew,
        es_num_arm_mapping: Dict[int, str],
        num_es_arm_mapping: Dict[str, int],
        arm_distr: np.ndarray,
        seed: int,
        all_xstar: List,
        all_ystar: np.ndarray,
        samples_global_ystar: np.ndarray,
        samples_global_xstar: List,
        task: str = "min",
        # do_cdcbo=False,
    ) -> None:
        """
        This is the class for the acquisition function
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        self.es = es
        self.model = model
        self.space = space
        self.grid = interventional_grid
        self.pre_kde = kde
        self.es_num_mapping = es_num_arm_mapping
        self.num_es_arm_mapping = num_es_arm_mapping
        self.prev_arm_distr = arm_distr
        self.seed = seed
        self.task = task
        self.init_posterior = current_posterior
        # self.node_parents = node_parents
        self.graphs = graphs
        self.all_sem_hat = all_sem_hat
        # self.all_emit_fncs = all_emit_fncs
        self.prev_all_ystar = all_ystar
        self.prev_all_xstar = all_xstar
        self.prev_global_samples_ystar = samples_global_ystar
        self.prev_global_samples_xstar = samples_global_xstar
        # self.do_cdcbo = do_cdcbo

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain: predicted change in entropy of p_min if we evaluate x.
        """

        # Make new aquisition points

        # grid = self.grid if len(self.es) == 1 else x  # this was wrong
        grid = self.grid

        initial_entropy = self.pre_kde.entropy  # A scalar really
        initial_graph_entropy = entropy(ceo_utils.normalize_log(self.init_posterior))
        n_fantasies = 5  # N. of fantasy observations
        # could  choose a subset of them to reduce computation
        n_acquisitions = x.shape[0]

        n_samples_mixture = self.prev_global_samples_ystar.shape[0]
        # first dimension is n anchor points
        new_entropies = np.empty((n_acquisitions,))
        # first dimension is n anchor points
        new_entropies_opt = np.empty((n_acquisitions,))
        # first dimension is n anchor points
        new_entropies_graph = np.empty((n_acquisitions,))

        # Stores the new samples from the updated p(y* | D, (x,y)).
        new_samples_global_ystar_list = np.empty(
            (n_acquisitions, n_fantasies, n_samples_mixture)
        )

        # Keeping track of these just because of plotting later
        # shape will be n_acquisitions x n_fantasies
        updated_models_list = [[] for _ in range(n_acquisitions)]

        const = np.pi**-0.5

        # Approx integral with GQ
        xx, w = np.polynomial.hermite.hermgauss(n_fantasies)

        curr_normalized_graph = ceo_utils.normalize_log(self.init_posterior)

        # not sure why you would only consider entry 0 here
        if curr_normalized_graph[0] > 0.90:
            logging.info("graph is found")
            # If you found the graph, optimize
            for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                # Get samples from p(y | D, do(x) )
                if single_x.shape[0] == 1:
                    x_inp = single_x.reshape(-1, 1)
                else:
                    x_inp = single_x.reshape(1, -1)

                m, v = self.model.predict(x_inp)
                m, v = m.squeeze(), v.squeeze()

                # Fantasy samples from sigma points xx
                fantasy_ys = 2**0.5 * np.sqrt(v) * xx + m

                new_entropies_unweighted = np.empty((n_fantasies,))

                for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                    updated_model = deepcopy(self.model)
                    prevx, prevy = updated_model.X, updated_model.Y

                    tempx = np.concatenate([prevx, x_inp])

                    fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(-1, 1)

                    tempy = np.vstack([prevy, fantasy_y])

                    updated_model.model.set_XY(tempx, tempy)

                    # Keeping track of them just for plotting ie. debugging reasons
                    updated_models_list[id_acquisition].append(updated_model)

                    # Arm distr gets updated only because model gets updated
                    new_arm_dist = ceo_utils.update_arm_dist_single_model(
                        arm_distribution=deepcopy(self.prev_arm_distr),
                        es=self.es,
                        single_updated_bo_model=updated_model,
                        inputs=grid[self.es],
                        arm_mapping_es_to_n=self.es_num_mapping,
                    )

                    pystar_samples, _ = ceo_utils.update_pystar_single_model(
                        arm_mapping=self.es_num_mapping,
                        es=self.es,
                        bo_model=updated_model,
                        inputs=grid[self.es],
                        all_xstar=self.prev_all_xstar,
                        all_ystar=deepcopy(self.prev_all_ystar),
                    )

                    new_samples_global_ystar, _ = ceo_utils.sample_global_xystar(
                        n_samples_mixture=n_samples_mixture,
                        all_ystar=pystar_samples,
                        arm_dist=ceo_utils.to_prob(
                            new_arm_dist, self.task  # checked , this works for min
                        ),
                    )

                    new_kde = ceo_utils.MyKDENew(new_samples_global_ystar)
                    try:
                        new_kde.fit()
                    except RuntimeError:
                        new_kde.fit(bw=0.5)

                    # this can be neg. as it's differential entropy
                    new_entropy_ystar = new_kde.entropy

                    new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                    new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = (
                        new_samples_global_ystar
                    )

                # GQ average
                new_entropies[id_acquisition] = np.sum(
                    w * const * new_entropies_unweighted
                )

                # Remove  when debugging with  batch
            assert new_entropies.shape == (n_acquisitions,) or new_entropies == (
                n_acquisitions,
                1,
            )
            # Represents the improvement in (averaged over fantasy observations!) entropy (it's good if it lowers)
            # It can be negative.
            entropy_changes = initial_entropy - new_entropies

        else:
            # still some uncertainty with regards to what is the right graph
            logging.info("graph is not found")

            # if not self.do_cdcbo:
            # Keep finding graph and optimize JOINTLY
            intervened_vars = [s for s in self.es]
            # Calc updated graph entropy
            for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
                if single_x.shape[0] == 1:
                    x_inp = single_x.reshape(-1, 1)
                else:
                    x_inp = single_x.reshape(1, -1)
                # print("+++++++++++++++++++++++++++++++++++++++")
                # print(self.init_posterior)
                updated_posterior = ceo_utils.fake_do_x(
                    x=x_inp,
                    graphs=self.graphs,
                    log_graph_post=deepcopy(self.init_posterior),
                    intervened_vars=tuple(intervened_vars),
                    all_sem=self.all_sem_hat,
                )
                # print(updated_posterior)
                new_entropies_graph[id_acquisition] = entropy(
                    ceo_utils.normalize_log(updated_posterior)
                )

            entropy_changes_graph = initial_graph_entropy - new_entropies_graph

            # Optimization part
            for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                # Get samples from p(y | D, do(x) )
                if single_x.shape[0] == 1:
                    x_inp = single_x.reshape(-1, 1)
                else:
                    x_inp = single_x.reshape(1, -1)

                m, v = self.model.predict(x_inp)
                m, v = m.squeeze(), v.squeeze()

                # Fantasy samples from sigma points xx
                fantasy_ys = 2**0.5 * np.sqrt(v) * xx + m

                new_entropies_unweighted = np.empty((n_fantasies,))

                for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                    updated_model: GPyModelWrapper = deepcopy(self.model)
                    prevx, prevy = updated_model.model.X, updated_model.model.Y

                    tempx = np.concatenate([prevx, x_inp])

                    fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(-1, 1)

                    tempy = np.vstack([prevy, fantasy_y])

                    updated_model.model.set_XY(tempx, tempy)

                    # Keeping track of them just for plotting ie. debugging reasons
                    updated_models_list[id_acquisition].append(updated_model)

                    # Arm distr gets updated only because model gets updated
                    # if self.es == ("X", "Z"):
                    #     print(grid)
                    new_arm_dist = ceo_utils.update_arm_dist_single_model(
                        deepcopy(self.prev_arm_distr),
                        self.es,
                        updated_model,
                        grid[self.es],
                        self.es_num_mapping,
                    )

                    # Use this to build p(y*, x* | D, (x,y) )
                    pystar_samples, _ = ceo_utils.update_pystar_single_model(
                        arm_mapping=self.es_num_mapping,
                        es=self.es,
                        bo_model=updated_model,
                        inputs=grid[self.es],
                        all_xstar=self.prev_all_xstar,
                        all_ystar=deepcopy(self.prev_all_ystar),
                    )

                    new_samples_global_ystar, _ = ceo_utils.sample_global_xystar(
                        n_samples_mixture=n_samples_mixture,
                        all_ystar=pystar_samples,
                        arm_dist=ceo_utils.to_prob(
                            new_arm_dist,  # checked , this works for min
                            self.task,
                        ),
                    )

                    new_kde = ceo_utils.MyKDENew(new_samples_global_ystar)
                    try:
                        new_kde.fit()
                    except RuntimeError:
                        new_kde.fit(bw=0.5)

                    new_entropy_ystar = (
                        new_kde.entropy
                    )  # this can be neg. as it's differential entropy

                    new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                    new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = (
                        new_samples_global_ystar
                    )

                # GQ average
                new_entropies_opt[id_acquisition] = np.sum(
                    w * const * new_entropies_unweighted
                )

            entropy_changes_opt = initial_entropy - new_entropies_opt

            entropy_changes = entropy_changes_graph + entropy_changes_opt
            # entropy_changes = entropy_changes_opt
            # else:
            #     # CD-CBO: only graph !
            #     # Keep finding graph and optimize jointly
            #     intervened_vars = [s for s in self.es]
            #     # Calc updated graph entropy
            #     for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
            #         if single_x.shape[0] == 1:
            #             x_inp = single_x.reshape(-1, 1)
            #         else:
            #             x_inp = single_x.reshape(1, -1)

            #         updated_posterior = ceo_utils.fake_do_x(
            #             x=x_inp,
            #             graphs=self.graphs,
            #             log_graph_post=deepcopy(self.init_posterior),
            #             intervened_vars=intervened_vars,
            #             all_sem=self.all_sem_hat,
            #         )
            #         new_entropies[id_acquisition] = entropy(
            #             ceo_utils.normalize_log(updated_posterior)
            #         )

            #     entropy_changes = initial_graph_entropy - new_entropies

        # end of inner if
        # end of outer if

        # Just in case any are negative, shift all, preserving the total order.
        if np.any(entropy_changes < 0.0):
            smallest = np.absolute(np.min(entropy_changes))
            entropy_changes = entropy_changes + smallest

        logging.info("Entropy changes for " + str(self.es) + ": ")
        logging.info(str(entropy_changes.tolist()))
        assert entropy_changes.shape[0] == x.shape[0]

        # manually changing the entropy values
        entropy_changes = np.nan_to_num(entropy_changes, nan=0.0)
        return entropy_changes

    # def worker_function(x):

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
