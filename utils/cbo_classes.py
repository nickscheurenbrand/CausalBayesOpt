import logging
from typing import Callable, List, OrderedDict, Tuple, Union

import numpy as np
import scipy
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core.optimization.anchor_points_generator import (
    ObjectiveAnchorPointsGenerator,
)
from emukit.core.optimization.context_manager import ContextManager
from emukit.core.optimization.optimizer import (
    OptLbfgs,
    OptTrustRegionConstrained,
    apply_optimizer,
)
from GPy.core import Param
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.kern.src.stationary import Stationary
from paramz.transformations import Logexp

from graphs.graph import GraphStructure
from utils.sem_sampling import sample_model


class DoFunctions:
    """
    This class synthesizes all the do functions into one class
    """

    def __init__(
        self,
        do_effects_functions: OrderedDict,
        observational_samples: OrderedDict,
        intervention_variables: List,
        # graph: GraphStructure,
    ) -> None:

        self.do_effects_functions = do_effects_functions
        self.observational_samples = observational_samples
        self.intervention_variables = intervention_variables
        self.set_do_effects_function()
        # self.graph = graph
        # this acts as a cache if it was previously computed for that specific value
        self.xi_dict_mean = {}
        self.xi_dict_var = {}

    def get_do_function_name(self) -> str:
        """
        returns to name of the do function, based on which variables are intervened
        upon
        """
        string = ""
        for i in range(len(self.intervention_variables)):
            string += str(self.intervention_variables[i])
        do_function_name = "compute_do_" + string
        return do_function_name

    def set_do_effects_function(self) -> None:
        """
        Set the do function if the current list of intervention changes, so that it
        is correctly computed
        """
        self.do_effects_function = self.do_effects_functions[
            self.get_do_function_name()
        ]

    def mean_function_do(self, x) -> np.float64:
        """
        Calculates the interventional mean based on the specific value
        """
        num_interventions = x.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in self.xi_dict_mean:
                mean_do[i] = self.xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = self.do_effects_function(
                    observational_samples=self.observational_samples, value=x[i]
                )
                self.xi_dict_mean[xi_str] = mean_do[i]
        return np.float64(mean_do)

    def var_function_do(self, x) -> np.float64:
        """
        Calculates the interventional variance based on the specific x value
        """
        num_interventions = x.shape[0]
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in self.xi_dict_var:
                var_do[i] = self.xi_dict_var[xi_str]
            else:
                _, var_do[i] = self.do_effects_function(
                    observational_samples=self.observational_samples, value=x[i]
                )
                self.xi_dict_var[xi_str] = var_do[i]
        return np.float64(var_do)


class CausalRBF(Stationary):
    """
    This is the causal rabial basis kernel function which inherits from the
    stationary class, mostly taken from the CBO package, but made some small changes
    This is pretty much taken directly from the github page
    """

    def __init__(
        self,
        input_dim: int,
        variance_adjustment: Callable,
        variance: float = 1.0,
        lenghtscale: float = None,
        rescale_variance: float = 1.0,
        ARD: bool = False,
        active_dims=None,
        name: str = "rbf",
        useGPU: bool = False,
        use_invLengthscale: bool = False,
    ):

        super(CausalRBF, self).__init__(
            input_dim, variance, lenghtscale, ARD, active_dims, name, useGPU=useGPU
        )

        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()

        self.use_invLengthscale = use_invLengthscale
        if self.use_invLengthscale:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param("inv_lengthscale", 1.0 / self.lengthscale**2, Logexp())
            self.link_parameter(self.inv_l)

        self.variance_adjustment = variance_adjustment
        self.rescale_variance = Param("rescale_variance", rescale_variance, Logexp())

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(CausalRBF, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def K(self, X: np.ndarray, X2: np.ndarray = None):
        """
        Computes the kernel between X and X2
        Similar to the CBO paper where K(., .) = k_rbf + sigma * sigma
        """
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        value = self.K_of_r(r)

        # calculating the do-variance of the interventions
        value_diagonal_X = self.variance_adjustment(X)
        value_diagonal_X2 = (
            value_diagonal_X if X2 is None else self.variance_adjustment(X2)
        )

        additional_matrix = np.outer(
            np.sqrt(value_diagonal_X), np.sqrt(value_diagonal_X2)
        )
        return value + additional_matrix

    def Kdiag(self, X):
        """
        Slightly different from the implementation of CBO
        """
        ret = np.empty(X.shape[0])
        ret[:] = np.repeat(0.1, X.shape[0])

        diagonal_terms = ret

        value = self.variance_adjustment(X)

        if X.shape[0] == 1 and X.shape[1] == 1:
            diagonal_terms = value
        else:
            if np.isscalar(value) == True:
                diagonal_terms = value
            else:
                diagonal_terms = value[:, 0]
        return self.variance + diagonal_terms

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r**2 - 1) * self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance

    def __getstate__(self):
        dc = super(CausalRBF, self).__getstate__()
        if self.useGPU:
            dc["psicomp"] = PSICOMP_RBF()
            dc["useGPU"] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(CausalRBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1  # TODO: higher dim spectra?
        return (
            self.variance
            * np.sqrt(2 * np.pi)
            * self.lengthscale
            * np.exp(-self.lengthscale * 2 * omega**2 / 2)
        )

    def parameters_changed(self):
        if self.use_invLengthscale:
            self.lengthscale[:] = 1.0 / np.sqrt(self.inv_l + 1e-200)
        super(CausalRBF, self).parameters_changed()

    def get_one_dimensional_kernel(self, dim):
        """
        Specially intended for Grid regression.
        """
        oneDkernel = GridRBF(
            input_dim=1, variance=self.variance.copy(), originalDimensions=dim
        )
        return oneDkernel

    # ---------------------------------------#
    #             PSI statistics            #
    # ---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(
            self, Z, variational_posterior, return_psi2_n=False
        )[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(
            self, Z, variational_posterior, return_psi2_n=True
        )[2]

    def update_gradients_expectations(
        self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
    ):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
        )[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale * (self.lengthscale**3 / -2.0)

    def gradients_Z_expectations(
        self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
    ):
        return self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
        )[2]

    def gradients_qX_expectations(
        self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
    ):
        return self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
        )[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(CausalRBF, self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (
                self.lengthscale**3 / -2.0
            )

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(CausalRBF, self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (
                self.lengthscale**3 / -2.0
            )


class TargetClass:
    """
    Compute the target of the class when specific interventions are performed,
    calculates the true effect after an intervention was selected
    """

    def __init__(
        self,
        sem_model: OrderedDict,
        interventions: List,
        variables: List,
        graph: GraphStructure,
        noiseless=True,
        use_iscm: bool = False,
    ) -> None:
        self.model = sem_model
        self.interventions = interventions
        self.variables = variables
        self.num_interventions = len(interventions)
        self.interventional_dict = {val: "" for val in self.interventions}
        self.graph = graph
        self.noiseless = noiseless
        self.use_iscm = use_iscm

    def compute_target(self, value: np.ndarray) -> np.ndarray:
        for i in range(self.num_interventions):
            self.interventional_dict[self.interventions[i]] = value[0, i]

        if self.noiseless:
            sample_count = 1000
        else:
            sample_count = 1

        new_samples = sample_model(
            self.model,
            interventions=self.interventional_dict,
            sample_count=sample_count,
            graph=self.graph,
            use_iscm=self.use_iscm,
        )
        return np.mean(new_samples[self.graph.target]).reshape(1, 1)

    def compute_all(self, value: np.ndarray):
        for i in range(self.num_interventions):
            self.interventional_dict[self.interventions[i]] = value[0, i]

        new_samples = sample_model(
            self.model,
            interventions=self.interventional_dict,
            sample_count=1000,
            graph=self.graph,
            use_iscm=self.use_iscm,
        )
        all_vars = {
            var: np.mean(new_samples[var]).reshape(1, 1) for var in self.variables
        }
        return all_vars


class CausalGradientAcquisitionOptimizer(AcquisitionOptimizerBase):
    """
    Optimizes the acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.
    """

    def __init__(self, space: ParameterSpace, num_anchor_points: int = 100) -> None:
        """
        param space: The parameter space spanning the search problem.
        """
        # print('self.num_anchor_points', num_anchor_points)
        self.num_anchor_points = num_anchor_points
        super().__init__(space)

    def _optimize(
        self, acquisition: Acquisition, context_manager: ContextManager
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        # Context validation
        if len(context_manager.contextfree_space.parameters) == 0:
            logging.warning("All parameters are fixed through context")
            x = np.array(context_manager.context_values)[None, :]
            return x, f(x)

        if acquisition.has_gradients:

            def f_df(x):
                f_value, df_value = acquisition.evaluate_with_gradients(x)
                return -f_value, -df_value

        else:
            f_df = None

        optimizer = self._get_optimizer(context_manager)
        anchor_points_generator = ObjectiveAnchorPointsGenerator(
            self.space, acquisition, self.num_anchor_points
        )

        # Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(
            num_anchor=1, context_manager=context_manager
        )

        logging.info(
            "Starting gradient-based optimization of acquisition function {}".format(
                type(acquisition)
            )
        )
        optimized_points = []
        for a in anchor_points:
            optimized_point = apply_optimizer(
                optimizer,
                a,
                space=self.space,
                f=f,
                df=None,
                f_df=f_df,
                context_manager=context_manager,
            )
            optimized_points.append(optimized_point)

        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        return x_min, -fx_min

    def _get_optimizer(self, context_manager):
        if len(self.space.constraints) == 0:
            return OptLbfgs(context_manager.contextfree_space.get_bounds())
        else:
            return OptTrustRegionConstrained(
                context_manager.contextfree_space.get_bounds(), self.space.constraints
            )


class CausalExpectedImprovement(Acquisition):
    def __init__(
        self,
        current_global_min: float,
        task: str,
        model: Union[IModel, IDifferentiable],
        jitter: float = float(0),
    ) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(
            self.current_global_min, mean, standard_deviation
        )
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(
            self.current_global_min, mean, standard_deviation
        )

        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))
            dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


class CausalUpperConfidenceBound(Acquisition):
    """
    Causal (Lower/Upper) Confidence Bound acquisition, matching the
    CausalExpectedImprovement interface so it is a drop-in in get_new_x_y_list.

    The acquisition optimiser MAXIMISES evaluate(x). For task == "min" we want
    points with LOW predicted mean and HIGH uncertainty, i.e. maximise
    beta*std - mean (the negative lower confidence bound). For task == "max" we
    maximise the usual UCB mean + beta*std.
    """

    def __init__(
        self,
        current_global_min: float,
        task: str,
        model: Union[IModel, IDifferentiable],
        beta: float = 2.0,
    ) -> None:
        self.model = model
        self.task = task
        self.beta = beta
        self.current_global_min = current_global_min  # kept for interface parity

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        if self.task == "min":
            return self.beta * standard_deviation - mean
        return mean + self.beta * standard_deviation

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)
        if self.task == "min":
            value = self.beta * standard_deviation - mean
            dvalue_dx = self.beta * dstandard_deviation_dx - dmean_dx
        else:
            value = mean + self.beta * standard_deviation
            dvalue_dx = dmean_dx + self.beta * dstandard_deviation_dx
        return value, dvalue_dx

    @property
    def has_gradients(self) -> bool:
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(
    x: np.array, mean: np.array, standard_deviation: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf


class Cost(Acquisition):
    def __init__(self, costs_functions, evaluated_set):
        self.costs_functions = costs_functions
        self.evaluated_set = evaluated_set

        assert len(self.evaluated_set) <= 3

    def evaluate(self, x):
        if len(self.evaluated_set) == 1:
            cost = self.costs_functions[self.evaluated_set[0]](x)
        if len(self.evaluated_set) == 2:
            cost = self.costs_functions[self.evaluated_set[0]](
                x[:, 0]
            ) + self.costs_functions[self.evaluated_set[1]](x[:, 1])
        if len(self.evaluated_set) == 3:
            cost = (
                self.costs_functions[self.evaluated_set[0]](x[:, 0])
                + self.costs_functions[self.evaluated_set[1]](x[:, 1])
                + self.costs_functions[self.evaluated_set[2]](x[:, 2])
            )
        return cost

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)
