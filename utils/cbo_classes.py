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
from GPy.kern.src.kern import Kern
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


class CausalSphericalLinear(Kern):
    r"""
    Spherical-linear surrogate kernel: an inverse stereographic projection of
    the (centred, lengthscale-scaled) inputs onto the unit sphere followed by a
    linear (dot-product) kernel with learnable constant/linear term weights and
    a global lengthscale.  This is a GPy port of the BoTorch/gpytorch
    ``SphericalLinearKernel`` (see ``spherical_linear.py``) so it drops into the
    existing GPy surrogate pipeline in place of the RBF core.

    Same do-variance adjustment as :class:`CausalRBF`, so the causal prior is
    preserved -- only the RBF core is swapped for the spherical-linear core::

        K(x, x') = variance * <phi(x), phi(x')>
                   + sqrt(vadj(x)) * sqrt(vadj(x'))

    where ``phi: R^D -> R^{D+2}`` maps ``x`` via the stereographic projection
    (which lands on the unit sphere, so ``||phi(x)|| = 1`` and ``K(x, x) =
    variance`` exactly like the RBF).

    As requested, only the outer ``variance`` is optimised.  The projection
    shape parameters (per-dimension ``lengthscale``, term ``coeffs`` and the
    global-lengthscale fraction) are fixed at their init values.  ``lengthscale``
    is registered (constrained fixed) purely so ``safe_optimization`` can read
    ``kern.lengthscale[0]``.
    """

    def __init__(
        self,
        input_dim: int,
        variance_adjustment: Callable,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        bounds: np.ndarray = None,
        coeffs: Tuple[float, float] = (0.5, 0.5),
        glob_ls_frac: float = 0.5,
        active_dims=None,
        name: str = "spherical_linear",
    ):
        super(CausalSphericalLinear, self).__init__(input_dim, active_dims, name)

        self.variance = Param("variance", variance, Logexp())
        self.link_parameter(self.variance)

        # lengthscale kept as a (fixed) parameter so that safe_optimization can
        # index kern.lengthscale[0]; it is never optimised.
        ls = np.asarray(lengthscale, dtype=float)
        if ls.ndim == 0:
            ls = np.full(input_dim, float(ls))
        self.lengthscale = Param("lengthscale", ls, Logexp())
        self.link_parameter(self.lengthscale)
        self.lengthscale.fix(warning=False)

        self.variance_adjustment = variance_adjustment

        # fixed projection geometry
        if bounds is None:
            bounds = np.tile(np.array([0.0, 1.0]), (input_dim, 1))
        bounds = np.asarray(bounds, dtype=float)
        self._mins = bounds[:, 0]
        self._maxs = bounds[:, 1]
        self._centers = 0.5 * (self._mins + self._maxs)

        # constant/linear term weights (normalised to sum to one, as in softmax)
        c = np.asarray(coeffs, dtype=float)
        c = c / c.sum()
        self._term0 = float(c[0])
        self._term1 = float(c[1])
        self._glob_ls_frac = float(glob_ls_frac)

    # ---- projection helpers -------------------------------------------------
    def _glob_ls(self, ls: np.ndarray) -> float:
        """Global lengthscale ~ O(sqrt(D)) from the input span (see BoTorch)."""
        half_span = (self._maxs - self._mins) / (2.0 * ls)
        max_sq_norm = np.sum(half_span**2)
        return float(np.sqrt(self._glob_ls_frac * max_sq_norm))

    def _project(self, X: np.ndarray):
        """Return z (N, D), p = 1/(1+||z||^2) (N,) and phi (N, D+2)."""
        ls = np.asarray(self.lengthscale)
        glob_ls = self._glob_ls(ls)
        z = (X - self._centers) / ls / glob_ls
        s = np.sum(z**2, axis=1)
        p = 1.0 / (1.0 + s)
        # inverse stereographic projection onto the unit sphere (R^{D+1})
        proj = np.concatenate(
            [2.0 * z * p[:, None], ((s - 1.0) * p)[:, None]], axis=1
        )
        n = X.shape[0]
        phi = np.concatenate(
            [
                np.sqrt(self._term1) * proj,
                np.full((n, 1), np.sqrt(self._term0)),
            ],
            axis=1,
        )
        return z, p, phi, glob_ls

    def _phi(self, X: np.ndarray) -> np.ndarray:
        return self._project(X)[2]

    # ---- kernel evaluations -------------------------------------------------
    def K(self, X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        phi1 = self._phi(X)
        phi2 = phi1 if X2 is None else self._phi(X2)
        value = float(self.variance) * (phi1 @ phi2.T)

        # do-variance adjustment (identical to CausalRBF)
        value_diagonal_X = self.variance_adjustment(X)
        value_diagonal_X2 = (
            value_diagonal_X if X2 is None else self.variance_adjustment(X2)
        )
        additional_matrix = np.outer(
            np.sqrt(value_diagonal_X), np.sqrt(value_diagonal_X2)
        )
        return value + additional_matrix

    def Kdiag(self, X: np.ndarray) -> np.ndarray:
        # <phi(x), phi(x)> == 1 (unit sphere), so the core diagonal is `variance`
        value = self.variance_adjustment(X)
        if X.shape[0] == 1 and X.shape[1] == 1:
            diagonal_terms = value
        elif np.isscalar(value):
            diagonal_terms = value
        elif np.ndim(value) >= 2:
            diagonal_terms = value[:, 0]
        else:
            diagonal_terms = value
        return float(self.variance) + diagonal_terms

    # ---- gradients wrt hyperparameters (only `variance` is free) ------------
    def update_gradients_full(self, dL_dK, X, X2=None):
        phi1 = self._phi(X)
        phi2 = phi1 if X2 is None else self._phi(X2)
        core = phi1 @ phi2.T  # dK/dvariance
        self.variance.gradient = np.sum(dL_dK * core)
        self.lengthscale.gradient = np.zeros_like(np.asarray(self.lengthscale))

    def update_gradients_diag(self, dL_dKdiag, X):
        # dKdiag/dvariance == 1 for every point
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = np.zeros_like(np.asarray(self.lengthscale))

    # ---- gradients wrt inputs (needed by the acquisition optimizer) ---------
    def _dphi_dx(self, X: np.ndarray):
        """Jacobian d proj / dx of the projected features, shape (N, D, D+1)."""
        z, p, _, glob_ls = self._project(X)
        ls = np.asarray(self.lengthscale)
        n, d = X.shape
        p2 = p**2

        # derivative of proj wrt z: Dz[n, q, k], k in 0..D (D+1 features)
        Dz = np.zeros((n, d, d + 1))
        idx = np.arange(d)
        Dz[:, idx, idx] += 2.0 * p[:, None]  # 2 p delta_{kq}
        Dz[:, :, :d] += -4.0 * (z[:, :, None] * z[:, None, :]) * p2[:, None, None]
        Dz[:, :, d] += 4.0 * z * p2[:, None]  # d proj_D / dz_q

        # chain rule z = (x - c) / ls / glob_ls  ->  dz_q/dx_q = 1/(ls_q glob_ls)
        inv_gl = 1.0 / (ls * glob_ls)  # (D,)
        # d phi_{0..D} / dx = sqrt(term1) * inv_gl_q * Dz
        return np.sqrt(self._term1) * Dz * inv_gl[None, :, None]

    def gradients_X(self, dL_dK, X, X2=None):
        phi2 = self._phi(X) if X2 is None else self._phi(X2)
        dproj = self._dphi_dx(X)  # (N, D, D+1)
        # W[n, k] = sum_m dL_dK[n, m] * phi2[m, k], only the first D+1 features
        # contract with the projected Jacobian (the constant term is flat)
        W = dL_dK @ phi2[:, : dproj.shape[2]]  # (N, D+1)
        grad = float(self.variance) * np.einsum("nqk,nk->nq", dproj, W)
        return grad

    def gradients_X_diag(self, dL_dKdiag, X):
        # core diagonal is constant (unit sphere), so its input-gradient is zero
        return np.zeros_like(X)

    def to_dict(self):
        input_dict = super(CausalSphericalLinear, self)._save_to_input_dict()
        input_dict["class"] = "utils.cbo_classes.CausalSphericalLinear"
        return input_dict


class CausalSphericalRBF(Kern):
    r"""
    Spherical-RBF surrogate kernel: the same inverse stereographic projection
    onto the unit sphere as :class:`CausalSphericalLinear`, but with an RBF
    (squared-exponential) kernel applied to the projected features instead of a
    linear (dot-product) one::

        K(x, x') = variance * exp(-0.5 * ||proj(x) - proj(x')||^2 / l^2)
                   + sqrt(vadj(x)) * sqrt(vadj(x'))

    ``proj: R^D -> R^{D+1}`` lands on the unit sphere, so ``K(x, x) = variance``
    exactly like the RBF and spherical-linear cores, and the do-variance
    adjustment of :class:`CausalRBF` plugs in identically (causal prior kept).

    The projection geometry (per-dimension ``proj_lengthscale`` and the global
    lengthscale) is fixed at init.  The outer ``variance`` and the RBF
    ``lengthscale`` are optimised -- the RBF lengthscale is cheap to fit because
    the projected features do not depend on it.
    """

    def __init__(
        self,
        input_dim: int,
        variance_adjustment: Callable,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        proj_lengthscale: float = 1.0,
        bounds: np.ndarray = None,
        glob_ls_frac: float = 0.5,
        active_dims=None,
        name: str = "spherical_rbf",
    ):
        super(CausalSphericalRBF, self).__init__(input_dim, active_dims, name)

        self.variance = Param("variance", variance, Logexp())
        self.link_parameter(self.variance)
        # RBF lengthscale on the projected (unit-sphere) features -- optimised
        self.lengthscale = Param("lengthscale", float(lengthscale), Logexp())
        self.link_parameter(self.lengthscale)

        self.variance_adjustment = variance_adjustment

        # fixed projection geometry
        pls = np.asarray(proj_lengthscale, dtype=float)
        if pls.ndim == 0:
            pls = np.full(input_dim, float(pls))
        self._proj_lengthscale = pls
        if bounds is None:
            bounds = np.tile(np.array([0.0, 1.0]), (input_dim, 1))
        bounds = np.asarray(bounds, dtype=float)
        self._mins = bounds[:, 0]
        self._maxs = bounds[:, 1]
        self._centers = 0.5 * (self._mins + self._maxs)
        self._glob_ls_frac = float(glob_ls_frac)

    # ---- projection helpers -------------------------------------------------
    def _glob_ls(self) -> float:
        half_span = (self._maxs - self._mins) / (2.0 * self._proj_lengthscale)
        return float(np.sqrt(self._glob_ls_frac * np.sum(half_span**2)))

    def _project(self, X: np.ndarray):
        """Return z (N, D), p = 1/(1+||z||^2) (N,), proj (N, D+1), glob_ls."""
        glob_ls = self._glob_ls()
        z = (X - self._centers) / self._proj_lengthscale / glob_ls
        s = np.sum(z**2, axis=1)
        p = 1.0 / (1.0 + s)
        proj = np.concatenate(
            [2.0 * z * p[:, None], ((s - 1.0) * p)[:, None]], axis=1
        )
        return z, p, proj, glob_ls

    @staticmethod
    def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = (
            np.sum(a**2, axis=1)[:, None]
            + np.sum(b**2, axis=1)[None, :]
            - 2.0 * (a @ b.T)
        )
        return np.maximum(sq, 0.0)

    def _dproj_dx(self, X: np.ndarray) -> np.ndarray:
        """Jacobian d proj / dx, shape (N, D, D+1)."""
        z, p, _, glob_ls = self._project(X)
        n, d = X.shape
        p2 = p**2
        Dz = np.zeros((n, d, d + 1))
        idx = np.arange(d)
        Dz[:, idx, idx] += 2.0 * p[:, None]
        Dz[:, :, :d] += -4.0 * (z[:, :, None] * z[:, None, :]) * p2[:, None, None]
        Dz[:, :, d] += 4.0 * z * p2[:, None]
        inv_gl = 1.0 / (self._proj_lengthscale * glob_ls)  # dz_q/dx_q
        return Dz * inv_gl[None, :, None]

    # ---- kernel evaluations -------------------------------------------------
    def K(self, X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        proj1 = self._project(X)[2]
        proj2 = proj1 if X2 is None else self._project(X2)[2]
        ls = float(self.lengthscale)
        r2 = self._sq_dist(proj1, proj2)
        value = float(self.variance) * np.exp(-0.5 * r2 / ls**2)

        value_diagonal_X = self.variance_adjustment(X)
        value_diagonal_X2 = (
            value_diagonal_X if X2 is None else self.variance_adjustment(X2)
        )
        additional_matrix = np.outer(
            np.sqrt(value_diagonal_X), np.sqrt(value_diagonal_X2)
        )
        return value + additional_matrix

    def Kdiag(self, X: np.ndarray) -> np.ndarray:
        value = self.variance_adjustment(X)
        if X.shape[0] == 1 and X.shape[1] == 1:
            diagonal_terms = value
        elif np.isscalar(value):
            diagonal_terms = value
        elif np.ndim(value) >= 2:
            diagonal_terms = value[:, 0]
        else:
            diagonal_terms = value
        return float(self.variance) + diagonal_terms

    # ---- gradients wrt hyperparameters (variance and RBF lengthscale) -------
    def update_gradients_full(self, dL_dK, X, X2=None):
        proj1 = self._project(X)[2]
        proj2 = proj1 if X2 is None else self._project(X2)[2]
        ls = float(self.lengthscale)
        r2 = self._sq_dist(proj1, proj2)
        expo = np.exp(-0.5 * r2 / ls**2)
        core = float(self.variance) * expo
        self.variance.gradient = np.sum(dL_dK * expo)
        self.lengthscale.gradient = np.sum(dL_dK * core * r2 / ls**3)

    def update_gradients_diag(self, dL_dKdiag, X):
        # diagonal distance is zero, so only `variance` contributes
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.0

    # ---- gradients wrt inputs (needed by the acquisition optimizer) ---------
    def gradients_X(self, dL_dK, X, X2=None):
        proj1 = self._project(X)[2]
        proj2 = proj1 if X2 is None else self._project(X2)[2]
        ls = float(self.lengthscale)
        r2 = self._sq_dist(proj1, proj2)
        core = float(self.variance) * np.exp(-0.5 * r2 / ls**2)
        G = dL_dK * core  # (N, M)
        A = self._dproj_dx(X)  # (N, D, D+1)
        # sum_m G[n,m] (proj1[n] - proj2[m]) contracted with the Jacobian
        inner = G.sum(axis=1)[:, None] * proj1 - (G @ proj2)  # (N, D+1)
        return (-1.0 / ls**2) * np.einsum("nqk,nk->nq", A, inner)

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros_like(X)

    def to_dict(self):
        input_dict = super(CausalSphericalRBF, self)._save_to_input_dict()
        input_dict["class"] = "utils.cbo_classes.CausalSphericalRBF"
        return input_dict


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
