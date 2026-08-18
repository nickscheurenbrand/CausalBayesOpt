r"""Geometry-aware surrogate: PFN prior mean + spherical BLR (appendix E.5-E.7).

Model, per intervention set I:

    f(x) = beta' psi(x~) + eps,   beta ~ N(0, Sigma_beta0),  eps ~ N(0, sigma^2)
    x~   = phi_pi(x)                                       (E.3, utils.geometry)
    m(x) = m_PFN(x~)                                       (E.4, utils.pfn_prior)

Bayesian linear regression is run on the PFN RESIDUAL, since the predictive mean
in E.5 is ``m_PFN(x) + psi(x~)' mu_beta`` -- the weights explain what the global
prior does not. With Gaussian noise the posterior is closed form:

    Lambda   = Psi' Psi / sigma^2 + Sigma_beta0^{-1}
    Sigma_beta = Lambda^{-1}
    mu_beta  = Sigma_beta Psi' (y - m_PFN) / sigma^2

giving p(f(x) | D) = N(m_PFN(x) + psi(x~)' mu_beta, psi(x~)' Sigma_beta psi(x~)),
exactly E.5. Because <psi(x~), psi(x~')> IS ``k_sphere``, this weight-space model
and the function-space GP of E.6, GP(m_PFN(phi_pi(x)), k_sphere(...)), are the
same object -- the BLR form is used because it is exact, cheap, and needs no
kernel inversion as the data concentrate near the boundary.

sigma^2, the prior weight variance and the scale of the projection are set by
maximising the log evidence over a small grid -- the analogue of
``gpy_model.optimize()`` in the RBF path, keeping the posterior calibrated rather
than fitted by hand. The projection scale matters more than the other two: it
controls how much of the sphere the data occupy, and hence how curved the
geometry is in the region being modelled, so leaving it fixed noticeably
underfits.

The class implements emukit's ``IModel``/``IDifferentiable``, so the EXISTING
acquisition code path (``CausalExpectedImprovement`` and the gradient optimizer
in ``utils.cbo_functions.get_new_x_y_list``) drives it unchanged -- which is the
claim of E.7: the acquisition is untouched, only the posterior it consumes
changes.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from emukit.core.interfaces import IDifferentiable, IModel

from utils.geometry import AdaptiveGeometry


class GeometryAwareSurrogate(IModel, IDifferentiable):
    """Unified geometry-aware surrogate for one intervention set.

    Parameters
    ----------
    geometry : the adaptive geometry; its ``pi`` is refreshed each iteration.
    prior : PFN-style prior with ``mean(X_ctx, y_ctx, X_query)``.
    variance_adjustment : optional do-variance from the causal prior, added to
        the predictive variance the way the repo's causal kernels add it to the
        kernel diagonal. Pass None for the plain E.6 surrogate.
    do_mean : optional do-mean. With ``prior_mean="pfn+do"`` the PFN models the
        residual over the causal prior mean instead of replacing it.
    prior_mean : "pfn" (E.4 as written), "pfn+do", "do", or "zero".
    include_noise : add sigma^2 to the predictive variance. E.5 states the
        latent variance, so this defaults to False.
    """

    def __init__(
        self,
        geometry: AdaptiveGeometry,
        prior,
        X: np.ndarray,
        Y: np.ndarray,
        variance_adjustment: Optional[Callable] = None,
        do_mean: Optional[Callable] = None,
        prior_mean: str = "pfn",
        include_noise: bool = False,
        noise_grid: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        weight_var_grid: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
        scale_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
        fd_eps: float = 1e-5,
    ):
        if prior_mean not in ("pfn", "pfn+do", "do", "zero"):
            raise ValueError(f"unknown prior_mean {prior_mean!r}")
        self.geometry = geometry
        self.prior = prior
        self.variance_adjustment = variance_adjustment
        self.do_mean = do_mean
        self.prior_mean = prior_mean
        self.include_noise = bool(include_noise)
        self.noise_grid = tuple(noise_grid)
        self.weight_var_grid = tuple(weight_var_grid)
        self.scale_grid = tuple(scale_grid)
        self.fd_eps = float(fd_eps)

        self.sigma2 = 1e-2
        self.weight_var = 1.0
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        self._Y = np.asarray(Y, dtype=float).reshape(-1, 1)
        self._fit()

    # -- prior mean ----------------------------------------------------------
    def _do_mean(self, X: np.ndarray) -> np.ndarray:
        if self.do_mean is None:
            return np.zeros(X.shape[0])
        return np.asarray(self.do_mean(X), dtype=float).reshape(-1)

    def _prior_mean(self, X: np.ndarray) -> np.ndarray:
        """m(x): the mean the BLR takes as given and models residuals around."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.prior_mean == "zero":
            return np.zeros(X.shape[0])
        if self.prior_mean == "do":
            return self._do_mean(X)

        base = self._do_mean(X) if self.prior_mean == "pfn+do" else 0.0
        # the PFN is evaluated in the ADAPTIVE GEOMETRY: m_pi(x) = m_PFN(phi(x))
        ctx_x = self.geometry.transform(self._X)
        ctx_y = self._Y.reshape(-1)
        if self.prior_mean == "pfn+do":
            ctx_y = ctx_y - self._do_mean(self._X)
        pfn = np.asarray(
            self.prior.mean(ctx_x, ctx_y, self.geometry.transform(X)), dtype=float
        ).reshape(-1)
        return base + pfn

    # -- posterior -----------------------------------------------------------
    def _log_evidence(self, Psi: np.ndarray, r: np.ndarray,
                      sigma2: float, weight_var: float) -> float:
        """log p(r | sigma^2, weight_var) for the linear-Gaussian model."""
        n = Psi.shape[0]
        C = weight_var * (Psi @ Psi.T) + sigma2 * np.eye(n)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        return float(
            -0.5 * r @ alpha
            - np.sum(np.log(np.diag(L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def _fit(self) -> None:
        # Joint evidence maximisation over the projection scale and the two
        # variances. The scale changes the geometry, so Psi and the PFN mean
        # (which is evaluated through the projection) are recomputed per candidate.
        best = (-np.inf, self.geometry.glob_ls_frac, self.sigma2, self.weight_var)
        for frac in self.scale_grid:
            self.geometry.glob_ls_frac = float(frac)
            Psi = self.geometry.features(self._X)
            r = self._Y.reshape(-1) - self._prior_mean(self._X)
            for s2 in self.noise_grid:
                for wv in self.weight_var_grid:
                    ev = self._log_evidence(Psi, r, s2, wv)
                    if ev > best[0]:
                        best = (ev, float(frac), s2, wv)

        self.log_evidence, self.glob_ls_frac, self.sigma2, self.weight_var = best
        self.geometry.glob_ls_frac = self.glob_ls_frac
        Psi = self.geometry.features(self._X)
        r = self._Y.reshape(-1) - self._prior_mean(self._X)

        F = Psi.shape[1]
        precision = Psi.T @ Psi / self.sigma2 + np.eye(F) / self.weight_var
        self.Sigma_beta = np.linalg.inv(precision)
        self.mu_beta = self.Sigma_beta @ Psi.T @ r / self.sigma2

    # -- IModel --------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Psi = self.geometry.features(X)
        mean = self._prior_mean(X) + Psi @ self.mu_beta
        var = np.einsum("ij,jk,ik->i", Psi, self.Sigma_beta, Psi)
        if self.include_noise:
            var = var + self.sigma2
        if self.variance_adjustment is not None:
            adj = np.asarray(self.variance_adjustment(X), dtype=float).reshape(-1)
            var = var + np.abs(adj)
        var = np.maximum(var, 1e-12)
        return mean.reshape(-1, 1), var.reshape(-1, 1)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        self._Y = np.asarray(Y, dtype=float).reshape(-1, 1)
        self._fit()

    def optimize(self) -> None:
        """Re-run the evidence maximisation (hyperparameters only)."""
        self._fit()

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    # -- IDifferentiable -----------------------------------------------------
    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """d mean / dx and d var / dx by central differences.

        The BLR terms are analytic, but the PFN mean is a neural net evaluated
        through the projection; one consistent finite-difference rule for both
        keeps the gradient the acquisition sees self-consistent, and the input
        dimension here is small enough that the 2D extra predicts are cheap.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        dmean = np.zeros((n, d))
        dvar = np.zeros((n, d))
        span = np.maximum(self.geometry.maxs - self.geometry.mins, 1e-9)
        for j in range(d):
            h = self.fd_eps * span[j]
            step = np.zeros(d)
            step[j] = h
            m_up, v_up = self.predict(X + step)
            m_dn, v_dn = self.predict(X - step)
            dmean[:, j] = (m_up - m_dn).reshape(-1) / (2 * h)
            dvar[:, j] = (v_up - v_dn).reshape(-1) / (2 * h)
        return dmean, dvar

    def __repr__(self) -> str:
        return (f"GeometryAwareSurrogate(n={self._X.shape[0]}, "
                f"prior={getattr(self.prior, 'name', type(self.prior).__name__)}, "
                f"prior_mean={self.prior_mean}, sigma2={self.sigma2:g}, "
                f"weight_var={self.weight_var:g}, "
                f"scale={self.glob_ls_frac:g}, {self.geometry})")
