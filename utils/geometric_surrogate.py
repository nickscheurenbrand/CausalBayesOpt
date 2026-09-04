r"""Geometry-aware surrogate: PFN prior mean + spherical BLR (appendix E.5-E.7). Runs BLR on the
PFN residual over spherical features psi(x~); <psi,psi'> IS k_sphere, so this equals the E.6 GP."""

from typing import Callable, Optional, Tuple

import numpy as np
from emukit.core.interfaces import IDifferentiable, IModel

from utils.geometry import AdaptiveGeometry


class GeometryAwareSurrogate(IModel, IDifferentiable):
    """Unified geometry-aware surrogate for one intervention set. prior_mean selects "pfn" (E.4),
    "pfn+do", "do", or "zero"; include_noise defaults False since E.5 states the latent variance."""

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
        """d mean / dx and d var / dx by central differences (uniform rule since the PFN mean
        isn't analytically differentiable, unlike the BLR terms); input dim is small so cheap."""
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
