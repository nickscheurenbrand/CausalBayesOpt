r"""Adaptive intervention geometry (appendix E.3).

The surrogate does not model the intervention space directly. Each intervention
set I carries a posterior over causal relevance, and the space is reshaped by it
before any modelling happens:

    x~ = phi_pi(x) = S(W(pi_I) x),        W(pi_I) = diag(pi_I),

with S the inverse stereographic projection onto the unit sphere (the map used
by the spherical kernels already in this repo, see ``utils.cbo_classes``). The
geometry therefore moves with the parent posterior: coordinates the posterior
believes are causal parents keep their extent, while coordinates it believes are
irrelevant are contracted toward the centre, so the surrogate spends its
capacity on directions that matter.

pi_I is a VECTOR, one weight per intervened coordinate -- W = diag(pi_I) only
makes sense that way. The natural per-coordinate quantity implied by
``pi_I^(t) = P(I contains causal parents | D_t)`` is the marginal inclusion
probability of each variable under the parent-set posterior, which is what
:func:`marginal_parent_probabilities` computes. A scalar is accepted too and is
broadcast across coordinates.

Two properties worth knowing:

* By default the global scale of the projection is renormalised by the same
  weights (``normalize_global_scale=True``), so the geometry depends only on the
  RELATIVE causal relevance across coordinates and is invariant to multiplying
  the whole posterior by a constant. Set it False for the literal ``W(pi) x``,
  where an all-low posterior also shrinks the space globally.
* ``pi_floor`` keeps a coordinate from collapsing to a point when its posterior
  probability hits zero, which would make the projection singular and erase the
  dimension entirely.

The feature map ``psi(x~)`` returned by :meth:`AdaptiveGeometry.features` is the
one whose inner product is the spherical linear kernel, so Bayesian linear
regression on these features (appendix E.5) and a GP with ``k_sphere`` (E.6) are
the same model in weight space and function space respectively.
"""

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def marginal_parent_probabilities(
    posterior: Dict[Tuple[str, ...], float] | Sequence[float],
    parent_sets: Iterable[Tuple[str, ...]] = None,
) -> Dict[str, float]:
    """Marginal P(v is a causal parent | D_t) for every variable in the support.

    Accepts either a {parent_set: probability} mapping or a (parent_sets,
    probabilities) pair, matching how ``PARENT_SCALE`` stores its posterior
    (``self.graphs.keys()`` alongside ``self.posterior``).
    """
    if parent_sets is not None:
        items = list(zip(parent_sets, posterior))
    elif isinstance(posterior, dict):
        items = list(posterior.items())
    else:
        raise TypeError("pass a {parent_set: prob} dict or (probs, parent_sets)")

    total = float(sum(p for _, p in items))
    if total <= 0:
        return {}
    marginals: Dict[str, float] = {}
    for parents, prob in items:
        for v in parents:
            marginals[str(v)] = marginals.get(str(v), 0.0) + float(prob) / total
    return marginals


def weights_for_set(
    intervention_set: Sequence[str],
    marginals: Dict[str, float],
    default: float = 1.0,
) -> np.ndarray:
    """pi_I as a vector aligned with the coordinates of `intervention_set`."""
    return np.array(
        [float(marginals.get(str(v), default)) for v in intervention_set],
        dtype=float,
    )


class AdaptiveGeometry:
    r"""x -> x~ = S(W(pi) (x - c)) and the induced feature map psi(x~).

    Parameters
    ----------
    bounds : (D, 2) array of per-coordinate (lower, upper) intervention limits.
    pi : (D,) posterior relevance weights, or a scalar to broadcast.
    lengthscale : per-coordinate scaling applied before the projection.
    glob_ls_frac : fraction controlling the global lengthscale, as in the
        BoTorch spherical kernel this projection follows.
    coeffs : (constant, linear) term weights of the spherical linear feature
        map; normalised to sum to one.
    pi_floor : lower clip on pi, so a zero-probability coordinate is contracted
        but not annihilated.
    normalize_global_scale : see module docstring.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        pi=1.0,
        lengthscale=1.0,
        glob_ls_frac: float = 0.5,
        coeffs: Tuple[float, float] = (0.5, 0.5),
        pi_floor: float = 1e-2,
        normalize_global_scale: bool = True,
    ):
        bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
        if bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (D, 2)")
        self.dim = bounds.shape[0]
        self.mins = bounds[:, 0].copy()
        self.maxs = bounds[:, 1].copy()
        degenerate = (self.maxs - self.mins) < 1e-9
        self.mins[degenerate] -= 0.5
        self.maxs[degenerate] += 0.5
        self.centers = 0.5 * (self.mins + self.maxs)

        ls = np.asarray(lengthscale, dtype=float)
        self.lengthscale = np.full(self.dim, float(ls)) if ls.ndim == 0 else ls
        self.glob_ls_frac = float(glob_ls_frac)
        self.pi_floor = float(pi_floor)
        self.normalize_global_scale = bool(normalize_global_scale)

        c = np.asarray(coeffs, dtype=float)
        c = c / c.sum()
        self.term0, self.term1 = float(c[0]), float(c[1])

        self.set_pi(pi)

    # -- posterior-driven weights -------------------------------------------
    def set_pi(self, pi) -> None:
        """Update pi between iterations; the geometry moves with the posterior."""
        pi = np.asarray(pi, dtype=float)
        if pi.ndim == 0:
            pi = np.full(self.dim, float(pi))
        if pi.shape[0] != self.dim:
            raise ValueError(f"pi has {pi.shape[0]} entries, expected {self.dim}")
        self.pi = np.clip(pi, self.pi_floor, None)

    @property
    def feature_dim(self) -> int:
        return self.dim + 2

    def _global_lengthscale(self) -> float:
        """O(sqrt(D)) global scale from the (optionally weighted) input span."""
        half_span = (self.maxs - self.mins) / (2.0 * self.lengthscale)
        if self.normalize_global_scale:
            half_span = half_span * self.pi
        max_sq_norm = float(np.sum(half_span ** 2))
        return float(np.sqrt(max(self.glob_ls_frac * max_sq_norm, 1e-12)))

    # -- the maps ------------------------------------------------------------
    def scaled(self, X: np.ndarray) -> np.ndarray:
        """z = W(pi) (x - c) / (lengthscale * global lengthscale)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self.centers) * self.pi / self.lengthscale / self._global_lengthscale()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """x~ = S(W(pi)(x - c)): points on the unit sphere in R^{D+1}."""
        z = self.scaled(X)
        s = np.sum(z ** 2, axis=1)
        p = 1.0 / (1.0 + s)
        return np.concatenate([2.0 * z * p[:, None], ((s - 1.0) * p)[:, None]], axis=1)

    def features(self, X: np.ndarray) -> np.ndarray:
        """psi(x~) in R^{D+2}; <psi(x), psi(x')> is the spherical linear kernel."""
        proj = self.transform(X)
        n = proj.shape[0]
        return np.concatenate(
            [
                np.sqrt(self.term1) * proj,
                np.full((n, 1), np.sqrt(self.term0)),
            ],
            axis=1,
        )

    def kernel(self, X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """k_sphere(x, x') = <psi(x~), psi(x~')>, unit diagonal by construction."""
        f1 = self.features(X)
        f2 = f1 if X2 is None else self.features(X2)
        return f1 @ f2.T

    def __repr__(self) -> str:
        return (f"AdaptiveGeometry(dim={self.dim}, "
                f"pi={np.round(self.pi, 3).tolist()})")
