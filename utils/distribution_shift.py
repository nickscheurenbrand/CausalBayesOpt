r"""Boundary-induced distribution shift, Delta_t = D(q_t(x) || p(x)) (appendix E.2), between the empirical sampling distribution and the uniform intervention distribution.
Since q_t is atomic, estimators here either bin the unit box (KL/JS with Laplace smoothing) or use a divergence well-defined against an empirical measure (Wasserstein-1, energy distance).
"""

from typing import Dict, Sequence

import numpy as np


def to_unit_box(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map each coordinate to [0, 1] using its intervention range."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return np.clip((X - lo) / span, 0.0, 1.0)


def _histogram(U: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-coordinate normalised histogram, shape (d, n_bins)."""
    d = U.shape[1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros((d, n_bins))
    for j in range(d):
        counts, _ = np.histogram(U[:, j], bins=edges)
        out[j] = counts
    return out


def binned_kl(U: np.ndarray, n_bins: int = 10, alpha: float = 1.0) -> float:
    """KL(q_t || p) on per-coordinate histograms, averaged over coordinates.
    Laplace smoothing (`alpha`) keeps empty bins from sending the KL to infinity once sampling concentrates."""
    counts = _histogram(U, n_bins)
    q = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_bins)
    p = np.full(n_bins, 1.0 / n_bins)
    return float(np.mean(np.sum(q * np.log(q / p), axis=1)))


def binned_js(U: np.ndarray, n_bins: int = 10, alpha: float = 1.0) -> float:
    """Jensen-Shannon divergence (bounded by log 2), same binning."""
    counts = _histogram(U, n_bins)
    q = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_bins)
    p = np.full(n_bins, 1.0 / n_bins)
    m = 0.5 * (q + p)
    js = 0.5 * np.sum(q * np.log(q / m), axis=1) + 0.5 * np.sum(p * np.log(p / m), axis=1)
    return float(np.mean(js))


def wasserstein1_uniform(U: np.ndarray) -> float:
    """Mean per-coordinate W_1(q_t, Uniform[0,1]) -- no binning needed.
    W_1 here is the mean absolute gap between sorted samples and uniform quantiles."""
    n, d = U.shape
    if n == 0:
        return 0.0
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    return float(np.mean([np.mean(np.abs(np.sort(U[:, j]) - quantiles))
                          for j in range(d)]))


def boundary_mass(U: np.ndarray, edge: float = 0.2) -> float:
    """Fraction of coordinates within `edge` of an end of their range.
    Not a divergence but what it's a proxy for; under p this equals 2*edge, so values above that are the concentration itself."""
    if U.size == 0:
        return 0.0
    return float(np.mean((U <= edge) | (U >= 1.0 - edge)))


def energy_distance_uniform(U: np.ndarray, n_ref: int = 512,
                            seed: int = 0) -> float:
    """Energy distance between q_t and p, estimated with a uniform reference."""
    n = U.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    V = rng.random((n_ref, U.shape[1]))

    def mean_dist(A, B):
        return float(np.mean(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)))

    return float(2 * mean_dist(U, V) - mean_dist(U, U) - mean_dist(V, V))


def distribution_shift(
    X: np.ndarray,
    bounds: np.ndarray,
    n_bins: int = 10,
    edge: float = 0.2,
    with_energy: bool = False,
) -> Dict[str, float]:
    """All Delta_t estimators for the points evaluated so far, as a dict with `kl`, `js`, `w1`, `boundary_mass`, `n` (and `energy` when requested).
    `js` is the default headline number: bounded, symmetric, and finite for any t including t = 1.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.size == 0:
        return {"kl": 0.0, "js": 0.0, "w1": 0.0, "boundary_mass": 0.0, "n": 0}
    U = to_unit_box(X, bounds)
    out = {
        "kl": binned_kl(U, n_bins),
        "js": binned_js(U, n_bins),
        "w1": wasserstein1_uniform(U),
        "boundary_mass": boundary_mass(U, edge),
        "n": int(U.shape[0]),
    }
    if with_energy:
        out["energy"] = energy_distance_uniform(U)
    return out
