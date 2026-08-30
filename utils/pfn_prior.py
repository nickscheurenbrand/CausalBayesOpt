r"""Prior-data Fitted Network prior mean (appendix E.4), backed by TabPFN.

A classical GP surrogate assumes ``m(x) = 0``. Under boundary-concentrated
observations that assumption extrapolates badly: everywhere away from the
cluster of evaluated points the posterior mean decays to zero, which is
precisely the region the optimizer must reason about. E.4 replaces it with

    m_pi(x) = m_PFN(phi_pi(x)),

a globally learned prior over optimization problems, evaluated in the adaptive
geometry of E.3.

A PFN is trained by supervised learning on functions SAMPLED FROM A PRIOR: given
a context of (x, y) pairs and a query x, it regresses the posterior mean of the
prior conditioned on that context. Once trained it is a fixed, amortised
predictor -- no per-run fitting -- and it stays informative with few, clustered
observations because the prior it learned is global rather than data-driven.

The prior used here is **TabPFN**, an externally trained, published PFN whose
own training prior is generated from structural causal models. Using it in place
of a bespoke, locally trained PFN removes "the authors chose their own prior"
as an objection to E.4, at the cost of a prior that is no longer matched to the
unit-sphere geometry the surrogate works in, and whose balance between
boundary-optimal and interior-optimal response surfaces is not under our
control. See :class:`TabPFNPrior` for the operational consequences.

This module provides

* :class:`TabPFNPrior` -- the E.4 prior: TabPFN's in-context posterior mean,
  evaluated on the already-projected inputs.
* :class:`ZeroPFN`, :class:`ConstantPFN` -- dependency-free baselines. ZeroPFN
  recovers the classical zero-mean GP exactly and is the ablation for E.4.

All priors share the interface

    prior.mean(X_ctx, y_ctx, X_query) -> (n_query,) ndarray

where the X are ALREADY projected (points on the unit sphere).
"""

import numpy as np


# --------------------------------------------------------------- baselines ---


class ZeroPFN:
    """m(x) = 0 -- the classical GP prior mean, kept as the E.4 ablation."""

    name = "zero"
    max_dim = None

    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        return np.zeros(np.atleast_2d(X_query).shape[0], dtype=float)


class ConstantPFN:
    """m(x) = mean(y_ctx): a trivially "global" prior, the weakest useful one."""

    name = "constant"
    max_dim = None

    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        n = np.atleast_2d(X_query).shape[0]
        y = np.asarray(y_ctx, dtype=float).reshape(-1)
        return np.full(n, float(y.mean()) if y.size else 0.0)


# ---------------------------------------------------------------- TabPFN -----


class TabPFNPrior:
    r"""TabPFN as the E.4 prior mean.

    ``TabPFNRegressor.fit(X_ctx, y_ctx)`` followed by ``.predict(X_query)`` IS
    in-context regression, so TabPFN satisfies the prior interface directly. No
    padding and no y-standardisation are applied here: TabPFN handles a variable
    number of features natively (up to ~500) and normalises the target itself.

    Fit caching is not an optimisation, it is a requirement. ``GeometryAware-
    Surrogate.predict`` evaluates the prior mean on EVERY acquisition
    evaluation, and the acquisition optimizer plus the finite-difference
    gradients issue on the order of 10^3-10^4 evaluations per trial -- while the
    context (the interventional data) changes only when ``set_data`` is called,
    once per rebuild. The context is therefore hashed and the fitted regressor
    reused until it actually changes, turning those 10^3-10^4 fits into one.
    Even so this prior is far more expensive than the analytic baselines; budget
    accordingly, and prefer ``device="cuda"`` where available.

    Two degenerate contexts are handled before TabPFN is called at all, since it
    is undefined on both: an empty context (no information -> 0) and a constant
    context (no shape -> that constant).
    """

    name = "tabpfn"
    max_dim = None

    def __init__(self, device: str = "cpu", random_state: int = 0, **kwargs):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "TabPFN is required for the PFN prior. Install it with "
                "`pip install tabpfn`, or pass prior='zero' / 'constant' to use "
                "the dependency-free baselines."
            ) from exc
        import os
        os.environ["TABPFN_TOKEN"] = "tabpfn_sk_6yLjEhHVGHW00Q16elWPEEX78991jqsX9N20K_hu7ns"
        self._regressor_cls = TabPFNRegressor
        self.device = device
        self.random_state = int(random_state)
        self._kwargs = kwargs
        self._fitted = None
        self._key = None

    # -- context caching -----------------------------------------------------
    @staticmethod
    def _context_key(X: np.ndarray, y: np.ndarray):
        """Cheap fingerprint of a context; equal keys => identical fit."""
        return (
            X.shape,
            float(X.sum()),
            float((X * X).sum()),
            float(y.sum()),
            float((y * y).sum()),
        )

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        key = self._context_key(X, y)
        if key == self._key and self._fitted is not None:
            return
        reg = self._regressor_cls(
            device=self.device, random_state=self.random_state, **self._kwargs
        )
        reg.fit(X, y)
        self._fitted, self._key = reg, key

    # -- prior interface -----------------------------------------------------
    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        """Posterior-mean prediction on the original y scale."""
        Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
        y = np.asarray(y_ctx, dtype=float).reshape(-1)
        if y.size == 0:                     # no context: nothing to condition on
            return np.zeros(Xq.shape[0], dtype=float)

        # shapes are checked BEFORE the constant-context shortcut, so a
        # malformed context is reported rather than silently short-circuited
        Xc = np.atleast_2d(np.asarray(X_ctx, dtype=float))
        if Xc.shape[0] != y.size:
            raise ValueError(
                f"context has {Xc.shape[0]} rows but {y.size} targets"
            )
        if Xq.shape[1] != Xc.shape[1]:
            raise ValueError(
                f"query has {Xq.shape[1]} features but the context has "
                f"{Xc.shape[1]}"
            )
        if y.std() < 1e-12:                 # a constant context carries no shape
            return np.full(Xq.shape[0], float(y.mean()))

        self._fit(Xc, y)
        pred = np.asarray(self._fitted.predict(Xq), dtype=float).reshape(-1)
        return pred


def build_prior(spec, device: str = "cpu"):
    """Resolve a prior from a name or an object.

    "pfn" and "tabpfn" both mean TabPFN -- the PFN prior is TabPFN now, and the
    former name is kept so existing configs keep working. Checkpoint paths are
    no longer accepted: there is no local PFN left to load.
    """
    if spec is None or spec == "zero":
        return ZeroPFN()
    if spec == "constant":
        return ConstantPFN()
    if spec in ("pfn", "tabpfn"):
        return TabPFNPrior(device=device)
    if isinstance(spec, str):
        raise ValueError(
            f"unknown prior {spec!r}. The local PFN (and its checkpoints) was "
            "replaced by TabPFN; use 'tabpfn', 'zero', 'constant', or pass a "
            "prior object implementing mean(X_ctx, y_ctx, X_query)."
        )
    return spec
