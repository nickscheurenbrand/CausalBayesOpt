r"""Prior-data Fitted Network prior mean (appendix E.4), backed by TabPFN. Replaces the classical
GP's m(x)=0, which extrapolates badly away from boundary-concentrated data, with a global prior."""

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
    r"""TabPFN as the E.4 prior mean: ``fit`` + ``predict`` IS in-context regression. Fit results
    are cached and reused until the context changes, since acquisition issues ~10^3-10^4 evals."""

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
    """Resolve a prior from a name or an object. "pfn" and "tabpfn" both mean TabPFN; checkpoint
    paths are no longer accepted."""
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
