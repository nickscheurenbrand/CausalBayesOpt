r"""GEOMETRY_SCALE: causal BO with a geometry-aware surrogate. Reuses
PARENT_SCALE's causal machinery; swaps in an adaptive-geometry, TabPFN-prior GP surrogate."""

import logging
from typing import Dict, List, Optional

import numpy as np

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.graph import GraphStructure
from utils.distribution_shift import distribution_shift
from utils.geometric_surrogate import GeometryAwareSurrogate
from utils.geometry import (
    AdaptiveGeometry,
    marginal_parent_probabilities,
    weights_for_set,
)
from utils.pfn_prior import build_prior


class GEOMETRY_SCALE(PARENT_SCALE):
    """PARENT_SCALE with the geometry-aware surrogate of appendix E."""

    def __init__(
        self,
        graph: GraphStructure,
        prior="zero",
        prior_mean: str = "pfn",
        use_do_variance: bool = True,
        adapt_geometry: bool = True,
        pi_floor: float = 1e-2,
        glob_ls_frac: float = 0.5,
        normalize_global_scale: bool = True,
        include_noise: bool = False,
        shift_bins: int = 10,
        shift_edge: float = 0.2,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(graph=graph, **kwargs)
        self.prior_spec = prior
        self.prior = build_prior(prior, device=device)
        self.prior_mean = prior_mean
        self.use_do_variance = bool(use_do_variance)
        self.adapt_geometry = bool(adapt_geometry)
        self.pi_floor = float(pi_floor)
        self.glob_ls_frac = float(glob_ls_frac)
        self.normalize_global_scale = bool(normalize_global_scale)
        self.include_noise = bool(include_noise)
        self.shift_bins = int(shift_bins)
        self.shift_edge = float(shift_edge)

        # per-iteration diagnostics
        self.shift_history: List[Dict[str, float]] = []
        self.pi_history: List[Dict[str, float]] = []
        self.geometries: Dict[tuple, AdaptiveGeometry] = {}
        self.surrogate_diagnostics: List[Dict] = []

    # -- geometry ------------------------------------------------------------
    def _bounds_for(self, intervention_set) -> np.ndarray:
        ranges = self.graph.interventional_range_data
        return np.array([[float(ranges[v][0]), float(ranges[v][1])]
                         for v in intervention_set], dtype=float)

    def current_marginals(self) -> Dict[str, float]:
        """pi^(t): marginal P(v is a causal parent | D_t) for each variable."""
        return marginal_parent_probabilities(self.posterior, list(self.graphs.keys()))

    def _geometry_for(self, intervention_set, marginals) -> AdaptiveGeometry:
        """Fetch or build the geometry for a set, refreshing pi in place."""
        key = tuple(intervention_set)
        pi = (weights_for_set(intervention_set, marginals, default=1.0)
              if self.adapt_geometry
              else np.ones(len(intervention_set)))
        geom = self.geometries.get(key)
        if geom is None:
            geom = AdaptiveGeometry(
                bounds=self._bounds_for(intervention_set),
                pi=pi,
                glob_ls_frac=self.glob_ls_frac,
                pi_floor=self.pi_floor,
                normalize_global_scale=self.normalize_global_scale,
            )
            self.geometries[key] = geom
        else:
            geom.set_pi(pi)          # the geometry adapts as the posterior moves
        return geom

    # -- E.2 -----------------------------------------------------------------
    def _record_shift(self, data_x_list, iteration: int) -> None:
        """Delta_t over the points evaluated so far, pooled across the sets."""
        rows, pooled = [], []
        for j, es in enumerate(self.exploration_set):
            X = np.atleast_2d(np.asarray(data_x_list[j], dtype=float))
            if X.size == 0:
                continue
            bounds = self._bounds_for(es)
            stats = distribution_shift(X, bounds, n_bins=self.shift_bins,
                                       edge=self.shift_edge)
            stats["set"] = tuple(es)
            rows.append(stats)
            pooled.append(distribution_shift(X, bounds, n_bins=self.shift_bins,
                                             edge=self.shift_edge))
        if not rows:
            return
        summary = {
            "iteration": iteration,
            "js": float(np.mean([r["js"] for r in pooled])),
            "kl": float(np.mean([r["kl"] for r in pooled])),
            "w1": float(np.mean([r["w1"] for r in pooled])),
            "boundary_mass": float(np.mean([r["boundary_mass"] for r in pooled])),
            "per_set": rows,
        }
        self.shift_history.append(summary)
        logging.info(
            f"Delta_t (iter {iteration}): JS={summary['js']:.4f} "
            f"KL={summary['kl']:.4f} W1={summary['w1']:.4f} "
            f"boundary_mass={summary['boundary_mass']:.3f}"
        )

    # -- the surrogate seam --------------------------------------------------
    def build_surrogates(self, trial_observed, data_x_list, data_y_list,
                         best_variable, input_space, iteration: int = 0):
        """Rebuild one geometry-aware surrogate per intervention set."""
        from functools import partial

        from utils.ceo_utils import aggregate_mean_function, aggregate_var_function

        marginals = self.current_marginals()
        self.pi_history.append({"iteration": iteration, **marginals})

        models = [None] * len(self.exploration_set)
        diagnostics = []
        for j, es in enumerate(self.exploration_set):
            X = np.atleast_2d(np.asarray(data_x_list[j], dtype=float))
            Y = np.asarray(data_y_list[j], dtype=float).reshape(-1, 1)
            geom = self._geometry_for(es, marginals)

            do_mean = partial(aggregate_mean_function, j,
                              self.do_effects_functions, self.posterior)
            do_var = partial(aggregate_var_function, j,
                             self.do_effects_functions, self.posterior)

            models[j] = GeometryAwareSurrogate(
                geometry=geom,
                prior=self.prior,
                X=X,
                Y=Y,
                variance_adjustment=do_var if self.use_do_variance else None,
                do_mean=do_mean if self.causal_prior else None,
                prior_mean=self.prior_mean,
                include_noise=self.include_noise,
            )
            diagnostics.append({
                "iteration": iteration,
                "set": tuple(es),
                "n": int(X.shape[0]),
                "pi": geom.pi.tolist(),
                "sigma2": models[j].sigma2,
                "weight_var": models[j].weight_var,
                "glob_ls_frac": models[j].glob_ls_frac,
                "log_evidence": models[j].log_evidence,
            })
            logging.info(f"[{tuple(es)}] {models[j]}")

        self.surrogate_diagnostics.extend(diagnostics)
        self._record_shift(data_x_list, iteration)
        return models

    # -- results -------------------------------------------------------------
    def geometry_results(self) -> Dict:
        """Extra diagnostics to store alongside the standard results dict."""
        return {
            "Distribution_Shift": self.shift_history,
            "Delta_t_JS": [s["js"] for s in self.shift_history],
            "Delta_t_KL": [s["kl"] for s in self.shift_history],
            "Delta_t_W1": [s["w1"] for s in self.shift_history],
            "Pi_History": self.pi_history,
            "Surrogate_Diagnostics": self.surrogate_diagnostics,
            "Prior": str(self.prior_spec),
            "Prior_Mean": self.prior_mean,
            "Adapt_Geometry": self.adapt_geometry,
            "Use_Do_Variance": self.use_do_variance,
        }
