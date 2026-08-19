"""Smoke test for utils/ground_truth.py against a hand-built graph.

GPy is not installed locally and utils/sem_sampling.py imports it at module
level, so stub out that import chain before importing anything real.
"""
import sys, types, os
from collections import OrderedDict

import numpy as np

ROOT = "/Users/nicholasscheurenbrand/Documents/Thesis Code/CausalBayesOpt"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

# --- stub GPy and graphs.graph so utils.sem_sampling imports ---
gpy = types.ModuleType("GPy"); models = types.ModuleType("GPy.models")
gpr = types.ModuleType("GPy.models.gp_regression"); gpr.GPRegression = object
models.gp_regression = gpr; gpy.models = models
sys.modules.update({"GPy": gpy, "GPy.models": models,
                    "GPy.models.gp_regression": gpr})
gmod = types.ModuleType("graphs.graph"); gmod.GraphStructure = object
pkg = types.ModuleType("graphs"); pkg.__path__ = [os.path.join(ROOT, "algorithms", "graphs")]
sys.modules.setdefault("graphs", pkg)
sys.modules["graphs.graph"] = gmod

from utils.ground_truth import (classify_nodes, stratified_pools,
                                stratified_non_parents, linear_total_effects,
                                find_true_optimum)

# ---------------------------------------------------------------- test graph --
#   0 -> 1 -> 3 <- 2 <- 7 -> 8      3 -> 6      4 -> 5
# target 3: parents {1,2}; mediated ancestors {0,7}; descendant 6;
# confounded 8 (shares ancestor 7); 4,5 disconnected
EDGES = [("0", "1"), ("1", "3"), ("2", "3"), ("3", "6"),
         ("7", "2"), ("7", "8"), ("4", "5")]
N = 9
W = np.zeros((N, N))
W[0, 1], W[1, 3], W[2, 3], W[3, 6], W[7, 2], W[7, 8], W[4, 5] = \
    0.5, 2.0, -1.0, 1.0, 3.0, 1.0, 1.0


class Stub:
    def __init__(self, nonlinear=False, quad_center=0.7):
        self.variables = [str(i) for i in range(N)]
        self.target = "3"
        self.nonlinear = nonlinear
        self.quad_center = quad_center
        self.weighted_adjacency_matrix = W
        self.parents = {v: [] for v in self.variables}
        for u, v in EDGES:
            self.parents[v].append(u)
        # NOT set here on purpose: the real graphs only gain this attribute via
        # set_interventional_range_data(D_O), called from PARENT_SCALE.set_values
        self.nodes = self.variables
        self.use_intervention_range_data = False
        self.rng = np.random.default_rng(0)
        self.SEM = self._sem()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_interventional_range_data(self, D_O):
        """Mirrors GraphStructure: per-variable [min, max] of the obs data."""
        self.interventional_range_data = {
            v: [float(np.min(D_O[v])), float(np.max(D_O[v]))] for v in self.variables
        }
        self.use_intervention_range_data = True

    def get_interventional_range(self):
        """The abstract default the real classes never override -- a trap."""
        return {v: [-5, 5] for v in self.variables}

    def get_error_distribution(self, noiseless=False):
        return {v: self.rng.normal(0.0, 1.0, size=1) for v in self.variables}

    def _sem(self):
        order = ["0", "7", "4", "1", "2", "5", "3", "6", "8"]
        sem = OrderedDict()
        for node in order:
            ps = self.parents[node]
            if not ps:
                sem[node] = lambda eps, s: eps
            elif node == "3" and self.nonlinear:
                # interior optimum at x1 = quad_center, x2 irrelevant
                c = self.quad_center
                sem[node] = lambda eps, s, c=c: (s["1"] - c) ** 2 + 0.0 * s["2"] + eps
            else:
                sem[node] = (lambda eps, s, node=node, ps=ps:
                             sum(W[int(p), int(node)] * s[p] for p in ps) + eps)
        return sem


fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


g = Stub()
D_O_FAKE = {v: np.array([-2.0, 2.0]) for v in g.variables}   # gives [-2, 2] bounds
rel, dist = classify_nodes(g)
check("relations", {rel["1"], rel["2"]} == {"parent"}
      and rel["0"] == "ancestor" and rel["7"] == "ancestor"
      and rel["6"] == "descendant" and rel["8"] == "confounded"
      and rel["4"] == "disconnected" and rel["5"] == "disconnected",
      str(rel))
check("distances", dist["1"] == 1 and dist["0"] == 2 and dist["7"] == 2
      and not np.isfinite(dist["4"]), str({k: dist[k] for k in "0147"}))

pools = stratified_pools(g)
check("pools", pools["ancestor"] == ["0", "7"]
      and pools["non_ancestor"] == ["4", "5", "6", "8"]
      and pools["any"] == ["0", "4", "5", "6", "7", "8"], str(pools))

check("draw ancestor", stratified_non_parents(g, 2, 71, "ancestor") == ("0", "7"))
drew = stratified_non_parents(g, 2, 71, "non_ancestor")
check("draw non_ancestor", all(v in pools["non_ancestor"] for v in drew), str(drew))
try:
    stratified_non_parents(g, 3, 71, "ancestor")
    check("pool too small raises", False)
except ValueError as e:
    check("pool too small raises", "only 2 candidates" in str(e), str(e))

te = linear_total_effects(g)
check("total effect direct", abs(te["1"] - 2.0) < 1e-9 and abs(te["2"] + 1.0) < 1e-9,
      f'1->{te["1"]}, 2->{te["2"]}')
check("total effect mediated", abs(te["0"] - 0.5 * 2.0) < 1e-9
      and abs(te["7"] - 3.0 * -1.0) < 1e-9, f'0->{te["0"]}, 7->{te["7"]}')
check("total effect non-ancestor", abs(te["4"]) < 1e-12 and abs(te["6"]) < 1e-12)

# ranges must be set first -- this is exactly the failure seen on the cluster
try:
    find_true_optimum(g, ["1", "2"], direction="min", n_mc=5, n_grid=3, seed=1)
    check("unset ranges raise", False)
except RuntimeError as e:
    check("unset ranges raise", "set_values" in str(e), str(e)[:60] + "...")

g.set_interventional_range_data(D_O_FAKE)
check("ranges from D_O", g.interventional_range_data["1"] == [-2.0, 2.0])

# linear SEM: E[Y|do(x)] affine -> optimum at a corner, every dim on boundary
o = find_true_optimum(g, ["1", "2"], direction="min", n_mc=40, n_grid=11, seed=1)
check("linear optimum on boundary", o["is_full_boundary"] and not o["is_interior"],
      f'pos={[round(p,3) for p in o["opt_pos"]]} EY={o["opt_EY"]:.3f}')
# minimise 2*x1 - 1*x2  ->  x1 = lo, x2 = hi
check("linear argmin correct", abs(o["opt_x"][0] + 2) < 1e-6
      and abs(o["opt_x"][1] - 2) < 1e-6, str(o["opt_x"]))
check("EY_range positive", o["EY_range"] > 1.0, f'{o["EY_range"]:.3f}')

# nonlinear mechanism with a genuine interior optimum at x1 = 0.7
gn = Stub(nonlinear=True, quad_center=0.7)
gn.set_interventional_range_data(D_O_FAKE)
on = find_true_optimum(gn, ["1", "2"], direction="min", n_mc=40, n_grid=41, seed=1)
check("nonlinear optimum interior", on["is_interior"],
      f'x={[round(v,3) for v in on["opt_x"]]} '
      f'bnd_dims={on["n_dims_on_boundary"]}/{on["k"]}')
check("nonlinear argmin near 0.7", abs(on["opt_x"][0] - 0.7) < 0.25,
      str(round(on["opt_x"][0], 3)))
check("linear_total_effects None for nonlinear", linear_total_effects(gn) is None)

# max direction flips the corner
omax = find_true_optimum(g, ["1", "2"], direction="max", n_mc=40, n_grid=11, seed=1)
check("max flips argopt", abs(omax["opt_x"][0] - 2) < 1e-6
      and abs(omax["opt_x"][1] + 2) < 1e-6, str(omax["opt_x"]))

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
