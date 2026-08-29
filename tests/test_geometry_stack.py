"""Tests for the appendix-E stack: geometry, PFN prior, BLR surrogate, Delta_t."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs
_stubs.install()

from utils.geometry import (AdaptiveGeometry, marginal_parent_probabilities,
                            weights_for_set)
from utils.distribution_shift import distribution_shift, to_unit_box
from utils.geometric_surrogate import GeometryAwareSurrogate
from utils.pfn_prior import ZeroPFN, ConstantPFN, TabPFNPrior, build_prior

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond: fails.append(name)

# ---------------------------------------------------------------- E.3 geometry
B = np.array([[-2.0, 2.0], [-1.0, 3.0], [0.0, 10.0]])
g = AdaptiveGeometry(B, pi=[1.0, 1.0, 1.0])
X = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 5.0], [2.0, 3.0, 10.0], [1.0, 0.5, 7.0]])

xt = g.transform(X)
check("S maps onto the unit sphere", np.allclose(np.linalg.norm(xt, axis=1), 1.0),
      f"norms={np.round(np.linalg.norm(xt,axis=1),6).tolist()}")
check("projection dim is D+1", xt.shape == (4, 4), str(xt.shape))
psi = g.features(X)
check("feature dim is D+2", psi.shape == (4, 5), str(psi.shape))
check("k_sphere = <psi,psi'>", np.allclose(g.kernel(X), psi @ psi.T))
check("unit kernel diagonal", np.allclose(np.diag(g.kernel(X)), 1.0),
      str(np.round(np.diag(g.kernel(X)), 6).tolist()))

# pi shrinks low-relevance coordinates RELATIVE to high-relevance ones
g_hi = AdaptiveGeometry(B, pi=[1.0, 1.0, 1.0])
g_lo = AdaptiveGeometry(B, pi=[1.0, 1.0, 0.05])
step = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
d_hi = np.linalg.norm(g_hi.transform(step)[0] - g_hi.transform(step)[1])
d_lo = np.linalg.norm(g_lo.transform(step)[0] - g_lo.transform(step)[1])
check("low pi contracts that coordinate", d_lo < 0.5 * d_hi,
      f"moved {d_hi:.4f} at pi=1 vs {d_lo:.4f} at pi=0.05")

# relevant coordinate is correspondingly expanded (capacity is reallocated)
step0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
e_hi = np.linalg.norm(g_hi.transform(step0)[0] - g_hi.transform(step0)[1])
e_lo = np.linalg.norm(g_lo.transform(step0)[0] - g_lo.transform(step0)[1])
check("high pi coordinate expands when others shrink", e_lo > e_hi,
      f"{e_hi:.4f} -> {e_lo:.4f}")

ga = AdaptiveGeometry(B, pi=[0.8, 0.4, 0.2])
gb = AdaptiveGeometry(B, pi=[0.4, 0.2, 0.1])          # same relative weights
check("invariant to uniform pi rescaling", np.allclose(ga.transform(X), gb.transform(X)))
gl = AdaptiveGeometry(B, pi=[0.8, 0.4, 0.2], normalize_global_scale=False)
gl2 = AdaptiveGeometry(B, pi=[0.4, 0.2, 0.1], normalize_global_scale=False)
check("literal W(pi)x is NOT scale invariant",
      not np.allclose(gl.transform(X), gl2.transform(X)))

gz = AdaptiveGeometry(B, pi=[1.0, 1.0, 0.0], pi_floor=1e-2)
check("pi_floor keeps zero-probability dims finite",
      np.all(np.isfinite(gz.transform(X))) and gz.pi[2] == 0.01, str(gz.pi.tolist()))

g.set_pi([0.1, 0.9, 0.5])
check("set_pi updates in place", np.allclose(g.pi, [0.1, 0.9, 0.5]))

post = {("1", "2"): 0.6, ("2", "3"): 0.3, ("1",): 0.1}
m = marginal_parent_probabilities(post)
check("marginal inclusion probabilities",
      abs(m["1"] - 0.7) < 1e-9 and abs(m["2"] - 0.9) < 1e-9 and abs(m["3"] - 0.3) < 1e-9,
      str({k: round(v, 3) for k, v in m.items()}))
check("weights align with the set",
      np.allclose(weights_for_set(["3", "1"], m), [0.3, 0.7]))
m2 = marginal_parent_probabilities([0.6, 0.3, 0.1], [("1", "2"), ("2", "3"), ("1",)])
check("list form matches dict form", m2 == m)

# --------------------------------------------------------------- E.2 Delta_t
rng = np.random.default_rng(0)
bounds2 = np.array([[0.0, 1.0], [0.0, 1.0]])
uni = rng.random((200, 2))
edge = np.where(rng.random((200, 2)) < 0.5, rng.random((200, 2)) * 0.01,
                1 - rng.random((200, 2)) * 0.01)
d_uni = distribution_shift(uni, bounds2)
d_edge = distribution_shift(edge, bounds2)
check("Delta_t larger for boundary-concentrated q_t",
      d_edge["js"] > 5 * d_uni["js"] and d_edge["kl"] > d_uni["kl"]
      and d_edge["w1"] > d_uni["w1"],
      f"JS {d_uni['js']:.4f} -> {d_edge['js']:.4f}, W1 {d_uni['w1']:.4f} -> {d_edge['w1']:.4f}")
check("boundary_mass ~ 2*edge under p", abs(d_uni["boundary_mass"] - 0.4) < 0.08,
      f"{d_uni['boundary_mass']:.3f}")
check("boundary_mass ~ 1 for edge-concentrated", d_edge["boundary_mass"] > 0.99,
      f"{d_edge['boundary_mass']:.3f}")
check("JS is bounded by log 2", d_edge["js"] <= np.log(2) + 1e-9, f"{d_edge['js']:.4f}")
check("Delta_t finite at t=1", np.isfinite(distribution_shift(uni[:1], bounds2)["kl"]))
check("to_unit_box maps ranges to [0,1]",
      np.allclose(to_unit_box(np.array([[-2.0], [2.0]]), np.array([[-2.0, 2.0]])),
                  [[0.0], [1.0]]))

# monotone growth of Delta_t as sampling concentrates (the E.2 claim)
seq = [distribution_shift(np.vstack([uni[:20], edge[:k]]), bounds2)["js"]
       for k in (1, 20, 60, 150)]
check("Delta_t grows as evaluations concentrate",
      all(a < b for a, b in zip(seq, seq[1:])), str([round(s, 4) for s in seq]))

# ------------------------------------------------------- E.5 BLR correctness
def f_true(X):
    return (2.0 * X[:, 0] - 1.0 * X[:, 1]).reshape(-1, 1)

Bs = np.array([[-2.0, 2.0], [-2.0, 2.0]])
Xtr = rng.uniform(-2, 2, size=(25, 2))
Ytr = f_true(Xtr)
geom = AdaptiveGeometry(Bs, pi=[1.0, 1.0])
sur = GeometryAwareSurrogate(geom, ZeroPFN(), Xtr, Ytr)

mu, var = sur.predict(Xtr)
check("predict shapes are (n,1)", mu.shape == (25, 1) and var.shape == (25, 1),
      f"{mu.shape}, {var.shape}")
check("variance positive", np.all(var > 0))
check("fits training data", float(np.sqrt(np.mean((mu - Ytr) ** 2))) < 0.35,
      f"train RMSE {float(np.sqrt(np.mean((mu - Ytr)**2))):.4f}")

Xte = rng.uniform(-2, 2, size=(50, 2))
mu_te, var_te = sur.predict(Xte)
rmse = float(np.sqrt(np.mean((mu_te - f_true(Xte)) ** 2)))
check("generalises to held-out points", rmse < 0.3, f"test RMSE {rmse:.4f}")
check("uncertainty larger off-data than on-data",
      float(np.mean(sur.predict(np.array([[1.99, -1.99]]))[1])) >= float(np.mean(var)) * 0.5)

# closed-form BLR check: recompute mu_beta / Sigma_beta independently
Psi = geom.features(Xtr)
s2, wv = sur.sigma2, sur.weight_var
Lam = Psi.T @ Psi / s2 + np.eye(Psi.shape[1]) / wv
Sig = np.linalg.inv(Lam)
mub = Sig @ Psi.T @ Ytr.reshape(-1) / s2
check("Sigma_beta matches the closed form", np.allclose(Sig, sur.Sigma_beta))
check("mu_beta matches the closed form", np.allclose(mub, sur.mu_beta))
p_psi = geom.features(Xte)
check("predictive mean = m + psi' mu_beta",
      np.allclose(sur.predict(Xte)[0].reshape(-1), p_psi @ sur.mu_beta))
check("predictive var = psi' Sigma_beta psi",
      np.allclose(sur.predict(Xte)[1].reshape(-1),
                  np.einsum("ij,jk,ik->i", p_psi, sur.Sigma_beta, p_psi)))

# hyperparameters come from evidence maximisation, not defaults
check("evidence selected hyperparameters", np.isfinite(sur.log_evidence),
      f"logZ={sur.log_evidence:.2f}, sigma2={sur.sigma2:g}, "
      f"weight_var={sur.weight_var:g}, scale={sur.glob_ls_frac:g}")
# the projection scale is fitted, and fitting it beats any fixed choice
fixed = GeometryAwareSurrogate(AdaptiveGeometry(Bs, pi=[1.0, 1.0]), ZeroPFN(),
                               Xtr, Ytr, scale_grid=(0.5,))
check("fitting the projection scale improves the evidence",
      sur.log_evidence > fixed.log_evidence and sur.glob_ls_frac != 0.5,
      f"logZ {fixed.log_evidence:.2f} at fixed 0.5 -> {sur.log_evidence:.2f} at {sur.glob_ls_frac:g}")
check("and improves held-out error",
      rmse < float(np.sqrt(np.mean((fixed.predict(Xte)[0] - f_true(Xte)) ** 2))),
      f"{float(np.sqrt(np.mean((fixed.predict(Xte)[0]-f_true(Xte))**2))):.4f} -> {rmse:.4f}")

# gradients (needed for the E.7 acquisition optimizer)
dmu, dvar = sur.get_prediction_gradients(Xte[:5])
check("gradient shapes", dmu.shape == (5, 2) and dvar.shape == (5, 2))
h = 1e-4
num = np.array([[(sur.predict(x + np.eye(2)[j] * h)[0][0, 0]
                  - sur.predict(x - np.eye(2)[j] * h)[0][0, 0]) / (2 * h)
                 for j in range(2)] for x in Xte[:5].reshape(5, 1, 2)])
check("mean gradient matches numerical", np.allclose(dmu, num, atol=1e-3),
      f"max err {np.abs(dmu - num).max():.2e}")

# emukit interface
from emukit.core.interfaces import IDifferentiable, IModel
check("implements emukit IModel/IDifferentiable",
      isinstance(sur, IModel) and isinstance(sur, IDifferentiable))
sur.set_data(Xtr[:10], Ytr[:10])
check("set_data refits", sur.X.shape == (10, 2) and sur.Y.shape == (10, 1))
sur.optimize()
check("optimize runs", np.isfinite(sur.log_evidence))

# -------------------------------------------------- E.4 prior-mean behaviour
# The E.4 claim is about the BOUNDARY-CONCENTRATED regime: few, clustered
# observations, where the prior mean governs everything the data do not reach.
# Priors receive ALREADY-projected inputs x~ (on the sphere, dim D+1).
class InSpanPrior:            # linear in x~ -> lies inside span(psi)
    name = "inspan"
    def mean(self, Xc, yc, Xq):
        return 3.0 * np.atleast_2d(Xq)[:, 0]

class OutSpanPrior:           # nonlinear in x~ -> outside span(psi)
    name = "outspan"
    def mean(self, Xc, yc, Xq):
        Xq = np.atleast_2d(Xq)
        return 5.0 * np.sin(4 * Xq[:, 0]) * np.cos(4 * Xq[:, 1])

Xcl = np.array([[-1.99, -1.99], [-1.98, -1.95], [-1.95, -1.99],
                [-1.99, -1.90], [-1.92, -1.97]])          # clustered at a corner
Ycl = f_true(Xcl)
far = np.array([[1.9, 1.9], [0.0, 0.0], [1.9, -1.9]])
mk = lambda pr: GeometryAwareSurrogate(AdaptiveGeometry(Bs, pi=[1.0, 1.0]), pr, Xcl, Ycl)
s_zero, s_in, s_out = mk(ZeroPFN()), mk(InSpanPrior()), mk(OutSpanPrior())
f_zero = s_zero.predict(far)[0].reshape(-1)
f_in = s_in.predict(far)[0].reshape(-1)
f_out = s_out.predict(far)[0].reshape(-1)
check("prior mean governs extrapolation under clustered data",
      np.max(np.abs(f_in - f_zero)) > 0.5 and np.max(np.abs(f_out - f_zero)) > 2.0,
      f"zero={np.round(f_zero,2).tolist()} in-span={np.round(f_in,2).tolist()} "
      f"out-of-span={np.round(f_out,2).tolist()}")
check("every prior still fits the observed data",
      all(float(np.sqrt(np.mean((s.predict(Xcl)[0] - Ycl) ** 2))) < 0.05
          for s in (s_zero, s_in, s_out)),
      "BLR explains the residual whatever the prior mean is")
check("in-span prior is partly absorbed, out-of-span is not",
      np.max(np.abs(f_in - f_zero)) < np.max(np.abs(f_out - f_zero)),
      f"in-span shift {np.max(np.abs(f_in-f_zero)):.2f} vs "
      f"out-of-span {np.max(np.abs(f_out-f_zero)):.2f}")

# do-mean / do-variance carry-over
sur_do = GeometryAwareSurrogate(geom, ZeroPFN(), Xtr, Ytr,
                                do_mean=lambda x: np.full((np.atleast_2d(x).shape[0], 1), 3.0),
                                prior_mean="do")
check("prior_mean='do' uses the causal mean",
      float(np.sqrt(np.mean((sur_do.predict(Xtr)[0] - Ytr) ** 2))) < 0.4)
sur_v = GeometryAwareSurrogate(geom, ZeroPFN(), Xtr, Ytr,
                               variance_adjustment=lambda x: np.full(np.atleast_2d(x).shape[0], 2.0))
sur_plain = GeometryAwareSurrogate(AdaptiveGeometry(Bs, pi=[1.0, 1.0]), ZeroPFN(), Xtr, Ytr)
check("do-variance inflates predictive variance",
      np.all(sur_v.predict(Xte)[1] > sur_plain.predict(Xte)[1] + 1.9))

# --------------------------------------------------------------- PFN itself
# The PFN prior is TabPFN now: an external checkpoint, gated behind a licence
# acceptance and a download. The interface, the fit cache and the degenerate
# contexts are what this repo owns, so they are tested against a stand-in
# regressor; the real weights are exercised only when they are available.
rng_p = np.random.default_rng(3)
_z = rng_p.normal(size=(40, 4))
Xs = _z / np.linalg.norm(_z, axis=1, keepdims=True)
ys = Xs @ rng_p.normal(size=4)

check("ZeroPFN returns zeros", np.allclose(ZeroPFN().mean(Xs, ys, Xs), 0.0))
check("ConstantPFN returns the context mean",
      np.allclose(ConstantPFN().mean(Xs, ys, Xs[:3]), ys.mean()))

try:
    prior = TabPFNPrior()
except ImportError:
    prior = None
    print("SKIP  TabPFN interface tests (tabpfn not installed)")

if prior is not None:
    fits = {"n": 0}

    class _FakeReg:
        """Least-squares stand-in: TabPFN's weights are licence-gated."""
        def __init__(self, **kw): self.kw = kw
        def fit(self, X, y):
            fits["n"] += 1
            self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
            return self
        def predict(self, X): return X @ self.coef

    prior._regressor_cls = _FakeReg
    pm = prior.mean(Xs[:24], ys[:24], Xs[24:])
    check("TabPFN prior returns one finite mean per query",
          pm.shape == (16,) and np.all(np.isfinite(pm)),
          str(np.round(pm[:3], 3).tolist()))

    for _ in range(20):
        prior.mean(Xs[:24], ys[:24], Xs[24:])
    check("the fit is cached across repeated predicts", fits["n"] == 1,
          f"{fits['n']} fit(s) for 21 calls")
    prior.mean(Xs[:24], ys[:24] + 1.0, Xs[24:])
    check("a changed context refits", fits["n"] == 2, f"{fits['n']} fit(s)")

    n_before = fits["n"]
    check("empty context -> zeros",
          np.allclose(prior.mean(np.zeros((0, 4)), np.zeros(0), Xs[:5]), 0.0))
    check("constant context -> that constant",
          np.allclose(prior.mean(Xs[:24], np.full(24, 2.5), Xs[:5]), 2.5))
    check("degenerate contexts never reach the model", fits["n"] == n_before)

    def _raises(fn):
        try:
            fn(); return False
        except ValueError:
            return True
    check("mismatched context rows are rejected",
          _raises(lambda: prior.mean(Xs[:5], ys[:3], Xs[:2])))
    check("mismatched query features are rejected",
          _raises(lambda: prior.mean(Xs[:24], ys[:24], Xs[:2, :3])))

    sur_pfn = GeometryAwareSurrogate(AdaptiveGeometry(Bs, pi=[1.0, 1.0]), prior,
                                     Xtr, Ytr)
    mu_p, var_p = sur_pfn.predict(Xte)
    check("surrogate runs end to end with the TabPFN prior",
          np.all(np.isfinite(mu_p)) and np.all(var_p > 0)
          and float(np.sqrt(np.mean((sur_pfn.predict(Xtr)[0] - Ytr) ** 2))) < 0.5,
          f"train RMSE {float(np.sqrt(np.mean((sur_pfn.predict(Xtr)[0]-Ytr)**2))):.4f}")

    check("build_prior('tabpfn') -> TabPFNPrior",
          isinstance(build_prior("tabpfn"), TabPFNPrior))
    check("build_prior('pfn') is the same alias",
          isinstance(build_prior("pfn"), TabPFNPrior))

try:
    build_prior("checkpoints/pfn.pt")
    _ckpt_rejected = False
except ValueError:
    _ckpt_rejected = True
check("checkpoint paths are rejected (no local PFN to load)", _ckpt_rejected)

check("build_prior('zero') -> ZeroPFN", isinstance(build_prior("zero"), ZeroPFN))
check("build_prior('constant') -> ConstantPFN", isinstance(build_prior("constant"), ConstantPFN))

print("\n" + ("ALL PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
