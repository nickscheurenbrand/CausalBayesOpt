"""E.7: drive the EXISTING acquisition code (CausalExpectedImprovement, the gradient optimizer, the
surrogate) against the new surrogate, stubbing only the GPy symbols cbo_classes needs to import."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs
_stubs.install()

from emukit.core import ContinuousParameter, ParameterSpace

from utils.cbo_classes import (CausalExpectedImprovement,
                               CausalGradientAcquisitionOptimizer,
                               CausalUpperConfidenceBound)
from utils.geometric_surrogate import GeometryAwareSurrogate
from utils.geometry import AdaptiveGeometry
from utils.pfn_prior import ZeroPFN

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond: fails.append(name)

rng = np.random.default_rng(0)
B = np.array([[-2.0, 2.0], [-2.0, 2.0]])
f = lambda X: (2.0 * X[:, 0] - 1.0 * X[:, 1]).reshape(-1, 1)   # min at (-2, +2)
X = rng.uniform(-2, 2, (12, 2)); Y = f(X)
sur = GeometryAwareSurrogate(AdaptiveGeometry(B, pi=[1.0, 1.0]), ZeroPFN(), X, Y)

space = ParameterSpace([ContinuousParameter("0", -2.0, 2.0),
                        ContinuousParameter("1", -2.0, 2.0)])
acq = CausalExpectedImprovement(float(Y.min()), "min", sur)
check("EI sees a differentiable model", acq.has_gradients)

val = acq.evaluate(rng.uniform(-2, 2, (5, 2)))
check("EI evaluates", val.shape == (5, 1) and np.all(np.isfinite(val)), str(val.shape))
v, dv = acq.evaluate_with_gradients(rng.uniform(-2, 2, (5, 2)))
check("EI evaluates with gradients", v.shape == (5, 1) and dv.shape == (5, 2))

x_new, _ = CausalGradientAcquisitionOptimizer(space).optimize(acq)
check("gradient optimizer returns a point in the space",
      x_new.shape == (1, 2) and np.all(x_new >= -2.001) and np.all(x_new <= 2.001),
      str(np.round(x_new, 3).tolist()))
check("EI drives toward the true optimum corner (-2, +2)",
      x_new[0, 0] < -1.0 and x_new[0, 1] > 1.0, str(np.round(x_new, 3).tolist()))

ucb = CausalUpperConfidenceBound(float(Y.min()), "min", sur, beta=2.0)
x_ucb, _ = CausalGradientAcquisitionOptimizer(space).optimize(ucb)
check("UCB path also runs", x_ucb.shape == (1, 2) and np.all(np.isfinite(x_ucb)))

# a short closed BO loop: EI + surrogate refit should improve the incumbent
best = float(Y.min()); Xc, Yc = X.copy(), Y.copy()
for _ in range(8):
    s = GeometryAwareSurrogate(AdaptiveGeometry(B, pi=[1.0, 1.0]), ZeroPFN(), Xc, Yc)
    xn, _ = CausalGradientAcquisitionOptimizer(space).optimize(
        CausalExpectedImprovement(best, "min", s))
    Xc = np.vstack([Xc, xn]); Yc = np.vstack([Yc, f(xn)])
    best = float(Yc.min())
check("closed BO loop reaches the optimum", abs(best - (-6.0)) < 0.15,
      f"best {best:.4f} vs true min -6.0")

# the adaptive geometry changes where the acquisition looks
sur_pi = GeometryAwareSurrogate(AdaptiveGeometry(B, pi=[1.0, 0.02]), ZeroPFN(), X, Y)
x_pi, _ = CausalGradientAcquisitionOptimizer(space).optimize(
    CausalExpectedImprovement(float(Y.min()), "min", sur_pi))
check("pi reshapes the acquisition surface",
      not np.allclose(x_pi, x_new, atol=1e-3),
      f"pi=[1,1] -> {np.round(x_new,3).tolist()}, pi=[1,0.02] -> {np.round(x_pi,3).tolist()}")

print("\n" + ("ALL PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
