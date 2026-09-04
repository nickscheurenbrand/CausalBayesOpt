"""Import shim so the appendix-E tests run with or without GPy installed: only the GPy
symbols that sibling modules touch at import time are faked, never the code under test."""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def install():
    """Put the repo on sys.path and stub GPy only if it is missing."""
    for p in (ROOT, os.path.join(ROOT, "algorithms")):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import GPy  # noqa: F401
        return False
    except Exception:
        pass

    class _Param:
        def __init__(self, *a, **k):
            pass

    class _Kern:
        def __init__(self, *a, **k):
            pass

    def _mk(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    sys.modules.update({
        "GPy": _mk("GPy"),
        "GPy.core": _mk("GPy.core", Param=_Param, Mapping=object),
        "GPy.kern": _mk("GPy.kern"),
        "GPy.kern.src": _mk("GPy.kern.src"),
        "GPy.kern.src.kern": _mk("GPy.kern.src.kern", Kern=_Kern),
        "GPy.kern.src.psi_comp": _mk("GPy.kern.src.psi_comp",
                                     PSICOMP_RBF=object, PSICOMP_RBF_GPU=object),
        "GPy.kern.src.stationary": _mk("GPy.kern.src.stationary", Stationary=_Kern),
        "GPy.kern.src.rbf": _mk("GPy.kern.src.rbf", RBF=object),
        "GPy.models": _mk("GPy.models"),
        "GPy.models.gp_regression": _mk("GPy.models.gp_regression",
                                        GPRegression=object),
        "paramz": _mk("paramz"),
        "paramz.transformations": _mk("paramz.transformations", Logexp=object),
    })
    if "graphs.graph" not in sys.modules:
        pkg = _mk("graphs")
        pkg.__path__ = [os.path.join(ROOT, "algorithms", "graphs")]
        sys.modules.setdefault("graphs", pkg)
        sys.modules["graphs.graph"] = _mk("graphs.graph", GraphStructure=object)
    return True
