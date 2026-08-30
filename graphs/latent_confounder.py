"""
Latent (hidden) confounder injection for the boundary experiments.

A confounder Z of a manipulable variable X and the target Y is a common cause
(Z->X, Z->Y). We make Z *latent*: it is present in the data-generating SEM (so it
correlates X and Y in the OBSERVATIONAL data) but is stripped from the observed
data and never given to the algorithm. Because neither the doubly-robust
parent-ID nor the do-effect estimation deconfounds, a hidden Z fools the method:
X shows a spurious X-Y association and can be mis-identified as a parent.

Key trick (no core-algorithm surgery): under do(X) the Z->X backdoor is cut, so
the *true* interventional effect of a non-parent X is flat. The confounding
therefore only needs to live in the observational data. We generate confounded
D_O from a SEM with Z, strip Z, and run PARENT_SCALE on the ORIGINAL base graph
(no Z) -- whose SEM already yields the correct interventional outcomes.

Injection is done at the SEM-function level: add a latent root Z ~ N(0, sigma_z)
and add w*Z to X's and Y's structural functions. This is graph-agnostic --
identical for the linear Erdos SEM and the NumPy nonlinear DREAM SEM.

Public API:
    pick_confounded_x(base_graph, x_kind)          -> chosen X (str)
    inject_latent_confounders(base_graph, confs)   -> ConfoundedSampler, meta
    generate_confounded_observational_data(...)     -> D_O (Z stripped)
"""

import logging
from collections import OrderedDict, defaultdict

import networkx as nx
import numpy as np

from utils.sem_sampling import sample_model


def _ancestors(graph, node):
    try:
        return set(nx.ancestors(graph.G, node))
    except Exception:
        return set()


def _descendants(graph, node):
    try:
        return set(nx.descendants(graph.G, node))
    except Exception:
        return set()


def pick_confounded_x(base_graph, x_kind: str) -> str:
    """
    Deterministically choose the observed variable X the hidden confounder
    attaches to.

    x_kind == "non_parent": a manipulable variable that is NOT a parent of the
        target and ideally neither an ancestor nor a descendant of it (a
        "spectator"), so the ONLY X-Y association is the injected Z. Tests the
        false-positive case.
    x_kind == "true_parent": one of the target's true parents. Tests the
        effect-corruption case.
    """
    target = base_graph.target
    manipulable = [v for v in base_graph.variables if v != target]
    parents = set(base_graph.parents[target])

    if x_kind == "true_parent":
        if not parents:
            raise ValueError("target has no true parents; cannot use x_kind=true_parent")
        # deterministic: lowest-index true parent
        return sorted(parents, key=lambda v: (len(v), v))[0]

    if x_kind == "non_parent":
        anc = _ancestors(base_graph, target)
        desc = _descendants(base_graph, target)
        spectators = [
            v for v in manipulable if v not in parents and v not in anc and v not in desc
        ]
        pool = spectators if spectators else [v for v in manipulable if v not in parents]
        if not pool:
            raise ValueError("no manipulable non-parent available for confounding")
        return sorted(pool, key=lambda v: (len(v), v))[0]

    raise ValueError(f"unknown x_kind {x_kind!r}")


class ConfoundedSampler:
    """
    Minimal object exposing exactly what sample_model needs (`.SEM` and
    `.get_error_distribution`) to draw observational data from the confounded
    SEM. Not a full GraphStructure -- it only generates D_O.
    """

    def __init__(self, base_graph, confounded_sem, latent, seed=None):
        self._base = base_graph
        self.SEM = confounded_sem
        self._latent = latent  # list of (z_name, sigma_z)
        self._rng = np.random.default_rng(seed)

    def get_error_distribution(self, noiseless=False):
        err = dict(self._base.get_error_distribution(noiseless))
        for z_name, sigma in self._latent:
            err[z_name] = self._rng.normal(scale=sigma, size=1)
        return err


def _wrap_add_terms(base_fn, terms):
    # add sum_k w_k * sample[z_k] to the base structural function's output
    return lambda epsilon, sample, base_fn=base_fn, terms=terms: base_fn(
        epsilon, sample
    ) + sum(w * sample[z_name] for z_name, w in terms)


def inject_latent_confounders(base_graph, confounders, seed=None):
    """
    confounders: list of dicts/tuples (x, w_zx, w_zy, sigma_z).
    Returns (sampler, meta) where sampler draws confounded observational data
    and meta records the confounder configuration for saving/analysis.
    """
    target = base_graph.target

    add_terms = defaultdict(list)  # observed node -> [(z_name, weight)]
    latent = []                    # [(z_name, sigma_z)]
    meta = []
    for k, conf in enumerate(confounders):
        x = conf["x"]
        w_zx = float(conf.get("w_zx", 2.0))
        w_zy = float(conf.get("w_zy", 2.0))
        sigma_z = float(conf.get("sigma_z", 1.0))
        z_name = f"Z_conf_{k}"
        add_terms[x].append((z_name, w_zx))
        add_terms[target].append((z_name, w_zy))
        latent.append((z_name, sigma_z))
        meta.append(
            {
                "x": x,
                "z_name": z_name,
                "w_zx": w_zx,
                "w_zy": w_zy,
                "sigma_z": sigma_z,
                "is_true_parent": x in set(base_graph.parents[target]),
            }
        )

    # build the confounded SEM: latent roots first (topological), then base
    # nodes with X/target wrapped to receive their Z terms
    confounded_sem = OrderedDict()
    for z_name, _ in latent:
        confounded_sem[z_name] = lambda epsilon, sample: epsilon
    for node, fn in base_graph.SEM.items():
        if node in add_terms:
            confounded_sem[node] = _wrap_add_terms(fn, add_terms[node])
        else:
            confounded_sem[node] = fn

    sampler = ConfoundedSampler(base_graph, confounded_sem, latent, seed=seed)
    logging.info(
        f"Injected {len(confounders)} latent confounder(s): "
        + ", ".join(f"{m['z_name']}->({m['x']},{target})" for m in meta)
    )
    return sampler, meta


def generate_confounded_observational_data(
    base_graph, confounders, n_obs, seed=None
):
    """
    Sample n_obs observational rows from the confounded SEM and strip the latent
    Z columns, returning D_O over the base graph's observed variables only.
    """
    sampler, meta = inject_latent_confounders(base_graph, confounders, seed=seed)
    D_O_full = sample_model(
        sampler.SEM, sample_count=n_obs, graph=sampler, use_iscm=False
    )
    # Preserve the SEM (topological) key order and drop the latent Z_conf_* keys.
    # Iterating base_graph.variables (numeric order) instead would put D_O keys
    # out of topological order, which breaks DoublyRobustModel.run_method's
    # groundtruth indexing when a true parent is the highest-indexed variable.
    base_vars = set(base_graph.variables)
    D_O = {v: D_O_full[v] for v in D_O_full if v in base_vars}
    return D_O, meta
