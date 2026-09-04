"""Latent (hidden) confounder injection for the boundary experiments: Z is a common
cause of X and Y, stripped from D_O so PARENT_SCALE runs unmodified on the base graph."""

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
    """Deterministically choose the observed variable X the confounder attaches to:
    "non_parent" picks a spectator, "true_parent" picks one of the target's true parents."""
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
    """Minimal `.SEM`/`.get_error_distribution` object for sample_model; not a full GraphStructure."""

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
    """Wrap base_graph's SEM with latent confounder terms from `confounders`.
    Returns (sampler, meta): sampler draws confounded D_O, meta logs the config."""
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
