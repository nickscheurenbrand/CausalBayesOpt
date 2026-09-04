"""Pure construction of the GWPS DAG subgraph (numpy + networkx only, no GPy/jax).
Shared by GwpsGraph and gwps_diagnose.py, so it stays dependency-light."""

import csv
import os

import networkx as nx
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# graphs/ lives at the repo root (moved from algorithms/graphs/), so data/ is one
# level up, not two.
DEFAULT_EDGE_CSV = os.path.join(_HERE, "..", "data", "gwps_direct_edges.csv")


def _load_edges(edge_csv):
    g = nx.DiGraph()
    with open(edge_csv, newline="") as f:
        for row in csv.DictReader(f):
            w = row["G_hat"]
            if w in ("", "NA"):
                continue
            g.add_edge(row["Exposure"], row["Outcome"], w=float(w))
    return g


def _sparsify(g, top_k_parents):
    s = nx.DiGraph()
    s.add_nodes_from(g.nodes())
    for node in g.nodes():
        in_edges = sorted(g.in_edges(node, data=True), key=lambda e: -abs(e[2]["w"]))
        for u, v, d in in_edges[:top_k_parents]:
            s.add_edge(u, v, w=d["w"])
    return s


def _auto_seed_target(g):
    return max(g.nodes(), key=lambda n: g.in_degree(n))


def _bounded_ancestors(g, target, max_nodes):
    keep = {target}
    frontier = [target]
    while frontier and len(keep) < max_nodes:
        nxt = []
        for n in frontier:
            for p in g.predecessors(n):
                if p not in keep:
                    keep.add(p)
                    nxt.append(p)
                    if len(keep) >= max_nodes:
                        break
            if len(keep) >= max_nodes:
                break
        frontier = nxt
    return keep


def _dagify(g):
    g = g.copy()
    for u, v in list(g.edges()):
        if g.has_edge(u, v) and g.has_edge(v, u):
            if abs(g[u][v]["w"]) >= abs(g[v][u]["w"]):
                if g.has_edge(v, u):
                    g.remove_edge(v, u)
            elif g.has_edge(u, v):
                g.remove_edge(u, v)
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g, orientation="original")
        edges = [(e[0], e[1]) for e in cycle]
        u, v = min(edges, key=lambda e: abs(g[e[0]][e[1]]["w"]))
        g.remove_edge(u, v)
    return g


def _non_ancestor_candidates(sparse, target, keep, n_wanted):
    """Up to `n_wanted` genes that provably CANNOT be ancestors of the target,
    ranked by how many edges they share with the kept genes so they stay wired in."""
    if n_wanted <= 0:
        return []
    forbidden = nx.ancestors(sparse, target) | {target}
    scored = []
    for n in sparse.nodes():
        if n in forbidden or n in keep:
            continue
        deg = sum(1 for k in keep if sparse.has_edge(k, n) or sparse.has_edge(n, k))
        if deg:
            scored.append((-deg, n))
    scored.sort()
    return [n for _, n in scored[:n_wanted]]


def build_gwps_dag(
    edge_csv=DEFAULT_EDGE_CSV,
    target=None,
    max_nodes=60,
    top_k_parents=8,
    weight_scale=1.0,
    n_non_ancestors=0,
):
    """Build a bounded DAG subgraph around `target` from the sparsified gene network.
    Returns dict with int_graph, weight matrix W, index_to_ensg, target_index, n_non_ancestors."""
    full = _load_edges(edge_csv)
    sparse = _sparsify(full, top_k_parents)

    if target is None:
        target = _auto_seed_target(sparse)
    elif target not in sparse:
        raise ValueError(f"target {target} not in the (sparsified) graph")

    n_non_ancestors = max(int(n_non_ancestors), 0)
    genes = _bounded_ancestors(sparse, target, max_nodes - n_non_ancestors)
    added = set(_non_ancestor_candidates(sparse, target, genes, n_non_ancestors))
    genes = set(genes) | added
    sub = _dagify(sparse.subgraph(genes).copy())

    if sub.in_degree(target) == 0:
        # fall back to the best-connected node, but never to one of the added
        # non-ancestors -- retargeting onto those would void the guarantee that
        # they have no causal path to the target
        pool = [n for n in sub.nodes() if n not in added] or list(sub.nodes())
        target = max(pool, key=lambda n: sub.in_degree(n))
    if sub.in_degree(target) == 0:
        raise ValueError("no node in the subgraph has parents; loosen the params")

    genes_sorted = sorted(sub.nodes())
    idx = {g: i for i, g in enumerate(genes_sorted)}
    N = len(genes_sorted)

    int_graph = nx.DiGraph()
    int_graph.add_nodes_from(range(N))
    W = np.zeros((N, N))
    for u, v, d in sub.edges(data=True):
        int_graph.add_edge(idx[u], idx[v])
        W[idx[u], idx[v]] = d["w"] * weight_scale

    return {
        "int_graph": int_graph,
        "W": W,
        "index_to_ensg": {i: g for g, i in idx.items()},
        "target_index": idx[target],
        "n_non_ancestors": len(added & set(sub.nodes())),
    }
