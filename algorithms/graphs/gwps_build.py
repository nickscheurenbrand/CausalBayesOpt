"""
Pure construction of the GWPS DAG subgraph (numpy + networkx only, no GPy/jax).

Shared by GwpsGraph (which adds the SEM) and gwps_diagnose.py (which needs only
the structure + weights). Keeping this dependency-light lets the diagnostic run
without the heavy modelling stack.
"""

import csv
import os

import networkx as nx
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EDGE_CSV = os.path.join(_HERE, "..", "..", "data", "gwps_direct_edges.csv")


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


def build_gwps_dag(
    edge_csv=DEFAULT_EDGE_CSV,
    target=None,
    max_nodes=60,
    top_k_parents=8,
    weight_scale=1.0,
):
    """
    Returns dict with:
      int_graph      : nx.DiGraph on integer nodes 0..N-1 (a DAG)
      W              : N x N weight matrix, W[parent, child] = G_hat*weight_scale
      index_to_ensg  : {int: ENSG}
      target_index   : int index of the (final) target
    """
    full = _load_edges(edge_csv)
    sparse = _sparsify(full, top_k_parents)

    if target is None:
        target = _auto_seed_target(sparse)
    elif target not in sparse:
        raise ValueError(f"target {target} not in the (sparsified) graph")

    genes = _bounded_ancestors(sparse, target, max_nodes)
    sub = _dagify(sparse.subgraph(genes).copy())

    if sub.in_degree(target) == 0:
        target = max(sub.nodes(), key=lambda n: sub.in_degree(n))
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
    }
