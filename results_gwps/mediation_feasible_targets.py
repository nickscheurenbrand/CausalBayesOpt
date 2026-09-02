"""Which targets can host BOTH mediation arms?

The mediation experiment needs, for one graph and one target, a mediated-ancestor
pool AND a non-ancestor pool each of size >= k = |parents(target)|, so the
"ancestor" and "non_ancestor" arms can be drawn at matched cardinality. Most
default targets fail: they were picked by in-degree or by hand, before ancestry
mattered.

This scans every node of a saved graph and lists the targets that work. Run it
after any topology change (a new Erdos exp_edges, a different DREAM network, a
re-carved GWPS) and feed the winners to run_mediation.py.

Graph structure is read from the saved random pickles (they store Edges,
Variables, Parents), so no GPy/modelling stack is needed.

Usage:
    python results_erdos/mediation_feasible_targets.py
    python results_erdos/mediation_feasible_targets.py --top 25
"""

import argparse
import os
import pickle
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    "Erdos50": "results/boundary_tracking_erdos_random/Erdos50/"
               "run1_cbo_unknown_dr2_boundary_EI_200_2.pickle",
    "Erdos100": "results/boundary_tracking_erdos_random/Erdos100/"
                "run1_cbo_unknown_dr2_boundary_EI_200_2.pickle",
    "Size50-Ecoli1": "results/boundary_tracking_dream_random/Size50-Ecoli1/"
                     "run1_cbo_unknown_dr2_boundary_EI_200_1_nonlinear.pickle",
    "Size100-Ecoli1": "results/boundary_tracking_dream_random/Size100-Ecoli1/"
                      "run1_cbo_unknown_dr2_boundary_EI_200_1_nonlinear.pickle",
}


def _bfs(adj, source):
    depth = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w not in depth:
                depth[w] = depth[u] + 1
                q.append(w)
    return depth


def feasible_targets(variables, edges):
    """[(target, k, mediated pool, non-ancestor pool, ancestor depth)]."""
    fwd = {v: [] for v in variables}
    rev = {v: [] for v in variables}
    for a, b in edges:
        fwd.setdefault(str(a), []).append(str(b))
        rev.setdefault(str(b), []).append(str(a))

    rows = []
    for t in variables:
        parents = set(rev.get(t, []))
        k = len(parents)
        if k == 0:
            continue
        depths = _bfs(rev, t)
        ancestors = set(depths) - {t}
        mediated = ancestors - parents
        non_ancestors = [v for v in variables if v != t and v not in ancestors]
        if len(mediated) >= k and len(non_ancestors) >= k:
            rows.append((t, k, len(mediated), len(non_ancestors),
                         max(depths.values())))
    # widest k first, then the largest mediated pool -- the scarce one
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="rows to show per graph (0 = all)")
    args = ap.parse_args()

    for label, rel in SOURCES.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"\n### {label}: pickle not found ({rel})")
            continue
        with open(path, "rb") as f:
            r = pickle.load(f)
        variables = [str(v) for v in r["Variables"]]
        rows = feasible_targets(variables, r["Edges"])
        print(f"\n### {label}: {len(rows)} feasible targets of {len(variables)} "
              f"(current: {r['Target']})")
        print(f"{'target':>7} {'k=|par|':>7} {'mediated':>8} {'non-anc':>8} "
              f"{'depth':>6}")
        for t, k, m, n, d in (rows if args.top == 0 else rows[:args.top]):
            print(f"{t:>7} {k:>7} {m:>8} {n:>8} {d:>6}")

    print("\nGWPS is not listed: its carve is the target's ancestor closure, so a "
          "null pool must be created with --n_non_ancestors rather than found "
          "by re-targeting.")


if __name__ == "__main__":
    main()
