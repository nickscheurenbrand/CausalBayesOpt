"""How far from the target are the RANDOM non-parents we intervene on?

The random runs (scripts_random/random_boundary_script.py) force a joint
intervention on k random NON-parents of the target. "Non-parent" says nothing
about graph position: a chosen node can still be a grandparent (real causal
effect on Y, just indirect), a descendant of Y (no causal effect at all), or
d-connected to Y only through a common ancestor (association, no effect).
That distinction is what makes the boundary result interpretable, so measure it.

Every random pickle already stores Edges / Parents / Target / Random_Set, so
this is post-hoc -- nothing has to be re-run.

Per chosen node v we report:
  dist_to_target   shortest DIRECTED path length v -> Y  (inf if not an ancestor)
  dist_from_target shortest DIRECTED path length Y -> v  (inf if not a descendant)
  skeleton_dist    shortest UNDIRECTED path length (inf if in another component)
  relation         ancestor / descendant / confounded / collider_only /
                   disconnected
                   confounded    = not an ancestor or descendant, but shares an
                                   ancestor with Y: associated with Y without
                                   affecting it (open backdoor path)
                   collider_only = joined to Y in the skeleton only through
                                   colliders, so marginally independent of Y
                   disconnected  = no undirected path to Y at all
  backdoor_dist    for 'confounded', hops from the nearest common ancestor to Y

Usage:
    python results_erdos/random_set_distance.py                 # per-node table
    python results_erdos/random_set_distance.py --by-run        # per-run summary
    python results_erdos/random_set_distance.py --csv out.csv
"""

import argparse
import os
import pickle
from collections import deque

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# family -> (results subdir, graph tags, pickle filename template)
FAMILIES = {
    "erdos": ("results/boundary_tracking_erdos_random",
              ["Erdos50", "Erdos100"],
              "run{run}_cbo_unknown_dr2_boundary_EI_200_2.pickle"),
    "dream": ("results/boundary_tracking_dream_random",
              ["Size50-Ecoli1", "Size100-Ecoli1"],
              "run{run}_cbo_unknown_dr2_boundary_EI_200_1_nonlinear.pickle"),
    "gwps":  ("results/boundary_tracking_gwps_random",
              ["gwps_n60_ws3"],
              "run{run}_cbo_unknown_dr2_boundary_EI_200_1.pickle"),
}
RUNS = [1, 2, 3, 4, 5]
EDGE = 0.20          # outer-20% edge zone, matches the notebooks


def adjacency(edges, variables):
    """Forward, reverse and undirected adjacency dicts over `variables`."""
    fwd = {v: [] for v in variables}
    rev = {v: [] for v in variables}
    und = {v: [] for v in variables}
    for u, v in edges:
        u, v = str(u), str(v)
        fwd.setdefault(u, []).append(v)
        rev.setdefault(v, []).append(u)
        und.setdefault(u, []).append(v)
        und.setdefault(v, []).append(u)
    return fwd, rev, und


def bfs_depths(adj, source):
    """{node: hop count} for every node reachable from `source`."""
    depth = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w not in depth:
                depth[w] = depth[u] + 1
                q.append(w)
    return depth


def per_node(r):
    """One row per chosen non-parent, with its graph distance to the target."""
    target = str(r["Target"])
    variables = [str(v) for v in r["Variables"]]
    fwd, rev, und = adjacency(r["Edges"], variables)

    anc_of_y = bfs_depths(rev, target)   # {v: directed hops v -> Y}
    desc_of_y = bfs_depths(fwd, target)  # {v: directed hops Y -> v}
    skel = bfs_depths(und, target)

    rows = []
    for v in [str(x) for x in r["Random_Set"]]:
        d_to = anc_of_y.get(v, np.inf)
        d_from = desc_of_y.get(v, np.inf)

        if np.isfinite(d_to):
            relation, backdoor = "ancestor", np.nan
        elif np.isfinite(d_from):
            relation, backdoor = "descendant", np.nan
        else:
            # nearest node that is an ancestor of BOTH v and Y -> a backdoor path
            anc_of_v = bfs_depths(rev, v)
            shared = set(anc_of_v) & set(anc_of_y)
            if shared:
                relation = "confounded"
                backdoor = min(anc_of_y[a] for a in shared)
            elif np.isfinite(skel.get(v, np.inf)):
                # joined to Y in the skeleton only via colliders
                relation, backdoor = "collider_only", np.nan
            else:
                relation, backdoor = "disconnected", np.nan

        rows.append(dict(
            node=v,
            dist_to_target=d_to,
            dist_from_target=d_from,
            skeleton_dist=skel.get(v, np.inf),
            relation=relation,
            backdoor_dist=backdoor,
            corr=r.get("Chosen_Corr_With_Target", {}).get(v, np.nan),
            n_parents_of_target=len(r["True_Parents"]),
        ))
    return pd.DataFrame(rows)


def outer20_pct(r):
    """% of intervened dimensions landing in the outer-20% of their range."""
    rng = r["Intervention_Ranges"]
    pos = []
    for vs, vals in zip(r["Intervention_Set"], r["Intervention_Value"]):
        for var, val in zip(vs, vals):
            lo, hi = rng[var]
            if hi > lo:
                pos.append((float(val) - lo) / (hi - lo))
    pos = np.asarray(pos)
    return 100 * float(((pos <= EDGE) | (pos >= 1 - EDGE)).mean())


def collect():
    rows = []
    for family, (subdir, tags, tmpl) in FAMILIES.items():
        for tag in tags:
            for run in RUNS:
                path = os.path.join(ROOT, subdir, tag, tmpl.format(run=run))
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    r = pickle.load(f)
                best = np.asarray(r["Best_Y"], float)
                d = per_node(r)
                d.insert(0, "family", family)
                d.insert(1, "graph", tag)
                d.insert(2, "run", run)
                d["target"] = str(r["Target"])
                d["strict_bnd_pct"] = 100 * float(
                    np.asarray(r["Boundary_Percentage"], float).mean())
                d["outer20_pct"] = outer20_pct(r)
                d["improvement"] = best[-1] - best[0]
                rows.append(d)
    if not rows:
        raise SystemExit("no random pickles found")
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--by-run", action="store_true",
                   help="Aggregate to one row per run instead of per node.")
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    nodes = collect()

    if args.by_run:
        out = (nodes.groupby(["family", "graph", "run"])
               .agg(k=("node", "size"),
                    n_ancestor=("relation", lambda s: (s == "ancestor").sum()),
                    n_descendant=("relation", lambda s: (s == "descendant").sum()),
                    n_confounded=("relation", lambda s: (s == "confounded").sum()),
                    n_collider=("relation", lambda s: (s == "collider_only").sum()),
                    n_disconnected=("relation", lambda s: (s == "disconnected").sum()),
                    min_dist_to_target=("dist_to_target", "min"),
                    mean_skeleton_dist=("skeleton_dist",
                                        lambda s: s.replace(np.inf, np.nan).mean()),
                    mean_abs_corr=("corr", lambda s: s.abs().mean()),
                    strict_bnd_pct=("strict_bnd_pct", "first"),
                    outer20_pct=("outer20_pct", "first"),
                    improvement=("improvement", "first"))
               .reset_index())
    else:
        out = nodes

    pd.set_option("display.width", 200, "display.max_rows", 400)
    print(out.round(3).to_string(index=False))

    print("\n--- relation counts (all chosen nodes, pooled) ---")
    print(nodes.groupby(["family", "graph"]).relation
          .value_counts().unstack(fill_value=0).to_string())

    print("\n--- skeleton distance to target (finite only) ---")
    fin = nodes[np.isfinite(nodes.skeleton_dist)]
    print(fin.groupby(["family", "graph"]).skeleton_dist
          .describe()[["count", "mean", "min", "50%", "max"]].round(2).to_string())

    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
