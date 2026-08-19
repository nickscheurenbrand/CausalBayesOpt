"""Ground-truth graph queries and true-response-surface search.

Two things the boundary experiments need but the algorithm never sees, both
computed from the TRUE graph/SEM rather than from any posterior or surrogate:

1. Ancestry stratification (`classify_node`, `stratified_non_parents`).
   "Non-parent" does not mean "no causal effect": a non-parent can still be an
   ANCESTOR, whose effect on the target is real but MEDIATED through other
   nodes. Only a non-ancestor satisfies P(Y | do(v)) = P(Y). Stratifying the
   random arm this way separates "intervened on something causally irrelevant"
   from "intervened on something causally relevant but indirect".

2. The true optimum of E[Y | do(X_S = x)] over the intervention box
   (`find_true_optimum`), and whether it is INTERIOR or on the BOUNDARY.
   Without this, an intervention landing on the range edge is uninterpretable:
   it may be the optimizer degenerating, or it may be the correct answer. For a
   linear SEM E[Y | do(x)] is affine in x, so the optimum is always at a corner
   and edge-seeking is CORRECT behaviour; with nonlinear mechanisms an interior
   optimum can exist and edge-seeking is then a genuine failure. This function
   tells the two cases apart per intervention set.

E[Y | do(x)] is estimated by Monte Carlo over the true SEM's noise. All
evaluations reuse the SAME noise draws (common random numbers, via
`graph.set_seed(crn_seed)` before each call), so the estimated surface is a
deterministic function of x and points are compared on equal footing. Note that
`--noiseless` does NOT suppress SEM noise for these graph classes -- their
`get_error_distribution` ignores the flag -- so the MC average is required, a
single draw would not give E[Y | do(x)].
"""

from collections import deque
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.sem_sampling import sample_model

# ---------------------------------------------------------------- ancestry ---

RELATIONS = (
    "parent",
    "ancestor",        # non-parent ancestor: real but MEDIATED effect on Y
    "descendant",      # no effect on Y
    "confounded",      # shares an ancestor with Y: associated, no effect
    "collider_only",   # skeleton-connected only via colliders: independent of Y
    "disconnected",
)


def _bfs(adj: Dict[str, List[str]], source: str) -> Dict[str, int]:
    """{node: hop count} for everything reachable from `source`."""
    depth = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w not in depth:
                depth[w] = depth[u] + 1
                q.append(w)
    return depth


def _adjacencies(graph):
    """Forward / reverse / undirected adjacency over the graph's variables."""
    fwd = {str(v): [] for v in graph.variables}
    rev = {str(v): [] for v in graph.variables}
    und = {str(v): [] for v in graph.variables}
    for child, parents in graph.parents.items():
        for p in parents:
            fwd.setdefault(str(p), []).append(str(child))
            rev.setdefault(str(child), []).append(str(p))
            und.setdefault(str(p), []).append(str(child))
            und.setdefault(str(child), []).append(str(p))
    return fwd, rev, und


def ancestor_depths(graph) -> Dict[str, int]:
    """{v: shortest directed hops v -> target}; only ancestors appear."""
    _, rev, _ = _adjacencies(graph)
    d = _bfs(rev, str(graph.target))
    d.pop(str(graph.target), None)
    return d


def classify_nodes(graph) -> Tuple[Dict[str, str], Dict[str, float]]:
    """(relation, directed distance to target) for every non-target node.

    `relation` is one of RELATIONS; the distance is the number of directed hops
    v -> target (inf when v is not an ancestor).
    """
    fwd, rev, und = _adjacencies(graph)
    target = str(graph.target)
    parents = {str(p) for p in graph.parents[graph.target]}

    anc = _bfs(rev, target)      # v -> Y
    desc = _bfs(fwd, target)     # Y -> v
    skel = _bfs(und, target)

    relation, distance = {}, {}
    for raw in graph.variables:
        v = str(raw)
        if v == target:
            continue
        distance[v] = float(anc.get(v, np.inf))
        if v in parents:
            relation[v] = "parent"
        elif v in anc:
            relation[v] = "ancestor"
        elif v in desc:
            relation[v] = "descendant"
        elif set(_bfs(rev, v)) & set(anc):
            relation[v] = "confounded"
        elif v in skel:
            relation[v] = "collider_only"
        else:
            relation[v] = "disconnected"
    return relation, distance


def stratified_pools(graph) -> Dict[str, List[str]]:
    """Candidate pools for the random arm, keyed by stratum.

    any          -- every non-parent, non-target node (the original behaviour)
    ancestor     -- non-parent ANCESTORS: mediated, causally relevant
    non_ancestor -- non-ancestors: the true causal null, P(Y | do(v)) = P(Y)
    """
    relation, _ = classify_nodes(graph)
    pools = {"any": [], "ancestor": [], "non_ancestor": []}
    for v, rel in relation.items():
        if rel == "parent":
            continue
        pools["any"].append(v)
        if rel == "ancestor":
            pools["ancestor"].append(v)
        else:
            pools["non_ancestor"].append(v)
    return {k: _sorted_nodes(v) for k, v in pools.items()}


def _sorted_nodes(nodes: Sequence[str]) -> List[str]:
    """Numeric ordering where possible, so tuple keys are stable."""
    try:
        return sorted(nodes, key=lambda v: int(v))
    except ValueError:
        return sorted(nodes)


def stratified_non_parents(graph, k: int, seed: int, category: str = "any"):
    """k distinct non-parents drawn from `category`, using `seed`.

    Drop-in replacement for the original pick_random_non_parents: category
    "any" reproduces it exactly (same pool, same rng, same ordering).
    """
    if category not in ("any", "ancestor", "non_ancestor"):
        raise ValueError(f"unknown category {category!r}")
    pool = stratified_pools(graph)[category]
    if len(pool) < k:
        raise ValueError(
            f"category {category!r} has only {len(pool)} candidates, need {k} "
            f"(target {graph.target} in this graph cannot support this arm)"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(pool, dtype=object), size=k, replace=False)
    return tuple(_sorted_nodes(chosen))


def linear_total_effects(graph) -> Optional[Dict[str, float]]:
    """{v: total causal effect of v on the target}, or None if not linear.

    For a linear SEM with weight matrix W[parent, child], the total effect is
    (I - W)^-1 [v, target] -- the sum over directed paths of the products of
    their edge weights, i.e. direct plus all MEDIATED contributions.
    """
    if getattr(graph, "nonlinear", False):
        return None
    W = getattr(graph, "weighted_adjacency_matrix", None)
    if W is None:
        W = getattr(getattr(graph, "causal_env", None),
                    "weighted_adjacency_matrix", None)
    if W is None:
        return None
    W = np.asarray(W, dtype=float)
    try:
        M = np.linalg.inv(np.eye(W.shape[0]) - W)
    except np.linalg.LinAlgError:
        return None
    t = int(graph.target)
    return {str(v): float(M[v, t]) for v in range(W.shape[0]) if v != t}


# ------------------------------------------------- true response surface ---


def intervention_bounds(graph, variables: Sequence[str]) -> List[Tuple[float, float]]:
    """The box the algorithm actually searches, per variable.

    `interventional_range_data` is the per-variable [min, max] of the
    OBSERVATIONAL data. It does not exist on a fresh graph -- it is created by
    graph.set_interventional_range_data(D_O), which PARENT_SCALE.set_values
    calls -- so this must run after set_values.

    There is deliberately no fallback: GraphStructure.get_interventional_range
    returns a flat [-5, 5] that no subclass overrides, which is NOT the box the
    run searches, so silently using it would label an optimum "interior" or
    "boundary" against the wrong bounds and make the result incomparable with
    the run's own Boundary_Percentage.
    """
    ranges = getattr(graph, "interventional_range_data", None)
    if not ranges:
        raise RuntimeError(
            "graph.interventional_range_data is not set: call this only after "
            "model.set_values(D_O, ...), which derives the intervention box "
            "from the observational data. (graph.get_interventional_range() is "
            "not a valid substitute -- it returns a flat [-5, 5].)"
        )
    missing = [v for v in variables if v not in ranges]
    if missing:
        raise KeyError(f"no intervention range for {missing}")
    return [(float(ranges[v][0]), float(ranges[v][1])) for v in variables]


class ExpectedYUnderDo:
    """Monte-Carlo estimator of E[Y | do(X_S = x)] on the TRUE SEM.

    Common random numbers: `graph.set_seed(crn_seed)` is called before every
    evaluation, so all x share one noise sample and the surface is a
    deterministic, comparable function of x. Results are cached.
    """

    def __init__(self, graph, variables: Sequence[str], n_mc: int = 50,
                 noiseless: bool = True, crn_seed: int = 12345,
                 round_to: int = 9):
        self.graph = graph
        self.variables = [str(v) for v in variables]
        self.n_mc = n_mc
        self.noiseless = noiseless
        self.crn_seed = crn_seed
        self.round_to = round_to
        self.target = str(graph.target)
        self._cache: Dict[Tuple[float, ...], float] = {}
        self.n_evals = 0

    def __call__(self, x: Sequence[float]) -> float:
        key = tuple(round(float(v), self.round_to) for v in x)
        if key in self._cache:
            return self._cache[key]
        # reset the SEM noise stream so every x sees the same draws
        self.graph.set_seed(self.crn_seed)
        np.random.seed(self.crn_seed)   # base-class noise path, if ever used
        samples = sample_model(
            self.graph.SEM,
            sample_count=self.n_mc,
            interventions=dict(zip(self.variables, [float(v) for v in x])),
            graph=self.graph,
            noiseless=self.noiseless,
        )
        val = float(np.mean(np.asarray(samples[self.target], dtype=float)))
        self._cache[key] = val
        self.n_evals += 1
        return val


def find_true_optimum(
    graph,
    variables: Sequence[str],
    direction: str = "min",
    eps_frac: float = 0.01,
    n_mc: int = 50,
    n_grid: int = 21,
    sweeps: int = 3,
    corner_budget: int = 512,
    n_random_starts: int = 8,
    seed: int = 0,
    noiseless: bool = True,
    crn_seed: int = 12345,
) -> dict:
    """Locate the optimum of E[Y | do(X_S = x)] and say if it is on the boundary.

    Search = all box corners (when 2^k fits in `corner_budget`) + the centre +
    `n_random_starts` random points, then coordinate-descent sweeps over a
    per-axis grid of `n_grid` points. Corners are enumerated explicitly because
    an affine surface always optimises at one, and coordinate descent is exact
    for a separable surface and a good refinement otherwise.

    A coordinate counts as on-boundary under the SAME rule the experiment uses
    for the interventions themselves: within eps_frac * (upper - lower) of
    either end. Returned keys:

      opt_x, opt_EY            argopt and its E[Y]
      opt_pos                  per-coordinate position in [0, 1]
      n_dims_on_boundary, k    how many coordinates sit on an end
      boundary_fraction        n_dims_on_boundary / k -- directly comparable to
                               the run's Boundary_Percentage
      is_interior              True iff NO coordinate is on a boundary
      is_full_boundary         True iff EVERY coordinate is on a boundary
      worst_x, worst_EY        the opposite extreme, for effect-size context
      EY_range                 |best - worst|: how much E[Y] moves over the box
      direction, n_mc, n_evals, variables
    """
    if direction not in ("min", "max"):
        raise ValueError("direction must be 'min' or 'max'")
    variables = [str(v) for v in variables]
    k = len(variables)
    bounds = intervention_bounds(graph, variables)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    width = hi - lo

    f = ExpectedYUnderDo(graph, variables, n_mc=n_mc, noiseless=noiseless,
                         crn_seed=crn_seed)
    sign = 1.0 if direction == "min" else -1.0
    better = lambda a, b: sign * a < sign * b          # noqa: E731

    # ---- candidate starts --------------------------------------------------
    candidates = [(lo + hi) / 2.0]
    if 2 ** k <= corner_budget:
        candidates += [np.where(np.array(c, dtype=bool), hi, lo)
                       for c in product([0, 1], repeat=k)]
    else:
        # too many corners to enumerate: sample some, plus the all-lo/all-hi ones
        rs = np.random.default_rng(seed)
        candidates += [lo.copy(), hi.copy()]
        candidates += [np.where(rs.random(k) < 0.5, lo, hi)
                       for _ in range(corner_budget)]
    rs = np.random.default_rng(seed + 1)
    candidates += [lo + rs.random(k) * width for _ in range(n_random_starts)]

    best_x = candidates[0]
    best_v = f(best_x)
    worst_x, worst_v = best_x, best_v
    for x in candidates[1:]:
        v = f(x)
        if better(v, best_v):
            best_x, best_v = x, v
        if better(worst_v, v):
            worst_x, worst_v = x, v

    # ---- coordinate-descent refinement on a per-axis grid ------------------
    grid_offsets = np.linspace(0.0, 1.0, max(int(n_grid), 3))
    x = np.array(best_x, dtype=float)
    for _ in range(max(int(sweeps), 0)):
        improved = False
        for j in range(k):
            keep = x[j]
            for t in grid_offsets:
                x[j] = lo[j] + t * width[j]
                v = f(x)
                if better(v, best_v):
                    best_v, keep, improved = v, x[j], True
                if better(worst_v, v):
                    worst_v, worst_x = v, x.copy()
            x[j] = keep
        best_x = x.copy()
        if not improved:
            break

    # ---- classify ----------------------------------------------------------
    pos, on_boundary = [], []
    for j, (l, h) in enumerate(bounds):
        w = h - l
        p = (best_x[j] - l) / w if w > 0 else np.nan
        pos.append(float(p))
        eps = eps_frac * w
        on_boundary.append(bool(best_x[j] <= l + eps or best_x[j] >= h - eps))

    n_bnd = int(sum(on_boundary))
    return {
        "variables": variables,
        "opt_x": [float(v) for v in best_x],
        "opt_EY": float(best_v),
        "opt_pos": pos,
        "on_boundary_per_dim": on_boundary,
        "n_dims_on_boundary": n_bnd,
        "k": k,
        "boundary_fraction": n_bnd / k,
        "is_interior": n_bnd == 0,
        "is_full_boundary": n_bnd == k,
        "worst_x": [float(v) for v in worst_x],
        "worst_EY": float(worst_v),
        "EY_range": float(abs(best_v - worst_v)),
        "direction": direction,
        "eps_frac": eps_frac,
        "n_mc": n_mc,
        "n_evals": f.n_evals,
    }
