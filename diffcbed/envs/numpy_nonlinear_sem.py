"""
Self-contained NumPy nonlinear-Gaussian SEM.

Drop-in replacement for the DiBS `DenseNonlinearGaussianJAX` conditionals object
that the repo expects at
`diffcbed.models.dibs.models.nonlinearGaussian.DenseNonlinearGaussianJAX`.
That dibs source is an orphaned, empty git submodule (its fork URL is lost), so
the nonlinear SEM path was un-runnable. This class reimplements just the narrow
API the repo actually calls, in pure NumPy (no jax, no dibs):

  - __init__(obs_noise, sig_param, hidden_layers)
  - sample_parameters(key, n_vars)            -> theta (per-node MLP weights)
  - eltwise_nn_forward(theta, x)              -> [N x d] per-node MLP outputs
  - sample_obs(key, n_samples, g, theta, node, values) -> [n_samples x d]

Semantics (matching how the callers use it):
  * Each variable j has its own MLP  f_j : R^d -> R  with one hidden layer of
    width `hidden_layers[0]` and tanh activation; weights ~ N(0, sig_param).
  * eltwise_nn_forward(theta, x) returns column j = f_j(x) for every j, so
    graph_chain.nn_forward can take `[:, node]`. The caller lays a node's parent
    values into the leading columns of x (parents in order, rest zero), exactly
    as define_SEM_causalenv_nonlinear does.
  * sample_obs performs additive-noise ancestral sampling over the (possibly
    mutilated) graph `g` (an igraph.Graph), honouring interventions given as a
    one-hot `node` mask + `values` vector. Used only to populate the env's
    held-out data at construction; the boundary experiment samples through
    graph.SEM (which uses eltwise_nn_forward), not through here.

`key` arguments (jax PRNGKeys in the original) are accepted and ignored; a
NumPy Generator seeded deterministically per instance is used instead, so a
given graph gets a fixed SEM.
"""

import numpy as np

_PARAM_SEED = 0  # deterministic per-instance MLP weights -> fixed SEM per graph


class DenseNonlinearGaussianNumpy:
    def __init__(self, obs_noise, sig_param=1.0, hidden_layers=(5,), param_seed=_PARAM_SEED):
        # obs_noise may be a scalar or a per-node list/array of std-devs
        self.obs_noise = obs_noise
        self.sig_param = float(sig_param)
        self.hidden_layers = list(hidden_layers) if hidden_layers else [5]
        self._rng = np.random.default_rng(param_seed)
        self.n_vars = None

    # -- noise std helper -------------------------------------------------
    def _noise_std(self, j, d):
        if np.isscalar(self.obs_noise):
            return float(self.obs_noise)
        arr = np.asarray(self.obs_noise, dtype=float).ravel()
        return float(arr[j]) if j < arr.size else float(arr[-1])

    # -- API expected by the repo ----------------------------------------
    def sample_parameters(self, key=None, n_vars=None):
        """Return theta: a list of per-node (W1, b1, W2, b2) MLP weight sets."""
        if n_vars is None:
            raise ValueError("n_vars is required")
        d = int(n_vars)
        H = int(self.hidden_layers[0])
        s = self.sig_param
        rng = self._rng
        theta = []
        for _ in range(d):
            W1 = rng.normal(0.0, s, size=(H, d))
            b1 = rng.normal(0.0, s, size=(H,))
            W2 = rng.normal(0.0, s, size=(H,))
            b2 = float(rng.normal(0.0, s))
            theta.append((W1, b1, W2, b2))
        self.n_vars = d
        return theta

    @staticmethod
    def _forward_one(params, x):
        """Single-node MLP: x [N x d] -> [N]."""
        W1, b1, W2, b2 = params
        h = np.tanh(x @ W1.T + b1)          # [N x H]
        return h @ W2 + b2                  # [N]

    def eltwise_nn_forward(self, theta, x):
        """x: [N x d] (or [d]) -> [N x d], column j = f_j(x)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        N = x.shape[0]
        out = np.zeros((N, len(theta)))
        for j, params in enumerate(theta):
            out[:, j] = self._forward_one(params, x)
        return out

    def sample_obs(self, key=None, n_samples=1, g=None, theta=None, node=None, values=None):
        """Additive-noise ancestral sampling over igraph `g`."""
        if theta is None:
            raise ValueError("theta is required")
        d = len(theta)
        N = int(n_samples)
        order, parents_of = _graph_order_and_parents(g, d)

        intervened = None
        if node is not None:
            node_arr = np.asarray(node).ravel()
            intervened = {j for j in range(d) if j < node_arr.size and node_arr[j]}

        samples = np.zeros((N, d))
        for v in order:
            if intervened is not None and v in intervened:
                samples[:, v] = float(np.asarray(values).ravel()[v])
                continue
            parents = parents_of[v]
            if parents:
                # lay parent values into leading columns, matching nn_forward,
                # then run only node v's MLP (avoids the O(d^2) full eltwise)
                x = np.zeros((N, d))
                for i, p in enumerate(parents):
                    x[:, i] = samples[:, p]
                mean = self._forward_one(theta[v], x)
            else:
                mean = np.zeros(N)
            noise = self._rng.normal(0.0, self._noise_std(v, d), size=N)
            samples[:, v] = mean + noise
        return samples


def _graph_order_and_parents(g, d):
    """Topological order + in-neighbours from an igraph.Graph (fallbacks for
    duck-typed test stubs / plain adjacency arrays)."""
    # igraph.Graph
    if hasattr(g, "topological_sorting") and hasattr(g, "predecessors"):
        order = list(g.topological_sorting())
        parents_of = {v: list(g.predecessors(v)) for v in range(d)}
        return order, parents_of
    # numpy adjacency matrix A[i, j] == edge i->j
    A = np.asarray(g)
    parents_of = {v: list(np.nonzero(A[:, v])[0]) for v in range(d)}
    # Kahn topological sort
    indeg = {v: len(parents_of[v]) for v in range(d)}
    ready = [v for v in range(d) if indeg[v] == 0]
    order = []
    children = {v: list(np.nonzero(A[v, :])[0]) for v in range(d)}
    while ready:
        v = ready.pop()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return order, parents_of
