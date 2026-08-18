r"""Prior-data Fitted Network prior mean (appendix E.4).

A classical GP surrogate assumes ``m(x) = 0``. Under boundary-concentrated
observations that assumption extrapolates badly: everywhere away from the
cluster of evaluated points the posterior mean decays to zero, which is
precisely the region the optimizer must reason about. E.4 replaces it with

    m_pi(x) = m_PFN(phi_pi(x)),

a globally learned prior over optimization problems, evaluated in the adaptive
geometry of E.3.

A PFN is trained by supervised learning on functions SAMPLED FROM A PRIOR: given
a context of (x, y) pairs and a query x, it regresses the posterior mean of the
prior conditioned on that context. Once trained it is a fixed, amortised
predictor -- no per-run fitting -- and it stays informative with few, clustered
observations because the prior it learned is global rather than data-driven.

This module provides

* :class:`SphericalPriorSampler` -- the prior the PFN is fitted to: random
  smooth functions on the unit sphere (random-feature draws plus a linear
  component), matching the geometry the surrogate actually sees, and biased to
  include monotone surfaces whose optimum sits on the boundary, since that is
  the regime the causal do-responses live in.
* :class:`PFN` -- a small transformer doing in-context regression.
* :func:`train_pfn` / :func:`load_pfn` -- fit and reload a checkpoint.
* :class:`ZeroPFN`, :class:`ConstantPFN` -- torch-free baselines. ZeroPFN
  recovers the classical zero-mean GP exactly and is the ablation for E.4.

All priors share the interface

    prior.mean(X_ctx, y_ctx, X_query) -> (n_query,) ndarray

where the X are ALREADY projected (points on the unit sphere).
"""

from typing import Optional

import numpy as np

try:  # torch is optional: the loop degrades to the torch-free priors without it
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is absent
    torch = None
    nn = object
    _TORCH = False


def _require_torch():
    if not _TORCH:
        raise ImportError(
            "PyTorch is required for the PFN prior. Install torch, or pass "
            "prior='zero' / 'constant' to use the torch-free baselines."
        )


# --------------------------------------------------------------- baselines ---


class ZeroPFN:
    """m(x) = 0 -- the classical GP prior mean, kept as the E.4 ablation."""

    name = "zero"
    max_dim = None

    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        return np.zeros(np.atleast_2d(X_query).shape[0], dtype=float)


class ConstantPFN:
    """m(x) = mean(y_ctx): a trivially "global" prior, the weakest useful one."""

    name = "constant"
    max_dim = None

    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        n = np.atleast_2d(X_query).shape[0]
        y = np.asarray(y_ctx, dtype=float).reshape(-1)
        return np.full(n, float(y.mean()) if y.size else 0.0)


# ------------------------------------------------------------------ prior ----


class SphericalPriorSampler:
    """Random smooth functions on the unit sphere -- the PFN's training prior.

    Each task draws f(x~) = w' sigma(A x~ + b) + v' x~, with random widths and
    scales, then optionally negates/steepens the linear part so that a
    controlled fraction of tasks are monotone (optimum on the boundary) rather
    than bump-shaped (interior optimum). Both regimes must be represented or the
    PFN would bake in one of the two answers the surrogate is meant to discover.
    """

    def __init__(
        self,
        max_dim: int = 10,
        n_features: int = 64,
        linear_fraction: float = 0.4,
        seed: int = 0,
    ):
        self.max_dim = int(max_dim)
        self.n_features = int(n_features)
        self.linear_fraction = float(linear_fraction)
        self.rng = np.random.default_rng(seed)

    def _sphere(self, n: int, dim: int) -> np.ndarray:
        """n points on the unit sphere in R^dim (the image of the projection)."""
        z = self.rng.normal(size=(n, dim))
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    def sample_task(self, n_points: int, dim: Optional[int] = None):
        """(X on the sphere, y) for one synthetic optimization problem."""
        dim = int(dim or self.rng.integers(2, self.max_dim + 1))
        X = self._sphere(n_points, dim)

        A = self.rng.normal(scale=self.rng.uniform(0.5, 3.0),
                            size=(self.n_features, dim))
        b = self.rng.normal(scale=1.0, size=self.n_features)
        w = self.rng.normal(scale=1.0 / np.sqrt(self.n_features),
                            size=self.n_features)
        nonlinear = np.tanh(X @ A.T + b) @ w

        v = self.rng.normal(size=dim)
        linear = X @ v
        if self.rng.random() < self.linear_fraction:
            # monotone-dominated task: optimum lies on the boundary
            y = 3.0 * linear + 0.2 * nonlinear
        else:
            y = nonlinear + 0.3 * linear

        y = y + self.rng.normal(scale=0.05, size=n_points)
        return X, y

    def sample_batch(self, batch_size: int, n_points: int):
        """Padded (B, n_points, max_dim) inputs and (B, n_points) targets."""
        Xs = np.zeros((batch_size, n_points, self.max_dim), dtype=np.float32)
        Ys = np.zeros((batch_size, n_points), dtype=np.float32)
        for b in range(batch_size):
            X, y = self.sample_task(n_points)
            Xs[b, :, : X.shape[1]] = X
            # standardise per task: the PFN predicts on a normalised y scale
            Ys[b] = (y - y.mean()) / (y.std() + 1e-8)
        return Xs, Ys


# -------------------------------------------------------------------- PFN ----


class _PFNModule(nn.Module if _TORCH else object):
    """Transformer that regresses E[y* | context] for in-context regression.

    One sequence of context tokens (x, y) followed by query tokens (x, mask).
    The attention mask lets everything attend to the context and stops queries
    attending to one another, so a query's prediction never depends on which
    other points happened to be queried alongside it.
    """

    def __init__(self, max_dim: int = 10, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        self.max_dim = max_dim
        self.d_model = d_model
        # separate embeddings: a context token carries y, a query token does not
        self.ctx_embed = nn.Linear(max_dim + 1, d_model)
        self.qry_embed = nn.Linear(max_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_ctx, y_ctx, x_qry):
        """x_ctx (B,C,max_dim), y_ctx (B,C), x_qry (B,Q,max_dim) -> (B,Q)."""
        b, c, _ = x_ctx.shape
        q = x_qry.shape[1]
        ctx = self.ctx_embed(torch.cat([x_ctx, y_ctx.unsqueeze(-1)], dim=-1))
        qry = self.qry_embed(x_qry)
        seq = torch.cat([ctx, qry], dim=1)

        # True = "not allowed to attend"; queries are invisible to everyone
        mask = torch.zeros(c + q, c + q, dtype=torch.bool, device=seq.device)
        mask[:, c:] = True
        mask[torch.arange(c, c + q), torch.arange(c, c + q)] = False

        out = self.encoder(seq, mask=mask)
        return self.head(out[:, c:, :]).squeeze(-1)


class PFN:
    """Trained PFN wrapped in the prior interface used by the surrogate."""

    name = "pfn"

    def __init__(self, module: "_PFNModule", max_dim: int, device: str = "cpu"):
        _require_torch()
        self.module = module.eval().to(device)
        self.max_dim = int(max_dim)
        self.device = device

    def _pad(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.shape[1] > self.max_dim:
            raise ValueError(
                f"projected input has {X.shape[1]} dims but the PFN was trained "
                f"for at most {self.max_dim}; retrain with a larger --max_dim"
            )
        out = np.zeros((X.shape[0], self.max_dim), dtype=np.float32)
        out[:, : X.shape[1]] = X
        return out

    def mean(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        """Posterior-mean prediction, returned on the original y scale."""
        y = np.asarray(y_ctx, dtype=float).reshape(-1)
        Xq = self._pad(X_query)
        if y.size == 0:
            return np.zeros(Xq.shape[0], dtype=float)
        mu, sd = float(y.mean()), float(y.std())
        if sd < 1e-12:                      # a constant context carries no shape
            return np.full(Xq.shape[0], mu)

        Xc = self._pad(X_ctx)
        yc = ((y - mu) / sd).astype(np.float32)
        with torch.no_grad():
            pred = self.module(
                torch.from_numpy(Xc).unsqueeze(0).to(self.device),
                torch.from_numpy(yc).unsqueeze(0).to(self.device),
                torch.from_numpy(Xq).unsqueeze(0).to(self.device),
            )
        return pred.squeeze(0).cpu().numpy().astype(float) * sd + mu


def train_pfn(
    max_dim: int = 10,
    steps: int = 2000,
    batch_size: int = 32,
    n_context: int = 24,
    n_query: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 200,
):
    """Fit a PFN on synthetic prior draws. Returns (PFN, history)."""
    _require_torch()
    torch.manual_seed(seed)
    sampler = SphericalPriorSampler(max_dim=max_dim, seed=seed)
    module = _PFNModule(max_dim=max_dim, d_model=d_model, n_heads=n_heads,
                        n_layers=n_layers).to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    module.train()
    history = []
    n_points = n_context + n_query
    for step in range(steps):
        X, Y = sampler.sample_batch(batch_size, n_points)
        X = torch.from_numpy(X).to(device)
        Y = torch.from_numpy(Y).to(device)
        x_ctx, y_ctx = X[:, :n_context], Y[:, :n_context]
        x_qry, y_qry = X[:, n_context:], Y[:, n_context:]

        pred = module(x_ctx, y_ctx, x_qry)
        loss = torch.mean((pred - y_qry) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        sched.step()

        history.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            recent = float(np.mean(history[-log_every:]))
            print(f"step {step:5d}  loss {loss.item():.4f}  mean {recent:.4f}",
                  flush=True)

    return PFN(module, max_dim=max_dim, device=device), history


def save_pfn(pfn: "PFN", path: str, **meta) -> None:
    _require_torch()
    torch.save(
        {
            "state_dict": pfn.module.state_dict(),
            "max_dim": pfn.max_dim,
            "d_model": pfn.module.d_model,
            "meta": meta,
        },
        path,
    )


def load_pfn(path: str, device: str = "cpu") -> "PFN":
    _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    d_model = ckpt.get("d_model", 64)
    n_layers = 1 + max(
        int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")
    )
    n_heads = ckpt.get("meta", {}).get("n_heads", 4)
    module = _PFNModule(max_dim=ckpt["max_dim"], d_model=d_model,
                        n_heads=n_heads, n_layers=n_layers)
    module.load_state_dict(sd)
    return PFN(module, max_dim=ckpt["max_dim"], device=device)


def build_prior(spec, device: str = "cpu"):
    """Resolve a prior from a name, a checkpoint path, or an object."""
    if spec is None or spec == "zero":
        return ZeroPFN()
    if spec == "constant":
        return ConstantPFN()
    if isinstance(spec, str):
        return load_pfn(spec, device=device)
    return spec
