"""
The learner: a linear relational-embedding model, shallow or deep.

Both models expose the same three methods -- `embedding()`, `sgd_epoch()` and
`fullbatch_step()` -- so training, measurement and theory never branch on
depth.

Shallow (N = 1) learns the (P x d) embedding matrix `E` directly.
Deep (N > 1) parameterises the same `E` as `W_1 W_2 ... W_N`; the end-to-end
solution is unchanged but the learning dynamics are not, which is what the
depth manipulation tests.
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULT, Settings


class ShallowModel:
    """E learned directly."""

    depth = 1

    def __init__(self, P, d, init_scale=1e-2, seed=0):
        self.E = np.random.default_rng(seed).standard_normal((P, d)) * init_scale

    def embedding(self):
        return self.E

    def sgd_epoch(self, sparse, C, lr, rng):
        for i in rng.permutation(len(sparse)):
            nz, vals = sparse[i]
            resid = vals @ self.E[nz] - C[i]
            self.E[nz] -= lr * np.outer(vals, resid)

    def fullbatch_step(self, A, C, lr):
        self.E -= lr * (A.T @ (A @ self.E - C))


class DeepModel:
    """E = W_0 @ W_1 @ ... @ W_{N-1}, hidden width d."""

    def __init__(self, P, d, depth=2, width=None, init_scale=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.depth = depth
        h = d if width is None else width
        dims = [P] + [h] * (depth - 1) + [d]
        self.W = [rng.standard_normal((dims[k], dims[k + 1])) * init_scale
                  for k in range(depth)]

    def embedding(self):
        return embed(self.W)

    def sgd_epoch(self, sparse, C, lr, rng):
        """Per-fact SGD that exploits row sparsity: a constraint row touches at
        most three tokens, so only those rows of W_0 participate and no
        P-sized matmul is ever formed.  All layer gradients are computed from
        the same pre-update weights, then applied together."""
        N = self.depth
        for i in rng.permutation(len(sparse)):
            nz, av = sparse[i]
            suf = suffixes(self.W)
            resid = av @ (self.W[0][nz] @ suf[0]) - C[i]
            G = np.outer(av, resid)                     # nonzero rows of dL/dE
            pre = [None] * N                            # (W_0..W_{k-1})[nz]
            if N > 1:
                pre[1] = self.W[0][nz]
                for k in range(2, N):
                    pre[k] = pre[k - 1] @ self.W[k - 1]
            grads = [G @ suf[0].T] + [pre[k].T @ G @ suf[k].T for k in range(1, N)]
            self.W[0][nz] -= lr * grads[0]
            for k in range(1, N):
                self.W[k] -= lr * grads[k]

    def fullbatch_step(self, A, C, lr):
        G = A.T @ (A @ self.embedding() - C)
        pre, suf = products(self.W)
        # every gradient must be read off the SAME pre-update weights, so
        # compute them all before applying any (prefix[1] *is* W_0, not a copy)
        grads = [prefix_T_dot(pre[k], G) @ suf[k].T for k in range(self.depth)]
        for k in range(self.depth):
            self.W[k] -= lr * grads[k]


# --------------------------------------------------------------------------- #
#  Layer-product helpers (shared with the deep mean-dynamics integrator)       #
# --------------------------------------------------------------------------- #

def embed(Ws):
    E = Ws[0]
    for k in range(1, len(Ws)):
        E = E @ Ws[k]
    return E


def suffixes(Ws):
    """suffix[k] = W_{k+1} ... W_{N-1}  (the `L_l` of the derivation)."""
    N = len(Ws)
    suf = [None] * N
    s = np.eye(Ws[-1].shape[1])
    for k in range(N - 1, -1, -1):
        suf[k] = s
        s = Ws[k] @ s
    return suf


def prefixes(Ws):
    """prefix[k] = W_0 ... W_{k-1}  (the `R_l` of the derivation).

    `prefix[0]` is the P x P identity.  It is returned as None rather than
    materialised: multiplying by it is a no-op, and building it would turn the
    cheapest gradient in the integrator's inner loop into by far the most
    expensive one (a P x P matmul per step).  Results are bit-identical.
    """
    pre = [None] * len(Ws)
    for k in range(1, len(Ws)):
        pre[k] = Ws[0] if k == 1 else pre[k - 1] @ Ws[k - 1]
    return pre


def products(Ws):
    """(prefixes, suffixes).  `prefix[0] is None` and means the identity."""
    return prefixes(Ws), suffixes(Ws)


def prefix_T_dot(pre_k, M):
    """`prefix_k^T @ M`, with a None prefix meaning the identity."""
    return M if pre_k is None else pre_k.T @ M


def make_model(world, depth, settings: Settings = DEFAULT, seed=None):
    """The model used everywhere: depth-appropriate class, default init scale,
    default init seed unless overridden."""
    seed = settings.init_seed if seed is None else seed
    scale = settings.init_scale(depth)
    if depth == 1:
        return ShallowModel(world.P, world.d, init_scale=scale, seed=seed)
    return DeepModel(world.P, world.d, depth=depth, init_scale=scale, seed=seed)
