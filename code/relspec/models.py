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
        """One pass of per-fact SGD.

        `np.outer` is not used, and the row block is gathered once rather than
        twice.  Both are pure overhead: `np.outer` is a Python wrapper that
        calls `asarray` and `ravel` twice before doing the multiply this does
        directly, and `E[nz] -= X` gathers a second time after the residual
        line already did.  At depth 1 this loop is 98 per cent of a cell's
        runtime, so the wrappers dominate the arithmetic.  Results are
        bit-identical: the operations and their order are unchanged.
        """
        E = self.E
        for i in rng.permutation(len(sparse)):
            nz, vals, col, c, j = sparse[i]
            if j is not None:            # anchor row: one token, no gather
                r = E[j]
                E[j] = r - lr * (vals[0] * (vals[0] * r - c))
                continue
            R = E[nz]
            resid = vals @ R - c
            E[nz] = R - lr * (col * resid)

    def fullbatch_step(self, A, C, lr):
        self.E -= lr * (A.T @ (A @ self.E - C))


class DeepModel:
    """E = W_0 @ W_1 @ ... @ W_{N-1}, hidden width d."""

    def __init__(self, P, d, depth=2, width=None, init_scale=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.depth = depth
        h = d if width is None else width
        dims = [P] + [h] * (depth - 1) + [d]
        self.W = [
            rng.standard_normal((dims[k], dims[k + 1])) * init_scale
            for k in range(depth)
        ]

    def embedding(self):
        return embed(self.W)

    def sgd_epoch(self, sparse, C, lr, rng):
        """Per-fact SGD that exploits row sparsity: a constraint row touches at
        most three tokens, so only those rows of W_0 participate and no P-sized
        matmul is ever formed.  All layer gradients are read off the SAME
        pre-update weights, then applied together.

        Every per-fact update changes W_1..W_{N-1}, so the suffix products
        cannot be hoisted out of the loop.  What can be avoided is building the
        trailing identity and then multiplying the last layer's gradient by it,
        which is a full d x d matmul and an allocation per fact for no effect.
        `suffix[N-1]` is therefore left implicit, as `prefix[0]` already is in
        `prefixes`.  Depth 2 is written out separately because it is the common
        case and needs no suffix product at all.  Results are bit-identical to
        the straightforward form.
        """
        W = self.W
        N = self.depth
        if N == 2:
            W0, W1 = W
            for i in rng.permutation(len(sparse)):
                nz, av, col, c, _ = sparse[i]
                R = W0[nz]
                resid = av @ (R @ W1) - c
                G = col * resid
                g0 = G @ W1.T
                g1 = R.T @ G
                W0[nz] = R - lr * g0
                W1 -= lr * g1
            return
        for i in rng.permutation(len(sparse)):
            nz, av, col, c, _ = sparse[i]
            suf = [None] * N              # suf[N-1] stays None: the identity
            s = None
            for k in range(N - 2, -1, -1):
                s = W[k + 1] if s is None else W[k + 1] @ s
                suf[k] = s
            R = W[0][nz]
            resid = av @ (R @ suf[0]) - c
            G = col * resid
            grads = [G @ suf[0].T]
            pre = R
            for k in range(1, N):
                gk = pre.T @ G
                if suf[k] is not None:
                    gk = gk @ suf[k].T
                grads.append(gk)
                if k < N - 1:
                    pre = pre @ W[k]
            W[0][nz] -= lr * grads[0]
            for k in range(1, N):
                W[k] -= lr * grads[k]

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
