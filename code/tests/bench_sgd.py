"""Are the trimmed SGD loops bit-identical to the ones they replaced?

The per-fact loop is 98 per cent of a depth-1 cell and 28 per cent of a
depth-3 one, and it is dominated by call overhead rather than arithmetic: each
row touches at most three tokens, so every operation is on a 3 x 16 block and
the Python wrappers around them cost more than the multiplies.

Two things were removed.  `np.outer(vals, resid)` is a wrapper that calls
`asarray` once and `ravel` twice before doing the multiply that `col * resid`
does directly, with `col` now cached on the row.  And `E[nz] -= X` gathered the
row block a second time, after the residual line had already gathered it.

Neither changes any operation or its order, so the two forms must agree to the
last bit, not merely closely.  Anything else is a bug.  The reference
implementations below are the originals.
"""

import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import copy

import numpy as np
from relspec import System, models, worlds
from relspec.config import override

SEED = 0
EPOCHS = 400
REPEATS = 5      # timings on this scale are noisy; the minimum is the signal


def old_sparse(A):
    """The two-element row format, without the cached column view."""
    return [(np.nonzero(A[i])[0], A[i][np.nonzero(A[i])[0]])
            for i in range(A.shape[0])]


def old_shallow_epoch(m, sparse, C, lr, rng):
    for i in rng.permutation(len(sparse)):
        nz, vals = sparse[i]
        resid = vals @ m.E[nz] - C[i]
        m.E[nz] -= lr * np.outer(vals, resid)


def old_deep2_epoch(m, sparse, C, lr, rng):
    W0, W1 = m.W
    for i in rng.permutation(len(sparse)):
        nz, av = sparse[i]
        R = W0[nz]
        resid = av @ (R @ W1) - C[i]
        G = np.outer(av, resid)
        g0 = G @ W1.T
        g1 = R.T @ G
        W0[nz] -= lr * g0
        W1 -= lr * g1


def old_deep_epoch(m, sparse, C, lr, rng):
    W, N = m.W, m.depth
    for i in rng.permutation(len(sparse)):
        nz, av = sparse[i]
        suf = [None] * N
        s = None
        for k in range(N - 2, -1, -1):
            s = W[k + 1] if s is None else W[k + 1] @ s
            suf[k] = s
        R = W[0][nz]
        resid = av @ (R @ suf[0]) - C[i]
        G = np.outer(av, resid)
        grads = [G @ suf[0].T]
        pre = R
        for k in range(1, N):
            gk = pre.T @ G
            grads.append(gk if suf[k] is None else gk @ suf[k].T)
            if k + 1 < N:
                pre = pre @ W[k]
        W[0][nz] -= lr * grads[0]
        for k in range(1, N):
            W[k] -= lr * grads[k]


def old_epoch(m, depth):
    if depth == 1:
        return old_shallow_epoch
    return old_deep2_epoch if depth == 2 else old_deep_epoch


def main():
    want = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    print("F2 world, seed %d, %d epochs of per-fact SGD" % (SEED, EPOCHS))
    for depth in want:
        S = override(init_seed=1000 + SEED)
        w = worlds.identifiability_world(SEED, K=16, k=4, closed_block="A")
        s = System.build(w, settings=S)
        lr = s.lr(depth, settings=S)
        m_new = models.make_model(w, depth, S)
        m_old = copy.deepcopy(m_new)
        fast, slow = s.sparse(), old_sparse(s.A)

        def time_it(run, model):
            best = float("inf")
            for _ in range(REPEATS):
                m = copy.deepcopy(model)
                rng = np.random.default_rng(7)
                t0 = time.perf_counter()
                run(m)
                best = min(best, time.perf_counter() - t0)
            return best, m

        base = copy.deepcopy(m_new)
        step = old_epoch(m_old, depth)

        def go_new(m):
            rng = np.random.default_rng(7)
            for _ in range(EPOCHS):
                m.sgd_epoch(fast, s.C, lr, rng)

        def go_old(m):
            rng = np.random.default_rng(7)
            for _ in range(EPOCHS):
                step.__get__(m) if False else old_epoch(m, depth)(
                    m, slow, s.C, lr, rng)

        t_new, m_new = time_it(go_new, base)
        t_old, m_old = time_it(go_old, base)

        d = float(np.abs(m_new.embedding() - m_old.embedding()).max())
        lone = sum(1 for r in fast if r[4] is not None)
        print("  N=%d  rows %d (%d single-token)   max |diff| %.3e %s   "
              "old %5.2fs  new %5.2fs  %.2fx" % (depth, len(fast), lone, d,
                         "BIT-IDENTICAL" if d == 0.0 else "*** DIFFERS ***",
                         t_old, t_new, t_old / t_new), flush=True)


if __name__ == "__main__":
    main()
