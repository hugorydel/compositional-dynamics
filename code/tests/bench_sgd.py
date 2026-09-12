"""Are the trimmed SGD loops bit-identical to the ones they replaced?

The per-fact loop is 98 per cent of a depth-1 cell and 28 per cent of a
depth-3 one, and it is dominated by call overhead rather than arithmetic: each
row touches at most three tokens, so every operation is on a 3 x 16 block and
the Python wrappers around them cost more than the multiplies.

Four things were removed.  `np.outer(vals, resid)` is a wrapper that calls
`asarray` once and `ravel` twice before doing the multiply that `col * resid`
does directly, with `col` now cached on the row.  `E[nz] -= X` gathered the row
block a second time after the residual line had already gathered it.  The
target row `C[i]` was indexed per update.  And anchor rows touch one token yet
paid for a one-element fancy index, where a scalar index does.

None of those changes any operation or its order, so the two forms must agree
to the last bit, not merely closely.  Anything else is a bug.  The reference
implementations below are the originals.

All three worlds are checked, not one.  They differ in size, in how many rows
carry a single token and in their spectra, and a loop that is exact on one is
not thereby exact on another.
"""

import copy
import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, worlds
from relspec.config import override

SEED = 0
REPEATS = 5      # timings on this scale are noisy; the minimum is the signal
# the emergence world is twenty times the size of the other two, so it gets
# fewer epochs; bit-identity needs few, and the timing is reported per run
EPOCHS = {"f1": 30, "f2": 400, "f3": 200}


def world_for(fig):
    if fig == "f1":
        return worlds.emergence_world(SEED)
    if fig == "f2":
        return worlds.identifiability_world(SEED, K=16, k=4, closed_block="A")
    return worlds.integration_world(SEED, link=True)


def old_sparse(A):
    """The two-element row format: no cached column, target or lone index."""
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


def old_epoch(depth):
    if depth == 1:
        return old_shallow_epoch
    return old_deep2_epoch if depth == 2 else old_deep_epoch


def best(run, model):
    """Minimum wall time over REPEATS, and the model the last run produced."""
    t, out = float("inf"), None
    for _ in range(REPEATS):
        m = copy.deepcopy(model)
        t0 = time.perf_counter()
        run(m)
        t = min(t, time.perf_counter() - t0)
        out = m
    return t, out


def main():
    figs = sys.argv[1:] or ["f1", "f2", "f3"]
    print("per-fact SGD, seed %d, min of %d runs" % (SEED, REPEATS))
    bad = 0
    for fig in figs:
        w = world_for(fig)
        n = EPOCHS[fig]
        for depth in (1, 2, 3):
            S = override(init_seed=1000 + SEED)
            s = System.build(w, settings=S)
            lr = s.lr(depth, settings=S)
            base = models.make_model(w, depth, S)
            fast, slow = s.sparse(), old_sparse(s.A)
            step = old_epoch(depth)

            def go_new(m, _f=fast, _s=s, _lr=lr, _n=n):
                rng = np.random.default_rng(7)
                for _ in range(_n):
                    m.sgd_epoch(_f, _s.C, _lr, rng)

            def go_old(m, _f=slow, _s=s, _lr=lr, _n=n, _step=step):
                rng = np.random.default_rng(7)
                for _ in range(_n):
                    _step(m, _f, _s.C, _lr, rng)

            t_new, m_new = best(go_new, base)
            t_old, m_old = best(go_old, base)
            d = float(np.abs(m_new.embedding() - m_old.embedding()).max())
            bad += d != 0.0
            lone = sum(1 for r in fast if r[4] is not None)
            print("  %s N=%d  %4d rows (%4d single-token) x %3d epochs   "
                  "max |diff| %.3e %s   old %5.2fs  new %5.2fs  %.2fx"
                  % (fig.upper(), depth, len(fast), lone, n, d,
                     "identical" if d == 0.0 else "*** DIFFERS ***",
                     t_old, t_new, t_old / t_new), flush=True)
    print(("all bit-identical" if not bad else "*** %d cases differ ***" % bad))


if __name__ == "__main__":
    main()
