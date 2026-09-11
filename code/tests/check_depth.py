"""Does the ladder still work at depth?

All three depths run at `lr_target = 0.03`.  Depth 1 needs ten times the
epochs of the others because depth accelerates emergence in epochs, not
because it is slower to converge.

The ladder was searched at depth 1.  Depth changes the dynamics, so the laws
could compress, reorder, or fall outside the training window.  This integrates
the assembled world at each depth at the corrected learning rate and reports
the emergence times, their spread, and whether the depth-1 ordering survives.
"""

import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import DEFAULT, override
from relspec.measure import law_plan

# One source of truth: the settled budgets live beside the ladder.
BUDGET = {d: (DEFAULT.lr_target, worlds.F1_EPOCHS[d], worlds.F1_EVERY[d])
          for d in (1, 2, 3)}


def run(depth):
    lrt, epochs, every = BUDGET[depth]
    S = override(lr_target=lrt, eval_every=every)
    w = worlds.emergence_world(0)
    s = System.build(w, settings=S)
    plan = law_plan(w, worlds.held_composites(w))
    t0 = time.time()
    tr, _ = theory.predict(s, depth, s.lr(depth, settings=S), epochs,
                           models.make_model(w, depth, S), settings=S,
                           eval_every=every, plan=plan)
    if tr is None:
        print("N=%d lr=%g: integration DIVERGED" % (depth, lrt), flush=True)
        return None
    ts = tr.emergence(S)
    t = np.array([ts[law.name] for law in w.laws], float)
    fin = t[np.isfinite(t)]
    print("N=%d lr=%-5g %6d epochs, %5.0fs | %2d/%d emerge"
          % (depth, lrt, epochs, time.time() - t0, len(fin), len(t)), flush=True)
    if len(fin):
        print("   t* %.0f to %.0f, spread %.1fx"
              % (fin.min(), fin.max(), fin.max() / fin.min()), flush=True)
        print("   %s" % " ".join("%.0f" % x for x in np.sort(fin)), flush=True)
    return t


def main():
    depths = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    out = {}
    for d in depths:
        out[d] = run(d)
    if 1 in out and out[1] is not None:
        base = out[1]
        for d in depths:
            if d == 1 or out[d] is None:
                continue
            m = np.isfinite(base) & np.isfinite(out[d])
            if m.sum() < 3:
                continue
            rank = lambda a: np.argsort(np.argsort(a))
            rho = float(np.corrcoef(rank(base[m]), rank(out[d][m]))[0, 1])
            print("ordering N=1 vs N=%d: Spearman %.3f over %d laws"
                  % (d, rho, m.sum()))


if __name__ == "__main__":
    main()
