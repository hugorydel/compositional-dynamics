"""Are depths 2 and 3 usable at `lr_target = 0.3`?

If they are, every depth column can share one learning rate and one epoch axis.
If they are not, the choice is between mixed rates, which makes the columns
non-comparable epoch for epoch, and dropping every depth to 0.03, which costs
depth 1 roughly ten times the epochs.

Three trajectories of the same quantity from the same initialisation:

  sgd     per-fact stochastic descent -- what the figures plot
  full    epoch-averaged full-batch descent -- the discrete process the
          continuous theory is the limit of
  theory  the integrated mean dynamics -- what the overlay plots

Stability is not the only test, and it is not the interesting one.  Per-fact
SGD can survive a step size at which full-batch descent diverges, because each
update is a fraction of an epoch-averaged step.  When that happens the network
is not running the process the theory describes, so the overlay is predicting
something else even though nothing overflowed.  The verdict therefore requires
full-batch to stay finite AND SGD to track the theory.
"""

import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan, law_plan

EPOCHS, EVERY = 3000, 50
TOL = 0.05  # median |log10 ratio| below this counts as tracking


def build(name):
    if name == "integration":
        w = worlds.integration_world(0, link=True)
        return w, lambda: cross_plan(w, worlds.held_cross(w))
    if name == "identifiability":
        w = worlds.identifiability_world(0, K=16, k=4, n_bridge=1)
        return w, lambda: law_plan(w, worlds.held_composites(w))
    w = worlds.emergence_world(0)
    return w, lambda: law_plan(w, worlds.held_composites(w))


def geo_matrix(traj):
    """(T, n_measures) of the geometric measure, or None if absent."""
    if traj is None or not traj.geometric:
        return None
    return np.column_stack([traj.geometric[k] for k in sorted(traj.geometric)])


def deviation(a, b):
    """Median |log10 ratio| over the window where either curve has moved off
    its starting value, so a shared flat head does not dilute it."""
    if a is None or b is None:
        return np.nan
    n = min(len(a), len(b))
    x, y = np.maximum(a[:n], 1e-16), np.maximum(b[:n], 1e-16)
    moved = (x < 0.95 * x[0]) | (y < 0.95 * y[0])
    if not moved.any():
        moved = np.ones(n, bool)
    return float(np.median(np.abs(np.log10(x[moved] / y[moved]))))


def cell(world_name, depth, lrt):
    S = override(lr_target=lrt, eval_every=EVERY)
    w, mk = build(world_name)
    s = System.build(w, settings=S)
    plan = mk()
    lr = s.lr(depth, settings=S)
    out = {}
    for mode in ("sgd", "full"):
        tr = train.train(models.make_model(w, depth, S), s, lr, EPOCHS, S,
                         order_seed=7, mode=mode, eval_every=EVERY, plan=plan)
        g = geo_matrix(tr)
        out[mode] = None if g is None or not np.all(np.isfinite(g)) else g
    tr, _ = theory.predict(s, depth, lr, EPOCHS, models.make_model(w, depth, S),
                           settings=S, eval_every=EVERY, plan=plan)
    g = geo_matrix(tr)
    out["theory"] = None if g is None or not np.all(np.isfinite(g)) else g
    return out


def main():
    names = sys.argv[1:] or ["integration", "identifiability", "emergence"]
    print("%d epochs, deviation = median |log10 ratio| over %d measures"
          % (EPOCHS, 0))
    print("%-16s %-2s %-6s %-8s %-8s %-8s %-12s %s"
          % ("world", "N", "lr", "sgd", "full", "theory", "sgd/theory", "verdict"))
    for name in names:
        for depth in (2, 3):
            for lrt in (0.30, 0.03):
                t0 = time.time()
                o = cell(name, depth, lrt)
                fin = {k: ("finite" if v is not None else "DIVERGED")
                       for k, v in o.items()}
                if o["sgd"] is not None and o["theory"] is not None:
                    d = np.median([deviation(o["sgd"][:, j], o["theory"][:, j])
                                   for j in range(o["sgd"].shape[1])])
                else:
                    d = np.nan
                ok = (o["full"] is not None and o["theory"] is not None
                      and np.isfinite(d) and d < TOL)
                print("%-16s %-2d %-6g %-8s %-8s %-8s %-12s %s  (%.0fs)"
                      % (name, depth, lrt, fin["sgd"], fin["full"], fin["theory"],
                         ("%.4f" % d) if np.isfinite(d) else "n/a",
                         "USABLE" if ok else "no", time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
