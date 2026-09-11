"""What will the three experiments cost to run?

Two parts.  For Figures 2 and 3 the epoch budgets are not settled, so this
first measures when their last item emerges at each depth, then proposes a
budget with headroom for the staged design.  Then it times both halves of the
work per depth and world, and multiplies out.

Training cost is rows times epochs, because per-fact descent walks every
constraint row once per epoch.  Prediction cost is substeps times epochs, and
the integrator takes 8 substeps per epoch at depth 2 and 32 at depth 3.  The
two scale differently, so both are measured rather than extrapolated.
"""

import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds
from relspec.config import DEFAULT, override
from relspec.measure import cross_plan, law_plan

DEPTHS = (1, 2, 3)
PROBE_EPOCHS = 100  # for timing only
DISCOVER = {1: 40000, 2: 8000, 3: 8000}  # generous, for budget discovery


def f1(depth):
    S = override(eval_every=worlds.F1_EVERY[depth])
    w = worlds.emergence_world(0)
    return w, System.build(w, settings=S), (lambda ww: law_plan(
        ww, worlds.held_composites(ww))), S


def f2(depth):
    S = override(eval_every=50)
    w = worlds.identifiability_world(0, K=16, k=4, n_bridge=1)
    return w, System.build(w, settings=S), (lambda ww: law_plan(
        ww, worlds.held_composites(ww))), S


def f3(depth):
    S = override(eval_every=50)
    w = worlds.integration_world(0, link=True)
    return w, System.build(w, settings=S), (lambda ww: cross_plan(
        ww, worlds.held_cross(ww))), S


FIGS = [("F1 emergence", f1), ("F2 identifiability", f2), ("F3 integration", f3)]


def discover(build, depth):
    """When does the last item emerge, integrating from scratch?"""
    w, s, mk, S = build(depth)
    plan = mk(w)
    ep = DISCOVER[depth]
    tr, _ = theory.predict(s, depth, s.lr(depth, settings=S), ep,
                           models.make_model(w, depth, S), settings=S,
                           eval_every=S.eval_every, plan=plan)
    if tr is None:
        return np.nan, w
    ts = np.array(list(tr.emergence(S).values()), float)
    return (np.nanmax(ts) if np.any(np.isfinite(ts)) else np.nan), w


def rate(build, depth):
    """Seconds per 1000 epochs, for training and for prediction."""
    w, s, mk, S = build(depth)
    plan = mk(w)
    lr = s.lr(depth, settings=S)
    t0 = time.time()
    train.train(models.make_model(w, depth, S), s, lr, PROBE_EPOCHS, S,
                order_seed=7, eval_every=PROBE_EPOCHS, plan=plan)
    sgd = (time.time() - t0) / PROBE_EPOCHS * 1000
    t0 = time.time()
    theory.predict(s, depth, lr, PROBE_EPOCHS, models.make_model(w, depth, S),
                   settings=S, eval_every=PROBE_EPOCHS, plan=plan)
    th = (time.time() - t0) / PROBE_EPOCHS * 1000
    return sgd, th, s.A.shape[0]


def main():
    print("rows and timescale per figure")
    budgets, rates = {}, {}
    for name, build in FIGS:
        for d in DEPTHS:
            sgd, th, rows = rate(build, d)
            rates[(name, d)] = (sgd, th)
            if name == "F1 emergence":
                bud = worlds.F1_EPOCHS[d]
                last = np.nan
            else:
                last, _ = discover(build, d)
                bud = int(np.ceil(last * 3 / 500.0) * 500) if np.isfinite(last) else DISCOVER[d]
            budgets[(name, d)] = bud
            print("  %-20s N=%d rows %4d | last emerges %7s | budget %6d "
                  "| sgd %6.2f s/1k | theory %6.2f s/1k"
                  % (name, d, rows,
                     ("%.0f" % last) if np.isfinite(last) else "n/a",
                     bud, sgd, th), flush=True)

    # runs per world: F1 one per depth; F2 two counterbalance arms; F3 one
    # shared phase then two branches, so two full-length equivalents.
    ARMS = {"F1 emergence": 1, "F2 identifiability": 2, "F3 integration": 2}
    print()
    print("per world, all three depths (minutes)")
    tot = 0.0
    for name, _ in FIGS:
        sub = 0.0
        for d in DEPTHS:
            sgd, th = rates[(name, d)]
            ep, arms = budgets[(name, d)], ARMS[name]
            sub += arms * ep / 1000.0 * (sgd + th) / 60.0
        tot += sub
        print("  %-20s %6.1f min" % (name, sub))
    print("  %-20s %6.1f min per world" % ("TOTAL", tot))
    print()
    for n_worlds in (1, 5, 10):
        print("  %2d worlds: %6.1f min = %.1f h" % (n_worlds, tot * n_worlds,
                                                    tot * n_worlds / 60.0))


if __name__ == "__main__":
    main()
