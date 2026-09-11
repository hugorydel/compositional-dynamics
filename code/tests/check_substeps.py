"""Is the integrator's substep count converged, and is it converged from below?

`ode_substeps` is 8 at depth 2 and 32 at depth 3, and it multiplies the whole
cost of a prediction: one epoch of integration is `substeps` RK4 steps, each
four calls to `ode_rhs`.  At depth 3 the prediction is 72 per cent of a cell's
runtime, so the count is the single largest lever on how long a figure takes to
produce.

An earlier check established that 32, 128 and 512 agree.  That shows the answer
has stopped changing as the count RISES, which is the wrong direction for
deciding whether 32 is needed.  This one comes up from the bottom.

The comparison is made on what the figure actually draws, not on the weights:
the geometric error curve and the retrieval curve of the plotted law, against
the highest count tested.  A count is acceptable only if its curve is
indistinguishable at the precision the figure is read at.
"""

import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.measure import cross_plan, law_plan

SEED = 0
COUNTS = {2: (2, 4, 8, 16, 32), 3: (4, 8, 16, 32, 64)}
# a window long enough to cover the steep descent, short enough to sweep.
# F1 is unstaged, so its second number is zero and only the first phase runs.
BUDGET = {2: (2000, 4000), 3: (1000, 2000)}
BUDGET_F1 = {2: (1500, 0), 3: (1000, 0)}


def f2_setup(depth, S):
    w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block="A")
    w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block="A",
                                      n_bridge=1)
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    return w0, w1, plan, "B"


def f1_setup(depth, S):
    w = worlds.emergence_world(SEED)
    plan = law_plan(w, worlds.held_composites(w))
    return w, w, plan, plan["names"][0]


def f3_setup(depth, S):
    w0 = worlds.integration_world(SEED, link=False)
    w1 = worlds.integration_world(SEED, link=True)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    return w0, w1, plan, "cross"


def curves(fig, depth, sub):
    """The predicted post-switch curves at one substep count."""
    S = override(init_seed=1000 + SEED,
                 ode_substeps={1: 1, 2: sub, 3: sub, 4: sub, 5: sub})
    t1, t2 = (BUDGET_F1 if fig == "f1" else BUDGET)[depth]
    setup = {"f1": f1_setup, "f2": f2_setup}.get(fig, f3_setup)
    w0, w1, plan, key = setup(depth, S)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    lr = s0.lr(depth, settings=S)
    every = 10
    ta, st = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                            settings=S, eval_every=every, plan=plan)
    if not t2:
        return (np.array(ta.geometric[key], float),
                np.array(ta.retrieval[key], float))
    tb, _ = theory.predict(s1, depth, lr, t2, st, settings=S,
                           eval_every=every, plan=plan)
    return (np.array(tb.geometric[key], float),
            np.array(tb.retrieval[key], float))


def main():
    figs = sys.argv[1:] or ["f2", "f3"]
    print("post-switch prediction, seed %d.  reference is the highest count"
          % SEED)
    for fig in figs:
        for depth in (2, 3):
            counts = COUNTS[depth]
            print()
            print("  %s  N=%d  %d then %d epochs"
                  % (fig.upper(), depth,
                     *(BUDGET_F1 if fig == "f1" else BUDGET)[depth]))
            got = {}
            for sub in counts:
                t0 = time.perf_counter()
                got[sub] = curves(fig, depth, sub)
                got[sub] += (time.perf_counter() - t0,)
            gref, rref, tref = got[counts[-1]]
            for sub in counts:
                g, r, t = got[sub]
                dg = float(np.abs(g - gref).max())
                # the error spans decades, so the relative miss is what matters
                rel = float(np.abs(g / np.maximum(gref, 1e-30) - 1.0).max())
                dr = float(np.abs(r - rref).max())
                print("     substeps %3d  %6.2fs  max |dgeo| %.2e  "
                      "max rel %.2e  max |dretrieval| %.3f%%%s"
                      % (sub, t, dg, rel, dr,
                         "   <- reference" if sub == counts[-1] else ""),
                      flush=True)


if __name__ == "__main__":
    main()
