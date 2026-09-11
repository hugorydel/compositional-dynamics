"""Where does the depth-3 gap in Figure 3f come from?

The network sits above the no-parameter prediction late in the window, and the
excess shrinks in proportion to the step size, so it is first order in the
learning rate.  That says the gap is an approximation error, but not WHICH
approximation.  Four distinct things separate the plotted red and dashed
curves, and they are separable by running the same post-switch window several
ways from the same starting point:

  inherited      the prediction has its own pre-switch trajectory, integrated
                 from the initialisation rather than branched from the trained
                 weights.  Anything the two accumulate before the fact arrives
                 is already there at the switch and is not caused by the fact
                 at all.
  discretisation full-batch gradient descent at step `lr` is exactly the Euler
                 discretisation of the mean dynamics, so comparing it against
                 the integrated flow isolates the cost of taking finite steps.
  sampling       per-fact SGD differs from full-batch descent only in visiting
                 facts one at a time, so comparing those two isolates gradient
                 noise.
  integration    the prediction is itself a numerical object, RK4 at a fixed
                 substep count.  Raising the count says whether the dashed
                 curve has converged or is simply under-resolved.

Everything after the switch starts from the SAME weights unless the line says
otherwise, so the four can be read off independently.
"""

import copy
import sys

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan

DEPTH = 3
BASE = 0.03
WINDOW = 3000
MARKS = (250, 500, 1000, 2000, 3000)
ORDER_SEED = 7
SUBSTEPS = (32, 128, 512)      # 32 is what the figure uses


def geo(tr):
    return np.array(tr.epochs, float), np.array(tr.geometric["cross"], float)


def main(rate=BASE, seed=0):
    sc = BASE / rate
    every = max(1, int(round(worlds.F3_EVERY[DEPTH] * sc)))
    t1 = int(round(worlds.F3_SWITCH[DEPTH] * sc))
    t2 = int(round(WINDOW * sc))
    S = override(init_seed=1000 + seed, eval_every=every)
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(DEPTH, target=rate, settings=S)

    print("depth %d, rate %g, switch at %d, window %d epochs (%d rescaled)"
          % (DEPTH, rate, t1, t2, WINDOW), flush=True)

    # -- pre-switch, the two trajectories that meet at the branch ----------- #
    m = models.make_model(w0, DEPTH, S)
    a = train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED,
                    eval_every=every, plan=plan)
    ta, st = theory.predict(s0, DEPTH, lr, t1, models.make_model(w0, DEPTH, S),
                            settings=S, eval_every=every, plan=plan)
    print("  at the switch: network %.4f, prediction %.4f  ->  inherited "
          "excess %.4f" % (a.geometric["cross"][-1], ta.geometric["cross"][-1],
                           a.geometric["cross"][-1] - ta.geometric["cross"][-1]),
          flush=True)

    runs = {}
    sgd = train.train(copy.deepcopy(m), s1, lr, t2, S,
                      order_seed=ORDER_SEED + 1, eval_every=every, plan=plan)
    runs["per-fact SGD"] = geo(sgd)

    gd, _ = theory.predict(s1, DEPTH, lr, t2, copy.deepcopy(m).W,
                           settings=override(init_seed=S.init_seed,
                                             eval_every=every,
                                             ode_method="euler",
                                             ode_substeps={DEPTH: 1}),
                           eval_every=every, plan=plan)
    runs["full-batch GD"] = geo(gd)

    for n in SUBSTEPS:
        tr, _ = theory.predict(s1, DEPTH, lr, t2, copy.deepcopy(m).W,
                               settings=override(init_seed=S.init_seed,
                                                 eval_every=every,
                                                 ode_substeps={DEPTH: n}),
                               eval_every=every, plan=plan)
        runs["flow, %d substeps" % n] = geo(tr)
        print("  integrated the flow at %d substeps" % n, flush=True)

    tr, _ = theory.predict(s1, DEPTH, lr, t2, st, settings=S,
                           eval_every=every, plan=plan)
    runs["flow, own pre-switch"] = geo(tr)

    order = ["per-fact SGD", "full-batch GD"] + \
            ["flow, %d substeps" % n for n in SUBSTEPS] + \
            ["flow, own pre-switch"]
    print()
    print("  geometric error, all branched from the trained weights except the last")
    print("  %-22s %s" % ("", "".join("%10s" % ("+%d" % t) for t in MARKS)))
    for k in order:
        ep, g = runs[k]
        idx = [int(np.argmin(np.abs(ep / sc - t))) for t in MARKS]
        print("  %-22s %s" % (k, "".join("%10.4f" % g[i] for i in idx)))

    ref = runs["flow, 512 substeps"]
    print()
    print("  excess over the converged flow, as a fraction of it")
    for k in order:
        ep, g = runs[k]
        idx = [int(np.argmin(np.abs(ep / sc - t))) for t in MARKS]
        print("  %-22s %s" % (k, "".join("%9.2fx" % (g[i] / max(ref[1][i], 1e-12))
                                         for i in idx)))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else BASE)
