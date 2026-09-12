"""What a Figure 3 cell spends its time on, and where the step size lets it stop.

Adopting a smaller rate multiplies every budget, so the question is which parts
of the cost have to grow with it.  Two do not.

The integrator takes `substeps` RK4 steps per epoch, each of size `lr /
substeps`.  Its accuracy depends on that size, not on the substep count, so a
rate ten times smaller already has steps ten times finer at a fixed count and
the count can come down by the same factor.  Left alone, the prediction pays
the full multiplier for resolution it no longer needs.

The pre-switch phase has no gap to close: at depth 3 it tracks the theory at
alpha = 1.004 at the locked rate.  Only the phase after the fact needs the fine
step, so the budget before it can stay coarse.

Reports measured per-epoch costs for each part, the accuracy of the integrator
at reduced substeps, and what the two changes do to a full cell.
"""

import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan

DEPTH = 3
BASE = 0.03
PROBE = 400          # epochs to time
SEED = 0


def setup(rate):
    S = override(init_seed=1000 + SEED, eval_every=PROBE)
    w0 = worlds.integration_world(SEED, link=False)
    w1 = worlds.integration_world(SEED, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    return S, s0, s1, plan, s0.lr(DEPTH, target=rate, settings=S)


def main():
    S, s0, s1, plan, lr = setup(0.003)
    m = models.make_model(worlds.integration_world(SEED), DEPTH, S)

    t0 = time.time()
    train.train(m, s0, lr, PROBE, S, order_seed=7, eval_every=PROBE, plan=plan)
    sgd = (time.time() - t0) / PROBE
    print("  per-fact SGD          %8.3f ms per epoch" % (1e3 * sgd), flush=True)

    ode = {}
    for n in (1, 2, 4, 8):
        mm = models.make_model(worlds.integration_world(SEED), DEPTH, S)
        t0 = time.time()
        theory.predict(s1, DEPTH, lr, PROBE, mm,
                       settings=override(init_seed=S.init_seed,
                                         eval_every=PROBE, ode_substeps={DEPTH: n}),
                       eval_every=PROBE, plan=plan)
        ode[n] = (time.time() - t0) / PROBE
        print("  flow, %d substeps      %8.3f ms per epoch" % (n, 1e3 * ode[n]),
              flush=True)

    print()
    print("  does the flow still agree at fewer substeps, at rate 0.003?")
    every = 250 * 10
    t2 = 3000 * 10
    ref = None
    for n in (8, 2, 1):
        mm = models.make_model(worlds.integration_world(SEED), DEPTH, S)
        tr, _ = theory.predict(s1, DEPTH, lr, t2, mm,
                               settings=override(init_seed=S.init_seed,
                                                 eval_every=every,
                                                 ode_substeps={DEPTH: n}),
                               eval_every=every, plan=plan)
        g = np.array(tr.geometric["cross"], float)
        if ref is None:
            ref = g
            print("    %d substeps is the reference" % n, flush=True)
        else:
            print("    %d substeps: worst relative deviation %.3e"
                  % (n, np.max(np.abs(g - ref) / np.maximum(ref, 1e-12))),
                  flush=True)

    print()
    print("  cost of one depth-3 cell at rate 0.003, both arms")
    t1 = worlds.F3_SWITCH[DEPTH]
    for lab, pre_rate, subs in (("as configured", 0.003, 8),
                                ("substeps scaled with the rate", 0.003, 1),
                                ("and the pre-switch phase left at 0.03",
                                 0.03, 1)):
        pre = int(t1 * BASE / pre_rate)
        post = 3000 * 10
        # one pre-switch pass of each kind, then two arms of each
        secs = ((pre * (sgd + ode[8 if pre_rate == 0.03 else subs]))
                + 2 * post * (sgd + ode[subs]))
        print("    %-38s %6.0f s   (pre %6d epochs, post %6d)"
              % (lab, secs, pre, post))


if __name__ == "__main__":
    main()
