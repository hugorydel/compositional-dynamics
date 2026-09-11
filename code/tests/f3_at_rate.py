"""Figure 3 at a smaller step size, to see the depth-3 gap close.

`check_lr3.py` establishes the arithmetic: the continuous mean dynamics depend
on the product of learning rate and epochs, so dividing the rate by `s` and
multiplying every budget by `s` leaves the prediction untouched and only makes
the discrete process finer.  The measured gap in panel f shrinks in proportion
to the step, which is a first-order discretisation lag rather than instability.

This produces the whole figure on that footing: both arms, all three depths,
written in the record format `fig3.py` reads, into `results/f3_lr<rate>/`.  The
main results are not touched, and the panels are rendered to their own file.

  usage:  python f3_at_rate.py 0.01
          python f3_at_rate.py 0.01 0.003
"""

import copy
import os
import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
from _paths import RESULTS
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan
from run_experiments import join, save, series, usable

import fig3

BASE = 0.03
WINDOW = 3000       # rescaled epochs after the fact, matching the figure
DEPTHS = (1, 2, 3)
ORDER_SEED = 7


def tag(rate):
    return "f3_lr%s" % ("%g" % rate).replace(".", "p")


def run(seed, depth, rate):
    """`run_experiments.run_f3` with the rate freed and the budgets rescaled.

    Kept separate rather than threading a rate through the locked runner: the
    figures in the paper all use one step size, and that property is easier to
    defend if the file that produces them cannot be asked for another.
    """
    s_ = BASE / rate
    every = max(1, int(round(worlds.F3_EVERY[depth] * s_)))
    t1 = int(round(worlds.F3_SWITCH[depth] * s_))
    t2 = int(round(WINDOW * s_))
    S = override(init_seed=1000 + seed, eval_every=every)
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(depth, target=rate, settings=S)

    m = models.make_model(w0, depth, S)
    a = train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED,
                    eval_every=every, plan=plan)
    ta, st = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                            settings=S, eval_every=every, plan=plan)
    arms = {}
    for name, sy in (("hold", s0), ("insert", s1)):
        mm = copy.deepcopy(m)
        b = train.train(mm, sy, lr, t2, S, order_seed=ORDER_SEED + 1,
                        eval_every=every, plan=plan)
        tb, _ = theory.predict(sy, depth, lr, t2, st, settings=S,
                               eval_every=every, plan=plan)
        arms[name] = dict(net=join(series(a, True), series(b, True), t1),
                          pred=join(series(ta, True), series(tb, True), t1))
    return dict(figure="f3", seed=seed, depth=depth, t_switch=t1,
                epochs_max=t1 + t2, lr_target=rate, init_seed=S.init_seed,
                order_seed=ORDER_SEED, n_pairs=len(plan["a"]), arms=arms)


def main():
    rates = [float(a) for a in sys.argv[1:]] or [0.01]
    for rate in rates:
        sub = tag(rate)
        d = os.path.join(RESULTS, sub)
        os.makedirs(d, exist_ok=True)
        for depth in DEPTHS:
            p = os.path.join(d, "w00_d%d.json" % depth)
            if usable(p):
                print("  skip N=%d at %g" % (depth, rate), flush=True)
                continue
            t0 = time.time()
            save(p, run(0, depth, rate))
            print("  wrote N=%d at %g  (%.0fs)" % (depth, rate, time.time() - t0),
                  flush=True)
        fig3.main(sub=sub, out="fig3_%s.png" % sub)


if __name__ == "__main__":
    main()
