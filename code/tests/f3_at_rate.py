"""Figure 3 at a smaller step size, to see the depth-3 gap close.

`check_lr3.py` establishes the arithmetic: the continuous mean dynamics depend
on the product of learning rate and epochs, so dividing the rate by `s` and
multiplying every budget by `s` leaves the prediction untouched and only makes
the discrete process finer.  The measured gap in panel f shrinks in proportion
to the step, which is a first-order discretisation lag rather than instability.

This produces the whole figure on that footing: both arms, all three depths,
written in the record format `fig3.py` reads, into `results/f3_lr<rate>/`.  The
main results are not touched, and the panels are rendered to their own file.

Three things keep the cost of a fine step off the parts that do not need it,
all measured by `check_cost.py`:

  substeps      the integrator's accuracy is set by `lr / substeps`, not by the
                count, so a rate ten times smaller is already ten times finer
                at a fixed count.  The count is scaled with the rate.  At 0.003
                one substep and eight agree to 5e-13 over the whole window.

  one rate      the pre-switch phase runs at the same rate as the rest.  An
                earlier version ran it at `BASE` to save the largest budget in
                the cell, on the argument that it has no gap to close, since it
                fits alpha = 1.004 at the locked rate.  That is true of its
                clock and false of its endpoint: running at the right speed for
                5,500 epochs still leaves the network 0.043 above the
                prediction when the two arrive at the branch, and that offset
                multiplies through the whole window.  Measured at depth 3, the
                error at the far edge was 1.07 with one rate throughout and
                1.23 with a coarse pre-switch phase, against 2.02 at 0.03.  The
                saving was five minutes across fifteen cells.

  processes     cells are independent, so they run in a pool, as in the main
                runner.

  usage:  python f3_at_rate.py 0.003
          python f3_at_rate.py 0.01 0.003 --seeds 0 1 2 3 4
          python f3_at_rate.py 0.003 --nproc 4
"""

import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import _boot  # noqa: F401
import _paths  # noqa: F401
from _paths import RESULTS
from relspec import System, models, theory, train, worlds
from relspec.config import DEFAULT, override
from relspec.measure import cross_plan
from run_experiments import REQUIRED, join, save, series

import fig3

BASE = 0.03         # the rate the pre-switch phase runs at, and the time unit
WINDOW = 4000       # rescaled epochs after the fact
DEPTHS = (1, 2, 3)
ORDER_SEED = 7
NPROC = 12


def tag(rate):
    return "f3_lr%s" % ("%g" % rate).replace(".", "p")


def substeps_for(depth, rate):
    """The configured count, scaled with the rate so the RK4 step is unchanged."""
    return {depth: max(1, int(round(DEFAULT.substeps(depth) * rate / BASE)))}


def run(seed, depth, rate):
    """`run_experiments.run_f3` with a two-rate schedule.

    Kept separate rather than threading a rate through the locked runner: the
    figures in the paper all use one step size, and that property is easier to
    defend if the file that produces them cannot be asked for another.
    """
    sc = BASE / rate
    t1 = int(round(worlds.F3_SWITCH[depth] * sc))
    t2 = int(round(WINDOW * sc))
    every = max(1, int(round(worlds.F3_EVERY[depth] * sc)))

    S = override(init_seed=1000 + seed, eval_every=every,
                 ode_substeps=substeps_for(depth, rate))
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
                epochs_max=t1 + t2, lr_target=rate, window=WINDOW,
                substeps=S.substeps(depth), init_seed=S.init_seed,
                order_seed=ORDER_SEED, n_pairs=len(plan["a"]), arms=arms)


def done(seed, depth, rate):
    """Is this cell already on disk under the schedule in force here?

    `run_experiments.usable` cannot answer this.  Its staleness test compares a
    record's budgets against the locked `F3_SWITCH` and `F3_AFTER`, which is
    right for the main runner and wrong here: every record this script writes
    has budgets rescaled by the rate, so all of them read as stale and a resume
    recomputes work it already has.  The same question asked against THIS
    script's schedule is what follows.
    """
    p = path_for(seed, depth, rate)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            r = json.load(f)
    except (ValueError, OSError):
        return False
    d = r["arms"][sorted(r["arms"])[0]]["net"]
    if not all(k in d for k in REQUIRED):
        return False
    sc = BASE / rate
    t1 = int(round(worlds.F3_SWITCH[depth] * sc))
    return (r.get("lr_target") == rate
            and r.get("window") == WINDOW
            and r.get("substeps") == substeps_for(depth, rate)[depth]
            and r.get("t_switch") == t1
            and r.get("epochs_max") == t1 + int(round(WINDOW * sc)))


def path_for(seed, depth, rate):
    d = os.path.join(RESULTS, tag(rate))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "w%02d_d%d.json" % (seed, depth))


def one(cell):
    seed, depth, rate = cell
    t0 = time.time()
    save(path_for(seed, depth, rate), run(seed, depth, rate))
    return ("w%02d N=%d at %g" % (seed, depth, rate), time.time() - t0)


def main():
    args = sys.argv[1:]
    seeds, nproc = [0], NPROC
    for flag, cast in (("--seeds", int), ("--nproc", int)):
        if flag in args:
            k = args.index(flag)
            if flag == "--nproc":
                nproc = int(args[k + 1])
                args = args[:k] + args[k + 2:]
            else:
                seeds = [cast(a) for a in args[k + 1:]]
                args = args[:k]
    rates = [float(a) for a in args] or [0.01]

    todo = [(s, d, r) for r in rates for s in seeds for d in DEPTHS
            if not done(s, d, r)]
    print("%d cells, window %d rescaled epochs, one rate throughout"
          % (len(todo), WINDOW), flush=True)
    if todo:
        nproc = max(1, min(nproc, len(todo)))
        with Pool(nproc) as pool:
            for what, dt in pool.imap_unordered(one, todo):
                print("  wrote %s  (%.0fs)" % (what, dt), flush=True)
    for rate in rates:
        fig3.main(sub=tag(rate),
                  out="fig3_eta%s.png" % ("%g" % rate).replace(".", "p"))


if __name__ == "__main__":
    main()
