"""Where does the time in one Figure 1 cell actually go?

A cell does three things: walk the constraint rows with per-fact SGD, integrate
the mean dynamics for the parameter-free prediction, and score 124 held-out
composites at every evaluation.  Which of those dominates decides whether the
epoch budget can be cut cheaply, or whether the run has to be split into a
cheap unmeasured pass for the emergence times and a short measured pass for the
curves, the way the earlier codebase did it.

Timed over a short common budget at each depth, so the three numbers are
directly comparable and the whole thing costs under a minute.  Scoring cost is
the difference between training with a plan and training without one, since
that is the only thing the plan adds.
"""

import time

import _boot  # noqa: F401
import _paths  # noqa: F401
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import law_plan

EPOCHS, EVERY = 500, 25


def clocked(fn):
    t = time.time()
    fn()
    return time.time() - t


def main():
    print("%d epochs, evaluating every %d.  Per-cell cost is these scaled to "
          "the real budget." % (EPOCHS, EVERY))
    print("%-5s %10s %10s %10s %10s" % ("depth", "sgd", "scoring", "predict",
                                        "total"))
    for depth in (1, 2, 3):
        S = override(init_seed=1000, eval_every=EVERY)
        w = worlds.emergence_world(0)
        s = System.build(w, settings=S)
        plan = law_plan(w, worlds.held_composites(w))
        lr = s.lr(depth, settings=S)

        bare = clocked(lambda: train.train(
            models.make_model(w, depth, S), s, lr, EPOCHS, S, order_seed=2000,
            eval_every=EVERY))
        full = clocked(lambda: train.train(
            models.make_model(w, depth, S), s, lr, EPOCHS, S, order_seed=2000,
            eval_every=EVERY, plan=plan))
        pred = clocked(lambda: theory.predict(
            s, depth, lr, EPOCHS, models.make_model(w, depth, S), settings=S,
            eval_every=EVERY, plan=plan))
        print("%-5d %9.1fs %9.1fs %9.1fs %9.1fs"
              % (depth, bare, full - bare, pred, full + pred), flush=True)


if __name__ == "__main__":
    main()
