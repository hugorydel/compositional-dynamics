"""Collect every proposed measure on the toy unified-lattice world.

Per-depth epoch budget.  `System.lr` is depth-independent, and every cell runs
at the frozen lr_target = 0.3, so what differs between depths is only how many
epochs each needs: depth preconditions the slowest mode, and the unanchored
copy IS the slowest mode.  N = 1 needs 120k epochs to reach the cross-lattice
error N = 2 reaches in 4k.

Writes one JSON per (depth, link), then calls `toy_theory.ensure` for the same
cell, so the prospective prediction is regenerated alongside the data and the
figures never plot an overlay computed from a different run.
"""
import json
import time

import numpy as np

import _paths                                                 # noqa: F401
from _paths import result                                     # noqa: E402
import toy_score                                              # noqa: E402
import toy_theory                                             # noqa: E402
from toy import build, S                                      # noqa: E402
from relspec import System, models, train                     # noqa: E402

EPOCHS = {1: 120000, 2: 4000, 3: 4000}
EVERY = {1: 50, 2: 20, 3: 20}
DENSE = 4000            # keep every evaluation up to here, then 1 in 10
SEEDS = (0, 1, 2)
LR_TARGET = S.lr_target


def one(depth, link, seed):
    w = build(link, seed)
    s = System.build(w, settings=S)
    m = models.make_model(w, depth, S)
    tr = train.train(m, s, s.lr(depth, settings=S), EPOCHS[depth], S,
                     order_seed=7, eval_every=EVERY[depth],
                     probes=dict(E=np.eye(w.P)))
    keep = [t for t, e in enumerate(tr.epochs)
            if e <= DENSE or t % 10 == 0 or t == len(tr.epochs) - 1]
    rec = toy_score.as_json(toy_score.score(w, tr.probes["E"][keep]))
    rec["epochs"] = [float(e) for e in tr.epochs[keep]]
    return rec


def main():
    for depth in (1, 2, 3):
        for link in (False, True):
            t0 = time.time()
            runs = [one(depth, link, sd) for sd in SEEDS]
            name = "toy_d%d_%s.json" % (depth, "link" if link else "nolink")
            with open(result(name), "w") as f:
                json.dump(dict(depth=depth, link=link, seeds=list(SEEDS),
                               epochs_max=EPOCHS[depth], every=EVERY[depth],
                               lr_target=LR_TARGET,
                               cross=[list(c) for c in toy_score.CROSS],
                               runs=runs), f)
            print("wrote %s  (%.0fs)" % (name, time.time() - t0), flush=True)
            toy_theory.ensure(depth, link, force=True)


if __name__ == "__main__":
    main()
