"""The theory's prediction, scored against all 18 candidates.

The control pass re-scored the NETWORK under the full candidate pool, but
Figure 3 draws the prediction beside it, and the stored records hold the
prediction's hits only for the nine destination candidates.  The prediction is
cheap to regenerate -- a closed form at depth 1, an integration at depths 2 and
3 -- so this recomputes it from the same initialization and scores it under
both pools on the original evaluation grid.

Every cell is checked against its stored record: the nine-candidate predicted
hits must match exactly, or the cell fails.  That is what makes the added
eighteen-candidate curve the same object the paper already plots.

  usage:  python pat_pred18.py --cells f3:0-199:1 f3:0-199:2 f3:0-199:3 --nproc 12
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import _boot  # noqa: F401, E402
import _paths  # noqa: F401, E402
import numpy as np  # noqa: E402
from _paths import RESULTS  # noqa: E402
from relspec import System, models, theory, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import apply_plan, cross_plan  # noqa: E402

OUT = Path(RESULTS) / "_pat_stage1"
SUB = "f3_lr0p003"


def one(spec):
    seed, depth = spec
    tag = "w%02d_d%d" % (seed, depth)
    out = OUT / ("pred18_%s_%s.npz" % (SUB, tag))
    if out.exists():
        return tag, 0.0
    t0 = time.time()
    record = json.loads((Path(RESULTS) / SUB / (tag + ".json")).read_text())
    net = record["arms"]["insert"]["net"]
    every = int(net["epochs"][1] - net["epochs"][0])
    S = override(init_seed=record["init_seed"], eval_every=every,
                 lr_target=record["lr_target"],
                 ode_substeps={depth: record.get("substeps", 1)})
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = (System.build(w, settings=S) for w in (w0, w1))
    pairs = worlds.held_cross(w0, reference=w1)
    plan9 = cross_plan(w1, pairs)
    plan18 = cross_plan(w1, pairs, candidates=w1.entities)
    lr = s0.lr(depth, target=record["lr_target"], settings=S)
    t1 = record["t_switch"]
    t2 = record["epochs_max"] - t1

    init = models.make_model(w0, depth, S)
    _, state = theory.predict(s0, depth, lr, t1, init, settings=S, eval_every=t1)
    arrays = {}
    probes = {"E": np.eye(w0.P)}
    for arm, system in (("hold", s0), ("insert", s1)):
        # integrate once, then score both pools off the same states: the
        # integration is the cost, the scoring is not
        traj, _ = theory.predict(system, depth, lr, t2, state, settings=S,
                                 eval_every=every, probes=probes)
        states = traj.probes["E"]
        for label, plan in (("pred9", plan9), ("pred18", plan18)):
            arrays["%s.%s" % (arm, label)] = np.array(
                [apply_plan(plan, E)[2]["cross"] for E in states])
        arrays["epochs"] = traj.epochs
        # the stored record is the check: same prediction, same hits
        old = record["arms"][arm]["pred"]
        ep = np.asarray(old["epochs"], float)
        keep = ep >= t1
        stored = np.asarray(old["hits"]["cross"], bool)[keep]
        mine = arrays["%s.pred9" % arm]
        if stored.shape != mine.shape or not np.array_equal(stored, mine):
            raise AssertionError("%s %s: predicted hits differ from the record" % (tag, arm))
    np.savez_compressed(out, **arrays)
    return tag, time.time() - t0


def parse(cell):
    parts = cell.split(":")
    lo, _, hi = parts[1].partition("-")
    seeds = range(int(lo), int(hi) + 1) if hi else [int(lo)]
    return [(s, int(parts[2])) for s in seeds]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["f3:0:1"])
    ap.add_argument("--nproc", type=int, default=1)
    ns = ap.parse_args()
    todo = [c for spec in ns.cells for c in parse(spec)]
    print("%d cells" % len(todo), flush=True)
    t0 = time.time()
    if ns.nproc > 1:
        with Pool(min(ns.nproc, len(todo))) as pool:
            for k, (tag, dt) in enumerate(pool.imap_unordered(one, todo), 1):
                if k % 25 == 0 or k == len(todo):
                    print("   %d of %d, %.1f min elapsed"
                          % (k, len(todo), (time.time() - t0) / 60.0), flush=True)
    else:
        for cell in todo:
            tag, dt = one(cell)
            print("   %s in %.0f s" % (tag, dt), flush=True)


if __name__ == "__main__":
    main()
