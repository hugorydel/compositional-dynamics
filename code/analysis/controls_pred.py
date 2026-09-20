"""The theory's prediction, scored the way the control pass scores the network.

The control pass re-scored the NETWORK under the full candidate pool and on the
trained facts.  The figures draw the prediction beside it, and the stored
records hold the prediction's hits only for the nine destination candidates, on
the evaluation grid the published run used.  The prediction is cheap to
regenerate -- a closed form at depth 1, an integration at depths 2 and 3 -- so
this recomputes it from the same initialization and scores it the same way:
against both candidate pools, and against each world's own training premises.

Two things differ from the stored prediction, and both are only resolution.
It is recorded every ten epochs to 500 and every fifty after that, rather than
every 2,500, 500 or 250, because a curve drawn on a 30,000-epoch axis with a
2,500-epoch stride is a sequence of straight segments through the one region
where the curve bends.  The stride does not change the trajectory: `substeps`
is per epoch, so the integrator takes the same steps whatever is recorded
(`relspec/theory.py`).  The second phase is entered from the state the first
phase returned, which reproduces an uninterrupted run exactly at every depth
(tests/check_checkpoint.py).

Every cell is checked against its stored record at the epochs the two grids
share: the nine-candidate predicted hits must match exactly, or the cell fails.
That is what makes the added curves the same object the paper already plots.

  usage:  python controls_pred.py --cells f3:0-199:1 f3:0-199:2 f3:0-199:3 --nproc 12
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

from controls import fact_hits, fact_queries  # noqa: E402

OUT = Path(RESULTS) / "f3"
SUB = "f3"
FINE = ((500, 10), (None, 50))     # (up to, stride); None is the end of the run
NEW = "insert.premise"             # a cell without this key predates the rewrite


def states(system, depth, lr, t2, state, S, probes):
    """Predicted embeddings on the drawing grid, and the state at the end."""
    epochs, out, done = [], [], 0
    for limit, step in FINE:
        span = (t2 if limit is None else min(limit, t2)) - done
        if span <= 0:
            break
        traj, state = theory.predict(system, depth, lr, span, state, settings=S,
                                     eval_every=step, probes=probes)
        keep = slice(1, None) if epochs else slice(None)   # the join is recorded twice
        epochs.append(np.asarray(traj.epochs)[keep] + done)
        out.extend(traj.probes["E"][keep])
        done += span
    return np.concatenate(epochs), out


def one(spec):
    seed, depth, force = spec
    tag = "w%02d_d%d" % (seed, depth)
    out = OUT / ("controls_pred_%s.npz" % tag)
    if out.exists() and not force:
        with np.load(out) as z:
            if NEW in z.files:
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
    facts = list(dict.fromkeys(w0.facts))
    premise = [f for f in facts if f[1].startswith(("x", "y"))]
    queries = {"premise": fact_queries(w0, premise), "facts": fact_queries(w0, facts)}
    lr = s0.lr(depth, target=record["lr_target"], settings=S)
    t1 = record["t_switch"]
    t2 = record["epochs_max"] - t1

    init = models.make_model(w0, depth, S)
    _, state = theory.predict(s0, depth, lr, t1, init, settings=S, eval_every=t1)
    arrays, probes = {}, {"E": np.eye(w0.P)}
    for arm, system in (("hold", s0), ("insert", s1)):
        # integrate once, then score everything off the same states: the
        # integration is the cost, the scoring is not
        ep, Es = states(system, depth, lr, t2, state, S, probes)
        arrays["epochs"] = ep
        for label, plan in (("pred9", plan9), ("pred18", plan18)):
            arrays["%s.%s" % (arm, label)] = np.array(
                [apply_plan(plan, E)[2]["cross"] for E in Es])
        for label, q in queries.items():
            arrays["%s.%s" % (arm, label)] = np.array([fact_hits(q, E)[0] for E in Es])
        # the stored record is the check: same prediction, same hits, at every
        # epoch the published grid and this one share
        old = record["arms"][arm]["pred"]
        oep = np.asarray(old["epochs"], float)
        keep = oep >= t1
        stored = np.asarray(old["hits"]["cross"], bool)[keep]
        where = np.searchsorted(ep, oep[keep] - t1)
        mine = arrays["%s.pred9" % arm][where]
        if (not np.array_equal(ep[where], oep[keep] - t1)
                or stored.shape != mine.shape or not np.array_equal(stored, mine)):
            raise AssertionError("%s %s: predicted hits differ from the record" % (tag, arm))
    np.savez_compressed(out, **arrays)
    return tag, time.time() - t0


def parse(cell, force):
    parts = cell.split(":")
    lo, _, hi = parts[1].partition("-")
    seeds = range(int(lo), int(hi) + 1) if hi else [int(lo)]
    return [(s, int(parts[2]), force) for s in seeds]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["f3:0:1"])
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="recompute cells already saved")
    ns = ap.parse_args()
    todo = [c for spec in ns.cells for c in parse(spec, ns.force)]
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
