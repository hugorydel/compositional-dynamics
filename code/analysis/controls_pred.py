"""The theory's prediction, scored the way the control pass scores the network.

The control pass re-scored the NETWORK under the full candidate pool, on the
trained facts, and on the geometric error.  The figures and the statistics draw
the prediction beside it, and the stored records hold the prediction's hits only
for the nine destination candidates, on the evaluation grid the published run
used.  The prediction is cheap to regenerate -- a closed form at depth 1, an
integration at depths 2 and 3 -- so this recomputes it from the same
initialization and scores it the same way: against both candidate pools,
against each world's own training premises, and on the geometric error.

It is evaluated on the union of two grids.  The first is the network's own
replay grid, point for point, so that network and prediction can be compared
where both were measured rather than through an interpolation: `shared` marks
those points, and every timing and trajectory statistic uses them.  The second
adds every tenth epoch to 500 and every fiftieth after that, because a curve
drawn on a 30,000-epoch axis from the published grid alone is a sequence of
straight segments through the one region where it bends.

Recording more often does not change the trajectory.  `substeps` is per epoch,
so the integrator takes the same steps whatever is recorded, and each segment is
entered from the state the previous one returned, which reproduces an
uninterrupted run exactly at every depth (tests/check_checkpoint.py).

Every cell is checked against its stored record at the epochs the published grid
and this one share: the nine-candidate predicted hits must match exactly and the
predicted geometric error to within 1e-9 relative, or the cell fails.  That is
what makes the added curves the same object the paper already plots.

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
NEW = "shared"                     # a cell without this key predates the rewrite


def grid_for(seed, depth, t2):
    """The network's replay grid, plus points fine enough to draw between them."""
    with np.load(OUT / ("controls_w%02d_d%d.npz" % (seed, depth))) as z:
        net = np.asarray(z["epochs"], float)
    fine = set()
    for limit, step in FINE:
        fine |= set(range(0, (t2 if limit is None else min(limit, t2)) + 1, step))
    grid = np.array(sorted({int(round(e)) for e in net} | fine), float)
    return grid, np.isin(grid, net)


def states(system, depth, lr, grid, state, S, probes):
    """Predicted embeddings at exactly `grid`, which starts at the intervention.

    Stepped gap by gap, since the grid is not uniform.  Each segment starts from
    the state the previous one ended at, so the sequence is one trajectory.
    """
    out, prev = [], 0.0
    for t in grid[1:]:
        span = int(round(t - prev))
        traj, state = theory.predict(system, depth, lr, span, state, settings=S,
                                     eval_every=span, probes=probes)
        rec = traj.probes["E"]
        if not out:
            out.append(rec[0])          # the state at the intervention itself
        out.append(rec[-1])
        prev = t
    return out


def one(spec):
    seed, depth, force = spec
    tag = "w%02d_d%d" % (seed, depth)
    out = OUT / ("controls_pred_%s.npz" % tag)
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
    grid, shared = grid_for(seed, depth, t2)
    if out.exists() and not force:
        # A cached cell is only reusable on the grid in force now.  `shared`
        # indexes into the network's grid, and that grid changes whenever the
        # replay's sampling does, so the key alone is not enough to trust it.
        with np.load(out) as z:
            if (NEW in z.files and np.array_equal(z["epochs"], grid)
                    and np.array_equal(z["shared"], shared)):
                return tag, 0.0

    init = models.make_model(w0, depth, S)
    _, state = theory.predict(s0, depth, lr, t1, init, settings=S, eval_every=t1)
    arrays = {"epochs": grid, "shared": shared}
    probes = {"E": np.eye(w0.P)}
    for arm, system in (("hold", s0), ("insert", s1)):
        # integrate once, then score everything off the same states: the
        # integration is the cost, the scoring is not
        Es = states(system, depth, lr, grid, state, S, probes)
        if len(Es) != len(grid):
            raise AssertionError("%s %s: %d states for %d grid points"
                                 % (tag, arm, len(Es), len(grid)))
        for label, plan in (("pred9", plan9), ("pred18", plan18)):
            scored = [apply_plan(plan, E) for E in Es]
            arrays["%s.%s" % (arm, label)] = np.array([s[2]["cross"] for s in scored])
            if label == "pred9":
                # the paper's geometric error, defined on the destination pool:
                # distance to the target in candidate spacings
                arrays["%s.geo9" % arm] = np.array([s[1]["cross"] for s in scored], float)
        for label, q in queries.items():
            arrays["%s.%s" % (arm, label)] = np.array([fact_hits(q, E)[0] for E in Es])
        # the stored record is the check: same prediction, same hits, same
        # geometry, at every epoch the published grid and this one share
        old = record["arms"][arm]["pred"]
        oep = np.asarray(old["epochs"], float)
        keep = oep >= t1
        where = np.searchsorted(grid, oep[keep] - t1)
        if not np.array_equal(grid[where], oep[keep] - t1):
            raise AssertionError("%s %s: the published grid is not a subset" % (tag, arm))
        stored = np.asarray(old["hits"]["cross"], bool)[keep]
        mine = arrays["%s.pred9" % arm][where]
        if stored.shape != mine.shape or not np.array_equal(stored, mine):
            raise AssertionError("%s %s: predicted hits differ from the record" % (tag, arm))
        g_old = np.asarray(old["geometric"]["cross"], float)[keep]
        g_new = arrays["%s.geo9" % arm][where]
        worst = float(np.max(np.abs(g_new - g_old) / np.maximum(g_old, 1e-12)))
        if worst > 1e-9:
            raise AssertionError("%s %s: predicted geometry differs from the record "
                                 "by %.2e" % (tag, arm, worst))
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
