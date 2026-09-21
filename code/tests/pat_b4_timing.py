"""B4: what resolution are Experiment 3's event times actually measured at?

Appendix B.4 carries a caveat that event times are multiples of the evaluation
interval, so a median or a quartile can only land on the grid.  On the published
grid that interval is 2,500, 500 and 250 epochs, which at depth 1 is coarse
enough that the quartiles read as "between the seventh and the tenth check".

The replay evaluates the same runs far more often, and the prediction is
evaluated at every point of the replay's grid (`shared`), so network and theory
are compared where both were measured rather than through an interpolation.
Nothing is re-trained: the replay reproduces each record at every epoch the two
grids share and only adds evaluations between them.

Resolution is reported as the interval a crossing actually fell in -- the gap
between the last evaluation below the criterion and the one that met it -- not
as an average of neighbouring intervals, since the grid is not uniform and it is
that gap, and only that gap, which bounds the error in the time.  The
qualification stays: an event time is known to within its crossing interval, and
the interval is quoted with it.

Both pools are reported.  The nine-candidate column is the published metric and
can be compared with the records directly; the eighteen-candidate column is the
metric E1 adopts, and the records cannot produce it at all.

  usage:  python pat_b4_timing.py
"""

import json
import os
import sys

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "analysis"))
from common import agreement_r2  # noqa: E402

OUT = os.path.join(RESULTS, "f3")
DEPTHS = (1, 2, 3)
SEEDS = range(200)
TARGET = 72          # of 80; the manuscript's unlock criterion


def crossing(epochs, counts):
    """(time of the first evaluation meeting the criterion, its interval).

    The interval is the gap from the previous evaluation, which is the window
    the true crossing lies in.  Zero when the criterion already held at the
    intervention, and infinite when it is never met within the run.
    """
    k = np.flatnonzero(np.asarray(counts) >= TARGET)
    if not len(k):
        return np.inf, np.nan
    i = int(k[0])
    return float(epochs[i]), (float(epochs[i] - epochs[i - 1]) if i else 0.0)


def replay(depth):
    """Network and predicted counts per world on the one grid both were
    measured on, under each candidate pool."""
    grid, net, pred = None, {}, {}
    for seed in SEEDS:
        a = np.load(os.path.join(OUT, "controls_w%02d_d%d.npz" % (seed, depth)))
        b = np.load(os.path.join(OUT, "controls_pred_w%02d_d%d.npz" % (seed, depth)))
        m = b["shared"]
        if not np.array_equal(b["epochs"][m], a["epochs"]):
            raise SystemExit("world %d at depth %d: the two grids do not match" % (seed, depth))
        if grid is None:
            grid = a["epochs"]
        elif not np.array_equal(grid, a["epochs"]):
            raise SystemExit("world %d at depth %d is on a different grid" % (seed, depth))
        for pool, nk, pk in (("nine", "cross9", "pred9"), ("all 18", "cross18", "pred18")):
            net.setdefault(pool, []).append(a["insert.%s" % nk].sum(axis=1))
            pred.setdefault(pool, []).append(b["insert.%s" % pk][m].sum(axis=1))
        a.close(), b.close()
    return grid, net, pred


def records(depth):
    """The same event, on the published grid, from the records themselves."""
    tn, tp, iv, step = [], [], [], None
    for seed in SEEDS:
        r = json.load(open(os.path.join(OUT, "w%02d_d%d.json" % (seed, depth))))
        for src, out in (("net", tn), ("pred", tp)):
            d = r["arms"]["insert"][src]
            ep = np.asarray(d["epochs"], float) - r["t_switch"]
            keep = ep >= 0
            t, w = crossing(ep[keep], np.asarray(d["hits"]["cross"], bool)[keep].sum(axis=1))
            out.append(t)
            if src == "net":
                iv.append(w)
        step = int(ep[1] - ep[0])
    return np.array(tn), np.array(tp), np.array(iv), step


def report(label, tn, tp, iv):
    """Event times and their agreement, at the resolution they were measured."""
    fn, fp = np.isfinite(tn), np.isfinite(tp)
    both = fn & fp
    pos = both & (tn > 0) & (tp > 0)
    err = np.abs(tn[both] - tp[both])
    win = iv[both]
    finite = tn[fn]
    print("    %-24s median %8s  IQR %8s-%-8s  distinct %3d"
          % (label, "{:,.0f}".format(np.median(finite)),
             "{:,.0f}".format(np.percentile(finite, 25)),
             "{:,.0f}".format(np.percentile(finite, 75)), len(np.unique(finite))))
    print("    %-24s crossing interval: median %s, worst %s (%.1f%% of the median time)"
          % ("", "{:,.0f}".format(np.median(win)), "{:,.0f}".format(np.max(win)),
             100.0 * np.median(win) / np.median(finite)))
    print("    %-24s vs theory: R2 of log10 %.4f | |dt| median %s, 95th %s | within "
          "its own crossing interval %.1f%% | both reach it %d of %d"
          % ("", agreement_r2(np.log10(tn[pos]), np.log10(tp[pos])),
             "{:,.0f}".format(np.median(err)), "{:,.0f}".format(np.percentile(err, 95)),
             100.0 * np.mean(err <= win), int(both.sum()), len(tn)))


def main():
    print("B4: unlock time, the epochs from the linking fact until %d of 80 are "
          "correct" % TARGET)
    print("200 worlds per depth; network and theory measured on the same grid\n")
    for depth in DEPTHS:
        print("  N=%d" % depth)
        tn, tp, iv, step = records(depth)
        report("records, nine", tn, tp, iv)
        grid, net, pred = replay(depth)
        for pool in ("nine", "all 18"):
            tn, iv = zip(*[crossing(grid, v) for v in net[pool]])
            tp = [crossing(grid, v)[0] for v in pred[pool]]
            report("replay, " + pool, np.array(tn), np.array(tp), np.array(iv))
        print()


if __name__ == "__main__":
    main()
