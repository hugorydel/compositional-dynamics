"""What Figure 1 actually contains: budgets, step sizes, and where each law lands.

Reads the saved cells rather than the picture, so the numbers are the ones the
panels are drawn from.  Three things it answers: how much of the standardised
axis each depth occupies, how coarse the staircase is (one held-out composite
is one step), and whether behaviour is running ahead of geometry the way it
did in Figure 3.
"""

import glob
import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence, resolved

NLAW = 3


def emergence(ep, rec):
    return {k: (float(ep[int(i)]) if np.isfinite(i) else np.inf)
            for k, i in ((k, detect_emergence(rec["retrieval"][k],
                                              rec["geometric"][k],
                                              S.hold, S.tau))
                         for k in rec["retrieval"])}


def first_at(ep, v, thr):
    i = np.where(v >= thr)[0]
    return float(ep[i[0]]) if len(i) else np.nan


def below(ep, v, thr):
    i = np.where(v < thr)[0]
    return float(ep[i[0]]) if len(i) else np.nan


for depth in (1, 2, 3):
    files = sorted(glob.glob(os.path.join(glob.escape(RESULTS), "f1", "w*_d%d.json" % depth)))
    if not files:
        print("N=%d  no cells" % depth)
        continue
    r = json.load(open(files[0]))
    ep = np.array(r["net"]["epochs"], float)
    laws = sorted(emergence(ep, r["net"]).items(), key=lambda kv: kv[1])
    names = [k for k, _ in laws]
    pick = [names[int(round(v))] for v in np.linspace(0, len(names) - 1, NLAW)]
    print()
    print("N=%d   run to %d epochs, %d evaluations, %d laws"
          % (depth, ep[-1], len(ep), len(names)))
    print("   drawn: %s" % ", ".join("Law %s = %s" % (c, p)
                                     for c, p in zip("ABC", pick)))
    for c, law in zip("ABC", pick):
        h = np.array(r["net"]["hits"][law], bool)
        n = h.shape[1]
        acc = resolved(ep, h, chance=1.0 / n)
        g = np.array(r["net"]["geometric"][law], float)
        print("    Law %s  %2d held-out items (one step = %5.1f%%)"
              "  100%% at %8s  final %5.1f%%   |  geo < tau at %8s"
              "  final %7.4f  peak %6.3f"
              % (c, n, 100.0 / n,
                 ("%.0f" % first_at(ep, acc, 99.9)) if np.isfinite(first_at(ep, acc, 99.9)) else "never",
                 acc[-1],
                 ("%.0f" % below(ep, g, S.tau)) if np.isfinite(below(ep, g, S.tau)) else "never",
                 g[-1], g.max()))
    # how much of the 0-10k axis is occupied
    print("    axis: run covers %.0f%% of the 0-10k window"
          % min(100.0, 100.0 * ep[-1] / 10000.0))
