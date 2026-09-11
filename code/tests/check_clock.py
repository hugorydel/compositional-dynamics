"""Does the network run on a slow clock everywhere, or only after the linking fact?

The depth-3 panel of Figure 3 separates from its prediction, and the separation
is almost entirely horizontal: fitting one constant in `net(t) = pred(alpha t)`
collapses most of it.  At the locked rate alpha is 0.83, so the network
traverses the predicted trajectory at 83% speed.

If that were a property of depth-3 per-fact SGD at this step size, Figure 1 at
depth 3 would show it too, on the same rate and the same optimiser.  It runs a
single unstaged phase from the initialisation, so it is the clean control.  And
Figure 3 contains its own control: the pre-switch phase, before any fact is
inserted, on the same world, depth and rate as the phase that misbehaves.

Three fits, all from saved records, all on the geometric measure:

  F1            one phase from initialisation, one alpha per law
  F3 before     the shared pre-switch phase, taken from either arm
  F3 after      the linked arm after the fact, which is the panel in question

A flat stretch of curve carries no information about a time rescaling, so the
fit is run on the log of the measure over the window where it actually moves.
"""

import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

DEPTHS = (1, 2, 3)
GRID = np.linspace(0.60, 1.20, 6001)


def fit(e, v, ep_p, p):
    """alpha minimising the rms gap in log between `v(t)` and `p(alpha t)`."""
    ok = (v > 0) & np.isfinite(v)
    en, ln = e[ok], np.log(v[ok])
    lp = np.log(np.maximum(p, 1e-12))
    c = np.array([np.sqrt(np.mean((ln - np.interp(a * en, ep_p, lp)) ** 2))
                  for a in GRID])
    k = int(np.argmin(c))
    return GRID[k], float(np.sqrt(np.mean((ln - np.interp(en, ep_p, lp)) ** 2))), c[k]


def moving(v, frac=0.02):
    """Where the measure has actually left its starting value.  A curve that
    has not moved fixes no clock, and including it drags every alpha toward 1
    for reasons that have nothing to do with agreement."""
    v = np.asarray(v, float)
    return np.abs(np.log(np.maximum(v, 1e-12) / max(v[0], 1e-12))) > frac


def f1(seed, depth):
    p = os.path.join(RESULTS, "f1", "w%02d_d%d.json" % (seed, depth))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    e = np.array(r["net"]["epochs"], float)
    ep = np.array(r["pred"]["epochs"], float)
    out = []
    for law in r["net"]["geometric"]:
        v = np.array(r["net"]["geometric"][law], float)
        q = np.array(r["pred"]["geometric"][law], float)
        m = moving(v)
        if m.sum() < 10:
            continue
        out.append(fit(e[m], v[m], ep, q))
    return out


def f3(seed, depth, phase):
    p = os.path.join(RESULTS, "f3", "w%02d_d%d.json" % (seed, depth))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    arm = "hold" if phase == "before" else "insert"
    d = r["arms"][arm]
    e = np.array(d["net"]["epochs"], float)
    ep = np.array(d["pred"]["epochs"], float)
    v = np.array(d["net"]["geometric"]["cross"], float)
    q = np.array(d["pred"]["geometric"]["cross"], float)
    t = r["t_switch"]
    if phase == "before":
        m = e <= t
        return [fit(e[m], v[m], ep[ep <= t], q[ep <= t])]
    m = e >= t
    return [fit(e[m] - t, v[m], ep[ep >= t] - t, q[ep >= t])]


def show(label, res):
    if not res:
        print("  %-26s no record" % label)
        return
    a = np.array([x[0] for x in res])
    r1 = np.array([x[1] for x in res])
    rb = np.array([x[2] for x in res])
    extra = ("" if len(a) == 1 else
             "  (%d curves, %.3f to %.3f)" % (len(a), a.min(), a.max()))
    print("  %-26s alpha %6.3f | rms at alpha=1 %6.3f -> %6.3f%s"
          % (label, np.median(a), np.median(r1), np.median(rb), extra))


def main():
    print("fitted clock rate, geometric measure, lr target 0.03")
    print("alpha below 1 means the network is slower than the prediction\n")
    for seed in (0, 1):
        any_ = False
        for depth in DEPTHS:
            res = f1(seed, depth)
            if res:
                if not any_:
                    print("Figure 1, world %d" % seed)
                    any_ = True
                show("N=%d, one phase" % depth, res)
        if any_:
            print()
    print("Figure 3, world 0")
    for depth in DEPTHS:
        for phase in ("before", "after"):
            show("N=%d, %s the fact" % (depth, phase), f3(0, depth, phase))
    print()


if __name__ == "__main__":
    main()
