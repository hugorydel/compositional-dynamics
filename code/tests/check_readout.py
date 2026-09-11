"""How the two rows of Figure 3 relate, and what the top row reads at the switch.

Two questions, both answered from the saved per-item records rather than from a
fresh run.

1.  Accuracy reaches one hundred per cent while the geometric error is still
    around one candidate spacing, and the error then falls two further decades
    with nothing left to show behaviourally.  That is either a legitimate
    consequence of a continuous distance being read through a discrete
    nearest-neighbour decision, or a scoring fault.  Pooling every item at
    every evaluation and asking what fraction is retrieved correctly at a given
    error separates the two: a clean step means the behavioural measure is a
    threshold on the geometric one, and the timing mismatch is the threshold,
    not a bug.

2.  The depth-2 panel starts above zero.  Both arms leave the same weights, and
    at the switch nothing in the world has determined the alignment between the
    two copies, so anything scored there is scored without the information the
    question asks about.  The chance correction subtracts the best CONSTANT
    answer, which is the right baseline only when the unresolved offset is
    large enough to collapse every query onto one corner entity.  This reports
    what each depth actually scores at the switch, so it is visible whether the
    correction is doing its job.
"""

import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from relspec.measure import resolved

DEPTHS = (1, 2, 3)
EDGES = np.arange(0.0, 3.01, 0.25)


def load(depth, arm="insert", seed=0):
    r = json.load(open(os.path.join(RESULTS, "f3", "w%02d_d%d.json" % (seed, depth))))
    n = r["arms"][arm]["net"]
    ep = np.array(n["epochs"], float) - r["t_switch"]
    m = ep >= 0
    return (ep[m], np.array(n["hits"]["cross"], bool)[m],
            np.array(n["errs"]["cross"], float)[m],
            np.array(n["geometric"]["cross"], float)[m], r)


def threshold(depth):
    """Is retrieval a step function of the error?"""
    ep, h, e, g, _ = load(depth)
    f, x = h.ravel(), e.ravel()
    print("  N=%d" % depth)
    print("    %-14s %8s %8s" % ("error band", "hit %", "n"))
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        s = (x >= a) & (x < b)
        if s.sum() >= 20:
            print("    %4.2f - %4.2f    %8.1f %8d" % (a, b, 100 * f[s].mean(), s.sum()))
    hi = x[f].max()
    lo = x[~f].min() if (~f).any() else np.inf
    k = int(np.argmax(h.mean(1) >= 1.0))
    j = np.where(g < 1.0)[0]
    print("    every miss has error >= %.2f; the best hit has error %.2f" % (lo, hi))
    print("    accuracy hits 100%% at +%d, mean error crosses one spacing at +%s"
          % (ep[k], ("%d" % ep[j[0]]) if len(j) else "never"))


def at_switch(depth):
    """What the top row reads at the moment the fact is inserted."""
    out = []
    for arm in ("insert", "hold"):
        ep, h, e, _, r = load(depth, arm)
        raw = 100.0 * h[0].mean()
        st = resolved(ep, h, chance=0.0)[0]
        out.append((arm, raw, st, e[0].mean(), e[0].max() - e[0].min()))
    print("  N=%d" % depth)
    for arm, raw, st, err, spread in out:
        print("    %-6s raw %5.1f%%  stable %5.1f%%  offset %6.3f spacings "
              "(spread across items %.3f)" % (arm, raw, st, err, spread))


def main():
    print("1. is the behavioural measure a threshold on the geometric one?")
    print("   pooled over every held-out item at every evaluation, linked arm\n")
    for d in DEPTHS:
        threshold(d)
        print()
    print("2. what is scored at the switch, before the fact can have any effect?")
    print("   chance as the best constant answer is 9/80 = 11.2%%\n")
    for d in DEPTHS:
        at_switch(d)
        print()


if __name__ == "__main__":
    main()
