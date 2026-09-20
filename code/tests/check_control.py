"""Why does the no-linking-fact control leave zero in some worlds?

The control is the claim of Figure 3: without the fact, no cross-structure
comparison becomes available.  At twenty worlds its band climbs to a quarter of
the eighty comparisons, so some worlds score without the fact.

The behavioural curve is chance-corrected against what the arm scores AT THE
SWITCH, on the argument that nothing there has constrained the alignment, so
anything retrieved is retrieved without the information being asked about.
That correction is a single number fixed at one instant.  It cannot catch an
arm whose score RISES afterwards, and the control has a reason to rise: the
free offset is still settling at the switch, and it settles onto the
minimum-norm point.  Whatever that final offset happens to get right, it gets
right permanently, and stable resolution -- counting an item from the first
evaluation after its last failure -- credits exactly that.

So the reading to separate is: has the control learnt anything, or has it come
to rest somewhere that scores better than where it started?  The two are
distinguishable, because learning would move the geometric error and coming to
rest would not.
"""

import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

import fig3

SUB = "f3"
WORLDS = range(20)
DEPTHS = (1, 2, 3)


def load(w, d, arm="hold"):
    p = os.path.join(RESULTS, SUB, "w%02d_d%d.json" % (w, d))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    e, res = fig3.after(r, arm, "net", "resolved")
    _, geo = fig3.after(r, arm, "net", "geometric")
    n = r["arms"][arm]["net"]
    ep = (np.array(n["epochs"], float) - r["t_switch"])
    m = ep >= 0
    hits = np.array(n["hits"]["cross"], bool)[m]
    return e, res, geo, hits


def main():
    print("the control arm, %s\n" % SUB)
    print("  %-4s %-4s %9s %9s %9s %11s %11s"
          % ("N", "world", "raw @0", "raw end", "resolved", "geo @0", "geo end"))
    risers = {d: [] for d in DEPTHS}
    for d in DEPTHS:
        for w in WORLDS:
            got = load(w, d)
            if got is None:
                continue
            e, res, geo, hits = got
            raw0, raw1 = 100 * hits[0].mean(), 100 * hits[-1].mean()
            if res[-1] > 0.5:
                risers[d].append((w, res[-1]))
                print("  %-4d %-4d %8.1f%% %8.1f%% %8.1f%% %11.4f %11.4f"
                      % (d, w, raw0, raw1, res[-1], geo[0], geo[-1]))
    print()
    for d in DEPTHS:
        n = len(risers[d])
        worst = max([v for _, v in risers[d]], default=0.0)
        print("  N=%d: %2d of %d worlds rise above zero, worst %.1f%%"
              % (d, n, len(WORLDS), worst))

    print()
    print("  is the control still moving while it scores?")
    print("  %-4s %-4s %11s %11s %11s"
          % ("N", "world", "geo @1k", "geo @3k", "change"))
    for d in DEPTHS:
        for w, _ in risers[d][:4]:
            e, res, geo, hits = load(w, d)
            a = geo[int(np.argmin(abs(e - 1000)))]
            b = geo[int(np.argmin(abs(e - 3000)))]
            print("  %-4d %-4d %11.4f %11.4f %10.1f%%"
                  % (d, w, a, b, 100 * (b / a - 1)))

    print()
    print("  where the score comes from: items correct at the END, and whether")
    print("  they were already correct at the switch")
    for d in DEPTHS:
        for w, _ in risers[d][:4]:
            e, res, geo, hits = load(w, d)
            end, start = hits[-1], hits[0]
            print("  N=%d w%02d  correct at end %2d of %d | of those, %2d were "
                  "correct at the switch, %2d are new"
                  % (d, w, end.sum(), len(end), (end & start).sum(),
                     (end & ~start).sum()))


if __name__ == "__main__":
    main()
