"""Do the two assumptions behind Figure 2's at-risk restriction actually hold?

Figure 2 scores only the held-out composites that were still WRONG when the
bridging fact arrived.  Two claims make that defensible, and neither is obvious
from the code that relies on them, so both are checked here against every
stored cell rather than argued.

  shared switch state   Both branches leave the switch holding identical
                        weights, so they must score identical hits there.  If
                        they did not, the excluded set would differ between the
                        arms and the restriction would bias the comparison it
                        is meant to clean up.

  distinct targets      The figure corrects against the best constant answer,
                        taken as one over the number of items scored.  That is
                        exact only while every held-out target is distinct, in
                        which case no constant answer can score more than one.
                        The records store the true rate computed at run time,
                        so the two can be compared directly.

Also reported, for the caption: how much the restriction removes, and which
arms it empties past the point of being scorable.
"""

import glob
import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

MIN_ITEMS = 2      # the chance correction is undefined over a single query


def post_switch_hits(rec, arm, law):
    d = rec["arms"][arm]["net"]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    return np.array(d["hits"][law], bool)[ep >= 0]


def main():
    print("Figure 2 at-risk restriction, checked against every stored cell")
    shared = differ = exact = wrong = 0
    for depth in (1, 2, 3):
        files = sorted(glob.glob(os.path.join(RESULTS, "f2",
                                              "w*_d%d_[AB].json" % depth)))
        if not files:
            continue
        dropped, items, kept_items, arms = [], 0, 0, 0
        for f in files:
            rec = json.load(open(f))
            law = rec["open"]
            h_hold = post_switch_hits(rec, "hold", law)[0]
            h_ins = post_switch_hits(rec, "insert", law)[0]
            shared += 1
            differ += not np.array_equal(h_hold, h_ins)

            n = h_hold.shape[0]
            exact += abs(rec["chance"][law] - 1.0 / n) <= 1e-12
            wrong += abs(rec["chance"][law] - 1.0 / n) > 1e-12

            keep = int((~h_hold).sum())
            arms += 1
            items += n
            kept_items += keep
            if keep < MIN_ITEMS:
                dropped.append((os.path.basename(f), n - keep, n))
        print()
        print("  N=%d  %d arms, %d items" % (depth, arms, items))
        print("     excluded as already correct: %d items (%.2f%%)"
              % (items - kept_items, 100.0 * (items - kept_items) / items))
        print("     arms left under %d scorable items: %d%s"
              % (MIN_ITEMS, len(dropped),
                 "".join("\n        %-16s %d of %d already correct"
                         % (a, c, t) for a, c, t in dropped)), flush=True)
    print()
    print("  arms whose branches disagree at the switch: %d of %d  %s"
          % (differ, shared, "OK" if not differ else "*** BIAS RISK ***"))
    print("  records whose stored chance is not 1/n_items: %d of %d  %s"
          % (wrong, exact + wrong, "OK" if not wrong else "*** CHANCE WRONG ***"))


if __name__ == "__main__":
    main()
