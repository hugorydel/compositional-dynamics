"""Does a shortlisted fast rung stay first at depths 2 and 3?

Figure 1 draws the same laws in all three columns and requires the same order
in each, so a rung that emerges before Law A at depth 1 is only usable if it
still does at depths 2 and 3.  `check_fast_rung.py` settles depth 1 in the
closed form; deeper networks need the mean dynamics integrated, so this runs
the full assembled world through the theory at depths 2 and 3.
"""

import sys
from multiprocessing import Pool

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.measure import law_plan

CANDIDATES = {"current": None,
              "8x6": dict(m=4, n=4, rep_x=8, rep_y=6),
              "8x8": dict(m=4, n=4, rep_x=8, rep_y=8),
              "12x12": dict(m=4, n=4, rep_x=12, rep_y=12)}
DEPTHS, SEEDS = (2, 3), (0, 1)
EPOCHS, EVERY = 6000, 25
DRAWN = {"A": "L3_1", "B": "L2_1", "C": "L5_0", "new": ("L8_0", "L8_1")}


def one(cell):
    tag, depth, seed = cell
    c = CANDIDATES[tag]
    ladder = worlds.LADDER + ((c,) if c else ())
    S = override(init_seed=1000 + seed, eval_every=EVERY)
    w = worlds.emergence_world(seed, ladder=ladder)
    s = System.build(w, settings=S)
    tr, _ = theory.predict(s, depth, s.lr(depth, settings=S), EPOCHS,
                           models.make_model(w, depth, S), settings=S,
                           eval_every=EVERY,
                           plan=law_plan(w, worlds.held_composites(w)))
    if tr is None:
        return tag, depth, seed, None
    ts = tr.emergence(S)
    out = {k: ts[v] for k, v in DRAWN.items() if k != "new"}
    if c:
        out["new"] = float(np.mean([ts[n] for n in DRAWN["new"]]))
    return tag, depth, seed, out


def main():
    cells = [(t, d, s) for t in CANDIDATES for d in DEPTHS for s in SEEDS]
    res = {}
    with Pool(min(12, len(cells))) as pool:
        for tag, depth, seed, out in pool.imap_unordered(one, cells):
            res[tag, depth, seed] = out
            print("  done %-8s N=%d seed %d" % (tag, depth, seed), flush=True)
    print()
    for depth in DEPTHS:
        print("depth %d, theory, median over seeds %s" % (depth, list(SEEDS)))
        for tag in CANDIDATES:
            runs = [res[tag, depth, s] for s in SEEDS if res[tag, depth, s]]
            if not runs:
                print("  %-8s diverged" % tag)
                continue
            m = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
            line = "  %-8s A %6.0f  B %6.0f  C %6.0f" % (tag, m["A"], m["B"], m["C"])
            if "new" in m:
                line += " | new %6.0f = %.2fx A | order new<A<B<C %s" % (
                    m["new"], m["new"] / m["A"],
                    "yes" if m["new"] < m["A"] < m["B"] < m["C"] else "NO")
            print(line)
        print()


if __name__ == "__main__":
    main()
