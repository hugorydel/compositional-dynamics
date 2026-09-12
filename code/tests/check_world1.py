"""Why does one Figure 3 world behave differently from the other four?

World 1 resolves all eighty comparisons almost immediately and then descends
far more slowly than the rest, and it is the only world whose fitted clock rate
came back at exactly 1.  Two candidate readings: it is a different structural
regime, or the fit window simply misses where its curve moves.

Three things are checked.

  structure   whether the seed changes the world at all.  The constraint matrix
              is built from the fact list and the anchor rows, neither of which
              the seed touches, so every seed should share one spectrum and the
              draw should only move the ground truth and the initialisation.

  geometry    where the pre-link solution sits relative to the post-link one,
              and how much of that separation lies along the single direction
              the linking fact adds.  Computed from the minimum-norm solutions
              of the two systems, so no training is needed.

  timing      when each world crosses fixed error levels, and the clock fit
              taken over the window where its curve actually moves rather than
              over a fixed window that may be flat.
"""

import glob
import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from relspec import System, worlds
from relspec.config import override

import fig3
from check_clock import fit

SEEDS = (0, 1, 2, 3, 4)
DEPTH = 3
LEVELS = (1.0, 0.1, 0.01)


def minnorm(s):
    return np.linalg.pinv(s.A) @ s.C


def main():
    S = override(eval_every=25)
    print("structure: does the seed change the world?\n")
    print("  %-5s %6s %7s %10s %12s %10s" % ("seed", "ents", "facts", "lam_max",
                                             "new mode", "isolation"))
    for w in SEEDS:
        s0 = System.build(worlds.integration_world(w, link=False), settings=S)
        s1 = System.build(worlds.integration_world(w, link=True), settings=S)
        e1 = np.sort(s1.evals_M)[::-1]
        n1 = int((e1 > 1e-9 * e1[0]).sum())
        print("  %-5d %6d %7d %10.4f %12.6f %9.1fx"
              % (w, s1.world.n_ent, s1.A.shape[0], e1[0], e1[n1 - 1],
                 e1[n1 - 2] / e1[n1 - 1]))

    print()
    print("geometry: the pre-link solution against the post-link one,")
    print("          and how much of the separation lies on the new direction\n")
    print("  %-5s %12s %14s %10s %12s"
          % ("seed", "separation", "on new mode", "share", "at the switch"))
    for w in SEEDS:
        s0 = System.build(worlds.integration_world(w, link=False), settings=S)
        s1 = System.build(worlds.integration_world(w, link=True), settings=S)
        ev = s1.evals_M
        order = np.argsort(ev)[::-1]
        nz = int((ev > 1e-9 * ev[order[0]]).sum())
        q = s1.Q[:, order[nz - 1]]            # the direction the link adds
        R = minnorm(s0) - minnorm(s1)
        on = q @ R
        rec = json.load(open(os.path.join(RESULTS, "f3", "w%02d_d%d.json"
                                          % (w, DEPTH))))
        e, g = fig3.after(rec, "insert", "net", "geometric")
        print("  %-5d %12.4f %14.4f %9.1f%% %12.4f"
              % (w, np.linalg.norm(R), np.linalg.norm(on),
                 100.0 * np.linalg.norm(on) / max(np.linalg.norm(R), 1e-12),
                 g[0]))

    print()
    print("timing: epochs after the fact to reach each error level, and the")
    print("        clock fit over the moving part of the curve\n")
    print("  %-5s %s %10s %10s" % ("seed",
                                   "".join("%10s" % ("<%g" % L) for L in LEVELS),
                                   "alpha 3k", "alpha all"))
    for w in SEEDS:
        rec = json.load(open(os.path.join(RESULTS, "f3", "w%02d_d%d.json"
                                          % (w, DEPTH))))
        e, g = fig3.after(rec, "insert", "net", "geometric")
        ep, q = fig3.after(rec, "insert", "pred", "geometric")
        hit = []
        for L in LEVELS:
            k = np.where(g < L)[0]
            hit.append("%10s" % (("%d" % e[k[0]]) if len(k) else "never"))
        m = e <= 3000
        a3, _, _ = fit(e[m], g[m], ep, q)
        aa, _, _ = fit(e, g, ep, q)
        print("  %-5d %s %10.3f %10.3f" % (w, "".join(hit), a3, aa))


if __name__ == "__main__":
    main()
