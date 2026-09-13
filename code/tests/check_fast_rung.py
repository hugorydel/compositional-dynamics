"""Is there a rung that would emerge well before Law A?

Figure 1's Law A (rung 3) is already the fastest of the sixteen laws to
within three per cent, so a law that emerges significantly earlier has to be a
new rung added to the world.  Two couplings decide whether that is cheap.

  learning rate   `lr = lr_target / sigma_max` over the whole world, so a new
                  rung whose largest eigenvalue exceeds the current maximum
                  slows EVERY law by the ratio.
  row budget      training cost is rows times epochs; the ladder was searched
                  under a 2,000-row cap and a repetition cap of 8.

Rungs share no token, so each rung's spectrum is independent and its
trajectory in an assembly is its solo trajectory rescaled in time by
sigma_max(assembly) / sigma_max(rung).  This integrates each candidate shape
alone at depth 1, where the theory is a closed form, and scores it against
Law A under that rescaling -- the coarse stage of `search_ladder.py`, which
notes that the initialisation slice moves t* a little in a real assembly, so a
shortlist here still needs a full integration before any data is collected.
"""

import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.measure import law_plan
from relspec.worlds import lattice_world

S = override(eval_every=1000)
Z_FRAC, MIN_HELD, SEED = worlds.Z_FRAC, 6, 0
ALONE_EPOCHS, ALONE_EVERY = 600000, 1000
REP_CAP, ROWS_MAX = 8, 2000
REPS = (1, 2, 3, 4, 6, 8, 12, 16)
FASTER = 0.67            # "significantly before": at most two thirds of Law A's time
OBSERVED = {3: 4025, 2: 5575, 6: 4200, 7: 4863}   # depth-1 medians, 200 worlds


def specs(c):
    return [dict(name="L0_%d" % k, x="b0_%d" % k, y="y0_%d" % k, z="z0_%d" % k,
                 prefix="B0L%d" % k, m=c["m"], n=c["n"], rep_x=c["rep_x"],
                 rep_y=c["rep_y"], z_rep=c["rep_y"], z_frac=Z_FRAC)
            for k in range(2)]


def solo(c):
    w = lattice_world(specs(c), d=S.d, seed=SEED)
    held = worlds.held_composites(w)
    n_held = min(len(v) for v in held.values())
    if n_held < MIN_HELD:
        return None
    s = System.build(w, settings=S)
    tr, _ = theory.predict(s, 1, s.lr(1, settings=S), ALONE_EPOCHS,
                           models.make_model(w, 1, S), settings=S,
                           eval_every=ALONE_EVERY, plan=law_plan(w, held))
    ts = tr.emergence(S)
    t = [ts[l.name] for l in w.laws]
    if not all(np.isfinite(t)):
        return None
    return dict(sigma=float(s.evals_M[0]), t=float(np.mean(t)),
                rows=s.A.shape[0], n_held=int(n_held))


def main():
    t0 = time.time()
    cur = [solo(c) for c in worlds.LADDER]
    smax = max(p["sigma"] for p in cur)
    A = cur[3]
    rows_now = System.build(worlds.emergence_world(0), settings=S).A.shape[0]
    tA = A["t"] * smax / A["sigma"]
    print("coarse model against the observed depth-1 medians")
    for bi, obs in sorted(OBSERVED.items()):
        p = cur[bi]
        print("  rung %d  predicted %6.0f   observed %6d" % (bi, p["t"] * smax / p["sigma"], obs))
    print("\nsigma_max %.1f | Law A at %.0f epochs | world has %d of %d rows\n"
          % (smax, tA, rows_now, ROWS_MAX))

    found = []
    for m in (4, 5):
        for n in (4, 5):
            for rx in REPS:
                for ry in REPS:
                    c = dict(m=m, n=n, rep_x=rx, rep_y=ry)
                    p = solo(c)
                    if p is None:
                        continue
                    s_new = max(smax, p["sigma"])
                    ratio = (p["t"] * s_new / p["sigma"]) / (A["t"] * s_new / A["sigma"])
                    if ratio <= FASTER:
                        found.append((rows_now + p["rows"], ratio, c, p["sigma"],
                                      s_new / smax, p["t"] * s_new / p["sigma"]))
    print("rungs emerging at most %.2fx Law A's time, fewest rows first" % FASTER)
    print("  %-26s %8s %10s %8s %8s %7s  %s" % ("shape", "sigma", "slows all", "t", "vs A", "rows", "within caps"))
    for rows, ratio, c, sg, slow, t in sorted(found, key=lambda x: (x[0], x[1]))[:20]:
        ok = (max(c["rep_x"], c["rep_y"]) <= REP_CAP) and rows <= ROWS_MAX
        print("  m=%d n=%d rx=%-2d ry=%-2d      %8.1f %9.2fx %8.0f %7.2fx %7d  %s"
              % (c["m"], c["n"], c["rep_x"], c["rep_y"], sg, slow, t, ratio, rows,
                 "yes" if ok else "no (%s)" % ", ".join(
                     x for x, bad in (("reps", max(c["rep_x"], c["rep_y"]) > REP_CAP),
                                      ("rows", rows > ROWS_MAX)) if bad)))
    print("\n%d shapes qualify, %d within both caps | %.0f s"
          % (len(found), sum(1 for r, _, c, *_ in found
                             if max(c["rep_x"], c["rep_y"]) <= REP_CAP and r <= ROWS_MAX),
             time.time() - t0))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
#  Refine: the coarse rescaling puts rungs 3 and 6 in the wrong order against   #
#  the observed times, so a shortlisted rung is integrated inside the whole     #
#  assembled world before it is trusted.                                        #
# --------------------------------------------------------------------------- #

REFINE = (dict(m=4, n=4, rep_x=8, rep_y=6),
          dict(m=4, n=4, rep_x=8, rep_y=8),
          dict(m=4, n=4, rep_x=12, rep_y=12))
REFINE_SEEDS = range(5)
REFINE_EPOCHS, REFINE_EVERY = 30000, 25


def assembled_times(ladder, seed):
    Sr = override(eval_every=REFINE_EVERY)
    w = worlds.emergence_world(seed, ladder=tuple(ladder))
    s = System.build(w, settings=Sr)
    tr, _ = theory.predict(s, 1, s.lr(1, settings=Sr), REFINE_EPOCHS,
                           models.make_model(w, 1, Sr), settings=Sr,
                           eval_every=REFINE_EVERY,
                           plan=law_plan(w, worlds.held_composites(w)))
    return tr.emergence(Sr), float(s.evals_M[0]), s.A.shape[0]


def refine():
    print("full assembled world, depth 1, median over seeds %s" % list(REFINE_SEEDS))
    base = [assembled_times(worlds.LADDER, sd) for sd in REFINE_SEEDS]
    med = lambda name, runs: float(np.median([r[0][name] for r in runs]))
    print("  %-30s sigma %6.1f rows %4d | A %6.0f  B %6.0f  C %6.0f"
          % ("current 16 laws", base[0][1], base[0][2],
             med("L3_1", base), med("L2_1", base), med("L5_0", base)))
    for c in REFINE:
        runs = [assembled_times(worlds.LADDER + (c,), sd) for sd in REFINE_SEEDS]
        new = float(np.median([np.mean([r[0]["L8_0"], r[0]["L8_1"]]) for r in runs]))
        a = med("L3_1", runs)
        print("  + m=%d n=%d rx=%-2d ry=%-2d          sigma %6.1f rows %4d | A %6.0f  B %6.0f  C %6.0f"
              " | new %6.0f = %.2fx A"
              % (c["m"], c["n"], c["rep_x"], c["rep_y"], runs[0][1], runs[0][2],
                 a, med("L2_1", runs), med("L5_0", runs), new, new / a))


if __name__ == "__main__" and "refine" in __import__("sys").argv:
    refine()
