"""Search lattice shapes for the Figure 1 ladder.

Figure 1 needs laws that emerge at clearly separated times, under the regime
the figure actually uses: `z_frac` of each law's diagonals trained on, the rest
withheld and scored by held-out composite retrieval.

Why the search integrates rather than scoring a spectral summary.  Scalar
summaries of a law's spectral support do not order emergence well enough to
optimise against: measured on the provisional ladder, the support-weighted
eigenvalue reaches Spearman 0.65 against t*, the slowest weighted mode 0.50,
and the weighted harmonic mean is negatively correlated.  The theory's
predictor is the full trajectory, not a formula over summary statistics, so the
search uses the depth-1 closed form directly.

Why it can afford to.  A ladder's lattices share no token across rungs, so
`A` is block diagonal and each rung's spectrum is independent.  The learning
rate is `lr_target / evals_M[0]` with `evals_M` the raw eigenvalues of `A^T A`,
so every rate is `lr_target * lambda_k / sigma_max`, and the fact-count
normalisation cancels.  A rung evaluated alone therefore has the same
trajectory it will have in an assembly, rescaled in time by exactly

    t_assembly = t_alone * sigma_max(assembly) / sigma_max(rung alone)

with `sigma_max(assembly)` the largest raw eigenvalue over the chosen rungs.
So each candidate shape is integrated once, and any assembly of eight is then
scored arithmetically.

Objective: maximise the smallest gap, in log space, between consecutive RUNG
emergence times, subject to every law emerging inside the epoch budget and
withholding at least `MIN_HELD` composites.  Maximin spacing favours a wide
range and an even one at the same time.  It is scored per rung because a
rung's two laws are structurally identical and differ only by the random draw,
so they are one design point and would otherwise tie at gap zero and flatten
the objective.
"""
import itertools
import json
import time

import numpy as np

import _paths                                                 # noqa: F401
from _paths import result                                     # noqa: E402
from relspec import System, models, theory, worlds            # noqa: E402
from relspec.config import override                           # noqa: E402
from relspec.measure import law_plan                          # noqa: E402
from relspec.worlds import lattice_world                      # noqa: E402

S = override(lr_target=0.3, eval_every=100)
Z_FRAC = 0.25
MIN_HELD = 6
N_RUNGS = 8
ALONE_EPOCHS, ALONE_EVERY = 200000, 200
T_MAX = 60000                       # assembled t* must fit this budget
SEED = 0

SHAPES = [dict(m=m, n=n, rep_x=rx, rep_y=ry)
          for m in (4, 5) for n in (4, 5)
          for rx in (1, 2, 3, 4, 6, 8, 12, 16, 24)
          for ry in (1, 2, 3, 4, 6, 8, 12, 16, 24)]


def rung_specs(c, bi=0, z_frac=Z_FRAC):
    return [dict(name="L%d_%d" % (bi, k), x="b%d" % bi,
                 y="y%d_%d" % (bi, k), z="z%d_%d" % (bi, k),
                 prefix="B%dL%d" % (bi, k), m=c["m"], n=c["n"],
                 rep_x=c["rep_x"], rep_y=c["rep_y"], z_rep=c["rep_y"],
                 z_frac=z_frac) for k in range(2)]


def evaluate(c):
    """Integrate one candidate shape alone.  Returns its raw sigma_max, its two
    laws' emergence times at that lr, and the held-out count, or None if the
    shape is unusable."""
    w = lattice_world(rung_specs(c), d=S.d, seed=SEED)
    held = worlds.held_composites(w)
    n_held = min(len(v) for v in held.values())
    if n_held < MIN_HELD:
        return None
    s = System.build(w, settings=S)
    plan = law_plan(w, held)
    tr, _ = theory.predict(s, 1, s.lr(1, settings=S), ALONE_EPOCHS,
                           models.make_model(w, 1, S), settings=S,
                           eval_every=ALONE_EVERY, plan=plan)
    ts = tr.emergence(S)
    t = [ts[l.name] for l in w.laws]
    if not all(np.isfinite(t)):
        return None
    return dict(shape=c, sigma=float(s.evals_M[0]), t=[float(x) for x in t],
                n_held=int(n_held))


def times(chosen):
    """The 16 assembled emergence times implied by a set of rungs."""
    smax = max(r["sigma"] for r in chosen)
    return np.array([t * smax / r["sigma"] for r in chosen for t in r["t"]])


def rung_times(chosen):
    """One time per rung.  The two laws in a rung are structurally identical --
    same shape, same repetitions, differing only in the random draw of their
    relation vectors and of which diagonals are shown -- so they are one design
    point, not two, and the objective must not be scored as if they were."""
    smax = max(r["sigma"] for r in chosen)
    return np.array([np.mean(r["t"]) * smax / r["sigma"] for r in chosen])


def objective(chosen):
    t = times(chosen)
    if t.max() > T_MAX:
        return -np.inf
    g = np.diff(np.sort(np.log10(rung_times(chosen))))
    return float(g.min())


def search(pool, iters=40000, rng=None):
    rng = rng or np.random.default_rng(0)
    order = sorted(range(len(pool)), key=lambda i: pool[i]["t"][0] / pool[i]["sigma"])
    pick = [order[int(round(k))] for k in
            np.linspace(0, len(order) - 1, N_RUNGS)]
    pick = list(dict.fromkeys(pick))
    while len(pick) < N_RUNGS:
        c = int(rng.integers(len(pool)))
        if c not in pick:
            pick.append(c)
    best, bs = list(pick), objective([pool[i] for i in pick])
    for _ in range(iters):
        cand = list(best)
        cand[int(rng.integers(N_RUNGS))] = int(rng.integers(len(pool)))
        if len(set(cand)) < N_RUNGS:
            continue
        v = objective([pool[i] for i in cand])
        if v > bs:
            best, bs = cand, v
    return [pool[i] for i in best], bs


def main():
    t0 = time.time()
    pool = []
    for k, c in enumerate(SHAPES):
        r = evaluate(c)
        if r is not None:
            pool.append(r)
        if (k + 1) % 60 == 0:
            print("  evaluated %d/%d, %d usable, %.0fs"
                  % (k + 1, len(SHAPES), len(pool), time.time() - t0), flush=True)
    print("pool: %d usable shapes of %d, %.0fs" % (len(pool), len(SHAPES),
                                                   time.time() - t0))

    chosen, sc = search(pool)
    t = np.sort(times(chosen))
    print()
    rt = np.sort(rung_times(chosen))
    print("chosen ladder: min log10 rung gap %.3f, t* %.0f to %.0f, spread %.1fx"
          % (sc, t.min(), t.max(), t.max() / t.min()))
    print("   rung times:   %s" % " ".join("%.0f" % x for x in rt))
    for r in sorted(chosen, key=lambda r: r["t"][0] / r["sigma"]):
        c = r["shape"]
        print("   m=%d n=%d rep_x=%-3d rep_y=%-3d  held %2d"
              % (c["m"], c["n"], c["rep_x"], c["rep_y"], r["n_held"]))
    print("   assembled t*: %s" % " ".join("%.0f" % x for x in t))

    rec = dict(z_frac=Z_FRAC, min_held=MIN_HELD, t_max=T_MAX, seed=SEED,
               score=sc, ladder=[r["shape"] for r in chosen],
               predicted_t=[float(x) for x in t])
    with open(result("ladder_search.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print()
    print("LADDER = (")
    for r in sorted(chosen, key=lambda r: r["t"][0] / r["sigma"]):
        c = r["shape"]
        print("    dict(m=%d, n=%d, rep_x=%d, rep_y=%d),"
              % (c["m"], c["n"], c["rep_x"], c["rep_y"]))
    print(")")


if __name__ == "__main__":
    main()
