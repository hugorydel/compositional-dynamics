"""How far must depth 1 run for every Figure 1 law to emerge?

Figure 1 gives every depth 10,000 epochs.  At depth 1, 1,075 of the 3,200
law-worlds have no t* inside that budget, in the network and the prediction
alike.  The prediction at depth 1 is closed form, and inside the budget it
matched the network's emergence status in every law-world and its time within
one evaluation in all 2,125 that emerged, so it can say how far the budget
would have to go without training anything.

  reproduce   the closed form, rebuilt here with the runner's settings, must
              give the stored predicted t* for every law inside the budget.
              If it does not, nothing below means anything.
  coarse      for each law-world without a t*, the criterion (every held-out
              composite retrieved, geometric error under tau) on a sparse
              geometric grid of epochs out to 10^8, which brackets the epoch
              after which it holds for good.
  exact       the 25-epoch evaluation grid from the end of the budget to just
              past that bracket, with `detect_emergence` exactly as the figure
              applies it.

The world has no null space, so every law is identifiable and the closed form
converges to the ground truth: every law must emerge eventually, and the
question is only when.  Network times beyond the budget are not measured here.

Writes results/_f1_horizon.json.  Trains nothing and changes no record.

  usage:  python f1_horizon.py
          python f1_horizon.py --seeds 0 1 2
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import _boot  # noqa: F401, E402
import _paths  # noqa: F401, E402
import numpy as np  # noqa: E402
from _paths import RESULTS  # noqa: E402
from relspec import System, models, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import apply_plan, detect_emergence, law_plan  # noqa: E402

BUDGET = 10000      # the budget this was measured against, before depth 1 was extended
EVERY = worlds.F1_EVERY[1]
FAR, RATIO = 1e8, 1.05
N_WORLDS = 200


class Closed:
    """The depth-1 closed form for one world, as `run_experiments.run_f1`
    builds it."""

    def __init__(self, seed):
        self.S = S = override(init_seed=1000 + seed, eval_every=EVERY)
        w = worlds.emergence_world(seed)
        s = System.build(w, settings=S)
        self.plan = law_plan(w, worlds.held_composites(w))
        lr = s.lr(1, settings=S)
        E0 = models.make_model(w, 1, S).embedding()
        self.base = 1.0 - lr * s.evals_M
        self.coef0 = s.Q.T @ (E0 - s.Estar)
        self.Q, self.Estar = s.Q, s.Estar

    def local(self, names):
        """A plan for `names` indexed into only the rows those laws read, and
        E(t) for those rows.  Identical measurements at a fraction of the
        cost of forming the whole embedding."""
        rows = set()
        for n in names:
            p = self.plan[n]
            rows |= set(p["cand"].tolist()) | set(p["a"].tolist()) | {p["x"], p["y"], p["z"]}
        rows = np.array(sorted(rows))
        pos = {r: i for i, r in enumerate(rows)}
        loc = dict(kind="law", names=list(names))
        for n in names:
            p = dict(self.plan[n])
            p["cand"] = np.array([pos[r] for r in p["cand"]])
            p["a"] = np.array([pos[r] for r in p["a"]])
            for k in ("x", "y", "z"):
                p[k] = pos[p[k]]
            loc[n] = p
        Qr, Er = self.Q[rows], self.Estar[rows]
        return loc, lambda t: Er + Qr @ ((self.base ** t)[:, None] * self.coef0)


def series(loc, E, ts):
    ret = {n: [] for n in loc["names"]}
    geo = {n: [] for n in loc["names"]}
    for t in ts:
        r, g, _, _, _ = apply_plan(loc, E(float(t)))
        for n in loc["names"]:
            ret[n].append(r[n])
            geo[n].append(g[n])
    return ({n: np.array(v) for n, v in ret.items()},
            {n: np.array(v) for n, v in geo.items()})


def t_of(ep, ret, geo, S):
    i = detect_emergence(ret, geo, S.hold, S.tau)
    return float(ep[int(i)]) if np.isfinite(i) else np.nan


def one(seed):
    t0 = time.time()
    c = Closed(seed)
    S = c.S
    rec = json.load(open(os.path.join(RESULTS, "f1", "w%02d_d1.json" % seed)))
    ep = np.array(rec["pred"]["epochs"], float)
    names = list(c.plan["names"])

    loc, E = c.local(names)
    ret, geo = series(loc, E, ep)
    stored, net, mismatch, gdiff = {}, {}, [], 0.0
    for n in names:
        stored[n] = t_of(ep, rec["pred"]["retrieval"][n], rec["pred"]["geometric"][n], S)
        net[n] = t_of(ep, rec["net"]["retrieval"][n], rec["net"]["geometric"][n], S)
        mine = t_of(ep, ret[n], geo[n], S)
        if not (mine == stored[n] or (np.isnan(mine) and np.isnan(stored[n]))):
            mismatch.append(n)
        g_s = np.array(rec["pred"]["geometric"][n], float)
        gdiff = max(gdiff, float(np.max(np.abs(geo[n] - g_s) / np.maximum(np.abs(g_s), 1e-12))))

    late = [n for n in names if np.isnan(stored[n])]
    exact, bracket = {}, {}
    if late:
        loc, E = c.local(late)
        k = int(np.ceil(np.log(FAR / BUDGET) / np.log(RATIO)))
        grid = np.unique(np.round(BUDGET * RATIO ** np.arange(1, k + 1) / EVERY) * EVERY)
        ret, geo = series(loc, E, grid)
        for n in late:
            ok = (ret[n] >= 100.0 - 1e-9) & (geo[n] < S.tau)
            bad = np.flatnonzero(~ok)
            bracket[n] = (None if len(bad) == len(ok)
                          else float(grid[bad[-1] + 1] if len(bad) else grid[0]))
        todo = [n for n in late if bracket[n] is not None]
        end = max([bracket[n] for n in todo], default=BUDGET) + 2 * S.hold * EVERY
        start = BUDGET + EVERY
        while todo:
            loc, E = c.local(todo)
            ts = np.arange(start, end + EVERY, EVERY)
            ret, geo = series(loc, E, ts)
            again = []
            for n in todo:
                i = detect_emergence(ret[n], geo[n], S.hold, S.tau)
                if np.isfinite(i) and int(i) + S.hold <= len(ts):
                    exact[n] = float(ts[int(i)])
                else:
                    again.append(n)          # not confirmed inside the scan
            todo, end = again, end * 1.5
        for n in late:
            exact.setdefault(n, None)
    return dict(seed=seed, mismatch=mismatch, geo_rel_diff=gdiff, stored=stored,
                net=net, late=exact, bracket=bracket, seconds=time.time() - t0)


def fmt(x):
    return "{:,}".format(int(round(x)))


def main():
    args = sys.argv[1:]
    seeds = ([int(a) for a in args[args.index("--seeds") + 1:]]
             if "--seeds" in args else list(range(N_WORLDS)))
    rows = []
    t0 = time.time()
    with Pool(min(12, len(seeds))) as pool:
        for r in pool.imap_unordered(one, seeds):
            rows.append(r)
            lt = [v for v in r["late"].values() if v is not None]
            print("  w%03d  %2d laws past the budget, slowest predicted t* %s  (%.0f s)"
                  % (r["seed"], len(r["late"]), fmt(max(lt)) if lt else "-", r["seconds"]),
                  flush=True)
    rows.sort(key=lambda r: r["seed"])
    names = sorted(rows[0]["stored"])

    print("\nreproduce: stored predicted t* matched in %d of %d law-worlds; largest "
          "relative difference in geometric error %.1e"
          % (sum(len(names) - len(r["mismatch"]) for r in rows), len(rows) * len(names),
             max(r["geo_rel_diff"] for r in rows)))

    inside_net = [v for r in rows for v in r["net"].values() if np.isfinite(v)]
    inside_pred = [v for r in rows for v in r["stored"].values() if np.isfinite(v)]
    print("inside the budget: slowest network t* %s, slowest predicted t* %s"
          % (fmt(max(inside_net)), fmt(max(inside_pred))))

    late = [(r["seed"], n, r["late"][n]) for r in rows for n in r["late"]]
    never = [(s, n) for s, n, t in late if t is None]
    found = [(s, n, t) for s, n, t in late if t is not None]
    print("past the budget: %d law-worlds; exact predicted t* found for %d; not "
          "emerged by %s epochs: %d %s"
          % (len(late), len(found), fmt(FAR), len(never), never[:10]))

    allt = np.array([v for r in rows for v in r["stored"].values() if np.isfinite(v)]
                    + [t for _, _, t in found])
    total = len(rows) * len(names)
    print("\nall %d law-worlds, predicted t*: every one emerged by %s epochs"
          % (total, fmt(allt.max()) if len(allt) == total else "n/a (some never)"))
    for q in (50, 90, 99, 99.9):
        print("  %5.1f%% have emerged by %s epochs" % (q, fmt(np.percentile(allt, q))))
    s, n, t = max(found, key=lambda x: x[2]) if found else (None, None, np.nan)
    if found:
        print("  slowest: world %d, law %s, t* = %s epochs" % (s, n, fmt(t)))

    print("\nper law, predicted t* over the 200 worlds (past the budget where needed)")
    print("  %-6s %10s %10s %10s %14s" % ("law", "median", "max", "past 10k", "never by 1e8"))
    for law in names:
        ts = []
        for r in rows:
            v = r["stored"][law]
            ts.append(v if np.isfinite(v) else (r["late"].get(law) or np.inf))
        ts = np.array(ts)
        fin = ts[np.isfinite(ts)]
        print("  %-6s %10s %10s %10d %14d"
              % (law, fmt(np.median(ts)) if np.isfinite(np.median(ts)) else "inf",
                 fmt(fin.max()) if len(fin) else "-",
                 sum(1 for r in rows if np.isnan(r["stored"][law])),
                 int((~np.isfinite(ts)).sum())))

    out = os.path.join(RESULTS, "_f1_horizon.json")
    with open(out, "w") as f:
        json.dump(dict(budget=BUDGET, every=EVERY, far=FAR, rows=rows), f)
    print("\nwrote %s  (%.0f s)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
