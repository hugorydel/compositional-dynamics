"""Why do Figure 2 arms come out already correct before the bridge so much more
often than Figure 3 worlds do before the link?

In both figures every held-out query for the probed quantity is displaced from
its target by ONE shared vector.  A Figure 2 query is `E_a + z`, which is
`E_b + (z - (x + y))`; a Figure 3 query is `E_b + t`, with `t` the offset
between the two copies.  Because the displacement is shared, the items pass or
fail together, which is why the Figure 2 distribution is bimodal rather than
graded.  So the question reduces to how large that shared displacement is,
relative to the spacing of the candidates, when the intervention arrives.

This scores both plans against the minimum-norm solution of the
pre-intervention system, `pinv(A) C`, which is where a depth-1 network settles
from a small initialisation.  No training is involved.  If the seeds that come
out already correct under minimum-norm are the seeds that came out degenerate
in training, the explanation is the geometry of the minimum-norm point, and
the network is simply finding it.
"""

import glob
import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from relspec import System, worlds
from relspec.config import DEFAULT
from relspec.measure import _spacing, apply_plan, cross_plan, law_plan

import fig2
import fig3

SEEDS = range(200)


def minnorm(w):
    s = System.build(w, settings=DEFAULT)
    return np.linalg.pinv(s.A) @ s.C


def shift(E, plan, law=None):
    """Per-item distance from query to target, in candidate spacings."""
    if law is not None:
        p = plan[law]
        Cand = E[p["cand"]]
        q = E[p["a"]] + E[p["z"]][None, :]
        return np.linalg.norm(q - Cand[p["b"]], axis=1) / _spacing(Cand)
    Cand = E[plan["cand"]]
    q = (E[plan["a"]] + plan["nx"][:, None] * E[plan["x"]][None, :]
         + plan["ny"][:, None] * E[plan["y"]][None, :])
    return np.linalg.norm(q - Cand[plan["bpos"]], axis=1) / _spacing(Cand)


def figure2():
    pred, dist_deg, dist_ok, spread = set(), [], [], []
    for seed in SEEDS:
        for closed in ("A", "B"):
            open_ = "B" if closed == "A" else "A"
            w0 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed)
            w1 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed,
                                              n_bridge=1)
            plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
            E = minnorm(w0)
            _, _, hits, _, _ = apply_plan(plan, E)
            wrong = int((~hits[open_]).sum())
            dd = shift(E, plan, open_)
            spread.append(dd.max() - dd.min())
            if wrong < 2:
                pred.add((seed, closed))
                dist_deg.append(dd.mean())
            else:
                dist_ok.append(dd.mean())
    obs = set()
    for f in glob.glob(os.path.join(RESULTS, "f2", "w*_d1_[AB].json")):
        r = json.load(open(f))
        if int(fig2.at_risk(r, r["open"]).sum()) < 2 and r["seed"] in SEEDS:
            obs.add((r["seed"], r["closed"]))
    print("Figure 2, open law, minimum-norm point before the bridge")
    print("  already correct under minimum-norm: %d of %d arms" % (len(pred), 2 * len(SEEDS)))
    print("  degenerate in depth-1 training:     %d" % len(obs))
    print("  agree %d | minimum-norm only %d | training only %d"
          % (len(pred & obs), len(pred - obs), len(obs - pred)))
    print("  shared shift, in candidate spacings: already correct %.2f, rest %.2f"
          % (np.mean(dist_deg), np.mean(dist_ok)))
    print("  spread of the shift across items within an arm: median %.3f spacings"
          % np.median(spread))


def figure3():
    pred, d_all = [], []
    for seed in SEEDS:
        w0 = worlds.integration_world(seed, link=False)
        w1 = worlds.integration_world(seed, link=True)
        plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
        E = minnorm(w0)
        _, _, hits, _, _ = apply_plan(plan, E)
        d_all.append(shift(E, plan).mean())
        if int((~hits["cross"]).sum()) < 2:
            pred.append(seed)
    d_all = np.array(d_all)
    print("Figure 3, cross comparisons, minimum-norm point before the link")
    print("  already correct under minimum-norm: %d of %d worlds %s"
          % (len(pred), len(SEEDS), pred))
    print("  shared shift, in candidate spacings: median %.2f, "
          "smallest five %s" % (np.median(d_all),
                                np.array2string(np.sort(d_all)[:5], precision=2)))


if __name__ == "__main__":
    figure2()
    print()
    figure3()
