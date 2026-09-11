"""Has the pre-link system settled by the time the linking fact arrives?

Figure 3 branches at `F3_SWITCH`, and the claim it makes is that everything
after that point is attributable to one added fact.  That only holds if the
pre-link system is quiet at the switch.  If it is still moving, both branches
carry the same residual dynamics and the comparison is confounded, which is
what the structured bumps in the no-link control suggest.

Three quantities are tracked on the UNLINKED world, none of which the linking
fact can touch:

  within A   both entities in the anchored block.  Its structure is fully
             determined by the facts, so this is ordinary learning.
  within B   both entities in the unanchored copy.  Its INTERNAL structure is
             also fully determined; only the copy's global position is free.
  cross      one entity from each block, which is the undetermined part.

That split answers whether both copies are learnt before the bridge arrives,
or whether the copy is still being assembled while the bridge is added.

A quantity counts as settled at the first epoch after which it changes by less
than `TOL` in relative terms over `WINDOW` consecutive evaluations, and stays
that quiet for the rest of the run.
"""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, train, worlds
from relspec.config import override
from relspec.measure import cross_plan

EPOCHS, EVERY = 40000, 200
TOL, WINDOW = 0.01, 10
TAU = 0.05      # within-block error counted as learnt, in spacings


def block_pairs(m, pre):
    """Every ordered pair inside one block, with its step offsets."""
    return [("%s_%d_%d" % (pre, i, j), "%s_%d_%d" % (pre, k, l), k - i, l - j)
            for i in range(m) for j in range(m)
            for k in range(m) for l in range(m)
            if (i, j) != (k, l)]


def ready(ep, out):
    """First epoch after which BOTH blocks are internally learnt and stay so.

    Learnt means every within-block query retrieves the right entity and the
    predicted point sits within `TAU` of it.  Those are the parts of the world
    the linking fact cannot touch, so they are what has to be quiet before the
    branch, otherwise both arms inherit the same unfinished assembly and the
    comparison is not about the fact.
    """
    ok = np.ones(len(ep), bool)
    for name in ("within A", "within B"):
        _, g, r, _ = out[name]
        ok &= (r >= 100.0) & (g < TAU)
    bad = np.where(~ok)[0]
    k = 0 if not len(bad) else bad[-1] + 1
    return float(ep[k]) if k < len(ep) else np.nan


def settled(ep, v):
    """First epoch after which the relative swing stays under TOL."""
    v = np.asarray(v, float)
    bad = -1
    for t in range(len(v) - WINDOW):
        seg = v[t:t + WINDOW + 1]
        rel = (seg.max() - seg.min()) / max(abs(seg.mean()), 1e-12)
        if rel >= TOL:
            bad = t
    return float(ep[bad + 1]) if bad + 1 < len(ep) else np.nan


def main():
    print("unlinked world, %d epochs.  settled = relative swing under %.0f%% "
          "for the rest of the run" % (EPOCHS, 100 * TOL))
    for depth in (1, 2, 3):
        S = override(eval_every=EVERY)
        w0 = worlds.integration_world(0, link=False)
        w1 = worlds.integration_world(0, link=True)
        m = w0.meta["m"]
        s = System.build(w0, settings=S)
        plans = {
            "within A": cross_plan(w0, block_pairs(m, "A"), name="within A"),
            "within B": cross_plan(w0, block_pairs(m, "B"), name="within B"),
            "cross": cross_plan(w1, worlds.held_cross(w0, reference=w1)),
        }
        out = {}
        for name, plan in plans.items():
            tr = train.train(models.make_model(w0, depth, S), s,
                             s.lr(depth, settings=S), EPOCHS, S, order_seed=7,
                             eval_every=EVERY, plan=plan)
            key = plan["name"]
            out[name] = (np.array(tr.epochs, float),
                         np.array(tr.geometric[key], float),
                         np.array(tr.retrieval[key], float),
                         np.array(tr.loss, float))
        sw = worlds.F3_SWITCH[depth]
        print()
        print("  N=%d   switch is at %d" % (depth, sw))
        for name in ("within A", "within B", "cross"):
            ep, g, r, _ = out[name]
            k = int(np.argmin(np.abs(ep - sw)))
            print("    %-8s settles %8s | at the switch: error %8.3f, "
                  "retrieval %5.1f%% | at the end: error %8.4f"
                  % (name, ("%.0f" % settled(ep, g)) if np.isfinite(settled(ep, g))
                     else "never", g[k], r[k], g[-1]), flush=True)
        rq = ready(out["within A"][0], out)
        print("    %-8s %s"
              % ("READY", ("both blocks learnt from epoch %.0f" % rq)
                 if np.isfinite(rq) else "never within the budget"), flush=True)
        ep, _, _, loss = out["cross"]
        print("    %-8s settles %8s | at the switch %.3e | at the end %.3e"
              % ("loss", ("%.0f" % settled(ep, loss))
                 if np.isfinite(settled(ep, loss)) else "never",
                 loss[int(np.argmin(np.abs(ep - sw)))], loss[-1]), flush=True)


if __name__ == "__main__":
    main()
