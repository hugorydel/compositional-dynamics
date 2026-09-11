"""Settle the epoch budgets for Figures 2 and 3, on their staged designs.

Both figures insert one fact mid-training, so a budget has two parts: how long
the pre-switch phase must run for the control to be established, and how long
after the switch the newly determined quantity takes to be recovered.

The two measures are reported SEPARATELY, not through the conjoined emergence
criterion.  That criterion requires retrieval at ceiling and the geometric
error below a threshold simultaneously, which is right for a composition law
where both are on the same scale.  It is not calibrated for Figure 3, whose
geometric measure is an offset error in units of one relation step and whose
rank test is invariant to a global scale the offset is not.  Reporting them
together there hides a real dissociation: the comparisons become behaviourally
available long before the geometry is metrically right.
"""

import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.measure import cross_plan, law_plan

DEPTHS = (1, 2, 3)
GEO_LEVELS = (1.0, 0.5, 0.1)
# discovery budgets: pre-switch, post-switch, eval spacing
BUD = {1: (40000, 400000, 500), 2: (4000, 40000, 100), 3: (4000, 40000, 100)}


def sustained(epochs, series, ok):
    """First epoch after which `ok` holds for the rest of the run."""
    good = np.asarray([bool(ok(v)) for v in series])
    bad = np.where(~good)[0]
    k = 0 if not len(bad) else bad[-1] + 1
    return float(epochs[k]) if k < len(epochs) else np.nan


def staged(w0, w1, mk_plan, depth, t1, t2, every):
    S = override(eval_every=every)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = mk_plan(w1)
    lr = s0.lr(depth, settings=S)
    a, st = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                           settings=S, eval_every=every, plan=plan)
    b, _ = theory.predict(s1, depth, lr, t2, st, settings=S,
                          eval_every=every, plan=plan)
    if a is None or b is None:
        return None
    ep = np.concatenate([a.epochs, t1 + b.epochs[1:]])
    out = {}
    for k in a.retrieval:
        out[k] = dict(
            ep=ep,
            ret=np.concatenate([a.retrieval[k], b.retrieval[k][1:]]),
            geo=np.concatenate([a.geometric[k], b.geometric[k][1:]]),
        )
    return out, t1


def report_f2():
    print("F2 identifiability, staged: bridge inserted at the switch")
    for d in DEPTHS:
        t1, t2, every = BUD[d]
        w0 = worlds.identifiability_world(0, K=16, k=4, n_bridge=0)
        w1 = worlds.identifiability_world(0, K=16, k=4, n_bridge=1)
        r = staged(w0, w1, lambda ww: law_plan(
            ww, worlds.held_composites(ww, reference=w1)), d, t1, t2, every)
        if r is None:
            print("  N=%d DIVERGED" % d)
            continue
        out, sw = r
        for name in sorted(out):
            o = out[name]
            rt = sustained(o["ep"], o["ret"], lambda v: v >= 100.0)
            gt = sustained(o["ep"], o["geo"], lambda v: v < 0.5)
            role = "open  " if np.isfinite(rt) and rt > sw else "closed"
            print("  N=%d law %s (%s): retrieval 100%% at %8s | geometric<0.5 at %8s"
                  % (d, name, role,
                     ("%.0f" % rt) if np.isfinite(rt) else "never",
                     ("%.0f" % gt) if np.isfinite(gt) else "never"), flush=True)


def report_f3():
    print()
    print("F3 integration, staged: linking fact inserted at the switch")
    print("  behavioural = all 80 cross comparisons ranked first and staying")
    print("  geometric   = offset error, in units of one x step, below a level")
    for d in DEPTHS:
        t1, t2, every = BUD[d]
        w0 = worlds.integration_world(0, link=False)
        w1 = worlds.integration_world(0, link=True)
        t0 = time.time()
        r = staged(w0, w1, lambda ww: cross_plan(
            ww, worlds.held_cross(ww, reference=w1)), d, t1, t2, every)
        if r is None:
            print("  N=%d DIVERGED" % d)
            continue
        out, sw = r
        o = out["cross"]
        rt = sustained(o["ep"], o["ret"], lambda v: v >= 100.0)
        line = "  N=%d switch %6d | behavioural %8s" % (
            d, sw, ("%.0f" % rt) if np.isfinite(rt) else "never")
        for lv in GEO_LEVELS:
            gt = sustained(o["ep"], o["geo"], lambda v, L=lv: v < L)
            line += " | geo<%.1f %9s" % (
                lv, ("%.0f" % gt) if np.isfinite(gt) else "never")
        print(line + "  (%.0fs, final geo %.2e)" % (time.time() - t0,
                                                    o["geo"][-1]), flush=True)


if __name__ == "__main__":
    report_f2()
    report_f3()
