"""Has the unbridged system settled by the time the bridging fact arrives?

Figure 2 branches at `F2_SWITCH` and claims that everything after it is
attributable to the one bridging fact.  That only holds if the unbridged system
is quiet at the switch.  It is not.  In the counterbalance arm where the open
block is A, the network answers every held-out composite of the NON-identifiable
law correctly at the switch and keeps doing so for tens of thousands of epochs
in the control, before collapsing to zero.  Averaged with the other arm, which
sits at zero throughout, that produces the meaningless fifty per cent plateau in
the top row.

The mechanism is scale, not structure.  The open law has rho near 0.26, so about
a quarter of its contrast is undetermined.  While the representation is still
small that quarter is small in absolute terms, the correct entity is still the
nearest candidate, and the rank test scores it right.  Only once the determined
part grows to full scale does the deficit exceed half the candidate spacing and
the answers go wrong.  A switch criterion based on when the CLOSED law resolves
fires long before that.

Both counterbalance arms are run, because the whole point is that they disagree.
Two quantities are tracked per arm on the UNBRIDGED world, neither of which the
bridging fact can touch before it is inserted:

  closed law   its composite facts close an x-y-z triangle, so the contrast lies
               in the row space.  Ordinary learning, and the evidence that
               premise and composite knowledge were acquired beforehand.
  open law     its composite facts sit on detached pairs.  This is the law the
               figure is about, and its control must have reached its asymptote
               before the branch, otherwise the control curve carries a decay
               that has nothing to do with the intervention.

`ready` is the first epoch after which the closed law is learnt AND the open law
has stopped being accidentally right, and both stay that way for the rest of the
run.  That is the earliest defensible switch.
"""

import sys

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, train, worlds
from relspec.config import override
from relspec.measure import law_plan

# depth 1 has to outlast the accidental-correctness plateau, which the stored
# figure data shows running past epoch 30,000 in the control, and then outlast
# its own closed law, which is still at 0.069 spacings after 60,000 epochs.
EPOCHS = {1: 120000, 2: 20000, 3: 10000}
EVERY = {1: 200, 2: 25, 3: 25}
TOL, WINDOW = 0.01, 10
TAU = 0.05      # miss distance counted as learnt, in candidate spacings
TAUS = (0.5, 0.2, 0.1, 0.05)   # the answer should not hinge on this choice
SEED = 0


def learnt(ret, geo):
    """Every held-out composite retrieved, and landing on top of its target."""
    return (np.asarray(ret) >= 100.0) & (np.asarray(geo) < TAU)


def quiet(ret):
    """No held-out composite of the open law is being answered correctly."""
    return np.asarray(ret) <= 0.0


def first_stable(ep, ok):
    """First epoch after which `ok` holds for the rest of the run."""
    bad = np.where(~np.asarray(ok, bool))[0]
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


def fmt(x):
    return "never" if not np.isfinite(x) else "%.0f" % x


def run(depth, closed):
    """One counterbalance arm of one depth on the unbridged world."""
    S = override(init_seed=1000 + SEED, eval_every=EVERY[depth])
    w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=closed)
    w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=closed,
                                      n_bridge=1)
    # exactly the plan the figure uses, so the measured epoch transfers
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    s = System.build(w0, settings=S)
    tr = train.train(models.make_model(w0, depth, S), s,
                     s.lr(depth, settings=S), EPOCHS[depth], S, order_seed=7,
                     eval_every=EVERY[depth], plan=plan)
    ep = np.array(tr.epochs, float)
    out = {n: (np.array(tr.retrieval[n], float),
               np.array(tr.geometric[n], float)) for n in plan["names"]}
    return ep, out, np.array(tr.loss, float), plan


def main():
    print("unbridged world, both counterbalance arms.  learnt = 100%% retrieval "
          "and miss under %.2f spacings" % TAU)
    # a depth can be named on the command line, because depth 1 costs two
    # orders of magnitude more epochs than depth 3 and is rarely the one
    # being re-measured.
    want = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for depth in want:
        print()
        print("N=%d   budget %d   current switch %d"
              % (depth, EPOCHS[depth], worlds.F2_SWITCH[depth]))
        worst = 0.0
        for closed in ("A", "B"):
            op = "B" if closed == "A" else "A"
            ep, out, loss, plan = run(depth, closed)
            cr, cg = out[closed]
            orr, og = out[op]
            ok = learnt(cr, cg) & quiet(orr)
            rq = first_stable(ep, ok)
            k = int(np.argmin(np.abs(ep - worlds.F2_SWITCH[depth])))
            print("  closed=%s open=%s  (%d/%d held-out items)"
                  % (closed, op, len(plan[closed]["a"]), len(plan[op]["a"])))
            print("    closed law  learnt from %8s | at the switch: "
                  "%5.1f%%, miss %7.3f | at the end: %5.1f%%, miss %7.4f"
                  % (fmt(first_stable(ep, learnt(cr, cg))), cr[k], cg[k],
                     cr[-1], cg[-1]))
            print("    open law    quiet  from %8s | at the switch: "
                  "%5.1f%%, miss %7.3f | at the end: %5.1f%%, miss %7.4f"
                  % (fmt(first_stable(ep, quiet(orr))), orr[k], og[k],
                     orr[-1], og[-1]))
            print("    open geom   settles %8s | loss settles %8s"
                  % (fmt(settled(ep, og)), fmt(settled(ep, loss))))
            # how much of `ready` is the arbitrary absolute cutoff?  If the
            # closed law creeps slowly across it, the switch epoch is a
            # property of TAU rather than of the system.
            print("    closed law  learnt at tau = %s"
                  % "  ".join("%.2f:%s" % (t, fmt(first_stable(
                      ep, (cr >= 100.0) & (cg < t)))) for t in TAUS))
            print("    READY       %s"
                  % (("both conditions hold from epoch %.0f" % rq)
                     if np.isfinite(rq) else "never within the budget"))
            worst = np.nan if not np.isfinite(rq) else max(worst, rq)
        print("  switch must clear BOTH arms: %s" % fmt(worst), flush=True)


if __name__ == "__main__":
    main()
