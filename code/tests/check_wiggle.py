"""Why does the geometric row wiggle, and why does its control drift upward?

Figure 2's bottom row plots the miss distance divided by the candidate spacing.
Both are properties of the LEARNED embedding and both move during training, so
the plotted curve is a ratio of two moving quantities:

    plotted  =  mean_i || (E_a + E_z) - E_b ||        <- numerator, the miss
                --------------------------------
                median nearest-neighbour distance     <- denominator, the scale
                        among the candidates

The predecessor code measured something else entirely, `|| E_z - (E_x + E_y) ||
/ || E_z ||`, which touches only the three relation vectors of the law and no
entity at all.  That quantity is a ratio of two things built from the same law,
so a uniform rescaling of the representation cancels exactly, and for an arm
with no bridging fact the composite relation never moves and the curve is flat
by construction.  The current measure has no such cancellation.

This script separates the two factors so the question can be answered from
data rather than from argument.  For each arm it records, at every evaluation:

  miss      the unnormalised mean distance to the correct entity
  spacing   the candidate spacing that divides it
  ratio     their quotient, which is what the figure draws
  old       the predecessor's relation-only composition error

If the wiggle lives in `miss`, it is real learning dynamics.  If it lives in
`spacing`, the figure is drawing a moving ruler.
"""

import sys

import _boot  # noqa: F401
import _paths  # noqa: F401
import copy

import numpy as np
from relspec import System, models, train, worlds
from relspec.config import override
from relspec.measure import law_plan

SEED = 0
AFTER = 4000          # the plotted window, not the full budget
SHOW = 12             # rows printed per arm


def spacing(Cand):
    d2 = ((Cand[:, None, :] - Cand[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def parts(E, p):
    """The two factors of the plotted ratio, plus the predecessor's measure."""
    Cand = E[p["cand"]]
    q = E[p["a"]] + E[p["z"]][None, :]
    miss = float(np.mean(np.linalg.norm(q - Cand[p["b"]], axis=1)))
    sp = spacing(Cand)
    old = float(np.linalg.norm(E[p["z"]] - (E[p["x"]] + E[p["y"]]))
                / (np.linalg.norm(E[p["z"]]) + 1e-8))
    return miss, sp, miss / sp, old


def turns(v):
    """Indices where a series reverses direction, ignoring flat stretches."""
    d = np.sign(np.diff(np.asarray(v, float)))
    d = d[d != 0]
    return int((np.diff(d) != 0).sum())


def run(depth, closed):
    """Reproduce one Figure 2 cell, keeping E(t) instead of only the measures."""
    S = override(init_seed=1000 + SEED, eval_every=worlds.F2_EVERY[depth])
    t1 = worlds.F2_SWITCH[depth]
    w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=closed)
    w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=closed,
                                      n_bridge=1)
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    lr = s0.lr(depth, settings=S)
    eye = np.eye(len(w0.tok_index))

    m = models.make_model(w0, depth, S)
    train.train(m, s0, lr, t1, S, order_seed=7, eval_every=t1, plan=None)

    out = {}
    for name, sy in (("hold", s0), ("insert", s1)):
        tr = train.train(copy.deepcopy(m), sy, lr, AFTER, S, order_seed=8,
                         eval_every=S.eval_every, probes={"E": eye}, plan=None)
        op = "B" if closed == "A" else "A"
        rows = [parts(E, plan[op]) for E in tr.probes["E"]]
        out[name] = (np.array(tr.epochs, float), np.array(rows, float))
    return out


def report(depth, closed, out):
    op = "B" if closed == "A" else "A"
    print("  closed=%s open=%s" % (closed, op))
    for name in ("insert", "hold"):
        ep, r = out[name]
        miss, sp, ratio, old = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        print("    %-6s  %8s %9s %9s %9s %9s"
              % (name, "epoch", "miss", "spacing", "ratio", "old"))
        for i in np.linspace(0, len(ep) - 1, SHOW).astype(int):
            print("            %8d %9.4f %9.4f %9.4f %9.4f"
                  % (ep[i], miss[i], sp[i], ratio[i], old[i]))
        print("            reversals: miss %d, spacing %d, ratio %d, old %d"
              % (turns(miss), turns(sp), turns(ratio), turns(old)))
        print("            end/start: miss %+.1f%%, spacing %+.1f%%, "
              "ratio %+.1f%%, old %+.1f%%"
              % tuple(100.0 * (v[-1] / v[0] - 1.0)
                      for v in (miss, sp, ratio, old)), flush=True)


def main():
    want = [int(a) for a in sys.argv[1:]] or [3]
    for depth in want:
        print()
        print("N=%d   switch %d, showing the first %d epochs after it"
              % (depth, worlds.F2_SWITCH[depth], AFTER))
        for closed in ("A", "B"):
            report(depth, closed, run(depth, closed))


if __name__ == "__main__":
    main()
