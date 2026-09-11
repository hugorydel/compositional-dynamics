"""Is the depth-3 gap in Figure 3f a discretisation artefact?

Panel f shows the network sitting above the no-parameter prediction from about
one thousand epochs after the linking fact onward.  The prediction is the
continuous-time mean dynamics, which per-fact SGD approaches only as the step
size goes to zero, so one candidate explanation is simply that 0.03 is too
large a step for depth three: the discrete process starts to ride above its own
continuous limit, which is what an approach to instability looks like.

The test rescales time.  The mean dynamics depend on the product of learning
rate and epochs, so dividing the rate by `s` and multiplying every budget by
`s` leaves the continuous trajectory unchanged while making the discrete one
three or ten times finer.  Plotted against rescaled epochs the predictions must
coincide; if the gap is discretisation, the network curve walks onto them as
the rate falls, and if it does not, the gap is a real property of the dynamics
that the mean-field approximation misses.

Only the linked arm is run.  The control has no gap to explain.
"""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan

DEPTH = 3
BASE = 0.03                    # the locked rate, and the unit of rescaled time
RATES = (0.03, 0.01, 0.003)
WINDOW = 3000                  # rescaled epochs after the fact, as in Figure 3
MARKS = (250, 500, 1000, 2000, 3000)
ORDER_SEED = 7


def run(rate, seed=0):
    """One rate.  Returns rescaled epochs since the fact, and the network and
    predicted geometric error and accuracy on that axis."""
    s_ = BASE / rate
    every = max(1, int(round(worlds.F3_EVERY[DEPTH] * s_)))
    t1 = int(round(worlds.F3_SWITCH[DEPTH] * s_))
    t2 = int(round(WINDOW * s_))
    S = override(init_seed=1000 + seed, eval_every=every)
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(DEPTH, target=rate, settings=S)

    m = models.make_model(w0, DEPTH, S)
    train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED,
                eval_every=every, plan=plan)
    _, st = theory.predict(s0, DEPTH, lr, t1,
                           models.make_model(w0, DEPTH, S), settings=S,
                           eval_every=every, plan=plan)
    b = train.train(m, s1, lr, t2, S, order_seed=ORDER_SEED + 1,
                    eval_every=every, plan=plan)
    tb, _ = theory.predict(s1, DEPTH, lr, t2, st, settings=S,
                           eval_every=every, plan=plan)
    ep = np.array(b.epochs, float) / s_
    return (ep,
            np.array(b.geometric["cross"], float),
            np.array(tb.geometric["cross"], float),
            np.array(b.retrieval["cross"], float))


def main():
    print("depth %d, linked arm only.  time is rescaled to rate %g, so the "
          "predictions should coincide" % (DEPTH, BASE))
    out = {}
    for rate in RATES:
        out[rate] = run(rate)
        ep, net, pred, ret = out[rate]
        lo = np.abs(np.log10(np.maximum(net, 1e-12) / np.maximum(pred, 1e-12)))
        sel = ep >= 250
        print()
        print("  rate %-6g  (%.0fx the epochs, %d evaluations)"
              % (rate, BASE / rate, len(ep)), flush=True)
        print("    %-10s %10s %10s %8s %8s"
              % ("tau", "network", "predicted", "ratio", "acc %"))
        for t in MARKS:
            k = int(np.argmin(np.abs(ep - t)))
            print("    +%-9d %10.4f %10.4f %8.2fx %8.1f"
                  % (t, net[k], pred[k], net[k] / max(pred[k], 1e-12), ret[k]))
        print("    median |log10 ratio| after tau=250: %.3f" % np.median(lo[sel]),
              flush=True)

    print()
    print("  agreement between predictions (should be ~0 if rescaling holds):")
    ref = out[BASE]
    for rate in RATES[1:]:
        ep, _, pred, _ = out[rate]
        p0 = np.interp(ep, ref[0], ref[2])
        sel = ep >= 250
        print("    rate %-6g vs %g: median |log10 ratio| %.4f"
              % (rate, BASE, np.median(np.abs(np.log10(
                  np.maximum(pred[sel], 1e-12) / np.maximum(p0[sel], 1e-12))))))


if __name__ == "__main__":
    main()
