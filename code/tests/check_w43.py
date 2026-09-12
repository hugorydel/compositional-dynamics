"""Does an unbridged control that answers the non-identifiable law ever stop?

One arm in Figure 2, world 43 with block B open, answers the law it should not
be able to answer.  Its behaviour differs by depth: at depth 1 it is correct
from the switch and stays correct for the whole 40,000-epoch window, at depth 2
it is correct at the switch and falls to zero within 3,000, and at depth 3 it
starts wrong and climbs to 80 per cent.

The obvious reading is that the undetermined offset drifts and the answers flip
as it passes through a correct configuration.  The stored miss distances say
otherwise: they barely move.  Over the depth-2 window the mean miss changes by
4 per cent while accuracy collapses from 100 to 0, which means the offset is
near a decision boundary rather than travelling across the space, and all
fifteen items flip together because they share one global offset.

That leaves the question this script answers: run the control far past its
figure budget and see whether the answers reverse, or whether the offset has
settled where it is.  The control is trained alone, with no bridging fact and
no prediction, from the same pre-switch weights and presentation order the
figure uses, so the first part of each trace reproduces the stored cell.
"""

import sys

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, train, worlds
from relspec.config import override
from relspec.measure import law_plan

SEED = 43
CLOSED = "A"          # the arm where block B is open, which is the one at issue
# far past the figure budgets of 40,000 / 8,000 / 4,000
AFTER = {1: 400000, 2: 60000, 3: 60000}
EVERY = {1: 2000, 2: 500, 3: 500}


def main():
    want = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    print("world %d, block B open, control arm only, no bridging fact" % SEED)
    for depth in want:
        S = override(init_seed=1000 + SEED, eval_every=EVERY[depth])
        t1 = worlds.F2_SWITCH[depth]
        w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=CLOSED)
        w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=CLOSED,
                                          n_bridge=1)
        law = "B" if CLOSED == "A" else "A"
        plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
        s0 = System.build(w0, settings=S)
        lr = s0.lr(depth, settings=S)

        m = models.make_model(w0, depth, S)
        train.train(m, s0, lr, t1, S, order_seed=7, eval_every=t1, plan=None)
        tr = train.train(m, s0, lr, AFTER[depth], S, order_seed=8,
                         eval_every=EVERY[depth], plan=plan)

        ep = np.array(tr.epochs, float)
        acc = np.array(tr.retrieval[law], float)
        g = np.array(tr.geometric[law], float)
        idx = np.linspace(0, len(ep) - 1, 11).astype(int)
        print()
        print("  N=%d  switch %d, control run to +%d" % (depth, t1, AFTER[depth]))
        print("     acc :", "  ".join("+%d:%.0f%%" % (ep[i], acc[i]) for i in idx))
        print("     miss:", "  ".join("+%d:%.2f" % (ep[i], g[i]) for i in idx))
        # a reversal is what the drift story predicts; a plateau is not
        flips = int((np.diff((acc > 0).astype(int)) != 0).sum())
        print("     accuracy start %.0f%%, end %.0f%%, crossings of zero: %d"
              % (acc[0], acc[-1], flips))
        print("     miss start %.3f, end %.3f, monotone: %s"
              % (g[0], g[-1],
                 "yes" if np.all(np.diff(g) >= -1e-12)
                 or np.all(np.diff(g) <= 1e-12) else "no"), flush=True)


if __name__ == "__main__":
    main()
