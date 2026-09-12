"""Is the integrator converged at the configured substep counts?

The prediction is a numerical object: RK4 on the mean dynamics at a fixed
number of substeps per epoch.  If that count is too low the dashed curve is
wrong, and every comparison against the network inherits the error.

`check_gap3.py` established that 32, 128 and 512 substeps agree to four decimal
places at depth 3 on the Figure 3 world.  It did NOT test anything below 32, so
a configuration below that range is unverified.  This checks the configured
counts directly, on the Figure 1 world, from the initialisation, against a
reference four times finer than the finest candidate.

Reported as the largest relative deviation over the whole run, taken over every
law, since a curve that agrees on average can still be wrong where it moves.
"""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import DEFAULT, override
from relspec.measure import law_plan

COUNTS = {2: (2, 4, 8, 32), 3: (4, 8, 32, 128)}
SEED = 0


def curves(depth, n, w, s, plan, lr, epochs, every, S):
    tr, _ = theory.predict(
        s, depth, lr, epochs, models.make_model(w, depth, S),
        settings=override(init_seed=S.init_seed, eval_every=every,
                          ode_substeps={depth: n}),
        eval_every=every, plan=plan)
    return {k: np.array(v, float) for k, v in tr.geometric.items()}


def main():
    w = worlds.emergence_world(SEED)
    for depth in (2, 3):
        counts = COUNTS[depth]
        every = worlds.F1_EVERY[depth]
        epochs = worlds.F1_EPOCHS[depth]
        S = override(init_seed=1000 + SEED, eval_every=every)
        s = System.build(w, settings=S)
        plan = law_plan(w, worlds.held_composites(w))
        lr = s.lr(depth, settings=S)
        ref = curves(depth, counts[-1] * 4, w, s, plan, lr, epochs, every, S)
        print("N=%d, %d epochs, reference %d substeps  (configured: %d)"
              % (depth, epochs, counts[-1] * 4, DEFAULT.substeps(depth)),
              flush=True)
        for n in counts:
            g = curves(depth, n, w, s, plan, lr, epochs, every, S)
            worst = max(np.max(np.abs(g[k] - ref[k]) / np.maximum(ref[k], 1e-12))
                        for k in ref)
            print("   %4d substeps   worst relative deviation %10.3e%s"
                  % (n, worst, "   <- configured" if n == DEFAULT.substeps(depth) else ""),
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
