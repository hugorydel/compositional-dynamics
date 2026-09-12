"""Does the retired two-chain world lag the theory the same way?

The account of the Figure 3 depth-3 gap is that the linking fact activates one
isolated, very slow mode, and that taking finite steps along a mode in that
regime accumulates a timing deficit.  The retired version of the experiment --
two chains of four entities on one relation, bridged by `D -r-> E` -- has the
same spectral signature, and by one measure a more extreme one: its new mode is
118 times below the next slowest, against 17.8 in the current world.

So it is the test that can refute the account.  If the old world runs on the
predicted clock at depth 3, an isolated slow mode is not sufficient to cause
the lag and the explanation is wrong.  If it lags too, the account survives a
case it did not come from.

Everything here uses the CURRENT machinery -- same optimiser, same integrator,
same locked rate -- so only the world differs.  The retired code ran at 0.3,
which is a step size this project rejected, so its own numbers are not
comparable and are not used.
"""

import copy

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train
from relspec.config import override
from relspec.worlds import World

from check_clock import fit

M = 4
C1 = ["A", "B", "C", "D"]
C2 = ["E", "F", "G", "H"]
ANCHOR = ("A", "B")          # the retired default
DEPTHS = (1, 2, 3)
SEED = 200                   # a seed the retired experiment actually used


def build(bridge, seed, d=16, anchor=ANCHOR):
    """The retired world, rebuilt against the current `World`.

    One relation, named `x` so the existing cross-plan can query it.  Ground
    truth puts all eight entities on one line at integer multiples of `r`; the
    facts never say where chain 2 sits relative to chain 1, and the bridge is
    the single fact that does.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(d)
    gt = {t: i * r for i, t in enumerate(C1 + C2)}
    facts = [(c[i], "x", c[i + 1]) for c in (C1, C2) for i in range(M - 1)]
    if bridge:
        facts.append((C1[-1], "x", C2[0]))
    return World(entities=C1 + C2, relations=["x"], facts=facts,
                 anchors={t: gt[t] for t in anchor}, laws=[], d=d,
                 meta=dict(gt_ent=gt, seed=seed, bridge=bool(bridge)))


def pairs_cross():
    """The cross-chain comparisons, minus the bridge itself."""
    out = []
    for i, a in enumerate(C1):
        for j, b in enumerate(C2):
            if (i, j) == (M - 1, 0):
                continue                      # this one is the bridge
            out.append((a, b, (M + j) - i, 0))
    return out


def pairs_within():
    return [(c[i], c[j], j - i, 0)
            for c in (C1, C2) for i in range(M) for j in range(M) if i != j]


def plan_for(w, pairs, name="cross"):
    """`cross_plan` assumes a two-relation lattice.  This world has one, so the
    plan is assembled here with both relation slots pointing at it and every
    `ny` zero.  Everything else -- candidates, scoring, spacing -- is the
    library's."""
    ti = w.tok_index
    dest = sorted({b for _, b, _, _ in pairs})
    pos = {e: k for k, e in enumerate(dest)}
    cnt = {}
    for _, b, _, _ in pairs:
        cnt[b] = cnt.get(b, 0) + 1
    return dict(
        kind="cross", name=name, names=[name],
        chance=max(cnt.values()) / float(len(pairs)),
        cand=np.array([ti[e] for e in dest]), n_cand=len(dest),
        a=np.array([ti[a] for a, _, _, _ in pairs]),
        bpos=np.array([pos[b] for _, b, _, _ in pairs]),
        nx=np.array([nx for _, _, nx, _ in pairs], float),
        ny=np.zeros(len(pairs)),
        x=ti["x"], y=ti["x"])


def main(t1=20000, t2=400000, every=500):
    w0, w1 = build(False, SEED), build(True, SEED)
    S = override(init_seed=1000, eval_every=every)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    pc = plan_for(w1, pairs_cross())
    pw = plan_for(w1, pairs_within(), name="within")

    ev = np.sort(s1.evals_M)[::-1]
    nz = ev[ev > 1e-9 * ev[0]]
    print("retired world, seed %d: %d cross comparisons, %d facts, lam_max %.4f"
          % (SEED, len(pc["a"]), s0.A.shape[0], ev[0]))
    print("  new mode %.5g, next slowest %.5g, a factor %.1f larger\n"
          % (nz[-1], nz[-2], nz[-2] / nz[-1]))

    for depth in DEPTHS:
        lr = s0.lr(depth, settings=S)
        m = models.make_model(w0, depth, S)
        a = train.train(m, s0, lr, t1, S, order_seed=7, eval_every=every,
                        plan=pw)
        _, st = theory.predict(s0, depth, lr, t1,
                               models.make_model(w0, depth, S), settings=S,
                               eval_every=every, plan=pw)
        print("  N=%d  within-chain error at the switch: network %.2e"
              % (depth, a.geometric["within"][-1]), flush=True)

        b = train.train(copy.deepcopy(m), s1, lr, t2, S, order_seed=8,
                        eval_every=every, plan=pc)
        tb, _ = theory.predict(s1, depth, lr, t2, st, settings=S,
                               eval_every=every, plan=pc)
        e = np.array(b.epochs, float)
        v = np.array(b.geometric["cross"], float)
        q = np.array(tb.geometric["cross"], float)
        al, r1, rb = fit(e, v, np.array(tb.epochs, float), q)
        print("        after the bridge: alpha %6.3f | rms at alpha=1 %6.3f "
              "-> %6.3f | end network %.4g predicted %.4g"
              % (al, r1, rb, v[-1], q[-1]), flush=True)


if __name__ == "__main__":
    main()
