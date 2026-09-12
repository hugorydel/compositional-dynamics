"""Why does the geometric curve rise before it falls?

The plotted quantity is a RATIO: the distance from the composed query point
`E_a + E_z` to the entity it should land on, divided by the median
nearest-neighbour spacing among the candidates, both measured in the LEARNED
embedding.  A ratio can rise for two quite different reasons, and the panel
cannot tell them apart:

  numerator grows    the composed point is drifting away from its target
  denominator lags   the target is right but the space around it expands more
                     slowly than the miss does

This splits them.  It also tracks how much of the composite relation has been
learned, as `|E_z|` IN UNITS OF THE LEARNED SPACING against the same ratio in
ground truth, because the obvious candidate explanation is that the lattice
organises while the composite relation is still missing.  Comparing `|E_z|` to
the true `|z|` instead would be meaningless: the whole embedding sits at a few
per cent of true scale for thousands of epochs, so every learned length looks
tiny whether or not it is correct relative to its neighbours.  If that is it, the query sits at
`E_a` with no composite step applied, the miss is then the full diagonal of a
lattice that is itself growing, and the ratio settles near the diagonal-to-
neighbour ratio of the ground-truth lattice until `E_z` catches up.
"""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, train, worlds
from relspec.config import override

DEPTH, EPOCHS, EVERY = 1, 3000, 50


def report(w, E, law, plan_a, plan_b, cand):
    """(ratio, numerator, spacing) for one law at one moment."""
    z = E[w.tok_index[law.z_rel]]
    q = E[plan_a] + z[None, :]
    C = E[cand]
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    sp = float(np.median(np.sqrt(d2.min(axis=1)))) + 1e-12
    num = float(np.linalg.norm(q - E[plan_b], axis=1).mean())
    return num / sp, num, sp


def main():
    S = override(init_seed=1000, eval_every=EVERY)
    w = worlds.emergence_world(0)
    s = System.build(w, settings=S)
    held = worlds.held_composites(w)
    by_name = {l.name: l for l in w.laws}

    # ground truth, for the scale each learned quantity is heading towards
    gt = w.meta["gt_ent"]
    print("law     true |z|/spacing   true diagonal/spacing")
    setup = {}
    for name in worlds.F1_LAWS:
        law = by_name[name]
        pool = worlds.law_entities(w, law)
        cand = np.array([w.tok_index[e] for e in pool])
        pa = np.array([w.tok_index[a] for a, _ in held[name]])
        pb = np.array([w.tok_index[b] for _, b in held[name]])
        G = np.array([gt[e] for e in pool])
        d2 = ((G[:, None, :] - G[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        sp = float(np.median(np.sqrt(d2.min(axis=1))))
        diag = float(np.linalg.norm(
            [gt[b] - gt[a] for a, b in held[name]], axis=1).mean())
        zt = float(np.linalg.norm(w.meta["gt_rel"][law.z_rel]))
        setup[name] = (law, cand, pa, pb, zt / sp, diag / sp)
        print("  %-6s %8.3f   %8.3f" % (name, zt / sp, diag / sp))

    model = models.make_model(w, DEPTH, S)
    lr = s.lr(DEPTH, settings=S)
    rows = {n: [] for n in worlds.F1_LAWS}
    for start in range(0, EPOCHS, EVERY):
        E = model.embedding()
        for n in worlds.F1_LAWS:
            law, cand, pa, pb, zt, _ = setup[n]
            ratio, num, sp = report(w, E, law, pa, pb, cand)
            zl = float(np.linalg.norm(E[w.tok_index[law.z_rel]]))
            rows[n].append((start, ratio, num, sp, (zl / sp) / zt))
        train.train(model, s, lr, EVERY, S, order_seed=2000,
                    eval_every=EVERY + 1)

    for n in worlds.F1_LAWS:
        _, _, _, _, _, gt_ratio = setup[n]
        print()
        print("%s   ground-truth diagonal/spacing = %.3f" % (n, gt_ratio))
        print("  %6s %8s %10s %10s %8s"
              % ("epoch", "ratio", "numerator", "spacing", "z done"))
        for ep, ratio, num, sp, zf in rows[n]:
            if ep % 250 == 0 or ep < 200:
                print("  %6d %8.3f %10.4f %10.4f %8.3f"
                      % (ep, ratio, num, sp, zf))


if __name__ == "__main__":
    main()
