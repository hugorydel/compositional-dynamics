"""Why do deeper networks answer more cross-copy queries correctly before the link?

At the switch, Figure 3's no-link count is 9.4 of 80 at depth 1 but 12.6 and
13.1 at depths 2 and 3, and every world scores exactly 9 or exactly 27.  Nothing
in the unlinked world says where the second copy sits relative to the first, so
those hits are luck; the question is why the luck depends on depth.

The prediction matches the network item for item at the switch, so this works
from the predicted state alone and trains nothing.  That state is split into

  E*        the minimum-norm solution of the unlinked facts
  null      the part no fact can reach: one rigid translation of copy B
  residual  whatever of the reachable part has not yet converged

and each piece is scored on its own, together with where the copy sits in the
lattice's own coordinates (steps of x and y).  9 of 80 is the copy pushed onto
one corner entity; 27 is the copy pushed onto a whole edge.

  usage:  python check_prelink.py [--seeds 0 1 2 ...]
"""

import sys
from multiprocessing import Pool

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.measure import _spacing, apply_plan, cross_plan

from f3_at_rate import substeps_for

RATE = 0.003
SWITCH = {1: 200000, 2: 65000, 3: 55000}   # the 0.003 records' switch times


def state_at_switch(seed, depth):
    S = override(init_seed=1000 + seed, eval_every=SWITCH[depth],
                 ode_substeps=substeps_for(depth, RATE))
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0 = System.build(w0, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(depth, target=RATE, settings=S)
    init = models.make_model(w0, depth, S)
    _, st = theory.predict(s0, depth, lr, SWITCH[depth], init,
                           settings=S, plan=plan)
    E = st if depth == 1 else models.embed(st)
    W0 = None if depth == 1 else init.W
    return w0, s0, plan, E, W0


def lattice_offset(E, plan):
    """Mean query-minus-target, in steps of the learned x and y, plus the
    out-of-plane remainder in spacings."""
    Cand = E[plan["cand"]]
    pts = (E[plan["a"]] + plan["nx"][:, None] * E[plan["x"]][None, :]
           + plan["ny"][:, None] * E[plan["y"]][None, :])
    delta = (pts - Cand[plan["bpos"]]).mean(0)
    X = np.stack([E[plan["x"]], E[plan["y"]]], 1)
    uv, *_ = np.linalg.lstsq(X, delta, rcond=None)
    perp = np.linalg.norm(delta - X @ uv) / _spacing(Cand)
    return uv, perp


def hits(E, plan):
    return int(apply_plan(plan, E)[2]["cross"].sum())


def one(cell):
    seed, depth = cell
    w0, s0, plan, E, W0 = state_at_switch(seed, depth)
    Er = s0.project_row(E)
    null = E - Er
    resid = Er - s0.Estar
    ti = w0.tok_index
    B = [ti[e] for e in w0.entities if e.startswith("B")]
    uv, perp = lattice_offset(E, plan)
    uv_star, _ = lattice_offset(s0.Estar, plan)
    uv_conv, _ = lattice_offset(s0.Estar + null, plan)
    row = dict(
        seed=seed, depth=depth,
        count=hits(E, plan),
        count_conv=hits(s0.Estar + null, plan),     # null kept, residual gone
        count_star=hits(s0.Estar, plan),            # neither
        uv=uv, uv_star=uv_star, uv_conv=uv_conv, perp=perp,
        centroid_B=float(np.linalg.norm(null[B].mean(0))),
        resid=float(np.linalg.norm(resid) / np.linalg.norm(s0.Estar)),
    )
    if W0 is not None:
        # the null row of the first layer never moves; the copy's position is
        # that frozen row carried through the later layers as they grow
        Q, mask = s0.row_basis()
        n = s0.Q[:, ~mask][:, 0]
        row["g0"] = float(np.linalg.norm(n @ W0[0]))
        rest0 = models.embed([np.eye(W0[0].shape[1])] + list(W0[1:]))
        row["later_sv0"] = float(np.linalg.svd(rest0, compute_uv=False)[0])
    return row


def main():
    args = sys.argv[1:]
    seeds = [int(a) for a in args[args.index("--seeds") + 1:]] if "--seeds" in args \
        else list(range(24))
    todo = [(s, d) for s in seeds for d in (1, 2, 3)]
    with Pool(min(12, len(todo))) as pool:
        rows = pool.map(one, todo)
    for depth in (1, 2, 3):
        rs = [r for r in rows if r["depth"] == depth]
        print("N=%d  %d worlds" % (depth, len(rs)))
        print("  seed  count  conv  E*   offset (x, y steps)   E* offset        "
              "out-of-plane  |B centroid|  unconverged")
        for r in rs:
            print("  %4d  %5d  %4d  %3d   (%6.2f, %6.2f)      (%6.2f, %6.2f)   "
                  "%6.2f        %7.3f       %.2e"
                  % (r["seed"], r["count"], r["count_conv"], r["count_star"],
                     r["uv"][0], r["uv"][1], r["uv_star"][0], r["uv_star"][1],
                     r["perp"], r["centroid_B"], r["resid"]))
        c = np.array([r["count"] for r in rs])
        print("  mean count %.2f | 27 in %d of %d | count equals converged state "
              "in %d of %d" % (c.mean(), (c == 27).sum(), len(rs),
                               sum(r["count"] == r["count_conv"] for r in rs), len(rs)))
        print()


if __name__ == "__main__":
    main()
