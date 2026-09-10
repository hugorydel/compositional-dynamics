"""Scorer sanity check.

On the ground-truth embedding every measure is perfect; on a random one none
is.  Also checks that the emergence criterion refuses a rank test that passes
while the geometry is still wrong, which is the failure the criterion exists to
catch.
"""

import _paths  # noqa: F401
import numpy as np
from relspec import worlds  # noqa: E402
from relspec.config import DEFAULT as S  # noqa: E402
from relspec.measure import (  # noqa: E402
    Trajectory,
    apply_plan,
    cross_plan,
    detect_emergence,
    law_plan,
)


def gt_embedding(world, rng=None):
    E = np.array(
        [world.meta["gt_ent"][e] for e in world.entities]
        + [world.meta["gt_rel"][r] for r in world.relations]
    )
    return E if rng is None else rng.standard_normal(E.shape)


def main():
    rng = np.random.default_rng(0)

    for lab, w, ref in (
        ("F1 emergence", worlds.emergence_world(0), None),
        (
            "F2 identifiability",
            worlds.identifiability_world(0, K=16, k=4),
            worlds.identifiability_world(0, K=16, k=4, n_bridge=1),
        ),
    ):
        plan = law_plan(w, worlds.held_composites(w, reference=ref))
        for tag, E in (
            ("ground truth", gt_embedding(w)),
            ("random", gt_embedding(w, rng)),
        ):
            r, g, _, _, q = apply_plan(plan, E)
            rv, gv = np.array(list(r.values())), np.array(list(g.values()))
            print(
                "%-20s %-13s retrieval min %5.1f max %5.1f | rank %5.1f | "
                "geometric max %.2e"
                % (lab, tag, rv.min(), rv.max(),
                   float(np.mean(list(q.values()))), gv.max())
            )

    w = worlds.integration_world(0, link=True)
    plan = cross_plan(w, worlds.held_cross(w))
    for tag, E in (("ground truth", gt_embedding(w)), ("random", gt_embedding(w, rng))):
        r, g, _, e, q = apply_plan(plan, E)
        print(
            "%-20s %-13s resolved %5.1f%% | offset error %.2e"
            % ("F3 integration", tag, r["cross"], e["cross"].mean())
        )

    # The criterion must reject retrieval that is perfect while geometry is not.
    ep = np.arange(0, 100, 10.0)
    perfect = np.full(len(ep), 100.0)
    bad_geo = np.full(len(ep), 3.0)
    good_geo = np.full(len(ep), 0.01)
    t = Trajectory(epochs=ep, retrieval=dict(L=perfect), geometric=dict(L=bad_geo))
    u = Trajectory(epochs=ep, retrieval=dict(L=perfect), geometric=dict(L=good_geo))
    print(
        "criterion          retrieval 100%% + geometry 3.00 -> t* %s (want nan)"
        % t.emergence(S)["L"]
    )
    print(
        "criterion          retrieval 100%% + geometry 0.01 -> t* %s (want 0.0)"
        % u.emergence(S)["L"]
    )
    assert np.isnan(t.emergence(S)["L"]) and u.emergence(S)["L"] == 0.0
    assert np.isnan(detect_emergence(perfect, bad_geo, S.hold, S.tau))


if __name__ == "__main__":
    main()
