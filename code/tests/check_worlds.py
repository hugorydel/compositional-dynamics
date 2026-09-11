"""Structural check on the three experiment worlds.  Run before any training."""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System  # noqa: E402
from relspec import worlds  # noqa: E402
from relspec.config import DEFAULT as S  # noqa: E402


def nulldim(world, system):
    return world.P - int(np.linalg.matrix_rank(system.A))


def main():
    w = worlds.emergence_world(0)
    h = worlds.held_composites(w)
    s = System.build(w, settings=S)
    print("F1 emergence_world")
    print(
        "   %d tokens, %d facts, %d laws, null dim %d"
        % (w.P, len(w.facts), len(w.laws), nulldim(w, s))
    )
    n = sorted(len(v) for v in h.values())
    print(
        "   held-out composites per law: min %d, median %d, max %d"
        % (n[0], n[len(n) // 2], n[-1])
    )
    rho = [s.identifiability(l, S)["rho"] for l in w.laws]
    print("   rho: max %.2e  (all laws should be identifiable)" % max(rho))
    print(
        "   emergence spread is not summarisable from the spectrum "
        "(see check_ladder.py)"
    )

    print("F2 identifiability_world  (K=16, k=4)")
    a = worlds.identifiability_world(0, K=16, k=4, closed_block="A")
    b = worlds.identifiability_world(0, K=16, k=4, closed_block="A", n_bridge=1)
    hi = worlds.held_composites(a, reference=b)
    sa, sb = System.build(a, settings=S), System.build(b, settings=S)
    zc = {r: sum(1 for _, x, _ in a.facts if x == r) for r in ("zA", "zB")}
    print(
        "   %d tokens, %d facts, z-facts %s, null dim %d -> %d"
        % (a.P, len(a.facts), zc, nulldim(a, sa), nulldim(b, sb))
    )
    print(
        "   rho before: %s"
        % {l.name: round(sa.identifiability(l, S)["rho"], 4) for l in a.laws}
    )
    print(
        "   rho after:  %s"
        % {l.name: round(sb.identifiability(l, S)["rho"], 4) for l in b.laws}
    )
    print("   held-out composites per law: %s" % {k: len(v) for k, v in hi.items()})

    print("F3 integration_world  (m=3)")
    c = worlds.integration_world(0, link=False)
    d = worlds.integration_world(0, link=True)
    hc = worlds.held_cross(c, reference=d)
    sc, sd = System.build(c, settings=S), System.build(d, settings=S)
    print(
        "   %d tokens, %d facts -> %d, null dim %d -> %d"
        % (c.P, len(c.facts), len(d.facts), nulldim(c, sc), nulldim(d, sd))
    )
    print("   held-out cross comparisons: %d" % len(hc))
    print(
        "   law rho: no link %.2e, link %.2e"
        % (
            sc.identifiability(c.laws[0], S)["rho"],
            sd.identifiability(d.laws[0], S)["rho"],
        )
    )


if __name__ == "__main__":
    main()
