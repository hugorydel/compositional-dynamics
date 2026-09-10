"""The toy unified-lattice world.

One world carries all three experimental roles at once:

  E1  two anchored 3x3 lattices sharing their base relation `a`; the composite
      is observed on one of four cells and held out on the rest
  E2  one anchored lattice whose composite is shown once (identifiable) and one
      whose composite is never shown (rho = 0.577)
  E3  an anchored lattice plus an UNANCHORED copy that reuses its relations, so
      no cross-lattice comparison is determined until one fact joins them

`build(link=True)` adds that joining fact.  Run this file directly for the
structural report: token and rank counts, null-space dimension, and the
identifiability rho of every law with and without the link.
"""
import numpy as np

import _paths                                     # noqa: F401  (sys.path)
from relspec import System
from relspec.config import override
from relspec.worlds import Law, World

S = override(lr_target=0.3, eval_every=10)
D = S.d
M = 3                                   # 3x3 lattices


def build(link=False, seed=0):
    rng = np.random.default_rng(seed)
    ents, facts, anchors, gt_e, gt_r, laws = [], [], {}, {}, {}, []
    held = {}                           # role -> list of (a, b, xrel, yrel)

    def lattice(pre, rx, ry, rz, anchored=True, z_cells=None):
        o = rng.standard_normal(D)
        for i in range(M):
            for j in range(M):
                e = "%s_%d_%d" % (pre, i, j)
                ents.append(e)
                gt_e[e] = o + i * gt_r[rx] + j * gt_r[ry]
        for i in range(M - 1):
            for j in range(M):
                facts.append(("%s_%d_%d" % (pre, i, j), rx,
                              "%s_%d_%d" % (pre, i + 1, j)))
        for i in range(M):
            for j in range(M - 1):
                facts.append(("%s_%d_%d" % (pre, i, j), ry,
                              "%s_%d_%d" % (pre, i, j + 1)))
        cells = [(i, j) for i in range(M - 1) for j in range(M - 1)]
        shown = cells if z_cells is None else cells[:z_cells]
        for (i, j) in shown:
            facts.append(("%s_%d_%d" % (pre, i, j), rz,
                          "%s_%d_%d" % (pre, i + 1, j + 1)))
        if anchored:
            for e in ("%s_0_0" % pre, "%s_1_0" % pre, "%s_0_1" % pre):
                anchors[e] = gt_e[e]
        return [("%s_%d_%d" % (pre, i, j), "%s_%d_%d" % (pre, i + 1, j + 1))
                for (i, j) in cells if (i, j) not in shown]

    def rels(*names):
        for n in names[:2]:
            gt_r[n] = rng.standard_normal(D)
        gt_r[names[2]] = gt_r[names[0]] + gt_r[names[1]]

    # E1: two lattices sharing base relation a; 1 of 4 z-cells shown
    rels("a", "b", "c")
    gt_r["d"] = rng.standard_normal(D)
    gt_r["e"] = gt_r["a"] + gt_r["d"]
    held["E1"] = [(p, q, "a", "b") for p, q in lattice("L1", "a", "b", "c", True, 1)]
    held["E1"] += [(p, q, "a", "d") for p, q in lattice("L2", "a", "d", "e", True, 1)]
    laws += [Law("E1_c", "a", "b", "c"), Law("E1_e", "a", "d", "e")]

    # E2: identifiable (1 z-fact) vs non-identifiable (0 z-facts)
    rels("f", "g", "h"); rels("m", "n", "o")
    held["E2_id"] = [(p, q, "f", "g") for p, q in lattice("Y", "f", "g", "h", True, 1)]
    held["E2_no"] = [(p, q, "m", "n") for p, q in lattice("Z", "m", "n", "o", True, 0)]
    laws += [Law("E2_id", "f", "g", "h"), Law("E2_no", "m", "n", "o")]

    # E3: anchored lattice + unanchored copy reusing p, q, s
    rels("p", "q", "s")
    lattice("A", "p", "q", "s", True, None)
    lattice("Acopy", "p", "q", "s", False, None)
    gt_e_copy_shift = gt_e["A_%d_0" % (M - 1)] + gt_r["p"] - gt_e["Acopy_0_0"]
    for i in range(M):
        for j in range(M):
            gt_e["Acopy_%d_%d" % (i, j)] = gt_e["Acopy_%d_%d" % (i, j)] + gt_e_copy_shift
    if link:
        facts.append(("A_%d_0" % (M - 1), "p", "Acopy_0_0"))
    held["E3"] = [("A_%d_%d" % (i, j), "Acopy_%d_%d" % (k, l), None, None)
                  for i in range(M) for j in range(M)
                  for k in range(M) for l in range(M)]
    laws += [Law("E3_s", "p", "q", "s")]

    return World(entities=ents, relations=list(gt_r), facts=facts,
                 anchors=anchors, laws=laws, d=D,
                 meta=dict(gt_ent=gt_e, gt_rel=gt_r, held=held))


def report():
    """The structural result: what the observed facts do and do not determine."""
    w0, w1 = build(False), build(True)
    s0, s1 = (System.build(w0, settings=S), System.build(w1, settings=S))
    print("world: %d tokens (%d entities, %d relations), %d facts"
          % (w0.P, len(w0.entities), len(w0.relations), len(w0.facts)))
    for lab, w, s in (("no link", w0, s0), ("with link", w1, s1)):
        nd = w.P - int(np.linalg.matrix_rank(s.A))
        rh = " ".join("%s %.3f" % (l.name, s.identifiability(l, S)["rho"])
                      for l in w.laws)
        print("  %-9s null dim %d | %s" % (lab, nd, rh))
    print()
    for k, v in w0.meta["held"].items():
        print("  held-out %-7s %d pairs" % (k, len(v)))


if __name__ == "__main__":
    report()
