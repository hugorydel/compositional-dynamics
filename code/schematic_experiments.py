"""Schematic: the three experimental designs, side by side.

  a  Experiment 1.  One representative law on a 2 x 2-cell grid: one diagonal
     trained and three withheld, the real quarter, and the three anchored
     corners.  Lattice size and presentation counts, which vary across the
     sixteen laws, are not drawn.
  b  Experiment 2, the open block.  No link beside Link, each on two lines: a
     known x-then-y path, which receives the linking z in the Link state, next to
     a z heard about entities whose x and y are never presented; below, a known
     path whose z is the test.  The two states leave the same pre-link state,
     so there is no arrow between them: they are conditions, not a sequence.
     The closed block is left to the caption.
  c  Experiment 3.  Two 3 x 3 lattices, the first anchored; every within-lattice
     fact is trained; two of the 80 cross-structure tests are drawn.

Everything a panel asserts about a design is checked against the world
constructors in `relspec.worlds` (world 0) before it is drawn, so a change to a
design that the drawing no longer matches stops the script.  Two things are
drawn by choice and said in the caption rather than asserted: Experiment 2's
z-triangle has a middle point that is not an entity in the world (an entity the
network never sees has no effect on it), and Experiment 3's link is drawn
between the middle rows, where the runs link the bottom rows.  Both are equally
true in the world and play the same role.

The drawing grammar is `schematic_world.py`'s.  Nothing here reads a record,
and the PDF carries no creation date, so a rerun is byte-identical.

  usage:  python schematic_experiments.py
"""

import matplotlib.pyplot as plt
import numpy as np

from relspec import worlds
from schematic_world import (DASH, DOT, FS, HELD, LINK, NOTEST, TRAIN, anchor_handle,
                             arrow_handle, bare, edge, node, save)

HEAD = dict(head=True)
LINK_LW = 0.8           # the linking fact: as thin as a trained fact, only blue
ANCHOR = 4.6            # marker size for an anchored entity (open nodes are 5-7)

STYLE = {"trained": dict(color=TRAIN, lw=0.8),
         "held": dict(color=HELD, lw=1.0, ls=DASH),
         "notest": dict(color=NOTEST, lw=0.9, ls=DOT)}


def link_edge(ax, p, q, shrink=2.4):
    """The linking fact: drawn exactly like a trained fact, in blue."""
    edge(ax, p, q, color=LINK, lw=LINK_LW, shrink=shrink, z=3, **HEAD)


# --------------------------------------------------------------------------- #

def experiment1(ax):
    bare(ax)
    sp = 0.42
    pos = lambda i, j: np.array([i * sp, j * sp])
    for i in range(3):
        for j in range(3):
            if i < 2:
                edge(ax, pos(i, j), pos(i + 1, j), lw=0.8, shrink=2.4, **HEAD)
            if j < 2:
                edge(ax, pos(i, j), pos(i, j + 1), lw=0.8, shrink=2.4, **HEAD)
    for i in range(2):
        for j in range(2):
            observed = (i, j) == (0, 0)
            edge(ax, pos(i, j), pos(i + 1, j + 1), shrink=2.4, z=3,
                 **(STYLE["trained"] if observed else STYLE["held"]), **HEAD)
    # anchors as every real lattice has them: the corners (0,0), (m-1,0), (0,n-1)
    w = worlds.emergence_world(0)
    assert {e for e in w.anchors if e.startswith("B3L1_")} == \
        {"B3L1_0_0", "B3L1_3_0", "B3L1_0_3"}
    corners = {(0, 0), (2, 0), (0, 2)}
    for i in range(3):
        for j in range(3):
            a = (i, j) in corners
            node(ax, pos(i, j), anchored=a, size=ANCHOR if a else 7)
    ax.set_anchor("W")                      # start at the left, under the letter
    ax.set_xlim(-0.12, 2 * sp + 0.3)
    ax.set_ylim(-0.14, 2 * sp + 0.14)


def triangle(ax, o, premises, z, labels=(), anchored=(False, False, False)):
    """a -x-> b -y-> c with its composite a -z-> c.  `premises` styles x and y
    together; `z` is 'link', a STYLE name, or None when it is not drawn;
    `anchored` marks a, b, c."""
    p = [o, o + np.array([0.42, 0.0]), o + np.array([0.56, 0.42])]
    edge(ax, p[0], p[1], shrink=2.4, **STYLE[premises], **HEAD)
    edge(ax, p[1], p[2], shrink=2.4, **STYLE[premises], **HEAD)
    if z == "link":
        link_edge(ax, p[0], p[2])
    elif z:
        edge(ax, p[0], p[2], shrink=2.4, z=3, **STYLE[z], **HEAD)
    for q, a in zip(p, anchored):
        node(ax, q, anchored=a, size=ANCHOR if a else 7)
    at = {"x": ((p[0] + p[1]) / 2 + [0, -0.05], "center", "top"),
          "y": ((p[1] + p[2]) / 2 + [0.05, -0.02], "left", "center"),
          "z": ((p[0] + p[2]) / 2 + [-0.05, 0.04], "right", "bottom")}
    for k in labels:
        xy, ha, va = at[k]
        ax.text(*xy, "$%s$" % k, ha=ha, va=va, fontsize=5.8)


def experiment2(ax):
    bare(ax)
    # Anchors as the open block has them: the first entity of every path, all
    # three of the path that receives the link (triad 0, which also grounds x and
    # y), and the first entity of each z-pair but not the second.
    w0 = worlds.identifiability_world(0, K=16, k=4, closed_block="A")
    w = worlds.identifiability_world(0, K=16, k=4, closed_block="A", n_bridge=1)
    A, before = w.anchors, set(w0.facts)
    linked_path = tuple("BT0_%d" % i in A for i in range(3))
    tested_path = tuple("BT1_%d" % i in A for i in range(3))
    z_pair = ("BF0_0" in A, False, "BF0_1" in A)
    assert linked_path == (True, True, True) and tested_path == (True, False, False)
    assert z_pair == (True, False, False)
    # and the facts as drawn: the link is one z on triad 0, whose x and y are
    # trained; triad 1's z is a test; the z-pair's z is trained
    assert [f for f in w.facts if f not in before] == [("BT0_0", "zB", "BT0_2")]
    assert {("BT0_0", "xB", "BT0_1"), ("BT0_1", "yB", "BT0_2")} <= before
    assert ("BT1_0", "BT1_2") in worlds.held_composites(w0, reference=w)["B"]
    assert ("BF0_0", "zB", "BF0_1") in before
    for x0, label, linked in ((0.0, "No link", False), (1.75, "Link", True)):
        triangle(ax, np.array([x0, 0.62]), "trained", "link" if linked else None,
                 labels=("z",) if linked else ("x", "y"), anchored=linked_path)
        triangle(ax, np.array([x0 + 0.78, 0.62]), "notest", "trained", labels=("z",),
                 anchored=z_pair)
        triangle(ax, np.array([x0, -0.1]), "trained", "held", anchored=tested_path)
        ax.text(x0 + 0.67, 1.32, label, ha="center", va="bottom")
    ax.set_xlim(-0.08, 3.2)
    ax.set_ylim(-0.22, 1.56)


def experiment3(ax):
    bare(ax)
    w0 = worlds.integration_world(0)
    w = worlds.integration_world(0, link=True)
    sp = 0.3
    off = {"A": np.array([0.0, 0.0]), "B": np.array([1.2, 0.0])}
    pos = lambda e: off[e[0]] + sp * np.array([int(e.split("_")[1]), int(e.split("_")[2])])
    before = set(w0.facts)
    # in the world's own order, not the set's: set order follows string hashing,
    # which changes between runs and would reorder the PDF
    for h, r, t in dict.fromkeys(w0.facts):
        edge(ax, pos(h), pos(t), lw=0.7, shrink=2.2, **HEAD)
    # Drawn between the middle rows, by choice: the runs link the bottom rows,
    # A(2,0) -x-> B(0,0).  The middle-row fact is equally true in the world and
    # equally fixes the alignment (one row, rank up by one).
    link = ("A_2_1", "x", "B_0_1")
    gt, gr = w.meta["gt_ent"], w.meta["gt_rel"]
    assert np.allclose(gt[link[0]] + gr["x"], gt[link[2]])
    link_edge(ax, pos(link[0]), pos(link[2]), shrink=2.2)
    tests = worlds.held_cross(w0, reference=w)
    for a, b, rad in (("A_1_2", "B_0_2", -0.35), ("A_0_2", "B_2_2", -0.42)):
        assert any(q[0] == a and q[1] == b for q in tests)
        edge(ax, pos(a), pos(b), color=HELD, lw=0.9, ls=DASH, shrink=2.5, rad=rad,
             z=3, **HEAD)
    for e in w.entities:
        node(ax, pos(e), anchored=e in w.anchors, size=ANCHOR if e in w.anchors else 5)
    ax.set_xlim(-0.1, 1.98)
    ax.set_ylim(-0.15, 1.05)


# Layout, in pixels at 300 dpi.  The manuscript's figure is a tight layout: the
# draft was 1650 x 558 with wider margins, a gap under the panel letters and a
# band above the legend, and was cropped by hand to 1577 x 413 by removing only
# white rows and columns.  The same layout is drawn here directly, every element
# at the pixel it has in that crop, so the figure is regenerated rather than
# edited.  The panel cells keep their draft width (so no drawing moves or
# rescales) and run past the right edge, where there is only white space.
DPI = 300
W, H = 1577, 413                   # figure
LEFT, RIGHT = 7.0, 1607.5          # the panel cells, from the left edge
TOP, BOTTOM = 25.38, 376.92        # the panel cells, from the top edge
LETTER = 41.01                     # bottom of the panel letters, from the top
LEGEND_X, LEGEND_Y = 799.0, -20.0  # legend centre from the left; bottom offset


def main():
    fig = plt.figure(figsize=(W / DPI, H / DPI))
    gs = fig.add_gridspec(1, 3, width_ratios=(0.7, 1.22, 1.08), wspace=0.14,
                          left=LEFT / W, right=RIGHT / W,
                          top=1 - TOP / H, bottom=1 - BOTTOM / H)
    for col, (draw, letter) in enumerate(((experiment1, "a"), (experiment2, "b"),
                                          (experiment3, "c"))):
        draw(fig.add_subplot(gs[col]))
        box = gs[col].get_position(fig)
        fig.text(box.x0, 1 - LETTER / H, letter, weight="bold", fontsize=9,
                 va="bottom", color="black")
    fig.legend(handles=[arrow_handle(TRAIN, 1.0, label="trained fact"),
                        arrow_handle(HELD, 1.0, DASH, "withheld (test)"),
                        arrow_handle(NOTEST, 1.0, DOT, "withheld (no test)"),
                        anchor_handle(3.5),
                        arrow_handle(LINK, LINK_LW, label="linking fact")],
               frameon=False, fontsize=FS, handlelength=2.2, numpoints=1,
               loc="lower center", ncol=5,
               bbox_to_anchor=(LEGEND_X / W, LEGEND_Y / H), columnspacing=1.2)
    save(fig, "schematic_experiments")


if __name__ == "__main__":
    main()
