"""Schematic: a relational world, the example of Section 3.1.

A 3 x 2 lattice named so that its trained triangle is the text's own example,
a -x-> b -y-> c closed by a -z-> c; the second path, b -x-> d -y-> e, carries
the withheld z.  The anchors are the three corners every real lattice anchors,
(0,0), (m-1,0) and (0,n-1).

This file also holds the drawing grammar both schematics share, so that
`schematic_experiments.py` draws every fact, entity and head the same way:

    grey solid      a trained fact
    orange dashed   a fact that holds in the world, withheld as a test
    grey dotted     a fact that holds in the world, never trained or tested
    blue            the linking fact
    black square    an anchored entity

Orange against blue was checked for every colour-vision type.  Relations are
told apart by direction and label, not colour.

Nothing here reads a record, so the figure needs no experimental results.  The
PDF is written without a creation date, so a rerun is byte-identical.

  usage:  python schematic_world.py
"""

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath

import style  # noqa: F401  (rcParams)
from style import DARK, IDENT

TRAIN = "#6f6f6f"          # a trained fact
HELD = "#d9731a"           # withheld, and tested
NOTEST = "#a6a6a6"         # holds in the world, never trained and never tested
LINK = IDENT               # the linking fact
DASH = (0, (2.2, 1.5))
DOT = (0, (1, 1.4))
FS = 7.0
plt.rcParams.update({"font.size": FS})

HEAD_L, HEAD_W = 2.0, 1.15      # arrowhead length and half-width, in points
ARROW = "-|>,head_length=%g,head_width=%g" % (HEAD_L, HEAD_W)


# --------------------------------------------------------------------------- #
#  drawing grammar                                                             #
# --------------------------------------------------------------------------- #

def edge(ax, p, q, color=TRAIN, lw=0.8, ls="-", head=False, shrink=2.5, z=2, rad=0.0):
    """A fact from p to q, straight or arced by `rad` (matplotlib's arc3).

    A dashed or dotted line cannot carry its own head: the dash pattern is applied
    to the head's outline too, which chops it up.  So a broken line is drawn
    headless, stopping inside where its head will sit, and the head is added as a
    solid patch by `finish_heads` once the figure's geometry is known.
    """
    solid = ls in ("-", "solid")
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=ARROW if head and solid else "-",
        connectionstyle="arc3,rad=%g" % rad, color=color, lw=lw, linestyle=ls,
        shrinkA=shrink, shrinkB=shrink + (0.7 * HEAD_L if head and not solid else 0),
        mutation_scale=1, zorder=z, capstyle="butt"))
    if head and not solid:
        if not hasattr(ax, "_pending_heads"):
            ax._pending_heads = []
        ax._pending_heads.append((np.asarray(p, float), np.asarray(q, float), rad,
                                  shrink, color, lw, z))


def finish_heads(fig):
    """Add the solid heads of every broken line in `fig`.

    Needs the final data-to-display transform, so the figure is drawn once first.
    Each head points along the line's end tangent -- for an arc3 curve, from its
    control point to the end -- and its tip sits where the line's own head would
    have: `shrink` points short of the end point.
    """
    fig.canvas.draw()
    px = fig.dpi / 72.0
    for ax in fig.axes:
        T = ax.transData
        for p, q, rad, shrink, color, lw, z in getattr(ax, "_pending_heads", []):
            P, Q = T.transform(p), T.transform(q)
            d = Q - P
            C = (P + Q) / 2 + rad * np.array([d[1], -d[0]])
            t = (Q - C) / np.linalg.norm(Q - C)
            tip = Q - t * shrink * px
            base = tip - t * HEAD_L * px
            a, b = T.inverted().transform(np.vstack([base, tip]))
            ax.add_patch(FancyArrowPatch(
                a, b, arrowstyle=ARROW, color=color, lw=lw, linestyle="-",
                shrinkA=0, shrinkB=0, mutation_scale=1, zorder=z))
        ax._pending_heads = []


def node(ax, p, anchored=False, size=7, z=4):
    """An entity: an open circle, or a filled square when anchored."""
    ax.scatter(*p, s=size * (1.3 if anchored else 1), marker="s" if anchored else "o",
               facecolor=DARK if anchored else "white", edgecolor=DARK, linewidth=0.6,
               zorder=z)


def bare(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def arrow_handle(color, lw, ls="-", label=""):
    """A legend swatch with the same small arrowhead as the facts."""
    return Line2D([], [], color=color, lw=lw, ls=ls, marker=">", markersize=2.6,
                  markerfacecolor=color, markeredgecolor=color, label=label)


def anchor_handle(size=3.5):
    return Line2D([], [], color="none", marker="s", markerfacecolor=DARK,
                  markeredgecolor=DARK, markersize=size, label="anchored entity")


def save(fig, stem, tight=False):
    """PNG and PDF into figures/, both reproducible byte for byte.

    The PDF is written without a creation date.  The PNG is flattened onto
    white and written as RGB, the form `manuscript/figures/` holds, so copying it
    there is the whole of the sync.  `tight` crops both to what is drawn, less a
    hair of margin, instead of keeping the figure's set size."""
    from PIL import Image
    finish_heads(fig)
    png, pdf = _paths.figure(stem + ".png"), _paths.figure(stem + ".pdf")
    crop = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    fig.savefig(png, dpi=300, **crop)
    fig.savefig(pdf, dpi=300, metadata={"CreationDate": None}, **crop)
    plt.close(fig)
    rgba = Image.open(png).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    Image.alpha_composite(white, rgba).convert("RGB").save(png)
    print("wrote", png)
    print("wrote", pdf)


# --------------------------------------------------------------------------- #
#  the world                                                                   #
# --------------------------------------------------------------------------- #

NAMES = {(0, 0): "a", (1, 0): "b", (1, 1): "c", (2, 0): "d", (2, 1): "e", (0, 1): "f"}
ANCHORED = {"a", "d", "f"}          # the corners (0,0), (m-1,0), (0,n-1)
X = [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((0, 1), (1, 1)), ((1, 1), (2, 1))]
Y = [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1))]
Z_TRAIN, Z_HELD = ((0, 0), (1, 1)), ((1, 0), (2, 1))


def centred_letter(ax, xy, s, size, color):
    """A letter whose drawn glyph, not its layout box, is centred on `xy`.

    The entity letters are math italic, as in the text.  Centring the text box
    (`ha`/`va="center"`) leaves them sitting high in their markers and leaning
    off-centre, and offsetting text by its outline's extent does not help,
    because the text renderer does not place the glyph exactly where the outline
    says.  So the glyph is drawn as its own outline instead: the outline is moved
    so that its ink box is centred on the origin and plotted as a marker, which
    matplotlib centres on its data point.  A marker is scaled so that its largest
    coordinate spans half its size, so a size of twice that coordinate, in points,
    keeps the letter at `size` points like the text it replaces."""
    outline = TextPath((0, 0), s, size=size)
    (x0, y0), (x1, y1) = outline.get_extents().get_points()
    verts = outline.vertices - [(x0 + x1) / 2, (y0 + y1) / 2]
    ax.plot(*xy, marker=Path(verts, outline.codes), markersize=2 * np.abs(verts).max(),
            color=color, markeredgewidth=0, linestyle="none", zorder=5)


def world_graph(ax, node_s=62, letter=5.6, label=6.3, shrink=5.4):
    """Entities and directed relations, one label per relation.  `node_s` is the
    marker area, `letter` the entity-letter size and `label` the relation-label
    size, in points."""
    bare(ax)
    pos = lambda k: np.array([k[0] * 1.0, k[1] * 0.95])
    for h, t in X + Y + [Z_TRAIN]:
        edge(ax, pos(h), pos(t), lw=0.9, head=True, shrink=shrink)
    edge(ax, pos(Z_HELD[0]), pos(Z_HELD[1]), color=HELD, lw=1.0, ls=DASH, head=True,
         shrink=shrink)
    for k, n in NAMES.items():
        anch = n in ANCHORED
        ax.scatter(*pos(k), s=node_s, marker="s" if anch else "o",
                   facecolor=DARK if anch else "white", edgecolor=DARK, linewidth=0.7,
                   zorder=4)
        centred_letter(ax, pos(k), "$%s$" % n, letter, "white" if anch else DARK)
    kw = dict(fontsize=label)
    ax.text(0.5, -0.1, "$x$", ha="center", va="top", **kw)         # a -x-> b
    ax.text(1.07, 0.5, "$y$", ha="left", va="center", **kw)        # b -y-> c
    ax.text(0.42, 0.56, "$z$", ha="center", va="center", **kw)     # a -z-> c
    ax.text(1.62, 0.36, "$z$", ha="center", va="center", **kw)     # b -z-> e, withheld
    # just past the nodes and the x label, so the saved crop has no empty band
    ax.set_xlim(-0.12, 2.12)
    ax.set_ylim(-0.24, 1.07)


def main():
    """The world, with its legend on two rows directly beneath it.

    The saved figure is cropped to what is drawn, so the manuscript sets its
    printed width (the wrap width in Section 3.1) and nothing around the drawing
    takes up space there.  The legend may be no wider than the diagram."""
    fig = plt.figure(figsize=(2.25, 2.0))
    ax = fig.add_axes([0.0, 0.3, 1.0, 0.7])
    world_graph(ax)
    # two columns, filled column-first: facts on the first row, the anchor below
    leg = ax.legend(handles=[arrow_handle(TRAIN, 1.0, label="trained fact"),
                             anchor_handle(3.0),
                             arrow_handle(HELD, 1.0, DASH, "withheld (test)")],
                    frameon=False, fontsize=5.6, handlelength=1.5, handletextpad=0.4,
                    numpoints=1, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                    borderaxespad=0.0, columnspacing=1.2, labelspacing=0.35)
    fig.canvas.draw()
    assert leg.get_window_extent().width <= ax.get_window_extent().width, \
        "legend is wider than the diagram"
    save(fig, "schematic_relational_world", tight=True)


if __name__ == "__main__":
    main()
