"""Figure S3: the geometric error behind Figures 1 to 3.

Three by three.  Each row is the geometric counterpart of one main-text figure
-- Experiment 1, 2 and 3 from top to bottom -- and columns are depths.  The main
figures show only the behavioural measure; the geometric one moved here to
free space in the main text.

Each row is drawn by its own figure's `row` function with the geometric
measure, so it uses exactly the worlds, records, replay files and conventions
that figure uses: the mean on the logarithms, shaded by its standard error, on
the same logarithmic range in every panel.  Nothing is recomputed differently
here, and a change to a figure's data path reaches this one automatically.

  usage:  python figS3.py
"""

import _paths  # noqa: F401
import matplotlib.pyplot as plt
from _paths import figure

import fig1
import fig2
import fig3


def main(out="figS3_geometry.png"):
    fig, axes = plt.subplots(3, 3, figsize=(9.8, 8.8))
    fig.subplots_adjust(wspace=0.14, hspace=0.62)
    print("Experiment 1")
    fig1.row(axes[0], "geometric", letters="abc")
    print("Experiment 2")
    fig2.row(axes[1], "geometric", letters="def")
    print("Experiment 3")
    fig3.row(axes[2], "geometric", letters="ghi")
    for k, ax in enumerate(axes[:, 0]):
        ax.set_ylabel("Experiment %d\nGeometric Error" % (k + 1), linespacing=1.6)
    # One legend per kind of curve: the laws in the first row, the two arms in
    # the second, which the third row shares.  The lower left of each first
    # panel is empty at this range, so the keys cover no data.
    for ax, (h, l) in ((axes[0][0], fig1.handles(axes[0][0])),
                       (axes[1][0], fig2.handles(axes[1][0]))):
        ax.legend(h, l, loc="lower left", fontsize=7.5, frameon=False,
                  handlelength=1.6, borderaxespad=0.3)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
