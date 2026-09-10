"""Shared figure style, so every panel in the paper looks like the others."""
from __future__ import annotations

import _paths                      # noqa: F401  (vendored: was the paper's _path)

import os                          # noqa: E402

import matplotlib                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from relspec import io                 # noqa: E402

FIGDIR = io.FIGURES
os.makedirs(FIGDIR, exist_ok=True)

# A restrained print palette.  Blue carries "the structure supports it / the
# theory agrees", red carries "it does not", and the rest are supporting roles.
IDENT = "#1b6ca8"      # identifiable / theory-agrees / depth 1
OTHER = "#c0392b"      # not identifiable / failure
ACCENT = "#7d5ba6"     # third series
GREEN = "#2a7f62"      # depth 2 / composition resolved
OCHRE = "#b8860b"      # depth 3 / training loss
GREY = "#7a7a7a"
LIGHT = "#c8c8c8"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 400,          # print resolution
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.titleweight": "regular",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.solid_capstyle": "round",
})


# An ordered blue -> red ramp for ordered conditions, dark enough at both ends
# to read on white.  Series that differ in degree use this rather than
# categorical colours, so the ordering is legible without reading the legend.
DARK = "#1a1a1a"
# Muted rather than saturated: the primary-ish blue/orange/red of a default
# plotting library reads as unconsidered on the page, so each hue is pulled
# toward its shade and the lightness steps are kept even.
_RAMPS = {
    1: ["#2c5985"],
    2: ["#2c5985", "#9d3b39"],
    3: ["#2c5985", "#c98b34", "#9d3b39"],
    4: ["#2c5985", "#6394ba", "#c98b34", "#9d3b39"],
    5: ["#2c5985", "#6394ba", "#d9b978", "#c98b34", "#9d3b39"],
    6: ["#2c5985", "#6394ba", "#a8c3d8", "#d9b978", "#c98b34", "#9d3b39"],
}


def ramp(n, reverse=False):
    """`n` colours running blue -> orange -> red, for conditions that differ in
    degree.  Chosen so the ordering is legible without reading the legend, and
    so every step stays dark enough to read on white."""
    out = list(_RAMPS[n])
    return out[::-1] if reverse else out


def panel(ax, letter, dx=-0.14, dy=1.06):
    """Journal-style bold panel letter, placed in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top", ha="left")


def note(ax, x, y, text, color=None, **kw):
    """A short inline annotation, in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.6,
            color=color or GREY, va="center", **kw)


def band(ax, x, curves, color, label, lw=1.8, lo=10, hi=90):
    """Median with a percentile band, for a set of trajectories."""
    Y = np.asarray(curves, float)
    ax.fill_between(x, np.nanpercentile(Y, lo, axis=0),
                    np.nanpercentile(Y, hi, axis=0), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.nanmedian(Y, axis=0), color=color, lw=lw, label=label)


def save(fig, name, also_pdf=True):
    """Write the figure.  A vector copy goes alongside the raster one, because
    that is what a journal will ask for."""
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    print("wrote %s" % path)
    if also_pdf and name.lower().endswith(".png"):
        pdf = path[:-4] + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("wrote %s" % pdf)
    return path
