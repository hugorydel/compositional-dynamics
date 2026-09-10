"""Shared loading and styling for the toy-world figures.

`data()` returns the trained runs, `pred()` the prospective theory prediction
for the same cell, computed on the same epoch grid from the same
initialisation and scored by the same function.  `pred()` computes and caches
it on first use, so the overlay appears without a separate step.
"""
import json

import numpy as np

import _paths                                               # noqa: F401
from _paths import figure, result                           # noqa: E402
from style import DARK, GREY, panel, ramp                   # noqa: E402,F401
import matplotlib.pyplot as plt                             # noqa: E402,F401

DEPTHS = (1, 2, 3)
NPAIRS = 81
NONE, BRIDGE = "#9d3b39", "#2c5985"
IDC, NOC = "#2c5985", "#9d3b39"
PRED = dict(color=DARK, lw=0.9, ls=(0, (2.4, 2.2)))


def data(depth, link):
    with open(result("toy_d%d_%s.json"
                     % (depth, "link" if link else "nolink"))) as f:
        return json.load(f)


def pred(depth, link):
    import toy_theory
    return toy_theory.load(depth, link)


def series(runs, path):
    """(runs x T) for a dotted key such as 'acc_xy.E1' or 'e3_acc'."""
    ks = path.split(".")

    def get(r):
        v = r
        for k in ks:
            v = v[k]
        return np.array(v, float)

    return np.array([get(r) for r in runs])


def gmean(Y, axis=0, floor=1e-16):
    return np.exp(np.log(np.maximum(Y, floor)).mean(axis=axis))


def unlocked(runs, ep, npairs=NPAIRS):
    """Mean number of held-out cross-lattice pairs permanently resolved by
    each epoch.  A pair counts from the first evaluation after its LAST
    failure, so the curve is monotone and a pair that flips back does not
    count until it stops flipping."""
    out = []
    for r in runs:
        H = np.array(r["e3_hit"], bool)
        t = np.full(H.shape[1], np.inf)
        for c in range(H.shape[1]):
            bad = np.where(~H[:, c])[0]
            k = 0 if not len(bad) else bad[-1] + 1
            if k < len(ep):
                t[c] = ep[k]
        out.append([(t <= e).sum() for e in ep])
    return np.array(out, float)


def head(ax, depth, lrt, dy=1.03):
    ax.text(0.5, dy, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
            % (depth, lrt), transform=ax.transAxes, fontsize=8.5,
            ha="center", va="bottom", color=DARK)


def save(fig, name):
    p = figure(name)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)
    return p
