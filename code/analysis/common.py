"""Quantities shared by the Results analyses.

Every number is read from the stored records the figures draw, over the same
200 worlds, so a value in the text and a curve in a panel come from the same
runs.  Nothing is trained.

What is reported, and why nothing else is:

  the world is the unit   Everything measured in one world shares its relation
                          vectors, lattices and initialisation.  A per-world
                          quantity is summarised across worlds, with a 95% t
                          interval over the 200 of them.

  estimates, not tests    Each claim is answered by the quantity it is about:
                          prediction error and R² for agreement; effect sizes,
                          intervals and recovery times for the interventions.
                          No P values.  The worlds share one fact structure and
                          differ only in random draws, so a test against a zero
                          effect is decided before it is run, and its P shrinks
                          with the number of worlds chosen rather than with
                          anything about the claim.

  R² is agreement         1 - SS(network - prediction) / SS(network - mean),
                          the share of the network's variance the prediction
                          reproduces with nothing fitted.  Not a squared
                          correlation, so a prediction that is offset or
                          rescaled is penalised rather than excused.  Reported
                          within each depth: pooling depths would credit the
                          prediction for the large timing differences between
                          them.
"""

import os
import sys

import numpy as np
from scipy import stats as ss

N_WORLDS = 200
DEPTHS = (1, 2, 3)
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# --------------------------------------------------------------------------- #
#  records and output                                                          #
# --------------------------------------------------------------------------- #

def seed_of(path):
    return int(os.path.basename(path).split("_")[0][1:])


def emit(lines):
    """Print a report.  The Windows console defaults to a code page with no
    R², ρ or superscripts, so switch it to UTF-8 first."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print("\n".join(lines), flush=True)


def lg(v):
    return np.log10(np.maximum(np.asarray(v, float), 1e-12))


def first_reach(ep, v, level):
    k = np.flatnonzero(np.asarray(v, float) >= level - 1e-9)
    return float(ep[k[0]]) if len(k) else np.inf


def at(ep, epoch):
    """Index of the evaluation nearest `epoch`."""
    return int(np.argmin(np.abs(np.asarray(ep, float) - epoch)))


# --------------------------------------------------------------------------- #
#  formatting                                                                  #
# --------------------------------------------------------------------------- #

def commas(x):
    return "{:,}".format(int(round(x)))


def power10(x, digits=2):
    m, e = ("%.*e" % (digits - 1, x)).split("e")
    return "%s × 10%s" % (m, str(int(e)).translate(_SUP))


def med_iqr(x, k=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    return "%.*f (IQR %.*f–%.*f)" % (k, np.median(x), k, np.percentile(x, 25),
                                     k, np.percentile(x, 75))


def fmt_ci(x, k=1, signed=False):
    """Mean over worlds and its 95% CI."""
    c = mean_ci(x)
    return (("%+.*f" if signed else "%.*f") + " [%.*f, %.*f]") % (
        k, c["b"], k, c["lo"], k, c["hi"])


def _fold(x):
    return "%.2f" % x if x < 10 else ("%.1f" % x if x < 100 else "%.0f" % x)


def fmt_fold(ratio):
    """A per-world ratio as a geometric mean with its 95% CI, and the median."""
    r = np.asarray(ratio, float)
    r = r[np.isfinite(r) & (r > 0)]
    c = mean_ci(np.log10(r))
    return "%s-fold (95%% CI %s–%s; median %s)" % (
        _fold(10 ** c["b"]), _fold(10 ** c["lo"]), _fold(10 ** c["hi"]),
        _fold(np.median(r)))


def fmt_factor(log_change):
    """A per-world change in log10 as a multiplicative factor with its CI."""
    c = mean_ci(log_change)
    return "×%.2f [%.2f, %.2f]" % (10 ** c["b"], 10 ** c["lo"], 10 ** c["hi"])


# --------------------------------------------------------------------------- #
#  estimation                                                                  #
# --------------------------------------------------------------------------- #

def mean_ci(x):
    """Mean over worlds with a 95% t interval.  With no spread at all the
    interval collapses onto the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    h = ss.t.ppf(0.975, len(x) - 1) * se if se > 0 else 0.0
    return dict(b=b, se=se, lo=b - h, hi=b + h, n=len(x))


def demean(A, groups):
    """Subtract each group's mean."""
    A = np.asarray(A, float)
    g = np.unique(groups, return_inverse=True)[1]
    cnt = np.bincount(g).astype(float)
    if A.ndim == 1:
        return A - (np.bincount(g, weights=A) / cnt)[g]
    s = np.zeros((len(cnt), A.shape[1]))
    np.add.at(s, g, A)
    return A - (s / cnt[:, None])[g]


def within_share(y, X, groups):
    """Share of the within-group variance of `y` that the columns of `X`
    account for: least squares after removing each group's mean from both.
    Descriptive; equal to the drop in residual sum of squares from a
    regression on group dummies to one on group dummies and `X`."""
    y, X = demean(y, groups), demean(X, groups)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return float(1.0 - (e @ e) / (y @ y))


def agreement_r2(net, pred):
    net, pred = np.ravel(net).astype(float), np.ravel(pred).astype(float)
    return float(1.0 - np.sum((net - pred) ** 2) / np.sum((net - net.mean()) ** 2))


def per_world_r2(net, pred):
    """R² of each world's trajectory (one row per world, any trailing shape).
    NaN for a world whose network trajectory is flat."""
    net = np.asarray(net, float).reshape(len(net), -1)
    pred = np.asarray(pred, float).reshape(len(pred), -1)
    sst = np.sum((net - net.mean(axis=1, keepdims=True)) ** 2, axis=1)
    sse = np.sum((net - pred) ** 2, axis=1)
    out = np.full(len(net), np.nan)
    ok = sst > 0
    out[ok] = 1.0 - sse[ok] / sst[ok]
    return out


def mean_abs_diff(net, pred):
    """Per row, the mean absolute network-minus-prediction difference over
    evaluations, in the measure's own units.  Defined for a flat trajectory,
    where R² is not."""
    net = np.asarray(net, float).reshape(len(net), -1)
    pred = np.asarray(pred, float).reshape(len(pred), -1)
    return np.mean(np.abs(net - pred), axis=1)


def r2_by_depth(per_depth, unit="world"):
    """`{depth: R² per unit}` as one line: each depth's mean.  The lowest
    single unit is a diagnostic (`lowest`), not part of this line."""
    allr = np.concatenate([per_depth[d] for d in DEPTHS])
    flat = int(np.isnan(allr).sum())
    return "mean R² per %s %s%s" % (
        unit, ", ".join("N=%d %.4f" % (d, np.nanmean(per_depth[d])) for d in DEPTHS),
        "" if not flat else " (%d flat trajector%s left out)"
                            % (flat, "y" if flat == 1 else "ies"))


def lowest(per_depth):
    """The lowest R² of any single unit at each depth.  R² depends on how much
    variance a trajectory has, so a steep step displaced by one evaluation can
    pull it far down while the prediction error stays small."""
    return ", ".join("N=%d %.4f" % (d, np.nanmin(per_depth[d])) for d in DEPTHS)


def timing_line(tn, tp, step, event, unit):
    """Observed against predicted times to an event, one per `unit`.

    First whether network and prediction agree that the event happens at all;
    then, where both say it does, the R² of log10 time and the absolute error
    in epochs, with the share within one evaluation, the resolution the grid
    allows.  `step` may be one interval, or one per unit when the grid is not
    uniform, in which case each event is tested against the interval it fell
    in.  A time of zero (the event already holds at the fact) has no logarithm
    and is left out of the R² only.
    """
    tn, tp = np.asarray(tn, float), np.asarray(tp, float)
    fn, fp = np.isfinite(tn), np.isfinite(tp)
    both = fn & fp
    pos = both & (tn > 0) & (tp > 0)
    err = np.abs(tn[both] - tp[both])
    zero = int(both.sum() - pos.sum())
    step = np.asarray(step, float)
    if step.ndim:
        # a non-uniform grid resolves each event differently, so the test is
        # against the interval that event actually fell in
        window, how = step[both], ("within its own crossing interval "
                                   "(median %s epochs)" % commas(np.median(step[both])))
    else:
        window, how = step, "within one evaluation (%s epochs)" % commas(step)
    return ("network and prediction agree on whether %s in %s of %s %s (both %s, "
            "neither %s). Where both do: R² of log10 epochs = %.4f%s; absolute "
            "error median %s epochs, 95th percentile %s; %s in %.1f%%"
            % (event, commas((fn == fp).sum()), commas(tn.size), unit,
               commas(both.sum()), commas((~fn & ~fp).sum()),
               agreement_r2(np.log10(tn[pos]), np.log10(tp[pos])),
               "" if not zero else " (%d already there at the fact left out)" % zero,
               commas(np.median(err)), commas(np.percentile(err, 95)), how,
               100.0 * np.mean(err <= window)))


def err_by_depth(per_depth, unit="", log=False):
    """`{depth: mean absolute difference per world}` as one line: the median
    over worlds and the 95th percentile at each depth.  With `log` the
    differences are in log10 and are shown as factors."""
    def f(v):
        return "×%.3f" % 10 ** v if log else "%.2f" % v
    return ("mean absolute difference from the prediction%s, median over worlds %s "
            "(95th percentile %s)"
            % (" as a factor" if log else " in " + unit,
               ", ".join("N=%d %s" % (d, f(np.median(per_depth[d]))) for d in DEPTHS),
               ", ".join(f(np.percentile(per_depth[d], 95)) for d in DEPTHS)))


def agreement_lines(data, key, label, window, unit="", log=False):
    """Network against prediction for each condition separately, from the fact
    to `window` epochs after it: the mean absolute difference per world, and
    R² per world for the linked condition.

    Never over both conditions together, where the gap between them would be
    credited to the prediction.  No R² for the no-link condition: its
    trajectory is flat or nearly flat, where R² is undefined or swings on a
    single item.  `data[d][key][condition, source]` holds (worlds x
    evaluations) arrays and `data[d]["ep"]` the epochs since the fact.

    Returns the report lines and, separately, the lowest single-world R² of
    the linked condition as diagnostic lines.
    """
    tf = lg if log else np.asarray
    out, diag = [], []
    for cond, name in (("insert", "linked"), ("hold", "no link")):
        r2, err = {}, {}
        for d in DEPTHS:
            m = data[d]["ep"] <= window + 1e-9
            net = tf(data[d][key][cond, "net"][:, m])
            pred = tf(data[d][key][cond, "pred"][:, m])
            r2[d], err[d] = per_world_r2(net, pred), mean_abs_diff(net, pred)
        line = "- %s, %s: %s" % (label, name, err_by_depth(err, unit, log))
        if cond == "insert":
            line += "; %s" % r2_by_depth(r2)
            diag.append("- %s, linked: %s" % (label, lowest(r2)))
        elif not log:
            line += "; network trajectory flat in %d of %d world-depths" % (
                sum(int(np.isnan(r2[d]).sum()) for d in DEPTHS),
                sum(len(r2[d]) for d in DEPTHS))
        out.append(line)
    return out, diag


AGREEMENT_HEADING = ("assessed separately by condition: R² for the linked "
                     "trajectories, and absolute error for the near-flat no-link "
                     "trajectories, where R² carries no information")
DIAGNOSTIC_HEADING = ("**Diagnostics: lowest R² of any single world** (R² depends on "
                      "how much variance a trajectory has, so a steep step displaced "
                      "by one evaluation pulls it down while the error stays small)")
