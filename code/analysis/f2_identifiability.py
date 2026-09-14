"""Results, section 2: structural identifiability determines whether
composition can emerge.

Reads Figure 2's records.  The claim is that removing one structural ambiguity
makes an already-experienced composition recoverable.

Main, for numbers the Results may quote:

  structure         the undetermined law's identifiability residual ρ before
                    and after the linking fact, which is what the intervention
                    changes
  effect            2,000 epochs after the fact, the end of the plotted window
                    and the same post-intervention time at every depth: linked
                    minus no-link accuracy, epochs to 90% accuracy, and the
                    decline in geometric error with the link
  control           how far the no-link arm moves from its level at the fact
  recovery time     network against prediction for the epochs to 90% accuracy
                    with the link, the event-time counterpart of Figure 1's
                    emergence result
  trajectories      network against prediction for each condition separately
                    (see `common.agreement_lines`)

Supplementary: the effect and the control's change at the end of each run,
40,000, 8,000 and 4,000 epochs after the fact, which differ by depth and so are
not compared across columns; and the lowest single-world R².

Worlds and records follow Figure 2: 200 worlds, each with two counterbalanced
records (the closed composites in one block or the other) averaged within the
world before anything is computed across worlds, accuracy arithmetically and
geometric error geometrically.  Accuracy is plain held-out accuracy in percent;
the panel draws the same curves relative to the mean at the fact.

  usage:  python f2_identifiability.py
"""

import json

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, worlds
from relspec.config import DEFAULT as S

import fig2
from common import (AGREEMENT_HEADING, DEPTHS, DIAGNOSTIC_HEADING, N_WORLDS,
                    agreement_lines, at, commas, emit, first_reach, fmt_ci,
                    fmt_factor, fmt_fold, lg, med_iqr, power10, timing_line)

CONDS = ("insert", "hold")
WINDOW = fig2.XLIM[1]
LEVEL = 90.0


def _window(ep):
    return at(ep, WINDOW)


def _end(ep):
    return len(ep) - 1


def load():
    seeds, _, arms = fig2.world_set(keep_all=True)
    if len(seeds) != N_WORLDS:
        raise SystemExit("Figure 2 has %d worlds, not %d" % (len(seeds), N_WORLDS))
    data = {}
    for d in DEPTHS:
        acc = {(c, s): [] for c in CONDS for s in ("net", "pred")}
        geo = {(c, s): [] for c in CONDS for s in ("net", "pred")}
        ep = None
        for seed in seeds:
            recs = [json.load(open(f)) for f in sorted(arms[d][seed])]
            for c, s in acc:
                A, G = [], []
                for r in recs:
                    dd = r["arms"][c][s]
                    e = np.array(dd["epochs"], float) - r["t_switch"]
                    m = e >= 0
                    A.append(100.0 * np.array(dd["hits"][r["open"]], bool)[m].mean(axis=1))
                    G.append(np.array(dd["geometric"][r["open"]], float)[m])
                    ep = e[m]
                acc[c, s].append(np.mean(A, axis=0))
                geo[c, s].append(np.exp(np.mean(np.log(np.maximum(G, 1e-12)), axis=0)))
        data[d] = dict(ep=ep, acc={k: np.array(v) for k, v in acc.items()},
                       geo={k: np.array(v) for k, v in geo.items()})
    return seeds, data


def structure(seeds):
    rho = {0: [], 1: []}
    for seed in seeds:
        for closed in ("A", "B"):
            open_ = "B" if closed == "A" else "A"
            for nb in (0, 1):
                w = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed,
                                                 n_bridge=nb)
                law = next(l for l in w.laws if l.name == open_)
                rho[nb].append(System.build(w, settings=S).identifiability(law, S)["rho"])
    b, a = np.array(rho[0]), np.array(rho[1])
    return ["**Structure** (every counterbalanced record, %d)" % len(b),
            "- ρ before the linking fact = %.4f (range %.4f–%.4f); after = %s at "
            "most, zero to machine precision" % (b.mean(), b.min(), b.max(),
                                                 power10(a.max()))]


def _difference(data, idx):
    parts = []
    for d in DEPTHS:
        k = idx(data[d]["ep"])
        a = data[d]["acc"]["insert", "net"][:, k]
        b = data[d]["acc"]["hold", "net"][:, k]
        parts.append("N=%d %s (%.1f%% against %.1f%%; higher in %d of %d worlds)"
                     % (d, fmt_ci(a - b, signed=True), a.mean(), b.mean(),
                        (a > b).sum(), len(a)))
    return "; ".join(parts)


def _control_change(data, idx):
    return "; ".join(
        "N=%d %s" % (d, fmt_ci(data[d]["acc"]["hold", "net"][:, idx(data[d]["ep"])]
                               - data[d]["acc"]["hold", "net"][:, 0], k=2, signed=True))
        for d in DEPTHS)


def _recovery(data, d, src):
    return np.array([first_reach(data[d]["ep"], v, LEVEL)
                     for v in data[d]["acc"]["insert", src]])


def effect(data):
    out = ["**Effect of the linking fact, %s epochs after it** (network; mean and "
           "95%% CI over worlds)" % commas(WINDOW),
           "- Linked minus no-link accuracy, percentage points: %s"
           % _difference(data, _window)]
    parts = []
    for d in DEPTHS:
        t = _recovery(data, d, "net")
        parts.append("N=%d %s, reached in %d of %d worlds"
                     % (d, med_iqr(t), np.isfinite(t).sum(), len(t)))
    out.append("- Epochs from the fact to %d%% accuracy with the link, median: %s"
               % (LEVEL, "; ".join(parts)))
    out.append("- Geometric error with the link, decline from the fact: %s" % "; ".join(
        "N=%d %s" % (d, fmt_fold(data[d]["geo"]["insert", "net"][:, 0]
                                 / data[d]["geo"]["insert", "net"][:, _window(data[d]["ep"])]))
        for d in DEPTHS))
    return out


def control(data):
    return ["**No-link control, change from the fact to %s epochs** (mean and 95%% "
            "CI over worlds)" % commas(WINDOW),
            "- Accuracy, percentage points: %s" % _control_change(data, _window),
            "- Geometric error, factor: %s" % "; ".join(
                "N=%d %s" % (d, fmt_factor(lg(data[d]["geo"]["hold", "net"][:, _window(data[d]["ep"])])
                                           - lg(data[d]["geo"]["hold", "net"][:, 0])))
                for d in DEPTHS)]


def recovery_time(data):
    return (["**Prediction agreement for recovery time** (epochs from the fact until "
             "accuracy first reaches %d%% with the link, per world)" % LEVEL]
            + ["- N=%d: %s" % (d, timing_line(_recovery(data, d, "net"),
                                              _recovery(data, d, "pred"),
                                              data[d]["ep"][1] - data[d]["ep"][0],
                                              "accuracy reaches %d%% within the run" % LEVEL,
                                              "worlds"))
               for d in DEPTHS])


def trajectories(data):
    acc, d_acc = agreement_lines(data, "acc", "Held-out accuracy", WINDOW,
                                 unit="percentage points")
    geo, d_geo = agreement_lines(data, "geo", "Geometric error", WINDOW, log=True)
    return (["**Prediction agreement for whole trajectories, 0 to %s epochs** (%s)"
             % (commas(WINDOW), AGREEMENT_HEADING)] + acc + geo, d_acc + d_geo)


def end_of_training(data):
    ends = ", ".join(commas(data[d]["ep"][-1]) for d in DEPTHS)
    return ["**End of training** (%s epochs after the fact at N = 1, 2 and 3, so not "
            "comparable across depths)" % ends,
            "- Linked minus no-link accuracy, percentage points: %s"
            % _difference(data, _end),
            "- No-link accuracy, change from the fact, percentage points: %s"
            % _control_change(data, _end)]


def report():
    seeds, data = load()
    traj, diag = trajectories(data)
    out = ["## Structural identifiability determines whether composition can emerge "
           "(Figure 2)", ""]
    for block in (structure(seeds), effect(data), control(data), recovery_time(data),
                  traj):
        out += block + [""]
    out += ["### Supplementary (Figure 2)", ""]
    for block in (end_of_training(data), [DIAGNOSTIC_HEADING] + diag):
        out += block + [""]
    return out


if __name__ == "__main__":
    emit(report())
