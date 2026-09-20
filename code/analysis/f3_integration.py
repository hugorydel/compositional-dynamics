"""Results, section 3: one linking fact unlocks many cross-structure inferences.

Reads Figure 3's records (`results/f3`).  The claim is about how much
one link unlocks.

Main, for numbers the Results may quote:

  effect            comparisons correct, of 80, with and without the link at
                    the end of the plotted window (30,000 epochs after the
                    fact), how many were already correct at the fact, epochs
                    until 72 of 80 are correct, and the decline in
                    cross-structure geometric error with the link
  control           how far the no-link arm moves from its level at the fact
  unlock time       network against prediction for the epochs until 72 of 80
                    are correct with the link, the event-time counterpart of
                    Figure 1's emergence result
  trajectories      network against prediction for each condition separately
                    (see `common.agreement_lines`)

Supplementary: the lowest single-world R².

"Declined by X-fold after linking" is the linked arm's error at the fact
divided by its error at the end of the window.

  usage:  python f3_integration.py
"""

import json

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np

import fig3
from common import (AGREEMENT_HEADING, DEPTHS, DIAGNOSTIC_HEADING, N_WORLDS,
                    agreement_lines, at, commas, emit, first_reach, fmt_ci,
                    fmt_factor, fmt_fold, lg, med_iqr, timing_line)

SUB = "f3"
CONDS = ("insert", "hold")


def load():
    seeds, dropped, have = fig3.world_set(SUB)
    if len(seeds) != N_WORLDS:
        raise SystemExit("Figure 3 has %d worlds, not %d" % (len(seeds), N_WORLDS))
    data = {}
    for d in DEPTHS:
        cnt = {(c, s): [] for c in CONDS for s in ("net", "pred")}
        geo = {(c, s): [] for c in CONDS for s in ("net", "pred")}
        ep = win = n_items = None
        for seed in seeds:
            r = json.load(open(have[d][seed]))
            for c, s in cnt:
                dd = r["arms"][c][s]
                e = np.array(dd["epochs"], float) - r["t_switch"]
                m = e >= 0
                cnt[c, s].append(np.array(dd["hits"]["cross"], bool)[m].sum(axis=1).astype(float))
                geo[c, s].append(np.array(dd["geometric"]["cross"], float)[m])
                ep = e[m]
            win, n_items = fig3.WINDOW * fig3.scale_of(r), r["n_pairs"]
        data[d] = dict(ep=ep, win=win, n_items=n_items,
                       cnt={k: np.array(v) for k, v in cnt.items()},
                       geo={k: np.array(v) for k, v in geo.items()})
    return data


def _most(data):
    return int(np.ceil(0.9 * data[1]["n_items"]))


def _unlock(data, d, src):
    return np.array([first_reach(data[d]["ep"], v, _most(data))
                     for v in data[d]["cnt"]["insert", src]])


def effect(data):
    win, n_items = data[1]["win"], data[1]["n_items"]
    out = ["**Effect of the linking fact** (network; mean and 95% CI over worlds)",
           "- Correct at the fact, in both conditions, before the link can act: %s"
           % "; ".join("N=%d %.1f of %d" % (d, data[d]["cnt"]["hold", "net"][:, 0].mean(),
                                            n_items) for d in DEPTHS)]
    parts = []
    for d in DEPTHS:
        k = at(data[d]["ep"], win)
        a = data[d]["cnt"]["insert", "net"][:, k]
        b = data[d]["cnt"]["hold", "net"][:, k]
        parts.append("N=%d linked %s, all %d correct in %d of %d worlds; no link %s"
                     % (d, fmt_ci(a), n_items, (a == n_items).sum(), len(a), fmt_ci(b)))
    out.append("- Correct of %d at %s epochs: %s" % (n_items, commas(win), "; ".join(parts)))
    parts = []
    for d in DEPTHS:
        t = _unlock(data, d, "net")
        parts.append("N=%d %s, reached in %d of %d worlds"
                     % (d, med_iqr(t), np.isfinite(t).sum(), len(t)))
    out.append("- Epochs from the fact until %d of %d are correct with the link, "
               "median: %s" % (_most(data), n_items, "; ".join(parts)))
    out.append("- Geometric error with the link, decline from the fact to %s epochs: %s"
               % (commas(win), "; ".join(
                   "N=%d %s" % (d, fmt_fold(data[d]["geo"]["insert", "net"][:, 0]
                                            / data[d]["geo"]["insert", "net"][:, at(data[d]["ep"], win)]))
                   for d in DEPTHS)))
    return out


def control(data):
    win = data[1]["win"]
    return ["**No-link control** (change from the fact to %s epochs; mean and 95%% "
            "CI over worlds)" % commas(win),
            "- Comparisons correct: %s" % "; ".join(
                "N=%d %s" % (d, fmt_ci(data[d]["cnt"]["hold", "net"][:, at(data[d]["ep"], win)]
                                       - data[d]["cnt"]["hold", "net"][:, 0], k=2, signed=True))
                for d in DEPTHS),
            "- Geometric error, factor: %s" % "; ".join(
                "N=%d %s" % (d, fmt_factor(lg(data[d]["geo"]["hold", "net"][:, at(data[d]["ep"], win)])
                                           - lg(data[d]["geo"]["hold", "net"][:, 0])))
                for d in DEPTHS)]


def unlock_time(data):
    n_items = data[1]["n_items"]
    return (["**Prediction agreement for unlock time** (epochs from the fact until "
             "%d of %d are correct with the link, per world)" % (_most(data), n_items)]
            + ["- N=%d: %s" % (d, timing_line(_unlock(data, d, "net"),
                                              _unlock(data, d, "pred"),
                                              data[d]["ep"][1] - data[d]["ep"][0],
                                              "%d of %d become correct within the run"
                                              % (_most(data), n_items), "worlds"))
               for d in DEPTHS])


def trajectories(data):
    win = data[1]["win"]
    cnt, d_cnt = agreement_lines(data, "cnt", "Comparisons correct", win,
                                 unit="comparisons")
    geo, d_geo = agreement_lines(data, "geo", "Geometric error", win, log=True)
    return (["**Prediction agreement for whole trajectories, 0 to %s epochs** (%s)"
             % (commas(win), AGREEMENT_HEADING)] + cnt + geo, d_cnt + d_geo)


def report():
    data = load()
    traj, diag = trajectories(data)
    out = ["## One linking fact unlocks many cross-structure inferences (Figure 3)", ""]
    for block in (effect(data), control(data), unlock_time(data), traj):
        out += block + [""]
    out += ["### Supplementary (Figure 3)", "", DIAGNOSTIC_HEADING] + diag + [""]
    return out


if __name__ == "__main__":
    emit(report())
