"""Reviewer 2, Q4 (second part): non-identifiability or optimizer selection?

  "Please clarify whether 'failure to settle the law' denotes true
   non-identifiability or optimizer selection bias."

Reads the stored Experiment 2 controls, `results/f2/controls_w{s}_d{N}_{A,B}.npz`
(worlds 0-29; written by `analysis/controls.py`), which hold the weights at the
moment of the intervention (`pre.*`) and per-evaluation hits and geometric
error of the no-link (`hold`) and linked (`insert`) arms.

For the open law, l^T E = (P+ l)^T E + (P0 l)^T E. The second term is the part
the data leave free; with v the null vector (see r2_q4_construction.py),
(P0 l)^T E = (l.v / v.v) v^T E. This script reports, per cell:

  open-law error      ||l^T E|| / ||r_z|| at the intervention, with r_z the
                      world's ground-truth z displacement
  free part at init   ||(P0 l)^T E(0)|| / ||r_z||, E(0) rebuilt from the
                      cell's init seed (1000 + s)
  free-part drift     ||(P0 l)^T (E - E(0))|| / ||r_z||  (zero for shallow SGD,
                      App. A.4; can move at depth, App. A.5)
  accuracy            open-law held-out accuracy, no-link and linked, at the
                      intervention and at the end of the stored horizon
  geometric change    no-link geometric error, end / start - 1
  training loss       at the intervention

Nothing is trained; a run takes about a minute.

  usage:  python r2_q4_selection.py
"""

import glob
import os
import re

import numpy as np

from common import write_outputs
from _paths import RESULTS
from r2_q4_construction import open_null_vector
from relspec import System, models, worlds
from relspec.config import override

NAME = "r2_q4_selection"
QUESTION = ("Q4, 'Please clarify whether \"failure to settle the law\" denotes "
            "true non-identifiability or optimizer selection bias.'")
PATTERN = re.compile(r"controls_w(\d+)_d(\d)_([AB])\.npz$")


def cell(path):
    s, N, closed = PATTERN.search(path).groups()
    s, N = int(s), int(N)
    op = "B" if closed == "A" else "A"
    d = np.load(path)
    Ws = [d["pre.%d" % i] for i in range(N)]
    E = models.embed(Ws) if N > 1 else Ws[0]
    w = worlds.identifiability_world(s, K=16, k=4, closed_block=closed)
    sysm = System.build(w)
    v, _ = open_null_vector(w, op)
    ell = sysm.law_contrast(op)
    P0l = (ell @ v) / (v @ v) * v
    rz = float(np.linalg.norm(w.ground_truth()[w.tok_index["z" + op]]))
    E0 = models.make_model(w, N, override(init_seed=1000 + s)).embedding()
    hold = d["hold.law_%s" % op].mean(1)
    ins = d["insert.law_%s" % op].mean(1)
    geo = d["hold.geo_%s" % op]
    return dict(
        seed=s, depth=N, closed=closed, epochs_after=int(d["epochs"][-1]),
        open_law_error_rel=float(np.linalg.norm(ell @ E) / rz),
        free_part_init_rel=float(np.linalg.norm(P0l @ E0) / rz),
        free_part_drift_rel=float(np.linalg.norm(P0l @ (E - E0)) / rz),
        nolink_acc_start=float(hold[0]), nolink_acc_end=float(hold[-1]),
        linked_acc_end=float(ins[-1]),
        nolink_geo_change=float(geo[-1] / geo[0] - 1),
        loss_at_intervention=float(sysm.loss(E)))


def main():
    # glob.escape: the project folder name contains "[2026-09]", which glob would
    # otherwise read as a character class
    files = sorted(glob.glob(os.path.join(glob.escape(os.path.join(RESULTS, "f2")),
                                          "controls_w*_d*_[AB].npz")))
    if not files:
        raise SystemExit("no results/f2/controls_*.npz found; run analysis/controls.py first")
    cells = [cell(f) for f in files]
    summ, rep = {}, [
        "%d stored cells (worlds x depths x block assignments)." % len(cells), "",
        "| Depth | Cells | Post-link epochs | Open-law error / ‖r_z‖, median (range) | "
        "Free part at init / ‖r_z‖, median | Free-part drift / ‖r_z‖, median (max) | "
        "No-link accuracy, start → end | Linked accuracy, end | "
        "No-link geometric change, geometric mean (max abs.) | Training loss at intervention, median |",
        "|" + "---|" * 10]
    for N in sorted({c["depth"] for c in cells}):
        C = [c for c in cells if c["depth"] == N]
        a = lambda k: np.array([c[k] for c in C])
        g = a("nolink_geo_change")
        summ["N%d" % N] = dict(
            n_cells=len(C), epochs_after=int(a("epochs_after").max()),
            open_law_error_rel_median=float(np.median(a("open_law_error_rel"))),
            open_law_error_rel_min=float(a("open_law_error_rel").min()),
            open_law_error_rel_max=float(a("open_law_error_rel").max()),
            free_part_init_rel_median=float(np.median(a("free_part_init_rel"))),
            free_part_drift_rel_median=float(np.median(a("free_part_drift_rel"))),
            free_part_drift_rel_max=float(a("free_part_drift_rel").max()),
            nolink_acc_start_mean=float(a("nolink_acc_start").mean()),
            nolink_acc_end_mean=float(a("nolink_acc_end").mean()),
            linked_acc_end_mean=float(a("linked_acc_end").mean()),
            nolink_geo_change_geomean=float(np.exp(np.mean(np.log1p(g))) - 1),
            nolink_geo_change_maxabs=float(np.abs(g).max()),
            loss_at_intervention_median=float(np.median(a("loss_at_intervention"))))
        S = summ["N%d" % N]
        rep.append("| N = %d | %d | %d | %.3f (%.3f–%.3f) | %.4f | %.2e (%.2e) | %.1f%% → %.1f%% | %.1f%% | %+.1f%% (%.1f%%) | %.1e |" % (
            N, S["n_cells"], S["epochs_after"], S["open_law_error_rel_median"], S["open_law_error_rel_min"],
            S["open_law_error_rel_max"], S["free_part_init_rel_median"], S["free_part_drift_rel_median"],
            S["free_part_drift_rel_max"], 100 * S["nolink_acc_start_mean"], 100 * S["nolink_acc_end_mean"],
            100 * S["linked_acc_end_mean"], 100 * S["nolink_geo_change_geomean"],
            100 * S["nolink_geo_change_maxabs"], S["loss_at_intervention_median"]))
    write_outputs(NAME, "Reviewer 2, Q4: what learning selects for the open law", QUESTION, rep, summ,
                  cells, "`results/f2/controls_w*_d*_[AB].npz` (pre-intervention weights, hold/insert arms)")


if __name__ == "__main__":
    main()
