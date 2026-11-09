"""Reviewer 2, Q2 (variance part): between-law vs within-law agreement.

  "... while distinguishing between-law variance from within-law realization
   variance?"

Reads `results/_timing_summary.npz` (written by `figS1.py`): observed
(`d{N}_net`) and predicted (`d{N}_pred`) Experiment 1 emergence times, 200
worlds x 16 laws per depth. On log10 epochs, per depth:

  between-law share    variance of the 16 law means (weighted by worlds) over
                       total variance
  R² law means         agreement of predicted and observed law means (16 values)
  R² spec means        the same over the 8 lattice specifications (laws
                       L<k>_0 and L<k>_1 share spec k)
  R² within-law        agreement of each world's deviation from its law mean,
                       predicted deviation against observed (3,200 values)

Nothing is trained; a run takes under a second.

  usage:  python r2_q2_variance.py
"""

import numpy as np

from common import r2, write_outputs
from _paths import result

NAME = "r2_q2_variance"
QUESTION = ("Q2, '... while distinguishing between-law variance from within-law "
            "realization variance?'")


def main():
    d = np.load(result("_timing_summary.npz"), allow_pickle=True)
    laws = [str(x) for x in d["laws"]]
    spec = np.array([int(l[1:].split("_")[0]) for l in laws])
    cells, summ = [], {}
    rep = ["200 worlds x 16 laws per depth; log10 emergence epochs.", "",
           "| Depth | R² all | Between-law share of variance | R² law means (16) | "
           "R² spec means (8) | R² within-law deviations | SD of within-law log10 t* | "
           "Median abs. within-law error (log10) |", "|" + "---|" * 8]
    for N in (1, 2, 3):
        y, yh = np.log10(d["d%d_net" % N]), np.log10(d["d%d_pred" % N])
        lm, lmh = y.mean(0), yh.mean(0)
        dev, devh = y - lm, yh - lmh
        specs = sorted(set(spec))
        sm = np.array([y[:, spec == k].mean() for k in specs])
        smh = np.array([yh[:, spec == k].mean() for k in specs])
        tot = np.sum((y - y.mean()) ** 2)
        S = dict(r2_all=r2(y, yh),
                 between_law_share=float(y.shape[0] * np.sum((lm - y.mean()) ** 2) / tot),
                 r2_law_means=r2(lm, lmh), r2_spec_means=r2(sm, smh), r2_within=r2(dev, devh),
                 within_sd=float(dev.std()),
                 within_median_abs_error=float(np.median(np.abs(dev - devh))))
        summ["N%d" % N] = S
        rep.append("| N = %d | %.4f | %.3f | %.4f | %.4f | %.4f | %.3f | %.4f |" % (
            N, S["r2_all"], S["between_law_share"], S["r2_law_means"], S["r2_spec_means"],
            S["r2_within"], S["within_sd"], S["within_median_abs_error"]))
        for j, law in enumerate(laws):
            cells.append(dict(depth=N, law=law, spec=int(spec[j]),
                              obs_mean_log10=float(lm[j]), pred_mean_log10=float(lmh[j]),
                              obs_within_sd=float(dev[:, j].std()),
                              r2_within_law=r2(dev[:, j], devh[:, j])))
    rep += ["", "`cells.csv` gives each law's observed and predicted mean, its within-law SD, "
            "and the within-law R² for that law alone."]
    write_outputs(NAME, "Reviewer 2, Q2: between-law vs within-law agreement", QUESTION, rep, summ,
                  cells, "`results/_timing_summary.npz`")


if __name__ == "__main__":
    main()
