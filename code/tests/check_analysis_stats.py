"""Do the estimators in `analysis/common.py` agree with statsmodels?

The analyses need nothing beyond numpy and scipy, and two of their quantities
involve an estimator that could be written wrongly:

  1. mean_ci        the 95% t interval of a mean over worlds, against
                    statsmodels' `DescrStatsW.tconfint_mean`
  2. within_share   the share of within-world variance that law identity
                    accounts for, against the drop in residual sum of squares
                    from a regression on world dummies to one on world dummies
                    and law dummies.  Checked on unbalanced data, as in Figure
                    1, where some law-worlds have no t*
"""

import os
import sys

import _boot  # noqa: F401
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "analysis"))
from common import mean_ci, within_share  # noqa: E402

rng = np.random.default_rng(0)
G, LAWS = 200, 16


def main():
    x = 1.0 + 3.0 * rng.standard_normal(G)
    c = mean_ci(x)
    lo, hi = DescrStatsW(x).tconfint_mean()
    print("1. mean_ci       mine [%.10f, %.10f] | statsmodels [%.10f, %.10f] | max diff %.1e"
          % (c["lo"], c["hi"], lo, hi, max(abs(c["lo"] - lo), abs(c["hi"] - hi))))

    g = np.repeat(np.arange(G), LAWS)
    law = np.tile(np.arange(LAWS), G)
    y = rng.standard_normal(G)[g] + 0.3 * law + rng.standard_normal(G * LAWS)
    keep = rng.random(len(y)) > 0.2
    y, g, law = y[keep], g[keep], law[keep]
    L = (law[:, None] == np.arange(1, LAWS)[None, :]).astype(float)
    D = (g[:, None] == np.unique(g)[None, :]).astype(float)
    reduced = sm.OLS(y, D).fit()
    full = sm.OLS(y, np.column_stack([D, L])).fit()
    ref = 1.0 - full.ssr / reduced.ssr
    mine = within_share(y, L, g)
    print("2. within_share  mine %.10f | statsmodels %.10f | diff %.1e"
          % (mine, ref, abs(mine - ref)))


if __name__ == "__main__":
    main()
