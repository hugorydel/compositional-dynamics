"""Verify the searched ladder on the real assembled world.

The search scored assemblies by rescaling each rung's solo trajectory, which is
exact only in continuous time.  This integrates the whole world once and
reports what the figure will actually show.
"""

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import law_plan  # noqa: E402

S = override(eval_every=250)
EPOCHS, EVERY = 30000, 100

w = worlds.emergence_world(0)
s = System.build(w, settings=S)
held = worlds.held_composites(w)
plan = law_plan(w, held)
tr, _ = theory.predict(
    s,
    1,
    s.lr(1, settings=S),
    EPOCHS,
    models.make_model(w, 1, S),
    settings=S,
    eval_every=EVERY,
    plan=plan,
)
ts = tr.emergence(S)
v = np.array([ts[l.name] for l in w.laws], float)
fin = v[np.isfinite(v)]
print(
    "assembled world: %d tokens, %d facts, %d laws" % (w.P, len(w.facts), len(w.laws))
)
print(
    "held-out composites per law: %d to %d"
    % (min(len(x) for x in held.values()), max(len(x) for x in held.values()))
)
print("emerging: %d of %d within %d epochs" % (len(fin), len(v), EPOCHS))
print("t*: %s" % " ".join("%.0f" % x for x in np.sort(fin)))
print(
    "range %.0f to %.0f, spread %.1fx" % (fin.min(), fin.max(), fin.max() / fin.min())
)
g = np.diff(np.sort(np.log10(fin)))
print("consecutive log10 gaps: min %.3f median %.3f" % (g.min(), np.median(g)))
