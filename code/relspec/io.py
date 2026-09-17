"""Result paths and JSON I/O.

All results live under `results/<experiment>/`.  Every experiment writes its
`Settings` alongside its data, so a result file is self-describing.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def results_dir(name):
    p = os.path.join(RESULTS, name)
    os.makedirs(p, exist_ok=True)
    return p


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, np.bool_)):
        return o.tolist() if isinstance(o, np.ndarray) else bool(o)
    raise TypeError(str(type(o)))


def save(path, obj, settings=None):
    if settings is not None and isinstance(obj, dict):
        obj = dict(obj, settings=settings.to_dict())
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


def load(path):
    with open(path) as f:
        return json.load(f)


def load_all(name, pattern="*.json", require=None):
    """Every result file in `results/<name>/` matching `pattern`.

    `require` is a field every row must have; a file lacking it raises rather
    than being silently treated as data.  Experiment directories hold results
    and nothing else -- analysis output goes to `results/analysis/`.
    """
    import glob

    out = []
    for p in sorted(glob.glob(os.path.join(glob.escape(results_dir(name)), pattern))):
        r = load(p)
        if require is not None and (not isinstance(r, dict) or require not in r):
            raise ValueError(
                "%s is in a results directory but is not a result row "
                "(no %r field). Analysis output belongs in results/analysis/."
                % (p, require)
            )
        out.append(r)
    return out
