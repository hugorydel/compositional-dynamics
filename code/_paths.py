"""Locate this repository's own `results/` and `figures/`, relative to this file.

Nothing stores an absolute path, so the repository can be moved or copied.
Imported for its side effect on `sys.path`; also exports `RESULTS`, `FIGURES`
and the two helpers below.

`relspec` is vendored into this folder, so there is nothing outside the
repository to find.
"""

import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

if CODE not in sys.path:
    sys.path.insert(0, CODE)

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def result(name):
    """Path of a file in `results/`."""
    return os.path.join(RESULTS, name)


def figure(name):
    """Path of an image in `figures/`."""
    return os.path.join(FIGURES, name)
