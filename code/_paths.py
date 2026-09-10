"""Locate the paper's `relspec` package and this folder's own results and
figures, all relative to this file.

Nothing in `toy_model/` stores an absolute path, so the folder can be moved or
copied as long as it stays one level below the paper root.  Imported for its
side effect on `sys.path`; also exports `RESULTS` and `FIGURES`.
"""
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))       # toy_model/code
ROOT = os.path.dirname(CODE)                            # toy_model
PAPER = os.path.dirname(ROOT)                           # the paper root
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

# `relspec` lives in the paper's code/, the shared figure style in its
# analysis/.  Appended, not prepended, so this folder's own modules always win
# a name clash.
for _p in (CODE, os.path.join(PAPER, "code"),
           os.path.join(PAPER, "code", "analysis")):
    if _p not in sys.path:
        sys.path.append(_p)

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def result(name):
    """Path of a JSON file in `toy_model/results/`."""
    return os.path.join(RESULTS, name)


def figure(name):
    """Path of an image in `toy_model/figures/`."""
    return os.path.join(FIGURES, name)
