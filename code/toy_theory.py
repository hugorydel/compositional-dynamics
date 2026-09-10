"""Prospective theory overlay for the toy-world figures.

The mean learning dynamics are integrated from the SAME initialisation the
network was trained from, at the same learning rate, and read out on the SAME
epoch grid the trained runs were recorded on.  The resulting embeddings are
then put through `toy_score.score`, the identical function that scored the
network, so every measure gets a prediction with no fitted parameters.

Cached in `toy_theory_d{N}_{link,nolink}.json`.  `ensure()` computes it once
and reuses it afterwards, and `toy_run.py` calls it after training, so a full
re-collection produces prediction and data together.
"""
import json
import os
import sys
import time

import numpy as np

import _paths                                                 # noqa: F401
from _paths import result                                     # noqa: E402
import toy_score                                              # noqa: E402
from toy import build, S                                      # noqa: E402
from relspec import System, models, theory                    # noqa: E402


def data_path(depth, link):
    return result("toy_d%d_%s.json" % (depth, "link" if link else "nolink"))


def theory_path(depth, link):
    return result("toy_theory_d%d_%s.json"
                  % (depth, "link" if link else "nolink"))


def _one(depth, link, seed, epochs, every, want):
    w = build(link, seed)
    s = System.build(w, settings=S)
    init = models.make_model(w, depth, S)
    tr, _ = theory.predict(s, depth, s.lr(depth, settings=S), epochs, init,
                           settings=S, eval_every=every,
                           probes=dict(E=np.eye(w.P)))
    if tr is None:
        return None
    keep = {float(e): i for i, e in enumerate(tr.epochs)}
    miss = [e for e in want if e not in keep]
    if miss:
        raise RuntimeError("theory grid missing %d epochs (e.g. %s)"
                           % (len(miss), miss[:3]))
    Es = [tr.probes["E"][keep[e]] for e in want]
    return toy_score.as_json(toy_score.score(w, Es))


def ensure(depth, link, force=False, verbose=True):
    """Compute (or load) the prediction for one (depth, link) cell."""
    out = theory_path(depth, link)
    if os.path.exists(out) and not force:
        with open(out) as f:
            return json.load(f)
    with open(data_path(depth, link)) as f:
        d = json.load(f)
    want = [float(e) for e in d["runs"][0]["epochs"]]
    t0, runs = time.time(), []
    for sd in d["seeds"]:
        runs.append(_one(depth, link, sd, d["epochs_max"], d["every"], want))
    rec = dict(depth=depth, link=link, seeds=d["seeds"], epochs=want,
               diverged=[r is None for r in runs],
               runs=[r for r in runs if r is not None])
    with open(out, "w") as f:
        json.dump(rec, f)
    if verbose:
        print("wrote %s  (%.0fs, %d/%d integrations finite)"
              % (os.path.basename(out), time.time() - t0,
                 len(rec["runs"]), len(runs)), flush=True)
    return rec


def load(depth, link):
    return ensure(depth, link, verbose=False)


if __name__ == "__main__":
    force = "--force" in sys.argv
    for depth in (1, 2, 3):
        for link in (False, True):
            ensure(depth, link, force=force)
