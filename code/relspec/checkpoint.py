"""
Checkpoints: the state a run needs in order to be continued rather than re-run.

A record stores what was measured; a checkpoint stores what was learnt.  With
both, a longer budget continues from the end of the old one instead of
starting again at epoch 0, and an interrupted run loses at most its last
chunk.  One `.npz` per cell under `results/checkpoints/<experiment>/`, holding
for every phase of the run (`run` for an unstaged cell; `pre`, `hold` and
`insert` for a staged one):

  net     the network's weights at the end of the phase, and the state of the
          generator that draws its presentation orders, so a continuation
          draws exactly the orders an uninterrupted run would
  pred    at depth 1, the embedding the phase's closed form is evaluated from,
          which is its STARTING state: an uninterrupted run evaluates every
          epoch from there, and evaluating onwards from the end state instead
          would differ in the last bits.  At depth > 1, the integrated weights
          at the end of the phase
  epochs  how far the phase has run

together with the settings the run was made under.  A checkpoint is used only
when those match the settings asked for, so a changed rate, seed or evaluation
interval starts again rather than continuing a different run.  Continuing
reproduces an uninterrupted run bit for bit (tests/check_checkpoint.py).
"""

from __future__ import annotations

import json
import os

import numpy as np


def weights(model):
    """The model's parameters as a list of arrays, whatever its depth."""
    return [model.E] if model.depth == 1 else list(model.W)


def restore(model, arrays):
    """Load the output of `weights` back into a freshly built model."""
    if model.depth == 1:
        model.E = np.array(arrays[0], float)
    else:
        model.W = [np.array(a, float) for a in arrays]
    return model


def save(path, phases, meta):
    """`phases` is `{name: dict(epochs, net, rng, pred)}`.  Written to a
    temporary file and renamed, so an interrupt cannot leave half a checkpoint."""
    arrays, info = {}, dict(meta=meta, phases={})
    for name, ph in phases.items():
        for src in ("net", "pred"):
            for i, a in enumerate(ph[src]):
                arrays["%s.%s.%d" % (name, src, i)] = np.asarray(a, float)
        info["phases"][name] = dict(epochs=int(ph["epochs"]), rng=ph["rng"],
                                    n_net=len(ph["net"]), n_pred=len(ph["pred"]))
    arrays["info"] = np.array(json.dumps(info))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[: -len(".npz")] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load(path):
    """`(phases, meta)` as `save` wrote them, or `(None, None)` if the file is
    missing or unreadable."""
    try:
        with np.load(path) as z:
            info = json.loads(str(z["info"]))
            phases = {
                name: dict(
                    epochs=p["epochs"], rng=p["rng"],
                    net=[z["%s.net.%d" % (name, i)] for i in range(p["n_net"])],
                    pred=[z["%s.pred.%d" % (name, i)] for i in range(p["n_pred"])])
                for name, p in info["phases"].items()}
        return phases, info["meta"]
    except (OSError, ValueError, KeyError):
        return None, None
