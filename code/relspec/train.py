"""
Training.

One loop, used by every experiment.  `mode="sgd"` is per-fact stochastic
gradient descent over randomised presentation orders at a fixed learning rate.
`mode="full"` is the epoch-averaged full-batch step, which is what the
closed-form theory solves exactly; having both in one function is what lets us
check the SGD-to-theory correspondence without a second code path.

`plan` selects what is measured at each evaluation, and comes from
`measure.law_plan` or `measure.cross_plan`.  With `plan=None` nothing is
measured and only probes and loss are recorded.

`probes` records an arbitrary linear functional of `E(t)` alongside the
measures.  Passing the identity records the whole embedding, which is how a
trajectory is re-scored afterwards without retraining.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT, Settings
from .measure import Trajectory, apply_plan
from .system import System


def train(
    model,
    system: System,
    lr,
    epochs,
    settings: Settings = DEFAULT,
    order_seed=0,
    mode="sgd",
    eval_every=None,
    probes=None,
    plan=None,
) -> Trajectory:
    """Train `model` on `system`, recording a `Trajectory`.

    probes : {name: v}  with `v` a (P,) vector; records `v^T E(t)` per evaluation.
    plan   : a measurement plan, or None to record probes and loss only.
    """
    every = settings.eval_every if eval_every is None else eval_every
    rng = np.random.default_rng(order_seed)
    sparse = system.sparse() if mode == "sgd" else None

    names = plan["names"] if plan else []
    rec_epochs, losses = [], []
    ret = {n: [] for n in names}
    geo = {n: [] for n in names}
    hit = {n: [] for n in names}
    err = {n: [] for n in names}
    rnk = {n: [] for n in names}
    probe_rec = {k: [] for k in (probes or {})}

    def record(ep):
        E = model.embedding()
        rec_epochs.append(ep)
        losses.append(system.loss(E))
        if plan:
            r, g, h, e, q = apply_plan(plan, E)
            for n in names:
                ret[n].append(r[n])
                geo[n].append(g[n])
                hit[n].append(h[n])
                err[n].append(e[n])
                rnk[n].append(q[n])
        for k, v in (probes or {}).items():
            probe_rec[k].append(np.asarray(v @ E).copy())

    record(0)
    for ep in range(1, epochs + 1):
        if mode == "sgd":
            model.sgd_epoch(sparse, system.C, lr, rng)
        else:
            model.fullbatch_step(system.A, system.C, lr)
        if ep % every == 0 or ep == epochs:
            record(ep)

    return Trajectory(
        epochs=np.asarray(rec_epochs, float),
        retrieval={n: np.asarray(v, float) for n, v in ret.items()},
        rank={n: np.asarray(v, float) for n, v in rnk.items()},
        geometric={n: np.asarray(v, float) for n, v in geo.items()},
        hits={n: np.asarray(v, bool) for n, v in hit.items()},
        errs={
            n: (None if err[n][0] is None else np.asarray(err[n], float)) for n in names
        },
        loss=np.asarray(losses, float),
        probes={k: np.asarray(v) for k, v in probe_rec.items()},
    )


def train_silently(model, system: System, lr, epochs, order_seed=0, mode="sgd"):
    """Train for its effect on the model, not its trajectory -- used for
    pretraining phases, where only the resulting network state matters.

    This is `train` with the evaluation schedule turned off (two recordings
    instead of thousands), so there is only one training loop in the codebase
    and the RNG stream is identical either way.  Returns the model.
    """
    train(
        model,
        system,
        lr,
        epochs,
        order_seed=order_seed,
        mode=mode,
        eval_every=max(epochs, 1),
    )
    return model
