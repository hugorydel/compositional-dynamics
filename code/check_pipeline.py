"""End-to-end check: training and prediction produce the same kind of object,
measured by the same plan, and agree.

Runs on the smallest world so this stays a quick sanity test rather than an
experiment.  If the plan threading through `train` or `theory.predict` ever
breaks, this fails loudly instead of silently recording nothing.
"""

import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, train, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import cross_plan  # noqa: E402

DEPTH, EPOCHS, EVERY = 1, 3000, 50


def main():
    S = override(lr_target=0.3, eval_every=EVERY)
    w = worlds.integration_world(0, link=True)
    plan = cross_plan(w, worlds.held_cross(w))
    s = System.build(w, settings=S)
    lr = s.lr(DEPTH, settings=S)

    obs = train.train(
        models.make_model(w, DEPTH, S),
        s,
        lr,
        EPOCHS,
        S,
        order_seed=7,
        eval_every=EVERY,
        plan=plan,
    )
    pre, _ = theory.predict(
        s,
        DEPTH,
        lr,
        EPOCHS,
        models.make_model(w, DEPTH, S),
        settings=S,
        eval_every=EVERY,
        plan=plan,
    )

    for lab, t in (("network", obs), ("prediction", pre)):
        r, g = t.retrieval["cross"], t.geometric["cross"]
        print(
            "%-11s %d evaluations | resolved %5.1f%% -> %5.1f%% | "
            "geometric %.3f -> %.2e | t* %s"
            % (lab, len(t.epochs), r[0], r[-1], g[0], g[-1], t.emergence(S)["cross"])
        )
        assert t.hits["cross"].shape == (len(t.epochs), len(plan["a"]))
        assert t.errs["cross"].shape == t.hits["cross"].shape

    gap = np.abs(obs.retrieval["cross"] - pre.retrieval["cross"]).max()
    ratio = np.median(
        np.abs(
            np.log10(
                np.maximum(obs.geometric["cross"], 1e-16)
                / np.maximum(pre.geometric["cross"], 1e-16)
            )
        )
    )
    print(
        "network vs prediction: max retrieval gap %.1f points | "
        "median |log10 ratio| on geometry %.4f" % (gap, ratio)
    )
    assert gap <= 10.0 and ratio < 0.1, "network and prediction disagree"
    print("pipeline OK")


if __name__ == "__main__":
    main()
