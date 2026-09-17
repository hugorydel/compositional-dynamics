"""Does the specialised `ode_rhs` agree with the naive one, and how much faster?

The deep integrator is 72 per cent of a depth-3 cell's runtime, and it calls
`ode_rhs` four times per RK4 step.  The naive form rebuilds every prefix and
suffix product from scratch on each call, materialises the trailing identity,
and computes `W_0 (W_1 ... W_{N-1})` inside `suffixes` only to throw it away.

The reference implementation below is the naive one, kept here rather than in
the library so the two can be compared.  Neither is expected to be bit-
identical to the other: they associate the same matrix products differently, so
they round differently, and the question is whether the difference stays at the
level of arithmetic noise over a whole integration rather than growing.

Reported per depth:

  agreement  max absolute difference between the two integrated trajectories,
             and the largest entry of E for scale
  speed      wall time of the same integration under each
"""

import sys
import time

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, models, theory, worlds
from relspec.config import override
from relspec.models import embed, prefix_T_dot, products

SEED = 0
# the emergence world is twenty times the size of the other two
EPOCHS = {"f1": {2: 60, 3: 30}, "f2": {2: 400, 3: 200},
          "f3": {2: 300, 3: 150}}


def world_for(fig):
    if fig == "f1":
        return worlds.emergence_world(SEED)
    if fig == "f2":
        return worlds.identifiability_world(SEED, K=16, k=4, closed_block="A")
    return worlds.integration_world(SEED, link=True)


def naive_rhs(Ws, M, B):
    """The form this replaced: full prefix and suffix products every call."""
    G = M @ embed(Ws) - B
    pre, suf = products(Ws)
    return [-(prefix_T_dot(pre[k], G) @ suf[k].T) for k in range(len(Ws))]


def integrate_with(rhs, Ws0, A, C, lr, epochs, every, substeps):
    """`theory.integrate`, with the right-hand side passed in."""
    Ws = [w.copy() for w in Ws0]
    M, B = A.T @ A, A.T @ C
    h = lr / substeps
    rec = [embed(Ws)]
    for ep in range(1, epochs + 1):
        for _ in range(substeps):
            k1 = rhs(Ws, M, B)
            k2 = rhs([Ws[i] + 0.5 * h * k1[i] for i in range(len(Ws))], M, B)
            k3 = rhs([Ws[i] + 0.5 * h * k2[i] for i in range(len(Ws))], M, B)
            k4 = rhs([Ws[i] + h * k3[i] for i in range(len(Ws))], M, B)
            Ws = [Ws[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                  for i in range(len(Ws))]
        if ep % every == 0 or ep == epochs:
            rec.append(embed(Ws))
    return np.array(rec)


def main():
    figs = sys.argv[1:] or ["f1", "f2", "f3"]
    print("seed %d.  naive = prefix/suffix products rebuilt per call" % SEED)
    for fig in figs:
        for depth in (2, 3):
            S = override(init_seed=1000 + SEED)
            w = world_for(fig)
            s = System.build(w, settings=S)
            lr = s.lr(depth, settings=S)
            Ws0 = models.make_model(w, depth, S).W
            ep, sub = EPOCHS[fig][depth], S.substeps(depth)
            args = (Ws0, s.A, s.C, lr, ep, 10, sub)

            out = {}
            for name, rhs in (("naive", naive_rhs), ("current", theory.ode_rhs)):
                t0 = time.perf_counter()
                out[name] = integrate_with(rhs, *args)
                out[name + "_t"] = time.perf_counter() - t0

            d = float(np.abs(out["naive"] - out["current"]).max())
            scale = float(np.abs(out["naive"]).max())
            print("  %s N=%d  %d epochs x %d substeps = %d rhs calls"
                  % (fig.upper(), depth, ep, sub, 4 * ep * sub))
            print("     agreement  max |diff| %.3e against max |E| %.3f  "
                  "(relative %.1e)" % (d, scale, d / scale))
            print("     speed      naive %6.2fs   current %6.2fs   %.2fx"
                  % (out["naive_t"], out["current_t"],
                     out["naive_t"] / out["current_t"]), flush=True)


if __name__ == "__main__":
    main()
