"""
Prospective predictions from the mean learning dynamics.

Shallow (N = 1).  Epoch-averaged learning is `E <- E - lr A^T (A E - C)`, which
in the eigenbasis of `M = A^T A` decouples completely:

    E_k(t) = E*_k + (1 - lr sigma_k)^t (E_k(0) - E*_k)

so the whole trajectory is available in closed form, with no simulation.

Deep (N > 1).  The layers do not decouple.  We integrate the coupled
epoch-averaged equation as derived,

    tau dW_l/dt = L_l^T (B - H E) R_l^T ,   L_l = W_N..W_{l+1}, R_l = W_{l-1}..W_1

which in this codebase's ordering (`E = W_0 W_1 ... W_{N-1}`) is
`dW_k = -lr * prefix_k^T A^T(A E - C) suffix_k^T`, with `tau = 1/(lr|D|)`.
RK4 with per-depth substep counts that were checked for step-size convergence.

Both return a `Trajectory`, so `traj.emergence(settings)` yields `t*_hat` under
exactly the criterion applied to SGD.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT, Settings
from .measure import Trajectory, trajectory_from_embeddings
from .models import embed, prefix_T_dot, products
from .system import System

# --------------------------------------------------------------------------- #
#  Shallow: closed form                                                        #
# --------------------------------------------------------------------------- #


def shallow_trajectory(
    system: System,
    lr,
    epochs,
    E0,
    settings: Settings = DEFAULT,
    eval_every=None,
    probes=None,
    plan=None,
    start=0,
) -> Trajectory:
    """Evaluations at `start`, at every multiple of `eval_every` after it, and
    at `epochs`, each computed directly from `E0`, the state at epoch 0 of the
    phase.  Continuing a phase from `start` therefore reproduces an
    uninterrupted run exactly."""
    every = settings.eval_every if eval_every is None else eval_every
    decay_base = 1.0 - lr * system.evals_M
    coef0 = system.Q.T @ (E0 - system.Estar)

    ts = [start] + list(range(every * (start // every + 1), epochs + 1, every))
    if ts[-1] != epochs:
        ts.append(epochs)
    Es = (system.Estar + system.Q @ ((decay_base**t)[:, None] * coef0) for t in ts)
    return trajectory_from_embeddings(
        system.world, ts, Es, system=system, probes=probes, plan=plan
    )


# --------------------------------------------------------------------------- #
#  Deep: integrate the coupled mean dynamics                                   #
# --------------------------------------------------------------------------- #


def ode_rhs(Ws, M, B):
    """dW_l/dt, up to the lr/tau factor.

    `M = A^T A` and `B = A^T C`, formed once by the caller.  The gradient is
    `A^T (A E - C) = M E - B`, and `M` is `P x P` where `A` is `rows x P`, so
    forming it once turns every substep from a pass over all the facts into one
    small square product.  On the sixteen-law world that is 6554 rows against
    354 columns, a 37-fold reduction in work per substep, and the deep
    integrator makes 8 substeps per epoch at depth 2 and 32 at depth 3.

    Those counts are converged: on the Figure 3 world at depth 3 and the locked
    rate, 32, 128 and 512 substeps agree on the geometric error to four decimal
    places across the whole window.  Measured by tests/check_gap3.py.
    """
    N = len(Ws)
    if N == 1:
        return [-(M @ Ws[0] - B)]
    # suffix products, with the trailing identity left implicit.  Building it
    # and multiplying the last layer's gradient by it is a d x d matmul and an
    # allocation per call for no effect, and the old loop also formed
    # W_0 (W_1..W_{N-1}) only to discard it.
    suf = [None] * N
    s = None
    for k in range(N - 2, -1, -1):
        s = Ws[k + 1] if s is None else Ws[k + 1] @ s
        suf[k] = s
    E = Ws[0] @ suf[0]                  # == embed(Ws), without a second pass
    G = M @ E - B
    out = [-(G @ suf[0].T)]
    # prefix_k^T G = W_{k-1}^T ... W_0^T G, so carrying the running product
    # never forms a prefix.  The old form built W_0 W_1 ... explicitly, which
    # costs a P-sized matmul per layer.
    T = Ws[0].T @ G
    for k in range(1, N):
        out.append(-(T if suf[k] is None else T @ suf[k].T))
        if k + 1 < N:
            T = Ws[k].T @ T
    return out


def _rk4_step(Ws, M, B, h):
    k1 = ode_rhs(Ws, M, B)
    k2 = ode_rhs([Ws[i] + 0.5 * h * k1[i] for i in range(len(Ws))], M, B)
    k3 = ode_rhs([Ws[i] + 0.5 * h * k2[i] for i in range(len(Ws))], M, B)
    k4 = ode_rhs([Ws[i] + h * k3[i] for i in range(len(Ws))], M, B)
    return [
        Ws[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(len(Ws))
    ]


def integrate(Ws0, A, C, lr, epochs, eval_every=5, substeps=1, method="rk4"):
    """Integrate from `Ws0`.  `substeps=1` with `method='euler'` reproduces
    full-batch gradient descent exactly; larger `substeps` converge to the
    continuous-time dynamics.  Returns (epochs, [E(t)], final weights)."""
    Ws = [w.copy() for w in Ws0]
    M, B = A.T @ A, A.T @ C
    h = lr / substeps
    rec_ep, rec_E = [0], [embed(Ws)]
    for ep in range(1, epochs + 1):
        for _ in range(substeps):
            if method == "rk4":
                Ws = _rk4_step(Ws, M, B, h)
            else:
                d = ode_rhs(Ws, M, B)
                Ws = [Ws[k] + h * d[k] for k in range(len(Ws))]
        if ep % eval_every == 0 or ep == epochs:
            rec_ep.append(ep)
            rec_E.append(embed(Ws))
    return np.array(rec_ep), rec_E, Ws


def deep_trajectory(
    system: System,
    lr,
    epochs,
    Ws0,
    depth,
    settings: Settings = DEFAULT,
    eval_every=None,
    probes=None,
    plan=None,
):
    """Trajectory of the integrated deep dynamics.  Returns
    `(Trajectory | None, final_weights)`; the trajectory is None if the
    integration diverged."""
    every = settings.eval_every if eval_every is None else eval_every
    ep, Es, Wend = integrate(
        Ws0,
        system.A,
        system.C,
        lr,
        epochs,
        eval_every=every,
        substeps=settings.substeps(depth),
        method=settings.ode_method,
    )
    if not np.all(np.isfinite(Es[-1])):
        return None, Wend
    return (
        trajectory_from_embeddings(
            system.world, ep, Es, system=system, probes=probes, plan=plan
        ),
        Wend,
    )


# --------------------------------------------------------------------------- #
#  One entry point                                                             #
# --------------------------------------------------------------------------- #


def predict(
    system: System,
    depth,
    lr,
    epochs,
    init,
    settings: Settings = DEFAULT,
    eval_every=None,
    probes=None,
    plan=None,
    start=0,
):
    """Prospective trajectory at any depth.

    `init` is a `ShallowModel`/`DeepModel`, a (P x d) array (N = 1) or a list of
    weight matrices (N > 1).  Returns `(Trajectory | None, final_state)`, where
    the final state can be fed straight back in as `init` for a second curriculum
    phase.

    `start` continues a phase that stopped at that epoch, which must sit on the
    evaluation grid.  At depth 1 `init` is still the phase's STARTING state,
    since the closed form evaluates every epoch from there; at depth > 1 it is
    the integrated state at `start`.  Either way the result reproduces an
    uninterrupted run exactly (tests/check_checkpoint.py).
    """
    if depth == 1:
        E0 = init.embedding() if hasattr(init, "embedding") else np.asarray(init)
        traj = shallow_trajectory(
            system, lr, epochs, E0, settings, eval_every, probes=probes, plan=plan,
            start=start,
        )
        return traj, traj_end_embedding(system, lr, epochs, E0)
    Ws0 = init.W if hasattr(init, "W") else [np.asarray(w) for w in init]
    traj, Wend = deep_trajectory(
        system, lr, epochs - start, Ws0, depth, settings, eval_every, probes=probes,
        plan=plan,
    )
    if traj is not None:
        traj.epochs = traj.epochs + start
    return traj, Wend


def traj_end_embedding(system: System, lr, epochs, E0):
    """E(epochs) under the shallow closed form, without recording anything."""
    decay = (1.0 - lr * system.evals_M) ** epochs
    return system.Estar + system.Q @ (
        decay[:, None] * (system.Q.T @ (E0 - system.Estar))
    )


# --------------------------------------------------------------------------- #
#  Diagnostic: do the environment's modes stay decoupled with depth?           #
# --------------------------------------------------------------------------- #


def mode_coupling(Ws, Q):
    """Off-diagonal mass of `M_k = Q^T (prefix_k prefix_k^T) Q`, per layer.

    Writing the deep dynamics in the mode basis gives
    `dZ/dt = -|D| sum_k M_k Lambda Z N_k`.  Modes evolve independently only if
    every `M_k` is diagonal; this returns how far from diagonal they are.
    """
    pre, _ = products(Ws)
    out = []
    for k in range(len(Ws)):
        # prefix[0] is the identity, for which M_0 = Q^T Q = I is diagonal
        Mk = np.eye(Q.shape[1]) if pre[k] is None else Q.T @ (pre[k] @ pre[k].T) @ Q
        off = np.linalg.norm(Mk - np.diag(np.diag(Mk))) / (np.linalg.norm(Mk) + 1e-30)
        out.append(float(off))
    return out
