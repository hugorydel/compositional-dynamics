"""
relspec -- a spectral theory of compositional learning.

The whole library in one screen:

    config      Settings: every configurable value, in one dataclass
    worlds      World, Law, the lattice construction and three experiments
    system      System: the world as `A E ~ C`, its spectrum, and identifiability
    models      ShallowModel / DeepModel -- one interface, any depth
    measure     held-out retrieval, geometric error, the criterion, Trajectory
    train       one training loop (per-fact SGD or full batch), with probes
    theory      prospective trajectories: shallow closed form, deep coupled ODE
    io          results paths and self-describing JSON

The design point worth knowing before reading anything else: SGD, the shallow
closed form and the deep ODE all produce a `Trajectory`, and `Trajectory.
emergence(settings)` applies one criterion to all of them.  A predicted `t*`
and an observed `t*` are therefore the same kind of measurement.

Typical use:

    from relspec import worlds, System, models, train, theory, DEFAULT
    from relspec.measure import law_plan

    world  = worlds.identifiability_world(seed=0)
    system = System.build(world)
    lr     = system.lr(depth=1)

    # what to score, and on which held-out items.  For a staged run, build the
    # plan from the POST-intervention world so the evaluation set cannot move.
    plan   = law_plan(world, worlds.held_composites(world))

    model  = models.make_model(world, depth=1)
    obs    = train.train(model, system, lr, epochs=4000, plan=plan)

    pred, _ = theory.predict(system, 1, lr, 4000, models.make_model(world, 1),
                             plan=plan)

    obs.emergence(DEFAULT), pred.emergence(DEFAULT)
"""

from . import io, measure, models, system, theory, train, worlds
from .config import DEFAULT, Settings, override
from .measure import (
    Trajectory,
    apply_plan,
    cross_plan,
    detect_emergence,
    law_plan,
    threshold_time,
    trajectory_from_embeddings,
)
from .models import DeepModel, ShallowModel, make_model
from .system import System, build_matrices, sparse_rows
from .worlds import Law, World

__all__ = [
    "DEFAULT",
    "Settings",
    "override",
    "World",
    "Law",
    "System",
    "Trajectory",
    "ShallowModel",
    "DeepModel",
    "make_model",
    "build_matrices",
    "sparse_rows",
    "law_plan",
    "cross_plan",
    "apply_plan",
    "detect_emergence",
    "threshold_time",
    "trajectory_from_embeddings",
    "worlds",
    "system",
    "models",
    "measure",
    "train",
    "theory",
    "io",
]
