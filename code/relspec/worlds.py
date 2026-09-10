"""
Synthetic relational worlds.

A `World` is a set of entity and relation tokens, a multiset of facts
`(head, relation, tail)`, a set of anchored entities, and the compositional
`Law`s (`z = x + y`) whose emergence we measure.  A fact imposes the linear
constraint `x_head + r - x_tail ~ 0`; an anchor imposes `x_e ~ c_e`.

Three experiment worlds are used, all variants of the same `lattice_world`
construction:

    emergence_world       (F1) a lattice family whose laws span a range of
                               spectral support, so emergence times differ
                               between laws in one world.  A fixed fraction of
                               each law's diagonals is withheld.
    identifiability_world (F2) two structurally identical blocks, one whose
                               composite evidence closes an x-y-z triangle and
                               one whose composite evidence is ungrounded.
    integration_world     (F3) one lattice duplicated, the copy unanchored, and
                               a single fact that may or may not link them.

All return the same `World` type, so everything downstream is shared.

Held-out items are derived from a world, not declared by hand: a pair is
held out for a law when ground truth places the two entities exactly one
composite step apart and no training fact states it.  That definition
works unchanged in all three, which is what lets one scorer serve them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Law:
    """One additive compositional law  z = x + y,  by relation-token name."""

    name: str
    x_rel: str
    y_rel: str
    z_rel: str


@dataclass
class World:
    entities: list  # entity token names, in row order
    relations: list  # relation token names, in row order
    facts: list  # (head, relation, tail), with multiplicity
    anchors: dict  # entity name -> fixed value in R^d
    laws: list  # list[Law]
    d: int  # embedding dimension
    meta: dict = field(default_factory=dict)

    # -- token indexing ----------------------------------------------------- #
    # theta is a (P x d) matrix whose rows are token embeddings: entities first,
    # then relations.  The index is *derived* from `entities` / `relations`, so
    # it is always well defined; nothing has to remember to build it, and no
    # function has to mutate the world as a side effect in order to use it.

    @property
    def n_ent(self):
        return len(self.entities)

    @property
    def n_rel(self):
        return len(self.relations)

    @property
    def P(self):
        return self.n_ent + self.n_rel

    @cached_property
    def ent_index(self):
        return {e: k for k, e in enumerate(self.entities)}

    @cached_property
    def rel_index(self):
        return {r: self.n_ent + k for k, r in enumerate(self.relations)}

    @cached_property
    def tok_index(self):
        return dict(self.ent_index, **self.rel_index)

    @property
    def law_names(self):
        return [l.name for l in self.laws]

    def law(self, name) -> Law:
        for l in self.laws:
            if l.name == name:
                return l
        raise KeyError(name)

    def ground_truth(self) -> np.ndarray:
        """The (P x d) embedding that the world was generated from.  Satisfies
        `A E_gt = C` exactly and `ell_j^T E_gt = 0` for every law."""
        E = np.zeros((self.P, self.d))
        for e, v in self.meta["gt_ent"].items():
            E[self.tok_index[e]] = v
        for r, v in self.meta["gt_rel"].items():
            E[self.tok_index[r]] = v
        return E


# --------------------------------------------------------------------------- #
#  Family 1: relational lattices  (the general constructor)                    #
# --------------------------------------------------------------------------- #


def lattice_world(law_specs, d=16, seed=0):
    """
    Build a world out of one relational lattice per law.

    Entities are lattice points (i, j); relation `x` is the +1 step along the
    first axis, `y` the +1 step along the second, and the composite `z = x + y`
    is the diagonal.  Fitting the x-, y- and z-edges therefore *forces*
    `z = x + y`: composition is imposed by the geometry, not supervised.

    Each spec is a dict:
        name        law name
        x, y, z     relation token names (reuse across specs to share structure)
        m, n        lattice size          -> diversity of the law's instances
        rep         repetition multiplier -> evidence strength
        rep_x/rep_y per-axis repetition (default `rep`)
        z_rep       repetition for the diagonal edges (default `rep`)
        z_frac      fraction of diagonal cells shown in training; the rest are
                    withheld and serve as the generalisation test
        prefix      entity-name prefix (default `name`)

    Three corner entities per lattice are anchored, which fixes the coordinate
    frame and makes the relations individually identifiable.
    """
    rng = np.random.default_rng(seed)

    entities, relations, facts, laws = [], [], [], []
    anchors, gt_ent, gt_rel = {}, {}, {}
    facts_by_law, anchors_by_law = {}, {}

    rel_names = []
    for spec in law_specs:
        for r in (spec["x"], spec["y"], spec["z"]):
            if r not in rel_names:
                rel_names.append(r)

    composites = {spec["z"] for spec in law_specs}
    for r in rel_names:
        if r not in composites:
            gt_rel[r] = rng.standard_normal(d)
    for spec in law_specs:
        gt_rel[spec["z"]] = gt_rel[spec["x"]] + gt_rel[spec["y"]]
    relations = list(rel_names)

    for spec in law_specs:
        name = spec["name"]
        rx, ry, rz = spec["x"], spec["y"], spec["z"]
        m, n = spec.get("m", 3), spec.get("n", 3)
        rep = spec.get("rep", 1)
        rep_x, rep_y = spec.get("rep_x", rep), spec.get("rep_y", rep)
        z_rep = spec.get("z_rep", rep)
        z_frac = spec.get("z_frac", 1.0)
        prefix = spec.get("prefix", name)

        origin = rng.standard_normal(d)
        u, w = gt_rel[rx], gt_rel[ry]

        for i in range(m):
            for j in range(n):
                e = "%s_%d_%d" % (prefix, i, j)
                entities.append(e)
                gt_ent[e] = origin + i * u + j * w

        law_facts = []
        for i in range(m - 1):
            for j in range(n):
                a, b = "%s_%d_%d" % (prefix, i, j), "%s_%d_%d" % (prefix, i + 1, j)
                law_facts += [(a, rx, b)] * rep_x
        for i in range(m):
            for j in range(n - 1):
                a, b = "%s_%d_%d" % (prefix, i, j), "%s_%d_%d" % (prefix, i, j + 1)
                law_facts += [(a, ry, b)] * rep_y

        diag = [(i, j) for i in range(m - 1) for j in range(n - 1)]
        rng.shuffle(diag)
        n_train = max(1, int(round(z_frac * len(diag))))
        for i, j in diag[:n_train]:
            a, b = "%s_%d_%d" % (prefix, i, j), "%s_%d_%d" % (prefix, i + 1, j + 1)
            law_facts += [(a, rz, b)] * z_rep

        facts += law_facts
        facts_by_law[name] = law_facts

        corners = ["%s_0_0" % prefix]
        if m > 1:
            corners.append("%s_%d_0" % (prefix, m - 1))
        if n > 1:
            corners.append("%s_0_%d" % (prefix, n - 1))
        law_anchors = {c: gt_ent[c] for c in corners}
        anchors.update(law_anchors)
        anchors_by_law[name] = law_anchors

        laws.append(Law(name=name, x_rel=rx, y_rel=ry, z_rel=rz))

    return World(
        entities=entities,
        relations=relations,
        facts=facts,
        anchors=anchors,
        laws=laws,
        d=d,
        meta=dict(
            seed=seed,
            law_specs=law_specs,
            gt_ent=gt_ent,
            gt_rel=gt_rel,
            facts_by_law=facts_by_law,
            anchors_by_law=anchors_by_law,
        ),
    )


# --------------------------------------------------------------------------- #
#  H2 family: structural identifiability                                       #
# --------------------------------------------------------------------------- #


def identifiability_world(
    seed=0, K=8, k=4, closed_block="A", n_bridge=0, d=16, rep=1, ground_open_tail=False
):
    """
    Two structurally identical blocks (A, B).  Each has its own premise
    relations `x_b`, `y_b`, its own composite `z_b = x_b + y_b`, `K` premise
    triads, and contributes exactly `k` composite facts.

        CLOSED block  its composite facts sit on triads whose x- and y-premises
                      are also observed, closing an x-y-z triangle, so
                      `ell = e_x + e_y - e_z` lies in row(A):  IDENTIFIABLE.
        OPEN block    its composite facts sit on separate pairs that share no
                      entity with the premise structure and whose tails are
                      ungrounded, so `ell` has a component in null(A):
                      NOT IDENTIFIABLE.

    Swapping `closed_block` produces a world that is an exact token permutation
    of the other, hence identical |D|, anchors, relation vocabulary and
    eigenvalue spectrum.  Only *which* law is identifiable differs.

    n_bridge          composite facts added to the OPEN block on triads that do
                      carry their premises -- "linking" evidence, which moves
                      `ell` into the row space.
    ground_open_tail  anchor both ends of the open block's pairs instead of just
                      the head.  This places the second structure in the
                      premises' coordinate frame, which also makes the law
                      identifiable -- with no bridging fact and still no shared
                      entity.  Used as a control that the criterion is the row
                      space, not entity sharing.
    """
    rng = np.random.default_rng(1000 + seed)
    open_block = "B" if closed_block == "A" else "A"

    entities, relations, facts, laws = [], [], [], []
    anchors, gt_ent, gt_rel = {}, {}, {}

    for b in ("A", "B"):
        gx, gy = rng.standard_normal(d), rng.standard_normal(d)
        gt_rel["x" + b], gt_rel["y" + b], gt_rel["z" + b] = gx, gy, gx + gy
        relations += ["x" + b, "y" + b, "z" + b]
        for m in range(K):
            o = rng.standard_normal(d)
            e0, e1, e2 = ("%sT%d_%d" % (b, m, i) for i in range(3))
            entities += [e0, e1, e2]
            gt_ent[e0], gt_ent[e1], gt_ent[e2] = o, o + gx, o + gx + gy
            for _ in range(rep):
                facts.append((e0, "x" + b, e1))
                facts.append((e1, "y" + b, e2))
            anchors[e0] = gt_ent[e0]
        # two further anchors ground x_b and y_b themselves
        for e in ("%sT0_1" % b, "%sT0_2" % b):
            anchors[e] = gt_ent[e]
        laws.append(Law(name=b, x_rel="x" + b, y_rel="y" + b, z_rel="z" + b))

    cb, ob = closed_block, open_block

    for m in range(k):  # closed: on triads
        facts.append(("%sT%d_0" % (cb, m), "z" + cb, "%sT%d_2" % (cb, m)))

    for m in range(k):  # open: on loose pairs
        f0, f1 = "%sF%d_0" % (ob, m), "%sF%d_1" % (ob, m)
        entities += [f0, f1]
        o = rng.standard_normal(d)
        gt_ent[f0], gt_ent[f1] = o, o + gt_rel["z" + ob]
        anchors[f0] = gt_ent[f0]
        if ground_open_tail:
            anchors[f1] = gt_ent[f1]
        facts.append((f0, "z" + ob, f1))

    for m in range(n_bridge):  # linking evidence
        facts.append(("%sT%d_0" % (ob, m), "z" + ob, "%sT%d_2" % (ob, m)))

    return World(
        entities=entities,
        relations=relations,
        facts=facts,
        anchors=anchors,
        laws=laws,
        d=d,
        meta=dict(
            seed=seed,
            K=K,
            k=k,
            rep=rep,
            closed_block=cb,
            open_block=ob,
            n_bridge=n_bridge,
            ground_open_tail=ground_open_tail,
            gt_ent=gt_ent,
            gt_rel=gt_rel,
        ),
    )


# --------------------------------------------------------------------------- #
#  Family 2: the F1 and F3 experiment worlds, and the held-out sets            #
# --------------------------------------------------------------------------- #

D = 16
TOL = 1e-9

# Searched by `search_ladder.py`, under the regime the figure uses: `Z_FRAC` of
# each law's diagonals trained on, the rest withheld and scored by held-out
# composite retrieval.  Three things are traded.  Every law must emerge between
# 2,000 and 20,000 epochs at `lr_target = 0.03`.  Within that, the smallest gap
# between consecutive rung emergence times, in log space, is maximised.  And it
# must not buy that spacing with evidence: per-fact descent walks every
# constraint row once per epoch, so training cost is rows times epochs, and the
# world is capped at 2,000 rows with repetitions capped at 8.  Achieved gap
# 0.072 against an ideal of 0.143, in 1,820 rows, with sixteen laws spanning
# 2,800 to 19,500 epochs.  The window itself caps the spread at tenfold.
LADDER = (
    dict(m=5, n=4, rep_x=1, rep_y=1),
    dict(m=4, n=4, rep_x=2, rep_y=2),
    dict(m=4, n=4, rep_x=3, rep_y=4),
    dict(m=4, n=4, rep_x=4, rep_y=6),
    dict(m=5, n=4, rep_x=2, rep_y=8),
    dict(m=4, n=4, rep_x=1, rep_y=4),
    dict(m=4, n=5, rep_x=4, rep_y=8),
    dict(m=4, n=4, rep_x=3, rep_y=6),
)
Z_FRAC = 0.25

# Experiment 1, settled.  Measured on the ladder above at `lr_target = 0.03`,
# by integrating the assembled world at each depth.  All sixteen laws emerge at
# every depth; the budgets leave headroom above the slowest.
#
#   depth 1   2,800 to 19,500 epochs   spread 7.0x
#   depth 2     800 to  1,750 epochs   spread 2.2x
#   depth 3   1,100 to  1,450 epochs   spread 1.3x
#
# Depths 2 and 3 emerge EARLIER in epochs than depth 1, so all three fit one
# shared log axis.  The between-law spread compresses with depth, and it does
# so toward the same floor on an independently searched ladder, so that is a
# property of the dynamics rather than of this particular ladder.
F1_EPOCHS = {1: 30000, 2: 4000, 3: 4000}
F1_EVERY = {1: 100, 2: 25, 3: 25}

# Experiments 2 and 3, settled.  Both are staged: one fact is inserted at the
# switch and training continues on the same weights.  Plot them against epochs
# SINCE the switch, so the three depth columns align at zero even though their
# budgets differ by an order of magnitude.
#
# The switch is set per depth, at 1.5x the measured point where that depth's
# pre-intervention control has settled.  Forcing one absolute switch epoch
# everywhere would make depths 2 and 3 sit flat for thousands of epochs waiting
# for depth 1, and once the axes are aligned to the switch it buys nothing.
#
#   control settles          depth 1   depth 2   depth 3
#   F2 closed law resolved     6,500     1,000       900
#   F3 no-link retrieval at 0  5,500       600       650
#
# After the switch, F2's open law reaches full retrieval in 3,800 / 300 / 140
# epochs, but its composition error takes 29,300 / 1,720 / 160 to fall below
# 0.5, and the post-switch budget is set by the slower of the two.
#
# Those recovery figures must be measured AT the switch point that will be
# used.  An earlier pass measured them with the switch at 40,000 and applied
# them to a switch at 10,000; recovery was three times longer than that
# predicted, because an earlier switch leaves the shared relational geometry
# less settled and the open law's composite has further to travel.  The first
# world 0 run was budgeted from those stale numbers and three of its six arms
# ended before the composition error crossed 0.5.
#
# F3 is different, and deliberately not sized to a geometric target.  Its two
# measures dissociate: the eighty comparisons become behaviourally available
# 3,500 / 700 / 600 epochs after the linking fact, but the offset error takes
# far longer, and at depth 1 it does not converge at any affordable budget --
# 310,500 epochs to reach half a step, still 0.32 after 400,000.  Depth 1 is
# therefore capped at 100,000, where the error has fallen from 3.2 to 1.6.
# The unconverged depth 1 panel IS the depth result and should not be padded
# out to hide it.
#
# One caveat for the F3 caption: the no-link control has no geometric plateau.
# Its offset error drifts upward at every depth, 2.94 to 3.66 at depth 1 over
# 30,000 epochs and to 3.85 and 4.08 at depths 2 and 3, as the free coordinate
# creeps toward the minimum-norm solution.  The control line rises; it does not
# sit flat.
F2_SWITCH = {1: 10000, 2: 1500, 3: 1500}
F2_AFTER = {1: 40000, 2: 3000, 3: 1000}
F2_EVERY = {1: 50, 2: 10, 3: 10}

F3_SWITCH = {1: 8000, 2: 1000, 3: 1000}
F3_AFTER = {1: 100000, 2: 20000, 3: 8000}
F3_EVERY = {1: 250, 2: 50, 3: 25}


def emergence_world(seed=0, ladder=LADDER, z_frac=Z_FRAC, d=D, share_base=True):
    """F1.  Two laws per ladder rung, each on its own lattice, with `z_frac` of
    the diagonals trained on and the rest withheld as the generalisation set."""
    specs = []
    for bi, c in enumerate(ladder):
        for k in range(2):
            base = "b%d" % bi if share_base else "b%d_%d" % (bi, k)
            specs.append(
                dict(
                    name="L%d_%d" % (bi, k),
                    x=base,
                    y="y%d_%d" % (bi, k),
                    z="z%d_%d" % (bi, k),
                    prefix="B%dL%d" % (bi, k),
                    m=c["m"],
                    n=c["n"],
                    rep_x=c["rep_x"],
                    rep_y=c["rep_y"],
                    z_rep=c["rep_y"],
                    z_frac=z_frac,
                )
            )
    return lattice_world(specs, d=d, seed=seed)


def _name(pre, i, j):
    return "%s_%d_%d" % (pre, i, j)


def integration_world(seed=0, link=False, m=3, d=D):
    """F3.  Two copies of one `m x m` lattice sharing relations `x, y, z`.  The
    first is anchored; the second is not, so its position relative to the first
    is undetermined.  `link=True` adds the one fact that joins them, using an
    existing relation between two existing entities, so it adds no token and no
    degree of freedom."""
    rng = np.random.default_rng(seed)
    gt_r = {"x": rng.standard_normal(d), "y": rng.standard_normal(d)}
    gt_r["z"] = gt_r["x"] + gt_r["y"]
    ents, facts, anchors, gt_e = [], [], {}, {}

    def block(pre, anchored):
        o = rng.standard_normal(d)
        for i in range(m):
            for j in range(m):
                ents.append(_name(pre, i, j))
                gt_e[_name(pre, i, j)] = o + i * gt_r["x"] + j * gt_r["y"]
        for i in range(m):
            for j in range(m):
                if i + 1 < m:
                    facts.append((_name(pre, i, j), "x", _name(pre, i + 1, j)))
                if j + 1 < m:
                    facts.append((_name(pre, i, j), "y", _name(pre, i, j + 1)))
                if i + 1 < m and j + 1 < m:
                    facts.append((_name(pre, i, j), "z", _name(pre, i + 1, j + 1)))
        if anchored:
            for c in (_name(pre, 0, 0), _name(pre, 1, 0), _name(pre, 0, 1)):
                anchors[c] = gt_e[c]

    block("A", True)
    block("B", False)
    shift = gt_e[_name("A", m - 1, 0)] + gt_r["x"] - gt_e[_name("B", 0, 0)]
    for i in range(m):
        for j in range(m):
            gt_e[_name("B", i, j)] = gt_e[_name("B", i, j)] + shift
    if link:
        facts.append((_name("A", m - 1, 0), "x", _name("B", 0, 0)))

    return World(
        entities=ents,
        relations=list(gt_r),
        facts=facts,
        anchors=anchors,
        laws=[Law("Lz", "x", "y", "z")],
        d=d,
        meta=dict(gt_ent=gt_e, gt_rel=gt_r, m=m, link=link, blocks=("A", "B")),
    )


# --------------------------------------------------------------------------- #
#  Held-out sets, derived from the world                                       #
# --------------------------------------------------------------------------- #


def _shown(world):
    return {(h, r, t) for h, r, t in world.facts}


def held_composites(world, reference=None):
    """{law: [(head, tail)]} for every pair one composite step apart in ground
    truth whose composite fact is absent from training.

    `reference` is the world whose training set defines the exclusion.  Pass
    the POST-intervention world so the evaluation set does not move when a fact
    is inserted mid-run.
    """
    ref = world if reference is None else reference
    shown, gt = _shown(ref), world.meta["gt_ent"]
    names = list(gt)
    X = np.array([gt[e] for e in names])
    out = {}
    for law in world.laws:
        z = world.meta["gt_rel"][law.z_rel]
        pairs = []
        for ai, a in enumerate(names):
            near = np.where(np.linalg.norm(X - (X[ai] + z)[None, :], axis=1) < TOL)[0]
            for bi in near:
                b = names[bi]
                if (a, law.z_rel, b) not in shown:
                    pairs.append((a, b))
        out[law.name] = pairs
    return out


def law_entities(world, law):
    """The entities a law's own structure touches, which is the candidate pool
    its held-out queries should be ranked against.

    Ranking a query against every entity in the world makes the measure depend
    on how many unrelated structures the world happens to contain: in the
    sixteen-law emergence world that is 280 candidates of which about 94 per
    cent belong to other lattices, and a rank flip against one of them says
    nothing about the law.  The pool is the entities touched by facts using the
    law's own y or z relation, which is its lattice or block.  The x relation is
    deliberately excluded because rungs share it.
    """
    keep = {law.y_rel, law.z_rel}
    out = {h for h, r, _ in world.facts if r in keep}
    out |= {t for _, r, t in world.facts if r in keep}
    return out


def freed_coefficients(system, pairs):
    """Each pair's coefficient on the null direction a linking fact removes.

    The unlinked world has a one-dimensional null space: one global offset of
    the unanchored copy that the facts do not fix.  A pair's contrast is
    `e_b - e_a`, so its exposure to that freedom is `n_b - n_a`.  Pairs with a
    large coefficient are the ones the linking fact actually settles.
    """
    import numpy as np

    _, sv, Vt = np.linalg.svd(system.A)
    tol = max(system.A.shape) * np.finfo(float).eps * sv.max()
    N = Vt[int((sv > tol).sum()):]
    if not len(N):
        return np.zeros(len(pairs))
    n = N[0]
    ti = system.world.tok_index
    return np.array([n[ti[b]] - n[ti[a]] for a, b, _, _ in pairs], float)


def held_cross(world, reference=None):
    """[(head, tail, n_x, n_y)] for every across-block comparison, excluding
    any pair the reference world states as a fact.  `n_x` and `n_y` are the
    steps of `x` and `y` separating the two in ground truth."""
    ref = world if reference is None else reference
    shown, m = _shown(ref), world.meta["m"]
    out = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    a, b = _name("A", i, j), _name("B", k, l)
                    if any((a, r, b) in shown for r in world.relations):
                        continue
                    out.append((a, b, m + k - i, l - j))
    return out
