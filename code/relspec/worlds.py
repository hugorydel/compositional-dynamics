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
    entities: list                  # entity token names, in row order
    relations: list                 # relation token names, in row order
    facts: list                     # (head, relation, tail), with multiplicity
    anchors: dict                   # entity name -> fixed value in R^d
    laws: list                      # list[Law]
    d: int                          # embedding dimension
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
        for (i, j) in diag[:n_train]:
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

    return World(entities=entities, relations=relations, facts=facts,
                 anchors=anchors, laws=laws, d=d,
                 meta=dict(seed=seed, law_specs=law_specs,
                           gt_ent=gt_ent, gt_rel=gt_rel,
                           facts_by_law=facts_by_law,
                           anchors_by_law=anchors_by_law))


# --------------------------------------------------------------------------- #
#  H2 family: structural identifiability                                       #
# --------------------------------------------------------------------------- #

def identifiability_world(seed=0, K=8, k=4, closed_block="A", n_bridge=0,
                          d=16, rep=1, ground_open_tail=False):
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

    for m in range(k):                                   # closed: on triads
        facts.append(("%sT%d_0" % (cb, m), "z" + cb, "%sT%d_2" % (cb, m)))

    for m in range(k):                                   # open: on loose pairs
        f0, f1 = "%sF%d_0" % (ob, m), "%sF%d_1" % (ob, m)
        entities += [f0, f1]
        o = rng.standard_normal(d)
        gt_ent[f0], gt_ent[f1] = o, o + gt_rel["z" + ob]
        anchors[f0] = gt_ent[f0]
        if ground_open_tail:
            anchors[f1] = gt_ent[f1]
        facts.append((f0, "z" + ob, f1))

    for m in range(n_bridge):                            # linking evidence
        facts.append(("%sT%d_0" % (ob, m), "z" + ob, "%sT%d_2" % (ob, m)))

    return World(entities=entities, relations=relations, facts=facts,
                 anchors=anchors, laws=laws, d=d,
                 meta=dict(seed=seed, K=K, k=k, rep=rep,
                           closed_block=cb, open_block=ob, n_bridge=n_bridge,
                           ground_open_tail=ground_open_tail,
                           gt_ent=gt_ent, gt_rel=gt_rel))

# --------------------------------------------------------------------------- #
#  Family 2: the F1 and F3 experiment worlds, and the held-out sets            #
# --------------------------------------------------------------------------- #

D = 16
TOL = 1e-9

# Searched by `search_ladder.py` over 324 lattice shapes, under the regime the
# figure uses: `Z_FRAC` of each law's diagonals trained on, the rest withheld
# and scored by held-out composite retrieval.  The objective was maximin
# spacing of the eight rungs' predicted emergence times in log space, subject
# to every law emerging inside the epoch budget.  Rung times step by about
# 1.5x each, spanning 23.6x across the sixteen laws.
LADDER = (
    dict(m=4, n=4, rep_x=24, rep_y=24),
    dict(m=4, n=5, rep_x=24, rep_y=12),
    dict(m=4, n=4, rep_x=6, rep_y=24),
    dict(m=5, n=5, rep_x=24, rep_y=6),
    dict(m=4, n=5, rep_x=3, rep_y=24),
    dict(m=4, n=5, rep_x=4, rep_y=2),
    dict(m=5, n=4, rep_x=16, rep_y=1),
    dict(m=5, n=4, rep_x=1, rep_y=8),
)
Z_FRAC = 0.25


def emergence_world(seed=0, ladder=LADDER, z_frac=Z_FRAC, d=D,
                    share_base=True):
    """F1.  Two laws per ladder rung, each on its own lattice, with `z_frac` of
    the diagonals trained on and the rest withheld as the generalisation set."""
    specs = []
    for bi, c in enumerate(ladder):
        for k in range(2):
            base = "b%d" % bi if share_base else "b%d_%d" % (bi, k)
            specs.append(dict(name="L%d_%d" % (bi, k), x=base,
                              y="y%d_%d" % (bi, k), z="z%d_%d" % (bi, k),
                              prefix="B%dL%d" % (bi, k), m=c["m"], n=c["n"],
                              rep_x=c["rep_x"], rep_y=c["rep_y"],
                              z_rep=c["rep_y"], z_frac=z_frac))
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
                    facts.append((_name(pre, i, j), "z",
                                  _name(pre, i + 1, j + 1)))
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

    return World(entities=ents, relations=list(gt_r), facts=facts,
                 anchors=anchors, laws=[Law("Lz", "x", "y", "z")], d=d,
                 meta=dict(gt_ent=gt_e, gt_rel=gt_r, m=m, link=link,
                           blocks=("A", "B")))


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
            near = np.where(np.linalg.norm(X - (X[ai] + z)[None, :],
                                           axis=1) < TOL)[0]
            for bi in near:
                b = names[bi]
                if (a, law.z_rel, b) not in shown:
                    pairs.append((a, b))
        out[law.name] = pairs
    return out


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
