"""md: Lennard-Jones molecular dynamics, grown from an empty repository.

The fourth formation domain, the largest, and the only one with **no oracle in the
scoring path**. A frozen test suite is the whole of the reward: sixty-five test
functions over six files, one task each, and a task's prompt is that test's own
source. Nothing here is compared against a reference implementation, because asking
a system to grow software you had to write first proves nothing. Upstream validates
against c-testsuite, LLVM and Csmith -- assertions, not expected outputs -- and this
is the same shape at a size that finishes in an afternoon.

What comes out is a program you can point at a box of argon::

    python md.py --particles 108 --density 0.8 --temperature 1.2 --steps 2000 \
                 --thermostat 0.5 --trajectory traj.xyz --rdf 40

Why this domain exists. minilang showed the mechanism, stackvm gave the recursion
somewhere to go, jqx came out as usable software -- and all three are scored by
matching a reference's output, which code that merely parses can sometimes do. This
one cannot be bluffed: many of its tests are **invariants** rather than values. The
forces must sum to zero and must match a central difference of the energy, energy
and momentum must survive a trajectory, reversing the velocities must retrace it --
and one test asserts that a far-too-large timestep does *not* conserve energy, so an
implementation that fakes conservation fails.

Invariants alone are not enough, which a real run demonstrated: a field that returns
**zero everywhere** satisfies every one of them (zero sums to zero, zero is the
gradient of a constant, nothing moves so nothing drifts), and a run shipped exactly
that from a minimum-image expression whose ``// 1`` bound after the multiplication.
Forty-two of the then forty-four tests passed, and the only test that noticed was the
negative control.

So the suite is checked against :data:`BASELINES` -- six deliberately wrong
implementations that all have to die, and to at least two tests each, because one
test between a false 1.000 and the truth is not a margin. Writing that check found
two more holes immediately: a ``round``-based geometry, the one thing the spec
explicitly forbids, passed everything; and the centre-of-mass-corrected temperature
died to a single test.

The held-out tail is an **audit set**, not a validation split. Fifteen further
tests live outside the repository entirely -- ``frozen`` stops a file being written,
not read, and the executor is handed the source of the test it is failing, so an
audit test in the tree is one it can write to. They are injected only while a
held-out task is scored: a different box, a brute-force image search, a harmonic
period, two layers composed. Every *driven* test is part of the specification and
every one of them drives the search, which is the opposite of what a train/validation
split does and the reason the jqx run once stalled at 0.923.

What is faithful and what is a surrogate is as it is for the other three, and the
reasoning is written out on :mod:`examples.genesis._domain`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from ._delegation import Brief, Delegation, Edit
from ._suite import (PYTHON_MODULE_SKILL, TEST_FAILURE, TestSuite,
                     plan_delegations, reward_test)
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR, normalise

__all__ = ["BASELINES", "CASE_NOUN", "CONTRACTS", "FROZEN",
           "SUITE_ONLY_BASELINES", "GROUP_NOUN", "HELD_OUT_FRAC",
           "MD",
           "SCORING", "build_tasks", "initial_files", "suite_failures",
           "llm_executor", "llm_manager", "make_runner", "offline_executor",
           "offline_manager", "reference_tree", "reward", "suite_review"]

#: The specification, the test suite and the driver. Human-supplied, refused to
#: every proposal, and restored pristine before scoring.
FROZEN = ("spec/**", "tests/**", "md.py")

#: What is **pushed into every brief**, in this order. Upstream an agent pulls files
#: with a read tool, so what it may not write and what it is handed unasked are two
#: different sets: the whole suite in every prompt would be 12 kB of tests the
#: episode was not asked about, and the one test that is failing arrives in the task
#: prompt already.
CONTRACTS = ("spec/**", "md.py")

#: How the run is scored and what the header says about it.
SCORING = "frozen test suite, no reference implementation in the scoring path"
CASE_NOUN = "test functions"
GROUP_NOUN = "test files"

ENTRY = "src/__init__.py"
CORE_INIT = "src/core/__init__.py"
VECTORS = "src/core/vectors.py"
STATE = "src/core/state.py"
POT_INIT = "src/potentials/__init__.py"
REGISTRY = "src/potentials/registry.py"
PAIR_INIT = "src/potentials/pair/__init__.py"
LJ = "src/potentials/pair/lennard_jones.py"
HARMONIC = "src/potentials/pair/harmonic.py"
INT_INIT = "src/integrate/__init__.py"
VERLET = "src/integrate/verlet.py"
OBS_INIT = "src/observe/__init__.py"
THERMO = "src/observe/thermo.py"
RDF = "src/observe/rdf.py"


# ---------------------------------------------------------------------------
# The human's repository: the physics, the decomposition, the tests, the driver
# ---------------------------------------------------------------------------

_SPEC = r'''# md -- the molecular dynamics this project implements

Lennard-Jones molecular dynamics in **reduced units**: sigma = epsilon = mass =
k_B = 1, so energies, lengths and temperatures are all dimensionless.

## Geometry

Periodic boundaries on every axis with a positive box length. A box length of `0`
means that axis is not periodic. The displacement from `a` to `b` is the
**nearest image**:

    d[i] = (b[i] - a[i]) - L[i] * floor((b[i] - a[i]) / L[i] + 0.5)

`floor(x + 0.5)`, never `round`: Python's `round` is banker's rounding, so two
implementations disagree exactly at half a box. A position is wrapped into
`[0, L)` with `x - L * floor(x / L)`.

## Pair interactions

A pair kernel takes the **squared** distance and returns `(u, dudr_over_r)`. The
caller puts the force on particle *i* at `dudr_over_r * d` and on *j* at its
negative, where `d` is the displacement from *i* to *j*.

**`lennard_jones`**, shifted so that `u(cutoff) == 0`, and zero beyond it:

    s6  = (sigma**2 / r2) ** 3          s12 = s6 * s6
    u            = 4*eps*(s12 - s6) - 4*eps*((sigma/rc)**12 - (sigma/rc)**6)
    dudr_over_r  = -24*eps*(2*s12 - s6) / r2

**`harmonic`**, with no cutoff, `u = 0.5 * k * (r - r0)**2`:

    dudr_over_r  = k * (r - r0) / r          (and `(0.0, 0.0)` at r == 0)

Energies and forces sum over every **distinct** pair once, which is what makes
the forces sum to zero. An unknown potential name must raise.

## Integration

Velocity Verlet, one step with timestep `dt`:

1. `v += 0.5 * dt * f / m` with the forces at the current positions;
2. `x += dt * v`, then **wrap** `x` into the box;
3. recompute the forces;
4. `v += 0.5 * dt * f / m`.

Recomputing the forces between the two half-kicks is the whole point: it is what
makes the scheme second-order and time-reversible. One force evaluation, or both
kicks with the same forces, is a different and wrong integrator.

## The public surface

`src/__init__.py` exposes exactly these seven functions. Nothing outside `src/`
reaches past them -- not the driver, and not a single test.

These are the signatures, exactly -- every keyword is part of the surface even
where a caller rarely overrides it, and the tests pass them **by name**:

    def displacement(a, b, box)
    def wrap(point, box)

    def energy(positions, box=(0.0, 0.0, 0.0), potential="lennard_jones",
               epsilon=1.0, sigma=1.0, cutoff=2.5, k=1.0, r0=1.0)
    def forces(positions, box=(0.0, 0.0, 0.0), potential="lennard_jones",
               epsilon=1.0, sigma=1.0, cutoff=2.5, k=1.0, r0=1.0)
    def step(positions, velocities, masses, dt, box=(0.0, 0.0, 0.0),
             potential="lennard_jones", epsilon=1.0, sigma=1.0, cutoff=2.5,
             k=1.0, r0=1.0)

    def observables(velocities, masses)
    def rdf(positions, box, bins, rmax)

`energy` returns a float; `forces` one vector per particle; `step` the pair
`(positions, velocities)` after one timestep; `rdf` a list of **integer** counts.
`observables` returns `{"kinetic": f, "temperature": f, "momentum": [f, f, f]}` --
it takes no positions and no box, with `KE = sum(0.5 * m * v.v)`,
`T = 2 * KE / (3 * N)` at `k_B = 1` and **no** centre-of-mass correction, and
`momentum` the plain mass-weighted sum. A pair at or beyond `rmax` is not counted
in the histogram.

The keyword set reaches all the way down: a layer that accepts `potential` and
then hard-codes Lennard-Jones fails, and so does one that takes `cutoff` but not
`epsilon`. Thread the parameters through rather than rediscovering them.

Every entry point MUST import its layer **lazily**, inside the function body.
The tests for one layer then pass while another layer is still missing, which is
how a partly-grown repository scores at all.

## How this is scored

`tests/` is frozen, and it is the whole of the reward: one task per test
function, and the fraction that pass. There is no reference implementation
anywhere in the loop -- the tests *are* the specification, made executable, and
the prompt you are shown for a unit of work is the source of the test that is
failing. Read it. It names the behaviour exactly, including the tolerance.

Several of the tests are **invariants** rather than values: the forces sum to
zero, every force component matches a central difference of the energy, energy
and momentum survive a trajectory, and reversing the velocities retraces it.
They cannot be satisfied by a lookup table, and one of them deliberately asserts
that a far-too-large timestep does *not* conserve energy -- so an implementation
that fakes conservation fails it.

Others pin **values in a periodic box**, because every invariant above is also
satisfied by a force field that returns zero everywhere. If `energy(..., box=L)`
comes out at zero for particles a sigma apart, the minimum image is wrong, however
many invariants still pass.
'''

_ROOT_CONTEXT = r'''# md -- root

## Intent
Implement the molecular dynamics in `spec/CONTEXT.md` as a library under `src/`.
`md.py` is the simulation driver and is already written; `tests/` is the suite
your work is scored against.

## API Surface
`md.py` -- the command-line driver. It imports `step`, `observables` and `rdf` from
`src` and nothing else.

## Routing Table
- `./src/` -> the library, and the seven public entry points

## Constraints
- `spec/`, `tests/` and `md.py` are read-only. They are the contract, not work
  items. A system that can edit its own tests has no tests.
- Pure Python, no third-party packages: this has to run anywhere.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = r'''# src -- library root

## Intent
Own `src/__init__.py`: the seven entry points `spec/CONTEXT.md` names, each a thin
wrapper that calls into a child node. Each MUST import its layer **lazily**, inside
the function body, so a missing layer does not stop the layers before it from
passing their tests.

## API Surface
`src/__init__.py` exposes exactly seven names, with the signatures
`spec/CONTEXT.md` gives verbatim: `displacement(a, b, box)`, `wrap(point, box)`,
`energy(positions, **params)`, `forces(positions, **params)`,
`step(positions, velocities, masses, dt, **params)`, `observables(velocities, masses)`,
`rdf(positions, box, bins, rmax)`. Nothing else is public, and the keyword set reaches
every layer below.

## Routing Table
- `./src/core/`       -> geometry and the system container
- `./src/potentials/` -> energies and forces
- `./src/integrate/`  -> advancing time
- `./src/observe/`    -> measuring
'''

_CORE_CONTEXT = r'''# src/core -- geometry and state

## Intent
`vectors.py` owns periodic geometry: the nearest-image displacement, squared
length, length, and wrapping a point into the box. Everything else in the project
goes through it rather than writing `floor` arithmetic of its own.

`state.py` owns the `System` container -- positions, velocities, masses, box, with
velocities defaulting to zeros and masses to ones -- and `make_spec`, which bundles
the potential parameters the pair kernels are handed.

## API Surface
- `vectors.py`: `minimum_image(a, b, box)`, `norm2(v)`, `norm(v)`, `wrap(point, box)`
- `state.py`: `System(positions, velocities=None, masses=None, box=(0,0,0))` with
  `.positions`, `.velocities`, `.masses`, `.box`, `__len__`, `copy()`;
  `make_spec(potential, epsilon, sigma, cutoff, k, r0) -> dict`
'''

_POT_CONTEXT = r'''# src/potentials -- energies and forces

## Intent
Own `registry.py`: the name-to-kernel table, the refusal of an unknown name, and
the loop over distinct pairs that turns a kernel into a total energy and a force
array. The loop applies Newton's third law once per pair, which is what makes the
forces sum to zero by construction rather than by luck.

## API Surface
- `registry.py`: `get(name) -> kernel` (raises on an unknown name),
  `evaluate(system, spec) -> (energy, forces)`

## Routing Table
- `./src/potentials/pair/` -> one module per pair interaction
'''

_PAIR_CONTEXT = r'''# src/potentials/pair -- the pair interactions

## Intent
One module per interaction, each exposing `pair(r2, spec) -> (u, dudr_over_r)`
exactly as the spec defines it. `lennard_jones.py` is shifted and cut off;
`harmonic.py` has no cutoff. Neither knows anything about boxes or loops.

## API Surface
- `lennard_jones.py`: `pair(r2, spec) -> (u, dudr_over_r)`
- `harmonic.py`: `pair(r2, spec) -> (u, dudr_over_r)`
'''

_INT_CONTEXT = r'''# src/integrate -- advancing time

## Intent
`verlet.py` owns velocity Verlet: `step` advances one timestep in place and returns
the system, `run` applies it `steps` times. The position update wraps back into the
box, and the forces are recomputed between the two half-kicks -- that recomputation
is what makes the scheme reversible, and the suite tests reversibility directly.

## API Surface
- `verlet.py`: `step(system, spec, dt) -> system` (in place),
  `run(system, spec, dt, steps) -> system`
'''

_OBS_CONTEXT = r'''# src/observe -- measuring

## Intent
`thermo.py` owns the thermodynamic observables: kinetic energy, temperature and
total momentum, in reduced units with k_B = 1 and no centre-of-mass correction.
`rdf.py` owns the pair-distance histogram, in raw integer counts of distinct pairs,
and takes its distances from `src/core/vectors.py` rather than recomputing them.

## API Surface
- `thermo.py`: `kinetic(velocities, masses)`, `temperature(velocities, masses)`,
  `momentum(velocities, masses)`
- `rdf.py`: `histogram(system, bins, rmax) -> [int]`
'''

_DRIVER = r'''#!/usr/bin/env python3
"""md -- a Lennard-Jones molecular dynamics driver.

Human-supplied, frozen, and never written by an agent: the library under ``src/``
is what the run grows, and this is the shell around it so the result is a program
you can run rather than a package nobody can invoke.

    python md.py --particles 32 --steps 500
    python md.py --particles 108 --density 0.8 --temperature 1.2 --steps 2000 \
                 --thermostat 0.2 --trajectory traj.xyz
    python md.py --particles 32 --steps 200 --rdf 40

Reduced Lennard-Jones units throughout: sigma = epsilon = mass = k_B = 1.

It touches the library only through the three entry points ``spec/CONTEXT.md``
pins -- ``step``, ``observables``, ``rdf`` -- so it runs against any
implementation that passes ``tests/``, whatever the internal layout turns out to
be. The kernel is pure Python and O(N^2), so keep the defaults small and do not
expect numpy speed.
"""

import argparse
import math
import random
import sys

#: The four-atom face-centred-cubic basis, in cell units.
_FCC_BASIS = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


def fcc_lattice(cells, density):
    """A full FCC lattice of ``4 * cells**3`` atoms at ``density``.

    FCC rather than simple cubic, and *full* rather than partly filled, because
    both shortcuts break the physics rather than the code. A simple-cubic lattice
    at rho=0.8 puts neighbours 0.855 sigma apart -- inside the repulsive core --
    so the run converts a huge potential energy into heat and reports T=22 from a
    T=1 start. FCC at the same density puts them at a/sqrt(2) = 1.21 sigma.
    """
    a = (4.0 / density) ** (1.0 / 3.0)
    points = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for bx, by, bz in _FCC_BASIS:
                    points.append([(i + bx) * a, (j + by) * a, (k + bz) * a])
    return points, [cells * a] * 3


def fcc_cells(requested):
    """The smallest FCC side giving at least ``requested`` atoms."""
    cells = 1
    while 4 * cells ** 3 < requested:
        cells += 1
    return cells


def maxwell_velocities(n, temperature, seed):
    """Gaussian velocities at ``temperature``, with the drift removed."""
    rng = random.Random(seed)
    sigma = math.sqrt(temperature)
    velocities = [[rng.gauss(0.0, sigma) for _ in range(3)] for _ in range(n)]
    drift = [sum(v[axis] for v in velocities) / n for axis in range(3)]
    for v in velocities:
        for axis in range(3):
            v[axis] -= drift[axis]
    return velocities


def rescale(velocities, target, current, strength):
    """Velocity rescaling, blended: ``strength=1`` hits the target exactly."""
    if current <= 0.0 or target <= 0.0 or strength <= 0.0:
        return velocities
    factor = math.sqrt(1.0 + strength * (target / current - 1.0))
    return [[c * factor for c in v] for v in velocities]


def build_parser():
    p = argparse.ArgumentParser(
        prog="md", description="Lennard-Jones molecular dynamics in reduced units.")
    p.add_argument("-n", "--particles", type=int, default=32,
                   help="rounded up to a full FCC lattice: 4, 32, 108, 256, ...")
    p.add_argument("-d", "--density", type=float, default=0.8)
    p.add_argument("-T", "--temperature", type=float, default=1.0,
                   help="initial (and, with --thermostat, target) temperature")
    p.add_argument("-s", "--steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--thermostat", type=float, default=0.0, metavar="STRENGTH",
                   help="velocity rescaling, 0 = off (constant energy), 1 = every "
                        "logged step rescaled exactly to the target")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--trajectory", metavar="FILE", help="write an XYZ trajectory")
    p.add_argument("--rdf", type=int, default=0, metavar="BINS",
                   help="print a pair-distance histogram over [0, cutoff) at the end")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.particles < 2:
        print("md: need at least two particles", file=sys.stderr)
        return 2

    from src import observables, rdf, step

    cells = fcc_cells(args.particles)
    positions, box = fcc_lattice(cells, args.density)
    count = len(positions)
    if count != args.particles:
        print(f"# {args.particles} rounded up to {count} "
              f"(a full {cells}x{cells}x{cells} FCC lattice)")
    velocities = maxwell_velocities(count, args.temperature, args.seed)
    masses = [1.0] * count

    trajectory = (open(args.trajectory, "w", encoding="utf-8")
                  if args.trajectory else None)
    print(f"# {count} particles, box {box[0]:.4f}, density {args.density}, "
          f"dt {args.dt}, cutoff {args.cutoff}"
          + (f", thermostat {args.thermostat} -> T={args.temperature}"
             if args.thermostat else ", constant energy"))
    print(f"# {'step':>8} {'T':>12} {'KE':>14} {'|P|':>12}")
    try:
        for n in range(args.steps + 1):
            if n % args.log_every == 0 or n == args.steps:
                obs = observables(velocities, masses)
                speed = math.sqrt(sum(c * c for c in obs["momentum"]))
                print(f"  {n:>8} {obs['temperature']:>12.6f} "
                      f"{obs['kinetic']:>14.6f} {speed:>12.3e}")
                if trajectory is not None:
                    trajectory.write(f"{count}\nstep {n}\n")
                    for p in positions:
                        trajectory.write(f"Ar {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                if args.thermostat:
                    velocities = rescale(velocities, args.temperature,
                                         obs["temperature"], args.thermostat)
            if n == args.steps:
                break
            positions, velocities = step(positions, velocities, masses, args.dt,
                                         box=box, cutoff=args.cutoff)
    finally:
        if trajectory is not None:
            trajectory.close()

    if args.rdf:
        counts = rdf(positions, box, args.rdf, args.cutoff)
        width = args.cutoff / args.rdf
        print(f"# pair-distance histogram, {args.rdf} bins over [0, {args.cutoff})")
        peak = max(counts) or 1
        for i, bin_count in enumerate(counts):
            bar = "#" * int(40 * bin_count / peak)
            print(f"  {i * width:6.3f}-{(i + 1) * width:6.3f} {bin_count:>6} {bar}")
    if args.trajectory:
        print(f"# wrote {args.trajectory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: The suite. Frozen, in the repository, and the whole of the reward.
_TESTS = {
    'tests/test_forces.py': r'''"""The force field as a whole: invariants no correct implementation can miss."""

from src import energy, forces

BOX = [6.0, 6.0, 6.0]
CONFIG = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]]


def test_forces_sum_to_zero():
    """Newton's third law, applied once per pair."""
    total = [sum(f[axis] for f in forces(CONFIG, box=BOX)) for axis in range(3)]
    assert max(abs(c) for c in total) < 1e-9, total


def test_every_force_component_matches_a_central_difference():
    """The force is minus the gradient of the energy. This is the real test."""
    analytic = forces(CONFIG, box=BOX)
    h = 1e-6
    for i in range(len(CONFIG)):
        for axis in range(3):
            plus = [list(p) for p in CONFIG]
            minus = [list(p) for p in CONFIG]
            plus[i][axis] += h
            minus[i][axis] -= h
            numeric = -(energy(plus, box=BOX) - energy(minus, box=BOX)) / (2.0 * h)
            assert abs(numeric - analytic[i][axis]) < 1e-4, (i, axis, numeric,
                                                            analytic[i][axis])


def test_the_harmonic_force_also_matches_its_gradient():
    kw = {"box": BOX, "potential": "harmonic", "k": 2.0, "r0": 1.0}
    analytic = forces(CONFIG, **kw)
    h = 1e-6
    for i in (0, 2):
        plus = [list(p) for p in CONFIG]
        minus = [list(p) for p in CONFIG]
        plus[i][1] += h
        minus[i][1] -= h
        numeric = -(energy(plus, **kw) - energy(minus, **kw)) / (2.0 * h)
        assert abs(numeric - analytic[i][1]) < 1e-4, (i, numeric, analytic[i][1])


def test_energy_does_not_change_when_everything_moves_together():
    shifted = [[c + 0.37 for c in p] for p in CONFIG]
    assert abs(energy(CONFIG, box=BOX) - energy(shifted, box=BOX)) < 1e-9


def test_energy_does_not_change_when_a_particle_crosses_the_boundary():
    """Periodicity: moving one particle by a whole box is the same configuration."""
    moved = [list(p) for p in CONFIG]
    moved[1][0] += BOX[0]
    assert abs(energy(CONFIG, box=BOX) - energy(moved, box=BOX)) < 1e-9


def test_an_isolated_pair_matches_the_closed_form():
    r = 1.3
    s6 = (1.0 / r) ** 6
    expected = 4.0 * (s6 * s6 - s6) - 4.0 * ((1.0 / 2.5) ** 12 - (1.0 / 2.5) ** 6)
    got = energy([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], box=[0.0, 0.0, 0.0])
    assert abs(got - expected) < 1e-9, (got, expected)


def test_pairs_beyond_the_cutoff_contribute_nothing():
    near = energy([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], box=[0.0, 0.0, 0.0])
    with_far = energy([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [40.0, 0.0, 0.0]],
                      box=[0.0, 0.0, 0.0])
    assert abs(near - with_far) < 1e-12, (near, with_far)


def test_a_single_particle_feels_nothing():
    assert energy([[1.0, 1.0, 1.0]], box=BOX) == 0.0
    assert forces([[1.0, 1.0, 1.0]], box=BOX) == [[0.0, 0.0, 0.0]]


def test_an_unknown_potential_is_refused():
    try:
        energy(CONFIG, box=BOX, potential="not_a_potential")
    except Exception:
        return
    raise AssertionError("an unknown potential name should raise")


def lj_closed_form(r, cutoff=2.5):
    """u(r) for epsilon = sigma = 1, shifted at `cutoff`."""
    s6 = (1.0 / r) ** 6
    sc6 = (1.0 / cutoff) ** 6
    return 4.0 * (s6 * s6 - s6) - 4.0 * (sc6 * sc6 - sc6)


def test_the_field_in_a_periodic_box_is_not_identically_zero():
    """Non-vacuity, stated as a test because it had to be.

    Every invariant above is satisfied by a field that returns zero everywhere:
    zero sums to zero, zero is the gradient of a constant, and nothing moves so
    nothing drifts. A run produced exactly that -- a minimum-image expression whose
    `// 1` bound after the multiplication, computing `floor(d + L/2)` instead of
    `L * floor(d/L + 0.5)`, which pushed every pair in a box past the cutoff. Two
    tests out of forty-four noticed.
    """
    u = energy(CONFIG, box=BOX)
    f = forces(CONFIG, box=BOX)
    # 1.1 and 1.2 sigma are just past the minimum at 2**(1/6), so this
    # configuration is bound: the energy is around -2 and the forces are order 1.
    assert u < -0.5, u
    assert max(abs(c) for vector in f for c in vector) > 1.0, f


def test_a_periodic_pair_in_the_middle_of_the_box_matches_the_closed_form():
    """A box must not change a pair that is nowhere near a boundary."""
    got = energy([[3.0, 3.0, 3.0], [4.1, 3.0, 3.0]], box=BOX)
    assert abs(got - lj_closed_form(1.1)) < 1e-9, (got, lj_closed_form(1.1))


def test_a_periodic_pair_across_the_seam_matches_the_closed_form():
    """5.9 to 0.3 is 0.4 the short way and 5.6 the long way."""
    got = energy([[5.9, 2.0, 2.0], [0.3, 2.0, 2.0]], box=BOX)
    assert abs(got - lj_closed_form(0.4)) < 1e-6, (got, lj_closed_form(0.4))


def test_a_pair_in_a_large_box_is_the_same_as_a_pair_in_no_box():
    """Periodicity that cannot be reached is periodicity that changes nothing."""
    pair = [[10.0, 10.0, 10.0], [11.3, 10.0, 10.0]]
    assert abs(energy(pair, box=[20.0, 20.0, 20.0])
               - energy(pair, box=[0.0, 0.0, 0.0])) < 1e-12
''',
    'tests/test_geometry.py': r'''"""Periodic geometry. Nothing here needs a force field."""

from src import displacement, wrap

BOX = [6.0, 6.0, 6.0]


def test_displacement_of_nearby_points_is_the_plain_difference():
    assert displacement([1.0, 2.0, 3.0], [1.5, 2.25, 3.5], BOX) == [0.5, 0.25, 0.5]


def test_displacement_takes_the_nearest_image_across_a_boundary():
    # 0.1 and 5.6 are 5.5 apart the long way and 0.5 apart the short way.
    d = displacement([0.1, 0.0, 0.0], [5.6, 0.0, 0.0], BOX)
    assert abs(d[0] - (-0.5)) < 1e-12, d


def test_displacement_is_antisymmetric():
    # No separation is exactly half a box: there the nearest image is genuinely
    # ambiguous and the formula has to pick a side, so antisymmetry does not hold.
    a, b = [0.3, 5.9, 1.0], [5.7, 0.2, 2.4]
    forward, backward = displacement(a, b, BOX), displacement(b, a, BOX)
    for i in range(3):
        assert abs(forward[i] + backward[i]) < 1e-12, (forward, backward)


def test_displacement_never_exceeds_half_a_box():
    for x in (0.0, 0.7, 2.9, 3.1, 5.5):
        d = displacement([0.0, 0.0, 0.0], [x, 0.0, 0.0], BOX)
        assert abs(d[0]) <= 3.0 + 1e-12, (x, d)


def test_a_zero_box_length_means_no_periodicity():
    d = displacement([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert d[0] == 100.0


def test_wrap_folds_a_point_into_the_box():
    assert wrap([6.5, -0.5, 3.0], BOX) == [0.5, 5.5, 3.0]


def test_wrap_leaves_an_interior_point_alone():
    assert wrap([1.0, 2.0, 3.0], BOX) == [1.0, 2.0, 3.0]


def test_a_separation_of_exactly_half_a_box_takes_the_negative_image():
    """The one place two correct-looking implementations disagree.

    This is why the spec pins `floor(d/L + 0.5)` and forbids `round`: Python's round
    is banker's rounding, so `round(0.5)` is 0 and `round(2.5)` is 2, and the sign of
    the image comes out the other way. Nothing else in the suite can tell the two
    apart -- a `round`-based geometry passed every other test here.
    """
    assert displacement([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], BOX)[0] == -3.0
    assert displacement([0.0, 0.0, 0.0], [15.0, 0.0, 0.0], BOX)[0] == -3.0
''',
    'tests/test_integration.py': r'''"""Velocity Verlet. The invariants here are what makes an integrator correct."""

from src import energy, observables, step

BOX = [6.0, 6.0, 6.0]
CONFIG = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]]
VEL = [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]
MASS = [1.0, 1.0, 1.0, 1.0]


def total_energy(positions, velocities):
    return (energy(positions, box=BOX)
            + observables(velocities, MASS)["kinetic"])


def run(positions, velocities, dt, steps):
    for _ in range(steps):
        positions, velocities = step(positions, velocities, MASS, dt, box=BOX)
    return positions, velocities


def test_a_free_particle_travels_at_constant_velocity():
    p, v = step([[1.0, 1.0, 1.0]], [[0.5, 0.0, 0.0]], [1.0], 0.1, box=BOX)
    assert abs(p[0][0] - 1.05) < 1e-12, p
    assert abs(v[0][0] - 0.5) < 1e-12, v


def test_energy_is_conserved_over_a_short_run():
    before = total_energy(CONFIG, VEL)
    p, v = run(CONFIG, VEL, 0.001, 40)
    after = total_energy(p, v)
    assert abs(after - before) / max(1.0, abs(before)) < 1e-4, (before, after)


def test_momentum_is_conserved_over_a_short_run():
    before = observables(VEL, MASS)["momentum"]
    _, v = run(CONFIG, VEL, 0.001, 40)
    after = observables(v, MASS)["momentum"]
    assert max(abs(after[a] - before[a]) for a in range(3)) < 1e-9, (before, after)


def test_a_far_too_large_timestep_does_not_conserve_energy():
    """The conservation tests must be able to fail, or they assert nothing."""
    before = total_energy(CONFIG, VEL)
    p, v = run(CONFIG, VEL, 0.25, 40)
    after = total_energy(p, v)
    assert abs(after - before) / max(1.0, abs(before)) > 1e-3, (before, after)


def test_reversing_the_velocities_retraces_the_trajectory():
    """Velocity Verlet is time-reversible, which no wrong update order is."""
    forward_p, forward_v = run(CONFIG, VEL, 0.002, 25)
    back_p, _ = run(forward_p, [[-c for c in v] for v in forward_v], 0.002, 25)
    for i in range(len(CONFIG)):
        for axis in range(3):
            gap = abs(back_p[i][axis] - CONFIG[i][axis]) % BOX[axis]
            assert min(gap, BOX[axis] - gap) < 1e-6, (i, axis, back_p[i], CONFIG[i])


def test_positions_come_back_inside_the_box():
    p, _ = step([[5.99, 0.01, 3.0]], [[1.0, -1.0, 0.0]], [1.0], 0.05, box=BOX)
    for axis in range(3):
        assert 0.0 <= p[0][axis] < BOX[axis], p


def test_a_heavier_particle_accelerates_less():
    # No periodicity, so the comparison is a displacement rather than a wrapped
    # coordinate; and a fresh velocity list per particle, because `[[0.0]*3]*2`
    # is the same list twice and the two updates would land on each other.
    start = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    rest = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    free = [0.0, 0.0, 0.0]
    light, _ = step(start, [list(free), list(free)], [1.0, 1.0], 0.01,
                    box=[0.0, 0.0, 0.0])
    heavy, _ = step(start, [list(free), list(free)], [4.0, 1.0], 0.01,
                    box=[0.0, 0.0, 0.0])
    assert rest is not None
    assert abs(heavy[0][0]) < abs(light[0][0]) - 1e-9, (light[0], heavy[0])


def test_zero_steps_change_nothing():
    p, v = run(CONFIG, VEL, 0.001, 0)
    assert p == CONFIG and v == VEL
''',
    'tests/test_observables.py': r'''"""Thermodynamics and structure, in reduced units with k_B = 1."""

from src import observables, rdf


def test_kinetic_energy_of_one_particle():
    obs = observables([[2.0, 0.0, 0.0]], [3.0])
    assert abs(obs["kinetic"] - 6.0) < 1e-12, obs


def test_kinetic_energy_adds_over_particles():
    obs = observables([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [1.0, 1.0])
    assert abs(obs["kinetic"] - 2.5) < 1e-12, obs


def test_temperature_is_two_thirds_of_the_kinetic_energy_per_particle():
    velocities = [[1.0, 1.0, 1.0], [-1.0, 0.0, 0.5]]
    obs = observables(velocities, [1.0, 1.0])
    assert abs(obs["temperature"] - 2.0 * obs["kinetic"] / 6.0) < 1e-12, obs


def test_momentum_of_opposing_particles_cancels():
    obs = observables([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], [1.0, 2.0])
    assert max(abs(c) for c in obs["momentum"]) < 1e-12, obs


def test_momentum_is_mass_weighted():
    obs = observables([[1.0, 0.0, 0.0]], [2.5])
    assert abs(obs["momentum"][0] - 2.5) < 1e-12, obs


def test_a_system_at_rest_has_no_temperature():
    obs = observables([[0.0, 0.0, 0.0]] * 3, [1.0] * 3)
    assert obs["kinetic"] == 0.0 and obs["temperature"] == 0.0


def test_the_histogram_counts_every_distinct_pair_once():
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    counts = rdf(positions, [0.0, 0.0, 0.0], 10, 5.0)
    assert sum(counts) == 3, counts


def test_the_histogram_ignores_pairs_at_or_beyond_rmax():
    positions = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert sum(rdf(positions, [0.0, 0.0, 0.0], 10, 2.0)) == 0


def test_the_histogram_puts_a_distance_in_the_right_bin():
    positions = [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]
    counts = rdf(positions, [0.0, 0.0, 0.0], 4, 2.0)   # bins 0.5 wide
    assert counts == [0, 0, 1, 0], counts


def test_the_histogram_uses_the_nearest_image():
    positions = [[0.1, 0.0, 0.0], [5.9, 0.0, 0.0]]     # 0.2 apart across the seam
    counts = rdf(positions, [6.0, 6.0, 6.0], 6, 3.0)   # bins 0.5 wide
    assert counts[0] == 1 and sum(counts) == 1, counts


def test_temperature_divides_by_three_n_and_not_by_the_degrees_of_freedom():
    """`T = 2 KE / (3N)`, with no centre-of-mass correction.

    The corrected form divides by `3N - 3`, which differs by `N / (N - 1)` -- 1.5 at
    N = 3. A ratio test between two temperatures cancels the factor and cannot see
    it, so the number is pinned here.
    """
    obs = observables([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1.0] * 3)
    assert abs(obs["kinetic"] - 1.5) < 1e-12, obs
    assert abs(obs["temperature"] - 1.0 / 3.0) < 1e-12, obs      # 2 * 1.5 / 9, not / 6
''',
    'tests/test_pair_potentials.py': r'''"""The pair interactions, probed through two-particle configurations."""

from src import energy, forces

FAR = [0.0, 0.0, 0.0]          # no periodicity: an isolated pair


def pair_energy(r, **kw):
    return energy([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], box=FAR, **kw)


def test_lennard_jones_is_zero_at_the_cutoff():
    """The potential is shifted, so it reaches the cutoff continuously."""
    assert abs(pair_energy(2.5, cutoff=2.5)) < 1e-12, pair_energy(2.5, cutoff=2.5)


def test_lennard_jones_is_zero_beyond_the_cutoff():
    assert pair_energy(3.0, cutoff=2.5) == 0.0


def test_lennard_jones_minimum_is_at_the_sixth_root_of_two():
    """u(2**(1/6) sigma) = -epsilon, the textbook minimum, before the shift."""
    r_min = 2.0 ** (1.0 / 6.0)
    shift_free = pair_energy(r_min, cutoff=1e6)
    assert abs(shift_free - (-1.0)) < 1e-9, shift_free


def test_lennard_jones_is_repulsive_inside_sigma():
    assert pair_energy(0.9, cutoff=1e6) > 0.0


def test_the_force_vanishes_at_the_lennard_jones_minimum():
    r_min = 2.0 ** (1.0 / 6.0)
    f = forces([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]], box=FAR, cutoff=1e6)
    assert abs(f[0][0]) < 1e-7, f


def test_particles_inside_sigma_push_apart():
    f = forces([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], box=FAR, cutoff=1e6)
    assert f[0][0] < 0.0 and f[1][0] > 0.0, f


def test_particles_outside_the_minimum_pull_together():
    f = forces([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], box=FAR, cutoff=2.5)
    assert f[0][0] > 0.0 and f[1][0] < 0.0, f


def test_harmonic_is_zero_at_its_rest_length():
    u = pair_energy(1.3, potential="harmonic", k=2.0, r0=1.3)
    assert abs(u) < 1e-12, u


def test_harmonic_grows_quadratically():
    """Doubling the extension quadruples the energy."""
    one = pair_energy(1.1, potential="harmonic", k=2.0, r0=1.0)
    two = pair_energy(1.2, potential="harmonic", k=2.0, r0=1.0)
    assert abs(two - 4.0 * one) < 1e-9, (one, two)


def test_a_stretched_harmonic_bond_pulls_inward():
    f = forces([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], box=FAR,
               potential="harmonic", k=2.0, r0=1.0)
    assert f[0][0] > 0.0 and f[1][0] < 0.0, f
''',
}

#: The audit set: never in the repository, injected only while a held-out task is
#: scored. `frozen` stops a file being written, not read.
_AUDIT = {
    'audit/test_physics.py': r'''"""The audit set: the same physics, asked differently, and never in the repository.

`tests/` is frozen but readable, and the executor is handed the source of the test
it is failing -- so a run could in principle satisfy one assertion at a time without
the physics underneath holding together. These tests are injected only while a
held-out task is being scored, so nothing that wrote the code has ever seen them.
Every probe here is deliberately *not* a restatement of a driven test: a different
configuration, a different cutoff, a brute-force comparison, or two layers composed.
"""

import math
import random

from src import displacement, energy, forces, observables, rdf, step, wrap

BOX = [7.3, 6.1, 5.4]          # deliberately not a cube
MASSES = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0]


def a_random_configuration(seed, n=6, box=BOX):
    """Spread out enough that nothing sits inside the repulsive core."""
    rng = random.Random(seed)
    while True:
        points = [[rng.uniform(0.0, box[axis]) for axis in range(3)] for _ in range(n)]
        if all(math.sqrt(sum(c * c for c in displacement(points[i], points[j], box)))
               > 0.95 for i in range(n) for j in range(i + 1, n)):
            return points


def test_displacement_agrees_with_a_brute_force_search_over_images():
    """The closed form, checked against the thing it is a closed form for."""
    rng = random.Random(11)
    for _ in range(40):
        # Both points inside the box, so one box of shift per axis is enough to
        # reach every candidate image -- and the search is then exhaustive.
        a = [rng.uniform(0.0, BOX[axis]) for axis in range(3)]
        b = [rng.uniform(0.0, BOX[axis]) for axis in range(3)]
        got = displacement(a, b, BOX)
        best = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    shift = (i * BOX[0], j * BOX[1], k * BOX[2])
                    candidate = [b[axis] + shift[axis] - a[axis] for axis in range(3)]
                    length = sum(c * c for c in candidate)
                    if best is None or length < best[0] - 1e-12:
                        best = (length, candidate)
        for axis in range(3):
            assert abs(got[axis] - best[1][axis]) < 1e-9, (a, b, got, best[1])


def test_wrapping_the_input_changes_no_displacement():
    """Two layers composed: wrapping is a change of representative, not of geometry."""
    points = a_random_configuration(3)
    shifted = [wrap([p[0] + 3 * BOX[0], p[1] - 2 * BOX[1], p[2] + BOX[2]], BOX)
               for p in points]
    for i in range(len(points)):
        for j in range(len(points)):
            here = displacement(points[i], points[j], BOX)
            there = displacement(shifted[i], shifted[j], BOX)
            for axis in range(3):
                assert abs(here[axis] - there[axis]) < 1e-9, (i, j, here, there)


def test_the_pair_energy_matches_the_closed_form_at_an_unusual_cutoff():
    """Not 2.5: a shift hard-coded for the default cutoff fails here."""
    for r in (0.95, 1.0, 1.4, 2.2, 2.9):
        s6 = (1.0 / r) ** 6
        shift = 4.0 * ((1.0 / 3.0) ** 12 - (1.0 / 3.0) ** 6)
        expect = 4.0 * (s6 * s6 - s6) - shift
        got = energy([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], box=[0.0, 0.0, 0.0], cutoff=3.0)
        assert abs(got - expect) < 1e-9, (r, got, expect)


def test_epsilon_scales_the_energy_linearly():
    config = a_random_configuration(5)
    one = energy(config, box=BOX, epsilon=1.0)
    three = energy(config, box=BOX, epsilon=3.0)
    assert abs(three - 3.0 * one) < 1e-9, (one, three)


def test_sigma_rescales_the_whole_problem():
    """u(r; sigma, rc) = u(r/sigma; 1, rc/sigma): a pure change of length unit."""
    r, sigma = 1.7, 1.3
    scaled = energy([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], box=[0.0, 0.0, 0.0],
                    sigma=sigma, cutoff=2.5 * sigma)
    plain = energy([[0.0, 0.0, 0.0], [r / sigma, 0.0, 0.0]], box=[0.0, 0.0, 0.0],
                   sigma=1.0, cutoff=2.5)
    assert abs(scaled - plain) < 1e-9, (scaled, plain)


def test_forces_match_a_central_difference_on_a_random_box():
    """The gradient check again, on a configuration no driven test names."""
    config = a_random_configuration(7)
    analytic = forces(config, box=BOX)
    h = 1e-6
    for i in range(len(config)):
        for axis in range(3):
            plus = [list(p) for p in config]
            minus = [list(p) for p in config]
            plus[i][axis] += h
            minus[i][axis] -= h
            numeric = -(energy(plus, box=BOX) - energy(minus, box=BOX)) / (2.0 * h)
            assert abs(numeric - analytic[i][axis]) < 1e-4, (i, axis, numeric,
                                                             analytic[i][axis])


def test_the_harmonic_trio_matches_its_analytic_energy():
    """Every pair is bonded, including the 1-3 pair: three springs, not two."""
    config = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]]
    expect = 0.5 * 3.0 * ((1.2 - 1.0) ** 2 + (1.2 - 1.0) ** 2 + (2.4 - 1.0) ** 2)
    got = energy(config, box=[0.0, 0.0, 0.0], potential="harmonic", k=3.0, r0=1.0)
    assert abs(got - expect) < 1e-9, (got, expect)


def test_a_harmonic_oscillator_keeps_its_period():
    """Two bonded particles, quarter period out and back: an integrator that is
    merely stable rather than correct drifts in phase."""
    k, mass = 4.0, 1.0
    omega = math.sqrt(2.0 * k / mass)            # reduced mass mu = m / 2
    positions = [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    dt, period = 1e-3, 2.0 * math.pi / omega
    for _ in range(int(round(period / dt))):
        positions, velocities = step(positions, velocities, [mass, mass], dt,
                                     box=[0.0, 0.0, 0.0], potential="harmonic",
                                     k=k, r0=1.0)
    separation = displacement(positions[0], positions[1], [0.0, 0.0, 0.0])[0]
    assert abs(separation - 1.3) < 1e-3, separation


def test_energy_is_conserved_on_a_configuration_no_driven_test_names():
    config = a_random_configuration(13)
    velocities = [[0.05 * (i + 1) * (-1) ** i, 0.03 * i, -0.02 * (i + 2)]
                  for i in range(len(config))]
    masses = MASSES[:len(config)]

    def total(p, v):
        return energy(p, box=BOX) + observables(v, masses)["kinetic"]

    before = total(config, velocities)
    p, v = config, velocities
    for _ in range(60):
        p, v = step(p, v, masses, 5e-4, box=BOX)
    assert abs(total(p, v) - before) / max(1.0, abs(before)) < 1e-4, (before,
                                                                     total(p, v))


def test_a_step_does_not_change_the_total_momentum():
    """Integration and observables composed: the forces cancel pairwise, so the
    two half-kicks move no momentum no matter what the configuration is."""
    config = a_random_configuration(17)
    masses = MASSES[:len(config)]
    velocities = [[0.1, -0.2, 0.05] for _ in config]
    before = observables(velocities, masses)["momentum"]
    _, after_v = step(config, velocities, masses, 1e-3, box=BOX)
    after = observables(after_v, masses)["momentum"]
    for axis in range(3):
        assert abs(after[axis] - before[axis]) < 1e-9, (before, after)


def test_doubling_every_velocity_quadruples_the_temperature():
    velocities = [[0.3, -0.1, 0.2], [0.0, 0.4, -0.3], [-0.2, 0.1, 0.1]]
    masses = [1.0, 2.0, 0.5]
    one = observables(velocities, masses)["temperature"]
    two = observables([[2.0 * c for c in v] for v in velocities],
                      masses)["temperature"]
    assert abs(two - 4.0 * one) < 1e-12, (one, two)


def test_the_histogram_accounts_for_every_pair_when_rmax_covers_the_box():
    """Half the shortest box side is the largest distance the nearest image can be,
    so nothing may fall outside the histogram here."""
    config = a_random_configuration(19)
    n = len(config)
    rmax = 0.5 * math.sqrt(sum(length * length for length in BOX)) + 1.0
    counts = rdf(config, BOX, 24, rmax)
    assert sum(counts) == n * (n - 1) // 2, (sum(counts), n)
    assert all(isinstance(c, int) for c in counts), counts


def test_the_histogram_agrees_with_distances_computed_through_displacement():
    """The structure layer against the geometry layer, bin by bin."""
    config = a_random_configuration(23)
    bins, rmax = 10, 3.0
    expect = [0] * bins
    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            r = math.sqrt(sum(c * c for c in displacement(config[i], config[j], BOX)))
            if r < rmax:
                expect[int(r / (rmax / bins))] += 1
    assert rdf(config, BOX, bins, rmax) == expect


def test_the_field_in_a_non_cubic_box_is_not_identically_zero():
    """The same non-vacuity check the driven suite makes, in a box with three
    different side lengths and across the shortest axis' seam."""
    close = [[1.0, 1.0, 5.3], [1.0, 1.0, 0.15]]        # 0.25 apart across z = 5.4
    u = energy(close, box=BOX)
    f = forces(close, box=BOX)
    assert u > 1000.0, u                                # deep in the repulsive core
    assert abs(f[0][2]) > 1000.0, f
    # and the closed form, to pin the number rather than its sign
    s6 = (1.0 / 0.25) ** 6
    shift = 4.0 * ((1.0 / 2.5) ** 12 - (1.0 / 2.5) ** 6)
    assert abs(u - (4.0 * (s6 * s6 - s6) - shift)) / u < 1e-9, u


def test_half_a_box_takes_the_negative_image_on_every_axis():
    """The banker's-rounding trap again, on a box with three different sides.

    `floor(d/L + 0.5)` at `d = L/2` gives `floor(1.0) = 1` and the image is `-L/2`;
    `round(0.5)` is 0 and it comes out `+L/2`. The magnitude is the same either way,
    so only a signed displacement can see this -- no distance, energy or histogram
    can. It is in the audit set as well as the suite because a requirement the spec
    states explicitly should not hang on one test.
    """
    for axis, length in enumerate(BOX):
        b = [0.0, 0.0, 0.0]
        b[axis] = length / 2.0
        assert displacement([0.0, 0.0, 0.0], b, BOX)[axis] == -length / 2.0
''',
}


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from."""
    files = {
        "CONTEXT.md": _ROOT_CONTEXT,
        "spec/CONTEXT.md": _SPEC,
        "md.py": _DRIVER,
        "src/CONTEXT.md": _SRC_CONTEXT,
        "src/core/CONTEXT.md": _CORE_CONTEXT,
        "src/potentials/CONTEXT.md": _POT_CONTEXT,
        "src/potentials/pair/CONTEXT.md": _PAIR_CONTEXT,
        "src/integrate/CONTEXT.md": _INT_CONTEXT,
        "src/observe/CONTEXT.md": _OBS_CONTEXT,
        f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL,
    }
    files.update(_TESTS)
    return files


# ---------------------------------------------------------------------------
# Deliberately wrong implementations, for checking the suite rejects them
# ---------------------------------------------------------------------------
# Not scoring, and not the algorithm: upstream has no objective function at all,
# and what stands in its place is a parent that reads the diff and runs the tests.
# This is the harness checking its own measurement. Every one of these has to die,
# and to more than one test -- the zero field originally died to exactly one, which
# is how close this domain came to reporting 1.000 for a repository with no force
# field in it.

_ZERO_FIELD = r'''"""The cheapest way to pass a suite of invariants: compute nothing."""

from ..core.vectors import minimum_image, norm2
from .pair import harmonic, lennard_jones

TABLE = {"lennard_jones": lennard_jones.pair, "harmonic": harmonic.pair}


def get(name):
    if name not in TABLE:
        raise ValueError("unknown potential: " + str(name))
    return TABLE[name]


def evaluate(system, spec):
    get(spec["name"])
    return 0.0, [[0.0, 0.0, 0.0] for _ in system.positions]
'''

_BANKERS_ROUNDING = r'''"""Geometry with `round` where the spec says `floor(x + 0.5)`."""

import math


def minimum_image(a, b, box):
    out = []
    for i in range(3):
        d = b[i] - a[i]
        length = box[i]
        if length > 0.0:
            d -= length * round(d / length)
        out.append(d)
    return out


def norm2(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def norm(v):
    return math.sqrt(norm2(v))


def wrap(point, box):
    out = []
    for i in range(3):
        length = box[i]
        out.append(point[i] - length * math.floor(point[i] / length)
                   if length > 0.0 else point[i])
    return out
'''

_PRECEDENCE_BUG = r'''"""The real one, from a real run: `// 1` binding after the multiplication.

`floor(d + L/2)` instead of `L * floor(d/L + 0.5)`, which pushes every pair in a
periodic box past the cutoff and leaves the force field identically zero.
"""

import math


def minimum_image(a, b, box):
    out = []
    for i in range(3):
        d = b[i] - a[i]
        length = box[i]
        if length > 0.0:
            d -= length * (d / length + 0.5) // 1
        out.append(d)
    return out


def norm2(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def norm(v):
    return math.sqrt(norm2(v))


def wrap(point, box):
    out = []
    for i in range(3):
        length = box[i]
        out.append(point[i] - length * math.floor(point[i] / length)
                   if length > 0.0 else point[i])
    return out
'''

_UNSHIFTED_LJ = r'''"""Lennard-Jones without the shift: discontinuous at the cutoff."""


def pair(r2, spec):
    cutoff = spec["cutoff"]
    if r2 >= cutoff * cutoff:
        return 0.0, 0.0
    epsilon, sigma = spec["epsilon"], spec["sigma"]
    s6 = (sigma * sigma / r2) ** 3
    s12 = s6 * s6
    u = 4.0 * epsilon * (s12 - s6)
    dudr_over_r = -24.0 * epsilon * (2.0 * s12 - s6) / r2
    return u, dudr_over_r
'''

_EULER = r'''"""One force evaluation per step: stable, plausible, first-order, irreversible."""

from ..core.vectors import wrap
from ..potentials.registry import evaluate


def step(system, spec, dt):
    _, forces = evaluate(system, spec)
    for i, mass in enumerate(system.masses):
        for axis in range(3):
            system.velocities[i][axis] += dt * forces[i][axis] / mass
            system.positions[i][axis] += dt * system.velocities[i][axis]
        system.positions[i] = wrap(system.positions[i], system.box)
    return system


def run(system, spec, dt, steps):
    for _ in range(steps):
        step(system, spec, dt)
    return system
'''

_NO_COM_DRIFT = r'''"""Temperature with a centre-of-mass correction the spec says not to apply."""


def kinetic(velocities, masses):
    total = 0.0
    for v, mass in zip(velocities, masses):
        total += 0.5 * mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return total


def temperature(velocities, masses):
    n = len(masses)
    if n == 0:
        return 0.0
    return 2.0 * kinetic(velocities, masses) / (3.0 * n - 3.0) if n > 1 else 0.0


def momentum(velocities, masses):
    total = [0.0, 0.0, 0.0]
    for v, mass in zip(velocities, masses):
        for axis in range(3):
            total[axis] += mass * v[axis]
    return total
'''

_SELF_CLOBBERING = r'''"""A public entry point that destroys itself on first call.

`from .rdf import histogram` imports the submodule `src.observe.rdf`, and importing a
submodule **rebinds that name on the package** -- over the function of the same name
defined right here. The first call works and every call after it raises. Taken from a
real run, unchanged.
"""


def observables(velocities, masses):
    from .thermo import kinetic, momentum, temperature
    return {"kinetic": kinetic(velocities, masses),
            "temperature": temperature(velocities, masses),
            "momentum": momentum(velocities, masses)}


def rdf(positions, box, bins, rmax):
    from .rdf import histogram
    from ..core.state import System
    return histogram(System(positions, box=box), bins, rmax)
'''

#: Each of these must fail at least two tests. See :meth:`TestSuite.kill_report`.
BASELINES = {
    "zero-field": {REGISTRY: _ZERO_FIELD},
    "precedence-bug": {VECTORS: _PRECEDENCE_BUG},
    "bankers-rounding": {VECTORS: _BANKERS_ROUNDING},
    "unshifted-lennard-jones": {LJ: _UNSHIFTED_LJ},
    "euler-not-verlet": {VERLET: _EULER},
    "com-corrected-temperature": {THERMO: _NO_COM_DRIFT},
}



MD = TestSuite(name="md", given=initial_files(), frozen=FROZEN, audit=_AUDIT,
               baselines=BASELINES)

build_tasks = MD.build_tasks
make_runner = MD.make_runner
suite_review = MD.review
#: The whole suite in one interpreter -- `mix test`, not one process per test.
suite_failures = MD.suite_failures
#: One test, pass or fail. No tolerance of its own -- the test owns that.
reward = reward_test
#: Exactly the audit set in the held-out tail, and every driven test in the search.
HELD_OUT_FRAC = MD.held_out_frac()


def llm_manager(complete):
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen,
                         failure=TEST_FAILURE)


# ---------------------------------------------------------------------------
# A reference implementation, for the two jobs that are not scoring
# ---------------------------------------------------------------------------
# It drives the offline rule-based actor, and it lets a test prove the suite is
# passable at all -- shipping a suite nobody has seen pass is its own dishonesty.
# Nothing in the scoring path reads it.

_PKG_INIT = '"""Package marker."""\n'

_REF_ENTRY = r'''"""Public surface. Each function imports its stage lazily -- see spec/CONTEXT.md."""


def displacement(a, b, box):
    """The nearest-image displacement from ``a`` to ``b``."""
    from .core.vectors import minimum_image
    return minimum_image(a, b, box)


def wrap(point, box):
    """``point`` folded into ``[0, L)`` on every periodic axis."""
    from .core.vectors import wrap as _wrap
    return _wrap(point, box)


def energy(positions, box=(0.0, 0.0, 0.0), potential="lennard_jones",
           epsilon=1.0, sigma=1.0, cutoff=2.5, k=1.0, r0=1.0):
    """The total potential energy of a configuration."""
    from .core.state import System, make_spec
    from .potentials.registry import evaluate
    system = System(positions, box=box)
    spec = make_spec(potential, epsilon, sigma, cutoff, k, r0)
    return evaluate(system, spec)[0]


def forces(positions, box=(0.0, 0.0, 0.0), potential="lennard_jones",
           epsilon=1.0, sigma=1.0, cutoff=2.5, k=1.0, r0=1.0):
    """One force vector per particle."""
    from .core.state import System, make_spec
    from .potentials.registry import evaluate
    system = System(positions, box=box)
    spec = make_spec(potential, epsilon, sigma, cutoff, k, r0)
    return evaluate(system, spec)[1]


def step(positions, velocities, masses, dt, box=(0.0, 0.0, 0.0),
         potential="lennard_jones", epsilon=1.0, sigma=1.0, cutoff=2.5,
         k=1.0, r0=1.0):
    """One velocity-Verlet step. Returns ``(positions, velocities)``."""
    from .core.state import System, make_spec
    from .integrate.verlet import step as one_step
    system = System(positions, velocities, masses, box)
    spec = make_spec(potential, epsilon, sigma, cutoff, k, r0)
    one_step(system, spec, dt)
    return system.positions, system.velocities


def observables(velocities, masses):
    """``{"kinetic": ..., "temperature": ..., "momentum": [...]}``."""
    from .observe.thermo import kinetic, momentum, temperature
    return {"kinetic": kinetic(velocities, masses),
            "temperature": temperature(velocities, masses),
            "momentum": momentum(velocities, masses)}


def rdf(positions, box, bins, rmax):
    """Integer counts of distinct pairs per bin over ``[0, rmax)``."""
    from .core.state import System
    from .observe.rdf import histogram
    return histogram(System(positions, box=box), bins, rmax)
'''

_REF_VECTORS = r'''"""Geometry under periodic boundaries."""

import math


def minimum_image(a, b, box):
    """The displacement b - a, taking the nearest periodic image of b.

    ``floor(d / L + 0.5)`` rather than ``round``: Python's round is
    banker's rounding, so two implementations disagree exactly at half a box.
    """
    out = []
    for i in range(3):
        d = b[i] - a[i]
        length = box[i]
        if length > 0.0:
            d -= length * math.floor(d / length + 0.5)
        out.append(d)
    return out


def norm2(v):
    """The squared length of a vector."""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def norm(v):
    return math.sqrt(norm2(v))


def wrap(point, box):
    """``point`` folded into ``[0, L)`` on every axis with a positive length."""
    out = []
    for i in range(3):
        length = box[i]
        out.append(point[i] - length * math.floor(point[i] / length)
                   if length > 0.0 else point[i])
    return out
'''

_REF_STATE = r'''"""The System container, and the potential's parameters."""


class System:
    """Positions, velocities, masses and the box. Mutable, and copied on demand."""

    def __init__(self, positions, velocities=None, masses=None, box=(0.0, 0.0, 0.0)):
        self.positions = [list(p) for p in positions]
        self.velocities = ([list(v) for v in velocities] if velocities is not None
                           else [[0.0, 0.0, 0.0] for _ in self.positions])
        self.masses = (list(masses) if masses is not None
                       else [1.0] * len(self.positions))
        self.box = list(box)

    def __len__(self):
        return len(self.positions)

    def copy(self):
        return System(self.positions, self.velocities, self.masses, self.box)


def make_spec(potential="lennard_jones", epsilon=1.0, sigma=1.0, cutoff=2.5,
              k=1.0, r0=1.0):
    """The parameter bundle every pair kernel is handed."""
    return {"name": potential, "epsilon": epsilon, "sigma": sigma,
            "cutoff": cutoff, "k": k, "r0": r0}
'''

_REF_REGISTRY = r'''"""Which pair interaction a name means, and the O(N^2) loop over pairs."""

from ..core.vectors import minimum_image, norm2
from .pair import harmonic, lennard_jones

TABLE = {"lennard_jones": lennard_jones.pair, "harmonic": harmonic.pair}


def get(name):
    if name not in TABLE:
        raise ValueError("unknown potential: " + str(name))
    return TABLE[name]


def evaluate(system, spec):
    """``(energy, forces)`` for every distinct pair, Newton's third law applied once."""
    kernel = get(spec["name"])
    positions, box = system.positions, system.box
    n = len(positions)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = minimum_image(positions[i], positions[j], box)
            r2 = norm2(d)
            if r2 == 0.0:
                continue
            u, dudr_over_r = kernel(r2, spec)
            energy += u
            for axis in range(3):
                component = dudr_over_r * d[axis]
                forces[i][axis] += component
                forces[j][axis] -= component
    return energy, forces
'''

_REF_LJ = r'''"""The shifted Lennard-Jones pair interaction."""


def pair(r2, spec):
    """``(u, dudr_over_r)`` for one pair at squared distance ``r2``.

    Shifted so that ``u(cutoff) == 0``; zero beyond the cutoff. The caller puts
    the force on particle *i* at ``dudr_over_r * d`` with ``d = x_j - x_i``.
    """
    cutoff = spec["cutoff"]
    if r2 >= cutoff * cutoff:
        return 0.0, 0.0
    epsilon, sigma = spec["epsilon"], spec["sigma"]
    s6 = (sigma * sigma / r2) ** 3
    s12 = s6 * s6
    sc6 = (sigma / cutoff) ** 6
    shift = 4.0 * epsilon * (sc6 * sc6 - sc6)
    u = 4.0 * epsilon * (s12 - s6) - shift
    dudr_over_r = -24.0 * epsilon * (2.0 * s12 - s6) / r2
    return u, dudr_over_r
'''

_REF_HARMONIC = r'''"""A harmonic spring between every pair, with no cutoff."""

import math


def pair(r2, spec):
    """``(u, dudr_over_r)`` for ``u = 0.5 * k * (r - r0) ** 2``."""
    k, r0 = spec["k"], spec["r0"]
    r = math.sqrt(r2)
    if r == 0.0:
        return 0.0, 0.0
    stretch = r - r0
    return 0.5 * k * stretch * stretch, k * stretch / r
'''

_REF_VERLET = r'''"""Velocity Verlet."""

from ..core.vectors import wrap
from ..potentials.registry import evaluate


def step(system, spec, dt):
    """One step, in place. Positions are wrapped back into the box."""
    _, forces = evaluate(system, spec)
    half = 0.5 * dt
    for i, mass in enumerate(system.masses):
        for axis in range(3):
            system.velocities[i][axis] += half * forces[i][axis] / mass
            system.positions[i][axis] += dt * system.velocities[i][axis]
        system.positions[i] = wrap(system.positions[i], system.box)
    _, forces = evaluate(system, spec)
    for i, mass in enumerate(system.masses):
        for axis in range(3):
            system.velocities[i][axis] += half * forces[i][axis] / mass
    return system


def run(system, spec, dt, steps):
    for _ in range(steps):
        step(system, spec, dt)
    return system
'''

_REF_THERMO = r'''"""Thermodynamic observables, in reduced units with k_B = 1."""


def kinetic(velocities, masses):
    """The total kinetic energy."""
    total = 0.0
    for v, mass in zip(velocities, masses):
        total += 0.5 * mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return total


def temperature(velocities, masses):
    """``2 * KE / (3 * N)`` -- no centre-of-mass correction, k_B = 1."""
    n = len(masses)
    if n == 0:
        return 0.0
    return 2.0 * kinetic(velocities, masses) / (3.0 * n)


def momentum(velocities, masses):
    """The total momentum vector."""
    total = [0.0, 0.0, 0.0]
    for v, mass in zip(velocities, masses):
        for axis in range(3):
            total[axis] += mass * v[axis]
    return total
'''

_REF_RDF = r'''"""Pair-distance histogram -- the raw counts an g(r) is built from."""

from ..core.vectors import minimum_image, norm


def histogram(system, bins, rmax):
    """Counts of distinct pairs per bin over ``[0, rmax)``, ``bins`` bins wide."""
    counts = [0] * bins
    width = rmax / bins
    positions = system.positions
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r = norm(minimum_image(positions[i], positions[j], system.box))
            if r < rmax:
                counts[int(r / width)] += 1
    return counts
'''

REFERENCE = {
    ENTRY: _REF_ENTRY,
    CORE_INIT: _PKG_INIT, VECTORS: _REF_VECTORS, STATE: _REF_STATE,
    POT_INIT: _PKG_INIT, REGISTRY: _REF_REGISTRY,
    PAIR_INIT: _PKG_INIT, LJ: _REF_LJ, HARMONIC: _REF_HARMONIC,
    INT_INIT: _PKG_INIT, VERLET: _REF_VERLET,
    OBS_INIT: _PKG_INIT, THERMO: _REF_THERMO, RDF: _REF_RDF,
}


def reference_tree() -> Dict[str, str]:
    """The repository as a finished run should leave it."""
    tree = initial_files()
    tree.update(REFERENCE)
    return tree


#: The public entry point routed through the package the way the real run wrote it,
#: which is what turns the self-clobbering import above into a visible failure.
_ROUTED_ENTRY = _REF_ENTRY.replace(
    """    from .core.state import System
    from .observe.rdf import histogram
    return histogram(System(positions, box=box), bins, rmax)""",
    """    from .observe import rdf as _rdf
    return _rdf(positions, box, bins, rmax)""")

#: The one a per-test score cannot catch, kept separate because it is not a failure of
#: the suite -- it is a failure of *how the suite is run*, and the guard for it is
#: `suite_failures` rather than `kill_report`.
SUITE_ONLY_BASELINES = {"self-clobbering-import": {ENTRY: _ROUTED_ENTRY,
                                                   OBS_INIT: _SELF_CLOBBERING}}


# ---------------------------------------------------------------------------
# The offline actors: rule-based, deterministic, and a surrogate
# ---------------------------------------------------------------------------

_PLAN = {
    "src": [(ENTRY, _REF_ENTRY)],
    "src/core": [(CORE_INIT, _PKG_INIT), (VECTORS, _REF_VECTORS), (STATE, _REF_STATE)],
    "src/potentials": [(POT_INIT, _PKG_INIT), (REGISTRY, _REF_REGISTRY)],
    "src/potentials/pair": [(PAIR_INIT, _PKG_INIT), (LJ, _REF_LJ),
                            (HARMONIC, _REF_HARMONIC)],
    "src/integrate": [(INT_INIT, _PKG_INIT), (VERLET, _REF_VERLET)],
    "src/observe": [(OBS_INIT, _PKG_INIT), (THERMO, _REF_THERMO), (RDF, _REF_RDF)],
}


def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    return {node: [path for path, _ in steps if path not in state]
            for node, steps in _PLAN.items()}


def _owes(state: Mapping[str, str], node: str) -> bool:
    node = normalise(node)
    return any(items for key, items in _outstanding(state).items()
               if key == node or key.startswith(node + "/"))


def offline_manager(brief: Brief) -> Sequence[Delegation]:
    """Decompose along the node's routing table; accountability writes its own.

    Cold-started there is no table, so it opens the nodes its plan needs -- see
    :func:`plan_delegations`.
    """
    return plan_delegations(brief, _owes, list(_PLAN))


def offline_executor(brief: Brief) -> Sequence[Edit]:
    """Write exactly one file, the way a bounded episode does."""
    path, state = normalise(brief.world.path), brief.state
    for target, content in _PLAN.get(path, ()):
        if target not in state:
            return [Edit(owner=path, path=target, content=content)]
    return ()
