"""md: Lennard-Jones molecular dynamics, grown from an empty repository.

The fourth formation domain and by some distance the largest: **ten** validated
stages, fifteen files the agents write across six nodes, and a physics kernel that
has to be right rather than merely parseable. A frozen driver sits on top, so a
finished run is a program you can point at a box of argon::

    python md.py --particles 108 --density 0.8 --temperature 1.2 --steps 1000 \
                 --thermostat 0.5 --trajectory traj.xyz --rdf 40

Why this domain exists. minilang showed the mechanism, stackvm gave the recursion
somewhere to go, jqx came out as usable software -- and all three can be passed by
code that merely parses. This one cannot: four of its ten stages are **invariants**
rather than values. Forces have to sum to zero, they have to match a central
difference of the energy, and energy and momentum have to survive a trajectory. A
stage that returns ``True`` without computing anything fails, because one ``cons``
case uses a timestep far too large and its answer is ``False``.

Floats are compared to six decimal places (``Suite.digits``), because two correct
implementations of one sum differ in the last bits by summation order alone.

What is faithful and what is a surrogate is as it is for the other three, and the
reasoning is written out on :mod:`examples.genesis._domain`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from ._delegation import Brief, Delegation, Edit
from ._suite import PYTHON_MODULE_SKILL, Suite, reward
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR, normalise

__all__ = ["FROZEN", "MD", "build_tasks", "initial_files", "llm_executor",
           "llm_manager", "make_runner", "offline_executor", "offline_manager",
           "reward", "suite_review"]

#: The specification and the driver. Human-supplied, refused to every proposal,
#: and restored pristine before scoring.
FROZEN = ("spec/**", "md.py")

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
CHECKS = "src/observe/checks.py"


# ---------------------------------------------------------------------------
# The human's repository: the physics, the decomposition, the driver
# ---------------------------------------------------------------------------

_SPEC = r'''# md -- the molecular dynamics this project implements

Lennard-Jones molecular dynamics in **reduced units**: sigma = epsilon = mass =
k_B = 1, so energies, lengths and temperatures are all dimensionless.

## Geometry

Periodic boundaries on every axis with a positive box length. The displacement
from `a` to `b` is the **nearest image**:

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

Energies and forces sum over every **distinct** pair once.

## Integration

Velocity Verlet, one step with timestep `dt`:

1. `v += 0.5 * dt * f / m` with the forces at the current positions;
2. `x += dt * v`, then **wrap** `x` into the box;
3. recompute the forces;
4. `v += 0.5 * dt * f / m`.

## Payload defaults

Every stage takes one JSON object as text. `positions` is required; `velocities`
defaults to zeros, `masses` to ones, `box` to `[0, 0, 0]` (no periodicity),
`potential` to `"lennard_jones"`, `epsilon`/`sigma`/`k`/`r0` to `1.0` and `cutoff`
to `2.5`.

## The ten validated stages

| stage | entry point | returns |
|---|---|---|
| `disp` | `src.displacement(p)` | the nearest-image vector from `p["a"]` to `p["b"]` |
| `pot` | `src.energy(p)` | the total potential energy |
| `force` | `src.forces(p)` | one force vector per particle |
| `newton` | `src.forces_sum_to_zero(p)` | `True` when `max abs(sum of forces)` <= 1e-9 |
| `fd` | `src.force_matches_gradient(p)` | `True` when every component agrees with a central difference of the energy (`h = 1e-6`) to 1e-4 |
| `step` | `src.step(p)` | `{"positions": ..., "velocities": ...}` after one step of `p["dt"]` |
| `obs` | `src.observables(p)` | `{"kinetic": ..., "temperature": ..., "momentum": [...]}` with `KE = sum(0.5*m*v.v)`, `T = 2*KE/(3*N)` and no centre-of-mass correction |
| `rdf` | `src.rdf(p)` | **integer** counts of distinct pairs per bin over `[0, p["rmax"])` in `p["bins"]` bins; a pair at or beyond `rmax` is not counted |
| `cons` | `src.energy_conserved(p)` | `True` when the total energy drifts by at most 1e-4 **relative** (`abs(after - before) / max(1, abs(before))`) over `p["steps"]` steps |
| `mom` | `src.momentum_conserved(p)` | `True` when the total momentum drifts by at most 1e-9 over `p["steps"]` steps |

Floats are compared to **six decimal places**, so a different summation order is
not a failure. The boolean stages are invariants, not opinions: one of the `cons`
cases uses a timestep far too large and its answer is `False`, so a stage that
always returns `True` fails the suite.

A stage is scored independently of the ones after it, so `src/__init__.py` MUST
import each stage lazily, inside the function that needs it.
'''

_ROOT_CONTEXT = r'''# md -- root

## Intent
Implement the molecular dynamics in `spec/CONTEXT.md` as a library under `src/`.
`md.py` is the simulation driver and is already written.

## Routing Table
- `./src/` -> the library, and the ten stage entry points

## Constraints
- `spec/` and `md.py` are read-only. They are the contract, not work items.
- Pure Python, no third-party packages: this has to run anywhere.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = r'''# src -- library root

## Intent
Own `src/__init__.py`: the ten entry points the spec names, each a thin wrapper
that decodes the payload and calls into a child node. Each MUST import its stage
lazily, so a missing layer does not stop the layers before it from scoring.

## Routing Table
- `./src/core/`       -> geometry and the system container
- `./src/potentials/` -> energies and forces
- `./src/integrate/`  -> advancing time
- `./src/observe/`    -> measuring and checking
'''

_CORE_CONTEXT = r'''# src/core -- geometry and state

## Intent
`vectors.py` owns periodic geometry: the nearest-image displacement, squared
length, length, and wrapping a point into the box. Everything else in the project
goes through it rather than writing `floor` arithmetic of its own.

`state.py` owns the `System` container -- positions, velocities, masses, box --
and the two functions that turn a decoded payload into one (`from_payload`) and
read the potential's parameters out of it with the spec's defaults
(`potential_spec`).
'''

_POT_CONTEXT = r'''# src/potentials -- energies and forces

## Intent
Own `registry.py`: the name-to-kernel table, and the loop over distinct pairs that
turns a kernel into a total energy and a force array. The loop applies Newton's
third law once per pair, so the forces sum to zero by construction.

## Routing Table
- `./src/potentials/pair/` -> one module per pair interaction
'''

_PAIR_CONTEXT = r'''# src/potentials/pair -- the pair interactions

## Intent
One module per interaction, each exposing `pair(r2, spec) -> (u, dudr_over_r)`
exactly as the spec defines it. `lennard_jones.py` is shifted and cut off;
`harmonic.py` has no cutoff. Neither knows anything about boxes or loops.
'''

_INT_CONTEXT = r'''# src/integrate -- advancing time

## Intent
`verlet.py` owns velocity Verlet: `step` advances one timestep in place and
returns the system, `run` applies it `steps` times. The position update wraps back
into the box, and the forces are recomputed between the two half-kicks.
'''

_OBS_CONTEXT = r'''# src/observe -- measuring and checking

## Intent
`thermo.py` owns the thermodynamic observables: kinetic energy, temperature and
total momentum. `rdf.py` owns the pair-distance histogram, in raw integer counts.

`checks.py` owns the invariants, **one function per invariant**, so that two
agents fixing two different checks are editing two different parts of the file
rather than the same one: forces summing to zero, forces against a finite
difference of the energy, and energy and momentum conserved across a run.
'''

_DRIVER = r'''#!/usr/bin/env python3
"""md -- a Lennard-Jones molecular dynamics driver.

Human-supplied, frozen, and never written by an agent: the library under ``src/``
is what the run grows, and this is the shell around it so the result is a program
rather than a package nobody can invoke.

    python md.py --particles 32 --steps 500
    python md.py --particles 64 --density 0.8 --temperature 1.2 --steps 1000 \
                 --thermostat 0.2 --trajectory traj.xyz
    python md.py --particles 32 --steps 200 --rdf 40

Reduced Lennard-Jones units throughout: sigma = epsilon = mass = k_B = 1.

It talks to the library only through the surface ``spec/CONTEXT.md`` pins --
``step``, ``observables``, ``rdf`` -- so it works against any implementation that
passes the suite, whatever the internal layout turns out to be. The price is a
JSON round trip per step, which is why the defaults are small; raise them when you
want a longer run and do not expect numpy speed from a pure-Python kernel.
"""

import argparse
import json
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
    common = {"box": box, "masses": masses, "cutoff": args.cutoff,
              "potential": "lennard_jones", "epsilon": 1.0, "sigma": 1.0}

    trajectory = open(args.trajectory, "w", encoding="utf-8") if args.trajectory else None
    print(f"# {count} particles, box {box[0]:.4f}, density {args.density}, "
          f"dt {args.dt}, cutoff {args.cutoff}"
          + (f", thermostat {args.thermostat} -> T={args.temperature}"
             if args.thermostat else ", constant energy"))
    print(f"# {'step':>8} {'T':>12} {'KE':>14} {'|P|':>12}")
    try:
        for n in range(args.steps + 1):
            if n % args.log_every == 0 or n == args.steps:
                obs = observables(json.dumps(
                    {"positions": positions, "velocities": velocities,
                     "masses": masses, "box": box}))
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
            out = step(json.dumps(dict(common, positions=positions,
                                       velocities=velocities, dt=args.dt)))
            positions, velocities = out["positions"], out["velocities"]
    finally:
        if trajectory is not None:
            trajectory.close()

    if args.rdf:
        counts = rdf(json.dumps({"positions": positions, "box": box,
                                 "bins": args.rdf, "rmax": args.cutoff}))
        width = args.cutoff / args.rdf
        print(f"# pair-distance histogram, {args.rdf} bins over [0, {args.cutoff})")
        peak = max(counts) or 1
        for i, count in enumerate(counts):
            bar = "#" * int(40 * count / peak)
            print(f"  {i * width:6.3f}-{(i + 1) * width:6.3f} {count:>6} {bar}")
    if args.trajectory:
        print(f"# wrote {args.trajectory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from."""
    return {
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


# ---------------------------------------------------------------------------
# The reference implementation: the oracle, and what the offline actors reveal
# ---------------------------------------------------------------------------

_PKG_INIT = '"""Package marker."""\n'

_REF_ENTRY = r'''"""Public surface. Each stage is imported lazily -- see spec/CONTEXT.md."""

import json


def _payload(source):
    return json.loads(source)


def displacement(source):
    from .core.vectors import minimum_image
    spec = _payload(source)
    return minimum_image(spec["a"], spec["b"], spec["box"])


def energy(source):
    from .core.state import from_payload, potential_spec
    from .potentials.registry import evaluate
    spec = _payload(source)
    return evaluate(from_payload(spec), potential_spec(spec))[0]


def forces(source):
    from .core.state import from_payload, potential_spec
    from .potentials.registry import evaluate
    spec = _payload(source)
    return evaluate(from_payload(spec), potential_spec(spec))[1]


def step(source):
    from .core.state import from_payload, potential_spec
    from .integrate.verlet import step as one_step
    spec = _payload(source)
    system = one_step(from_payload(spec), potential_spec(spec), spec["dt"])
    return {"positions": system.positions, "velocities": system.velocities}


def observables(source):
    from .core.state import from_payload
    from .observe.thermo import kinetic, momentum, temperature
    system = from_payload(_payload(source))
    return {"kinetic": kinetic(system.velocities, system.masses),
            "temperature": temperature(system.velocities, system.masses),
            "momentum": momentum(system.velocities, system.masses)}


def rdf(source):
    from .core.state import from_payload
    from .observe.rdf import histogram
    spec = _payload(source)
    return histogram(from_payload(spec), spec["bins"], spec["rmax"])


def forces_sum_to_zero(source):
    from .core.state import from_payload, potential_spec
    from .observe.checks import forces_sum_to_zero as check
    spec = _payload(source)
    return check(from_payload(spec), potential_spec(spec))


def force_matches_gradient(source):
    from .core.state import from_payload, potential_spec
    from .observe.checks import force_matches_gradient as check
    spec = _payload(source)
    return check(from_payload(spec), potential_spec(spec))


def energy_conserved(source):
    from .core.state import from_payload, potential_spec
    from .observe.checks import energy_conserved as check
    spec = _payload(source)
    return check(from_payload(spec), potential_spec(spec), spec["dt"], spec["steps"])


def momentum_conserved(source):
    from .core.state import from_payload, potential_spec
    from .observe.checks import momentum_conserved as check
    spec = _payload(source)
    return check(from_payload(spec), potential_spec(spec), spec["dt"], spec["steps"])
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

_REF_STATE = r'''"""The System container, and the payload shape every stage is given."""


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


def from_payload(payload):
    """A System from a decoded stage payload."""
    return System(payload["positions"], payload.get("velocities"),
                  payload.get("masses"), payload.get("box", [0.0, 0.0, 0.0]))


def potential_spec(payload):
    """The potential's name and its parameters, with the spec's defaults."""
    return {
        "name": payload.get("potential", "lennard_jones"),
        "epsilon": payload.get("epsilon", 1.0),
        "sigma": payload.get("sigma", 1.0),
        "cutoff": payload.get("cutoff", 2.5),
        "k": payload.get("k", 1.0),
        "r0": payload.get("r0", 1.0),
    }
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

#: The shape `src/observe/CONTEXT.md` asks for: one function per invariant, four
#: of them, each independently fillable and each reached by a different stage --
#: so four agents holding four different failing stages edit four different parts
#: of one file, which is what a keyed union cannot fuse.
_CHECKS_SKELETON = r'''"""Invariants a correct force field and integrator must satisfy."""

from ..integrate.verlet import run
from ..observe.thermo import kinetic, momentum
from ..potentials.registry import evaluate


def forces_sum_to_zero(system, spec, tol=1e-9):
    raise NotImplementedError("forces_sum_to_zero")


def force_matches_gradient(system, spec, h=1e-6, tol=1e-4):
    raise NotImplementedError("force_matches_gradient")


def energy_conserved(system, spec, dt, steps, tol=1e-4):
    raise NotImplementedError("energy_conserved")


def momentum_conserved(system, spec, dt, steps, tol=1e-9):
    raise NotImplementedError("momentum_conserved")
'''

_STUBS = {'forces_sum_to_zero': ('def forces_sum_to_zero(system, spec, tol=1e-9):\n    raise NotImplementedError("forces_sum_to_zero")\n', 'def forces_sum_to_zero(system, spec, tol=1e-9):\n    """Newton\'s third law, pair by pair: the net force on the box is zero."""\n    _, forces = evaluate(system, spec)\n    total = [sum(f[axis] for f in forces) for axis in range(3)]\n    return max(abs(c) for c in total) <= tol\n'), 'force_matches_gradient': ('def force_matches_gradient(system, spec, h=1e-6, tol=1e-4):\n    raise NotImplementedError("force_matches_gradient")\n', 'def force_matches_gradient(system, spec, h=1e-6, tol=1e-4):\n    """Every force component against a central difference of the energy."""\n    _, forces = evaluate(system, spec)\n    worst = 0.0\n    for i in range(len(system)):\n        for axis in range(3):\n            original = system.positions[i][axis]\n            system.positions[i][axis] = original + h\n            plus, _ = evaluate(system, spec)\n            system.positions[i][axis] = original - h\n            minus, _ = evaluate(system, spec)\n            system.positions[i][axis] = original\n            worst = max(worst, abs(-(plus - minus) / (2.0 * h) - forces[i][axis]))\n    return worst <= tol\n'), 'energy_conserved': ('def energy_conserved(system, spec, dt, steps, tol=1e-4):\n    raise NotImplementedError("energy_conserved")\n', 'def energy_conserved(system, spec, dt, steps, tol=1e-4):\n    """Total energy before and after ``steps`` steps, as a relative drift."""\n    potential, _ = evaluate(system, spec)\n    before = potential + kinetic(system.velocities, system.masses)\n    run(system, spec, dt, steps)\n    potential, _ = evaluate(system, spec)\n    after = potential + kinetic(system.velocities, system.masses)\n    return abs(after - before) / max(1.0, abs(before)) <= tol\n'), 'momentum_conserved': ('def momentum_conserved(system, spec, dt, steps, tol=1e-9):\n    raise NotImplementedError("momentum_conserved")\n', 'def momentum_conserved(system, spec, dt, steps, tol=1e-9):\n    """Total momentum before and after ``steps`` steps."""\n    before = momentum(system.velocities, system.masses)\n    run(system, spec, dt, steps)\n    after = momentum(system.velocities, system.masses)\n    return max(abs(after[a] - before[a]) for a in range(3)) <= tol\n')}

#: Which stage's failure points at which invariant. Routing by the stage rather
#: than by the payload text, because an MD payload is numbers and mentions no
#: function by name.
_STAGE_STUB = {"newton": "forces_sum_to_zero", "fd": "force_matches_gradient",
               "cons": "energy_conserved", "mom": "momentum_conserved"}


def _filled_checks() -> str:
    body = _CHECKS_SKELETON
    for stub, filled in _STUBS.values():
        body = body.replace(stub, filled)
    return body


_CASES = (
    ('disp', '{"a": [0.1, 0.0, 0.0], "b": [5.6, 0.0, 0.0], "box": [6.0, 6.0, 6.0]}'),
    ('pot', '{"box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('force', '{"box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('newton', '{"box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('fd', '{"box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('step', '{"box": [6.0, 6.0, 6.0], "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('obs', '{"box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('rdf', '{"bins": 8, "box": [6.0, 6.0, 6.0], "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "rmax": 2.5, "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('cons', '{"box": [6.0, 6.0, 6.0], "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "steps": 40, "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('mom', '{"box": [6.0, 6.0, 6.0], "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "steps": 40, "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('cons', '{"box": [6.0, 6.0, 6.0], "dt": 0.25, "masses": [1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [2.0, 2.0, 2.0]], "steps": 40, "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.2], [0.0, 0.0, -0.2]]}'),
    ('disp', '{"a": [1.0, 2.0, 3.0], "b": [1.5, 2.5, 3.5], "box": [8.0, 8.0, 8.0]}'),
    ('pot', '{"box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('force', '{"box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('newton', '{"box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('fd', '{"box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('step', '{"box": [8.0, 8.0, 8.0], "dt": 0.002, "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('obs', '{"box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('rdf', '{"bins": 8, "box": [8.0, 8.0, 8.0], "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "rmax": 2.5, "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('cons', '{"box": [8.0, 8.0, 8.0], "dt": 0.002, "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "steps": 30, "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('mom', '{"box": [8.0, 8.0, 8.0], "dt": 0.002, "masses": [1.0, 2.0], "positions": [[1.0, 1.0, 1.0], [2.15, 1.0, 1.0]], "steps": 30, "velocities": [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]}'),
    ('disp', '{"a": [0.0, 6.9, 0.0], "b": [0.0, 0.2, 0.0], "box": [7.0, 7.0, 7.0]}'),
    ('pot', '{"box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('force', '{"box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('newton', '{"box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('fd', '{"box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('step', '{"box": [7.0, 7.0, 7.0], "dt": 0.001, "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('obs', '{"box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('rdf', '{"bins": 8, "box": [7.0, 7.0, 7.0], "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "rmax": 2.5, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('cons', '{"box": [7.0, 7.0, 7.0], "dt": 0.001, "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "steps": 30, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('mom', '{"box": [7.0, 7.0, 7.0], "dt": 0.001, "k": 2.0, "masses": [1.0, 1.0, 1.5], "positions": [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.9, 0.4]], "potential": "harmonic", "r0": 1.0, "steps": 30, "velocities": [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0], [0.0, -0.1, 0.0]]}'),
    ('disp', '{"a": [4.9, 4.9, 4.9], "b": [0.1, 0.1, 0.1], "box": [5.0, 5.0, 5.0]}'),
    ('pot', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('force', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('newton', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('fd', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('step', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('obs', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('rdf', '{"bins": 8, "box": [5.0, 5.0, 5.0], "cutoff": 2.0, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "rmax": 2.5, "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('cons', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "steps": 40, "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
    ('mom', '{"box": [5.0, 5.0, 5.0], "cutoff": 2.0, "dt": 0.001, "masses": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "positions": [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, 0.0, 1.15], [2.3, 0.0, 0.0], [1.15, 1.15, 1.15]], "steps": 40, "velocities": [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]}'),
)

MD = Suite(
    name="md",
    given=initial_files(),
    frozen=FROZEN,
    stages={"disp": "displacement", "pot": "energy", "force": "forces",
            "newton": "forces_sum_to_zero", "fd": "force_matches_gradient",
            "step": "step", "obs": "observables", "rdf": "rdf",
            "cons": "energy_conserved", "mom": "momentum_conserved"},
    cases=_CASES,
    reference={
        ENTRY: _REF_ENTRY,
        CORE_INIT: _PKG_INIT, VECTORS: _REF_VECTORS, STATE: _REF_STATE,
        POT_INIT: _PKG_INIT, REGISTRY: _REF_REGISTRY,
        PAIR_INIT: _PKG_INIT, LJ: _REF_LJ, HARMONIC: _REF_HARMONIC,
        INT_INIT: _PKG_INIT, VERLET: _REF_VERLET,
        OBS_INIT: _PKG_INIT, THERMO: _REF_THERMO, RDF: _REF_RDF,
        CHECKS: _filled_checks(),
    },
    digits=6,
)

build_tasks = MD.build_tasks
make_runner = MD.make_runner
suite_review = MD.review


def llm_manager(complete):
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen)


def _reference_tree() -> Dict[str, str]:
    return MD.reference_tree()


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
    "src/observe": [(OBS_INIT, _PKG_INIT), (THERMO, _REF_THERMO), (RDF, _REF_RDF),
                    (CHECKS, _CHECKS_SKELETON)],
}


def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    owed = {node: [path for path, _ in steps if path not in state]
             for node, steps in _PLAN.items()}
    if CHECKS in state:
        body = state[CHECKS]
        owed["src/observe"] += [f"{CHECKS}#{name}"
                                for name, (stub, _) in _STUBS.items() if stub in body]
    return owed


def _owes(state: Mapping[str, str], node: str) -> bool:
    node = normalise(node)
    return any(items for key, items in _outstanding(state).items()
               if key == node or key.startswith(node + "/"))


def offline_manager(brief: Brief) -> Sequence[Delegation]:
    """Decompose along the node's routing table; accountability writes its own."""
    return [Delegation(node, f"clear the outstanding work under {node}/")
            for node in brief.world.routing(brief.state)
            if _owes(brief.state, node)]


def offline_executor(brief: Brief) -> Sequence[Edit]:
    """Write exactly one file, the way a bounded episode does."""
    path, state = normalise(brief.world.path), brief.state
    for target, content in _PLAN.get(path, ()):
        if target not in state:
            return [Edit(owner=path, path=target, content=content)]
    if path == "src/observe" and CHECKS in state:
        name = _stub_for(brief, state)
        if name is not None:
            stub, filled = _STUBS[name]
            return [Edit(owner=path, path=CHECKS,
                         content=state[CHECKS].replace(stub, filled))]
    return ()


def _stub_for(brief: Brief, state: Mapping[str, str]) -> Optional[str]:
    """Which invariant this episode's failing stage points at."""
    body = state.get(CHECKS, "")
    open_stubs = [n for n, (stub, _) in _STUBS.items() if stub in body]
    if not open_stubs:
        return None
    meta = getattr(brief.task, "meta", None) or {}
    wanted = _STAGE_STUB.get(meta.get("kind"))
    return wanted if wanted in open_stubs else open_stubs[0]
