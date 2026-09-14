"""The fifth formation domain: a fruit fly, its brain, and a web page to watch it on.

The four domains before this one grow a *library*. This one grows a **product**: a
simulated animal, the tasks it is set, the dopamine that changes what it does next, an
HTTP backend and the page a person actually looks at -- and it is the first where the
specification alone had to carry the design, because nothing in this port has ever been
asked for a frontend before.

It is also the first domain shipped **without a reference implementation**. The others
keep one beside them for two jobs that are not scoring: driving the offline rule-based
actor, and letting a test prove the suite is passable at all. Here there is none, and
the cost is stated rather than hidden. There is no `--domain fly` run without a model,
and no `kill_report`: nothing in the repository checks what this suite *rejects*, which
is the check that caught a molecular-dynamics run reporting 1.000 for a force field
that was identically zero. What stands in its place is that every number in the
specification was measured on a throwaway implementation while the suite was being
written -- 91 assertions passing, and three of the specification's least obvious
paragraphs are there because that implementation failed them first:

* a ring attractor sharpened with any *pointwise* nonlinearity quantises heading to the
  nearest wedge, so the lateral term has to be linear;
* an avoidance assay with the punished source in the middle of a walled arena measures
  the arena, not the animal -- the model's valence had correctly reversed by tick 40 and
  it still took 61 shocks an episode, because the walls kept reflecting it back through
  the gradient;
* and `src/learn/` beside a public `learn()` is two things called `src.learn`, where the
  submodule import rebinds the name and every caller after it gets `TypeError: 'module'
  object is not callable`.

The last of those is the same collision this port had already been bitten by twice from
the other direction, which is why the specification names it and `test_integration.py`
asserts it.
"""

from __future__ import annotations

from typing import Dict

from ._suite import PYTHON_MODULE_SKILL, TEST_FAILURE, TestSuite, reward_test
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR

__all__ = ["CASE_NOUN", "CONTRACTS", "FLY", "FROZEN", "GROUP_NOUN", "HELD_OUT_FRAC",
           "OBJECTIVE", "REQUIRES_MODEL", "SCORING", "build_tasks", "initial_files",
           "llm_executor", "llm_manager", "make_runner", "reward", "suite_failures",
           "suite_review"]

#: The specification, the suite and the driver. Human-supplied, refused to every
#: proposal, and restored pristine before scoring.
FROZEN = ("spec/**", "tests/**", "fly.py")

#: What is pushed into every brief, in this order.
CONTRACTS = ("spec/**", "fly.py")

SCORING = "frozen test suite, no reference implementation anywhere in the port"
CASE_NOUN = "test functions"
GROUP_NOUN = "test files"

ENTRY = "src/__init__.py"

#: There is no offline actor for this domain, and pretending otherwise would mean
#: shipping the decomposition the run is meant to invent. See the module docstring.
REQUIRES_MODEL = True

OBJECTIVE = (
    "Build the Drosophila model the frozen driver `fly.py` and the frozen test suite "
    "both import: a Python package rooted at `src/`, whose `src/__init__.py` exposes "
    "exactly `make_brain`, `step_brain`, `sense`, `act`, `learn`, `make_world`, "
    "`step_world`, `run_episode`, `train` and `handle`. It simulates a fly's olfactory "
    "brain -- antennal lobe, mushroom body, lateral horn, central complex -- walking in "
    "an arena, being punished or rewarded, and changing what it does next; and it "
    "serves the whole thing as a web page a person can watch it learn on. "
    "`spec/CONTEXT.md` pins those signatures, the circuit and the assays; everything "
    "below `src/` -- how many modules, what they are called, who owns what -- is yours "
    "to decide. Pure Python, no third-party packages."
)

#: The specification. The only design a person wrote.
_SPEC = r'''# Specification — a Drosophila brain you can watch learn

Build the library the frozen driver `fly.py` and the frozen suite in `tests/` import: a
pure-Python package rooted at `src/` that simulates a fruit fly — its olfactory brain,
its body in an arena, the tasks it is set, and the dopamine that changes what it does
next — and serves the whole thing as a web page.

Reduced units throughout: one body length, one second, one arbitrary unit of odour.
Pure Python, no third-party packages, no numpy. Determinism is a requirement, not a
convenience: the same seed must give the same trajectory on any machine.

## What the animal is

Four structures, and the division of labour between them is the point of the model.

**Antennal lobe.** 24 glomeruli. Olfactory receptor neurons respond across orders of
magnitude of concentration and the projection neurons they drive do not: lateral
inhibition divides each glomerulus by the activity of the whole lobe, so the PN
*pattern* says which odour this is and carries almost nothing about how much of it
there is. Use `r^1.5 / (sigma^1.5 + sum_j r_j^1.5)` with `sigma = 12`.

**Mushroom body.** 400 Kenyon cells, each sampling 6 glomeruli at random, fixed for the
life of the animal. Feedback inhibition (the APL neuron) leaves the most strongly driven
**5%** active — a rank, not a fixed threshold, so the code stays sparse for a faint
odour and a strong one alike. Two output neurons: MBON 0 is *approach*, MBON 1 is
*avoid*, each reading the active cells through plastic weights that start at 1.0.

**Learning is depression, and this is the part that is usually modelled backwards.** A
Kenyon cell active at the same moment as a dopaminergic neuron has its synapse onto that
DAN's MBON *weakened*, floored at zero — `w -= 0.5 * kc * da`. Punishment therefore does
not build an avoid response; it removes the approach response for that odour and lets
the untouched avoid pathway win. Nothing is learned about an odour that was absent,
because its cells were silent and the rule is a coincidence rule.

**Lateral horn.** The projection neurons fork. One branch goes to the mushroom body,
where experience can change what an odour means; the other goes to the lateral horn,
which is stereotyped and does not learn. A naive fly walks towards food odour without
ever having met it. Innate valence is `1.0 * sum(projection)` — positive, so every
odour here starts attractive.

Keeping these two apart is what lets learning show up as a **reversal**: the lateral
horn says approach, the mushroom body after punishment subtracts more than that, the sum
changes sign, and the animal turns around. Fuse them into one number and there is
nothing left to reverse.

**Central complex.** A ring of 16 wedges holding one bump of activity whose position is
the heading. There is no compass: the bump is *shifted* by the animal's own turning, so
heading is dead reckoning and it runs in the dark. Two properties, and both are tested.
Turning by an angle and back must return the bump where it started. And the bump must
neither smear flat nor collapse onto a single wedge — hold it with a **linear** filter,
`(1+2a)` on the wedge against `a` on each neighbour at `a = 0.1`. Any *pointwise*
nonlinearity — squaring, thresholding — snaps the bump toward the nearest wedge and
quantises every heading; measured over four full turns, no lateral term drifts 0.45 rad,
a pointwise power loses the heading entirely, and the linear filter drifts 0.04.

## What the animal has for a body

A square arena, half-width 20, with **reflecting** walls: a fly in a dish has edges, and
an animal that learns to leave the arena has learned nothing. Odour sources each carry a
receptor profile and a concentration falling off as `exp(-distance / 6)`.

**Two antennae**, 0.6 apart, and the animal steers on the difference between them:
`turn = wanted * 12 * (left - right) / (left + right)`, where `wanted` is `+1` when
valence is positive and `-1` when it is negative. That sign is the entire behavioural
consequence of learning.

Bilateral steering has a blind spot exactly where it matters most — pointed straight at
a source both antennae read the same thing — so the temporal gradient covers the head-on
case: when the odour is changing the way the animal does *not* want, it turns, and which
way is one bit of memory it carries between ticks. The relative change is about 2.5% a
tick, so that term needs a gain around 200 to be worth anything.

Walking speed is `4 * (1 + 0.6 * tanh(valence))`, turning is capped at 3 rad/s, the tick
is 0.05 s and an episode is at most 400 ticks.

## Three assays

| task | arena | paid |
|---|---|---|
| `taxis` | one safe source at the centre | `+1` on arrival, which ends the episode |
| `avoid` | one punished source, off centre | `-1` every tick within 5 of it |
| `choice` | a punished source and a safe one, opposite ends | both of the above |

`avoid` and `choice` put the punished source **off centre on purpose**. With it in the
middle of a walled arena there is nowhere to go: the animal learns the odour is bad,
turns away, hits the wall, is reflected back through the gradient and is punished again
— 61 shocks an episode from a model whose valence had correctly reversed by tick 40. An
avoidance assay has to contain somewhere safe or it measures the arena.

Reward is paid **per tick**, not per episode, because the plasticity rule is a
coincidence rule: it can only act on what the animal was smelling when the dopamine
arrived.

## `src/__init__.py` exposes exactly these ten names

```python
make_brain(config=None) -> dict
    # config keys: seed, glomeruli, kenyon_cells, wedges, heading. All optional.

step_brain(brain, sensory, dt=0.05) -> activity     # mutates brain
sense(world) -> sensory
act(activity) -> (forward, turn)
learn(brain, activity, reward) -> brain             # mutates brain

make_world(task="taxis", seed=0, glomeruli=24) -> world
step_world(world, action, dt=0.05) -> (world, reward, done)

run_episode(brain, task="taxis", seed=0, dt=0.05, learning=True) -> result
train(task="taxis", episodes=20, seed=0) -> {"task", "episodes": [...], "brain"}

handle(method, path, body=None) -> (status, content_type, body)
```

Each MUST import its layer **lazily**, inside the function body, so a missing layer does
not stop the layers before it from passing their tests.

**No module below `src/` may be named after one of these ten.** A package laid out with
`src/learn/` beside a public `learn()` has two things called `src.learn`; the moment
anything inside imports the submodule, the name on the package is rebound to the module
and every caller gets `TypeError: 'module' object is not callable`. Tests that call the
function first still pass. The one that calls `train()` first does not.

### `sensory`
`odor` (one float per glomerulus), `odor_left`, `odor_right`, `angular_velocity`.

### `activity`
`projection` (24 floats), `kenyon` (one 0.0/1.0 per cell), `mbon` (2 floats, approach
then avoid), `ring` (one float per wedge, summing to 1), `heading` (the compass, radians
in `(-pi, pi]`), `innate`, `learned` (= `mbon[0] - mbon[1]`), `valence` (= `innate +
learned`), `delta_drive`, `tumble_sign`, `odor_left`, `odor_right`.

### `world`
`task`, `seed`, `glomeruli`, `t`, `step`, `position` (x, y), `heading`,
`angular_velocity`, `sources` (each `name`, `position`, `profile`, `strength`,
`punished`), `shocked`.

### `result` from an episode
`reward`, `shocks`, `steps`, `nearest`, `distance`, `reached`.

## The web page

`handle` is a **pure function** — it opens no socket. `fly.py` wraps it in
`http.server` and does nothing else, so the part worth testing is testable without one.

| route | does |
|---|---|
| `GET /` | the page: one HTML file, no build step, no CDN, no network |
| `GET /api/tasks` | `{"tasks": [...]}` |
| `GET /api/state` | the arena, the brain, and the episode history |
| `POST /api/reset` | `{task, seed, brain}` — `brain: "new"` forgets what was learned |
| `POST /api/step` | `{steps, learning}` — advance the live animal |
| `POST /api/train` | `{task, episodes}` — run episodes, append to the history |

Anything else is `404` with a JSON body; a body that is not JSON is `400`.

The page must draw, from `/api/state` and nothing else: the **arena** with the animal,
the sources and the punished region; the **Kenyon-cell code** as a raster with the
active cells marked; the **ellipsoid body** as a ring with the bump and the compass
heading; the **valence** panel breaking the number into innate, approach, avoid and net;
and the **learning curve**, shocks per episode. Controls for task, play/pause, step,
train, and a fresh brain. It must work at phone width and follow the viewer's colour
scheme.

## How this is scored

A frozen suite in `tests/`, and no reference implementation anywhere in the loop.
One task per test function; the prompt is that test's own source.

| file | asserts |
|---|---|
| `test_antennal_lobe.py` | bounds, compression in the range the arena uses, order preserved under rescaling |
| `test_mushroom_body.py` | sparsity at every concentration, the same odour lights the same cells, separation, depression only, compartment specificity |
| `test_central_complex.py` | one bump, neither flat nor collapsed, reversible, accurate over four turns, runs in the dark |
| `test_body.py` | two antennae, the sign of steering, walls, containment, determinism |
| `test_tasks.py` | each assay's shape, that `avoid` has a safe region, what is paid and when |
| `test_learning.py` | valence reverses, other odours keep theirs, shocks fall, and it transfers to arenas it never saw |
| `test_server.py` | every route, every JSON shape, the page, 404 and 400 |
| `test_integration.py` | the loop closes by hand, and no module shadows an entry point |

A further set of assertions is **held out**: the same requirements asked differently, in
files that are not in the repository and that no agent can read, injected only at
scoring time. That is the honest number, because an executor is shown the source of the
test it is failing and could otherwise write to that one assertion.
'''

#: The driver: a program you can run, rather than a package nobody can invoke.
_DRIVER = r'''#!/usr/bin/env python3
"""fly -- a Drosophila brain you can watch learn.

Human-supplied, frozen, and never written by an agent: the model under ``src/`` is what
the run grows, and this is the shell around it, so the result is a program you can run
rather than a package nobody can invoke.

    python fly.py train --task avoid --episodes 20
    python fly.py probe --task avoid --episodes 10 --trials 5
    python fly.py serve --port 8000

It touches the library only through the entry points ``spec/CONTEXT.md`` pins, so it
runs against any implementation that passes ``tests/``, whatever the internal layout
turns out to be. Pure Python and no third-party packages: this has to run anywhere.
"""

import argparse
import json
import sys


def cmd_train(args):
    from src import train
    result = train(args.task, args.episodes, args.seed)
    rows = result["episodes"]
    print(f"{'ep':>4} {'shocks':>7} {'reward':>8} {'steps':>6}  nearest")
    for row in rows:
        print(f"{row['episode']:>4} {row['shocks']:>7} {row['reward']:>8.1f} "
              f"{row['steps']:>6}  {row['nearest']}")
    head = sum(r["shocks"] for r in rows[:3]) / max(1, len(rows[:3]))
    tail = sum(r["shocks"] for r in rows[-3:]) / max(1, len(rows[-3:]))
    print(f"\nfirst 3 episodes: {head:.1f} shocks    last 3: {tail:.1f}")
    return 0


def cmd_probe(args):
    """Train one animal, then test it and a naive one with plasticity switched off.

    The honest measurement of learning: same arenas, seeds neither animal was trained
    on, and nothing changing during the test.
    """
    from src import make_brain, run_episode, train
        trained = train(args.task, args.episodes, args.seed)["brain"]
    naive = make_brain({"seed": args.seed})
    out = {}
    for label, brain in (("naive", naive), ("trained", trained)):
        shocks = [run_episode(brain, args.task, 1000 + i, learning=False)["shocks"]
                  for i in range(args.trials)]
        out[label] = sum(shocks)
        print(f"{label:>8}: {sum(shocks):>5} shocks over {args.trials} unseen arenas  {shocks}")
    print(f"\nlearned reduction: {out['naive'] - out['trained']} shocks")
    return 0


def cmd_serve(args):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from src import handle

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else None
            status, content_type, payload = handle(method, self.path, body)
            data = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._respond("GET")

        def do_POST(self):
            self._respond("POST")

        def log_message(self, *a):
            pass

    server = HTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/")
    sys.stdout.flush()
    server.serve_forever()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("probe", cmd_probe), ("serve", cmd_serve)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("train", "probe"):
            p.add_argument("--task", default="avoid")
            p.add_argument("--episodes", type=int, default=20)
            p.add_argument("--seed", type=int, default=0)
        if name == "probe":
            p.add_argument("--trials", type=int, default=5)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: The suite. One task per test function; the prompt is the test's own source.
_TESTS = {
    'tests/test_antennal_lobe.py': r'''"""The antennal lobe: what survives a change of concentration, and what does not."""

from src import make_brain, step_brain

GLOMERULI = 24


def odour(seed, n=GLOMERULI):
    """A reproducible receptor pattern. Squared uniforms: a few glomeruli dominate."""
    import random
    rng = random.Random(seed)
    return [rng.random() ** 2 for _ in range(n)]


def present(brain, receptors, scale=1.0):
    return step_brain(brain, {"odor": [r * scale for r in receptors]})


def test_projection_is_one_rate_per_glomerulus():
    brain = make_brain({"seed": 0})
    activity = present(brain, odour(1))
    assert len(activity["projection"]) == GLOMERULI


def test_projection_rates_are_bounded_below_one():
    """Divisive normalisation: a glomerulus is a share of the lobe, never more than it."""
    brain = make_brain({"seed": 0})
    for scale in (0.01, 1.0, 100.0, 10000.0):
        activity = present(brain, odour(2), scale)
        assert all(0.0 <= p < 1.0 for p in activity["projection"])
        assert sum(activity["projection"]) < 1.0


def test_a_silent_antenna_gives_a_silent_lobe():
    brain = make_brain({"seed": 0})
    activity = step_brain(brain, {"odor": [0.0] * GLOMERULI})
    assert all(p == 0.0 for p in activity["projection"])


def test_the_lobe_compresses_concentration_where_the_arena_puts_it():
    """Ten times the odour is nowhere near ten times the response -- in range.

    The range matters and the test states it. Far below saturation the lobe is if
    anything expansive; what the arena actually delivers at a source is an order of
    magnitude above that, and there ten times the odour buys a few per cent.
    """
    brain = make_brain({"seed": 0})
    weak = sum(present(brain, odour(3), 10.0)["projection"])
    strong = sum(present(brain, odour(3), 100.0)["projection"])
    assert weak < strong < 1.5 * weak


def test_more_odour_is_never_less_response():
    brain = make_brain({"seed": 0})
    totals = [sum(present(brain, odour(7), s)["projection"])
              for s in (0.1, 1.0, 10.0, 100.0, 1000.0)]
    assert totals == sorted(totals)


def test_the_strongest_glomerulus_stays_the_strongest():
    """Normalisation rescales the pattern; it must not reorder it."""
    brain = make_brain({"seed": 0})
    receptors = odour(4)
    loudest = max(range(GLOMERULI), key=lambda g: receptors[g])
    for scale in (0.5, 5.0, 50.0):
        projection = present(brain, receptors, scale)["projection"]
        assert max(range(GLOMERULI), key=lambda g: projection[g]) == loudest


def test_two_odours_give_two_different_patterns():
    brain = make_brain({"seed": 0})
    a = present(brain, odour(5))["projection"]
    b = present(brain, odour(6))["projection"]
    assert any(abs(x - y) > 1e-6 for x, y in zip(a, b))
''',
    'tests/test_body.py': r'''"""The body: two antennae, a pair of legs, and walls that do not let go."""

import math

from src import act, make_world, sense, step_world

SIZE = 20.0


def test_sensing_gives_one_reading_per_glomerulus_and_one_per_antenna():
    world = make_world("taxis", 0)
    sensory = sense(world)
    assert len(sensory["odor"]) == world["glomeruli"]
    assert sensory["odor_left"] >= 0.0 and sensory["odor_right"] >= 0.0


def test_the_antennae_read_more_odour_nearer_the_source():
    near = make_world("taxis", 0)
    near["position"] = (2.0, 0.0)
    far = make_world("taxis", 0)
    far["position"] = (18.0, 0.0)
    assert sum(sense(near)["odor"]) > 10.0 * sum(sense(far)["odor"])


def test_the_antenna_facing_the_source_reads_more():
    """Bilateral comparison is the only thing that tells the animal which way to turn."""
    world = make_world("taxis", 0)
    world["position"] = (10.0, 0.0)
    # Facing +y from x=+10 with the source at the origin puts it to the animal's left.
    world["heading"] = math.pi / 2.0
    assert sense(world)["odor_left"] > sense(world)["odor_right"]
    # Turn it around and the same source is on the other side.
    world["heading"] = -math.pi / 2.0
    assert sense(world)["odor_right"] > sense(world)["odor_left"]


def test_an_attractive_odour_turns_the_animal_toward_the_stronger_antenna():
    left = act({"mbon": [1.0, 0.0], "valence": 1.0, "odor_left": 2.0,
                "odor_right": 1.0, "delta_drive": 0.0})[1]
    right = act({"mbon": [1.0, 0.0], "valence": 1.0, "odor_left": 1.0,
                 "odor_right": 2.0, "delta_drive": 0.0})[1]
    assert left > 0.0 > right


def test_an_aversive_odour_turns_the_animal_the_other_way():
    """The same gradient, the opposite sign -- and that is all learning changes."""
    attracted = act({"mbon": [1.0, 0.0], "valence": 1.0, "odor_left": 2.0,
                     "odor_right": 1.0, "delta_drive": 0.0})[1]
    repelled = act({"mbon": [0.0, 1.0], "valence": -1.0, "odor_left": 2.0,
                    "odor_right": 1.0, "delta_drive": 0.0})[1]
    assert attracted > 0.0 > repelled


def test_the_animal_walks_faster_toward_something_it_likes():
    liked = act({"mbon": [1.0, 0.0], "valence": 1.0, "odor_left": 1.0,
                 "odor_right": 1.0, "delta_drive": 0.0})[0]
    disliked = act({"mbon": [0.0, 1.0], "valence": -1.0, "odor_left": 1.0,
                    "odor_right": 1.0, "delta_drive": 0.0})[0]
    assert liked > disliked > 0.0


def test_the_animal_stays_in_the_arena():
    world = make_world("taxis", 0)
    for _ in range(600):
        world, _, _ = step_world(world, (8.0, 0.3))
        assert abs(world["position"][0]) <= SIZE + 1e-6
        assert abs(world["position"][1]) <= SIZE + 1e-6


def test_walking_into_a_wall_turns_the_animal_around():
    world = make_world("taxis", 0)
    world["position"] = (SIZE - 0.1, 0.0)
    world["heading"] = 0.0
    world, _, _ = step_world(world, (8.0, 0.0))
    assert math.cos(world["heading"]) < 0.0


def test_the_world_is_a_function_of_its_seed():
    one = make_world("taxis", 3)
    two = make_world("taxis", 3)
    assert one["position"] == two["position"]
    assert make_world("taxis", 4)["position"] != one["position"]


def test_a_step_reports_reward_and_whether_the_episode_is_over():
    world = make_world("taxis", 0)
    world, reward, done = step_world(world, (0.0, 0.0))
    assert isinstance(reward, float) and isinstance(done, bool)
''',
    'tests/test_central_complex.py': r'''"""The ellipsoid body: one bump, and what it costs to move it accurately."""

import math

from src import make_brain, step_brain

SILENT = {"odor": [0.0] * 24}


def turn(brain, rate, seconds, dt=0.05):
    """Spin the animal at `rate` rad/s for `seconds` and return the last activity."""
    activity = None
    for _ in range(int(round(seconds / dt))):
        activity = step_brain(brain, dict(SILENT, angular_velocity=rate), dt)
    return activity


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def test_the_ring_is_a_distribution():
    brain = make_brain({"seed": 0})
    ring = step_brain(brain, SILENT)["ring"]
    assert len(ring) >= 8
    assert all(a >= 0.0 for a in ring)
    assert abs(sum(ring) - 1.0) < 1e-6


def test_there_is_exactly_one_bump():
    brain = make_brain({"seed": 0})
    ring = turn(brain, 0.8, 1.3)["ring"]
    n = len(ring)
    peaks = sum(1 for i in range(n) if ring[i] > ring[i - 1] and ring[i] >= ring[(i + 1) % n])
    assert peaks == 1, ring


def test_the_bump_neither_smears_flat_nor_collapses_to_one_wedge():
    """Both failures look fine on a reversibility test and neither is a compass.

    Flat is what happens with no lateral term at all; one wedge is what a pointwise
    nonlinearity gives, and it quantises every heading to a multiple of 22.5 degrees.
    """
    brain = make_brain({"seed": 0})
    ring = turn(brain, 1.0, 2.0)["ring"]
    uniform = 1.0 / len(ring)
    assert max(ring) > 2.0 * uniform
    assert max(ring) < 0.6


def test_turning_back_returns_the_bump_where_it_started():
    """The integration is reversible, which is the property a dead-reckoner needs."""
    brain = make_brain({"seed": 0})
    start = step_brain(brain, SILENT)["heading"]
    turn(brain, 1.0, 1.0)
    back = turn(brain, -1.0, 1.0)["heading"]
    assert abs(wrapped(back - start)) < 0.05


def test_the_heading_follows_the_turn():
    brain = make_brain({"seed": 0})
    start = step_brain(brain, SILENT)["heading"]
    after = turn(brain, 1.0, 1.0)["heading"]
    assert wrapped(after - start) > 0.5
    back = turn(brain, -2.0, 1.0)["heading"]
    assert wrapped(back - after) < -0.5


def test_the_compass_survives_several_full_turns():
    """Four turns of dead reckoning, and the error stays a fraction of a wedge."""
    brain = make_brain({"seed": 0})
    start = step_brain(brain, SILENT)["heading"]
    total = 4.0 * 2.0 * math.pi
    after = turn(brain, 1.0, total)["heading"]
    assert abs(wrapped(after - start - total)) < 0.4


def test_standing_still_does_not_move_the_compass():
    brain = make_brain({"seed": 0})
    start = step_brain(brain, SILENT)["heading"]
    after = turn(brain, 0.0, 5.0)["heading"]
    assert abs(wrapped(after - start)) < 1e-6


def test_the_compass_runs_in_the_dark():
    """Nothing external drives it: with no odour at all the bump still tracks turning."""
    brain = make_brain({"seed": 0})
    after = turn(brain, 1.5, 1.0)["heading"]
    assert abs(wrapped(after - 1.5)) < 0.3
''',
    'tests/test_integration.py': r'''"""The whole animal, and the one way a package can break itself."""

import json
import random

from src import (act, handle, learn, make_brain, make_world, run_episode, sense,
                 step_brain, step_world, train)


def test_every_public_name_is_callable():
    """A module is not a function. This catches the one import mistake that hides.

    A package laid out so that `src/learn/` sits beside a public `learn()` has two
    things called `src.learn`, and the moment anything inside imports the submodule the
    name on the package is rebound to the module. Every test that calls the function
    first still passes; the one that calls `train()` first gets `TypeError: 'module'
    object is not callable`. No module below `src/` may be named after an entry point.
    """
    for name in (act, handle, learn, make_brain, make_world, run_episode, sense,
                 step_brain, step_world, train):
        assert callable(name), name


def test_training_first_does_not_break_the_other_entry_points():
    """The order that catches it: touch the deep layers, then use the shallow names."""
    train("taxis", episodes=1, seed=0)
    handle("GET", "/api/state")
    brain = make_brain({"seed": 0})
    rng = random.Random(0)
    activity = step_brain(brain, {"odor": [rng.random() for _ in range(24)]})
    learn(brain, activity, -1.0)
    world = make_world("taxis", 0)
    step_world(world, act(activity))
    run_episode(brain, "taxis", 0, learning=False)


def test_the_loop_closes_by_hand():
    """Sense, think, act, move, be paid, learn -- in that order, with nothing hidden."""
    brain = make_brain({"seed": 0})
    world = make_world("avoid", 0)
    total = 0.0
    for _ in range(120):
        activity = step_brain(brain, sense(world))
        world, reward, done = step_world(world, act(activity))
        total += reward
        if reward:
            learn(brain, activity, reward)
        if done:
            break
    assert world["step"] > 0
    assert total <= 0.0


def test_the_animal_the_server_shows_is_the_animal_that_learns():
    handle("POST", "/api/reset", json.dumps({"task": "avoid", "seed": 0, "brain": "new"}))
    handle("POST", "/api/step", json.dumps({"steps": 1}))
    before = json.loads(handle("GET", "/api/state")[2])["brain"]["learned"]
    handle("POST", "/api/train", json.dumps({"task": "avoid", "episodes": 4}))
    handle("POST", "/api/step", json.dumps({"steps": 1}))
    after = json.loads(handle("GET", "/api/state")[2])["brain"]["learned"]
    assert before == 0.0 > after


def test_a_brain_is_plain_data():
    """No classes required, and nothing that cannot be written to a file."""
    brain = make_brain({"seed": 0})
    assert isinstance(brain, dict)


def test_the_model_scales_down():
    """The sizes are configuration, not constants baked into the wiring."""
    small = make_brain({"seed": 0, "kenyon_cells": 60, "wedges": 8})
    activity = step_brain(small, {"odor": [1.0] * 24})
    assert len(activity["kenyon"]) == 60
    assert len(activity["ring"]) == 8
''',
    'tests/test_learning.py': r'''"""What the animal does differently after it has been punished."""

import random

from src import learn, make_brain, run_episode, step_brain, train

GLOMERULI = 24


def odour(seed, n=GLOMERULI):
    rng = random.Random(seed)
    return [rng.random() ** 2 for _ in range(n)]


def present(brain, receptors, scale=10.0):
    return step_brain(brain, {"odor": [r * scale for r in receptors]})


def test_a_naive_animal_finds_every_odour_attractive():
    """Innate valence comes from the lateral horn, and it is there before experience."""
    brain = make_brain({"seed": 0})
    for seed in range(5):
        assert present(brain, odour(seed))["valence"] > 0.0


def test_valence_is_innate_plus_learned_and_nothing_else():
    brain = make_brain({"seed": 0})
    activity = present(brain, odour(30))
    assert abs(activity["valence"] - (activity["innate"] + activity["learned"])) < 1e-9


def test_a_naive_animal_has_learned_nothing():
    brain = make_brain({"seed": 0})
    assert abs(present(brain, odour(31))["learned"]) < 1e-9


def test_punishment_reverses_an_odour_the_animal_used_to_approach():
    """The headline: the sign of valence flips, and the innate part never moved."""
    brain = make_brain({"seed": 0})
    before = present(brain, odour(32))
    for _ in range(8):
        learn(brain, present(brain, odour(32)), -1.0)
    after = present(brain, odour(32))
    assert before["valence"] > 0.0 > after["valence"]
    assert abs(after["innate"] - before["innate"]) < 1e-9
    assert after["learned"] < -0.5


def test_an_odour_that_was_never_punished_keeps_its_valence():
    brain = make_brain({"seed": 0})
    before = present(brain, odour(34))["valence"]
    for _ in range(8):
        learn(brain, present(brain, odour(33)), -1.0)
    assert present(brain, odour(34))["valence"] > 0.8 * before


def test_training_reduces_the_shocks_an_animal_takes():
    result = train("avoid", episodes=12, seed=0)
    shocks = [e["shocks"] for e in result["episodes"]]
    first = sum(shocks[:3]) / 3.0
    last = sum(shocks[-3:]) / 3.0
    assert first > 0.0
    assert last < 0.5 * first


def test_what_was_learned_transfers_to_arenas_it_never_saw():
    """Plasticity off, seeds it was not trained on: the only honest test of learning."""
    trained = train("avoid", episodes=10, seed=0)["brain"]
    naive = make_brain({"seed": 0})
    unseen = range(1000, 1005)
    before = sum(run_episode(naive, "avoid", s, learning=False)["shocks"] for s in unseen)
    after = sum(run_episode(trained, "avoid", s, learning=False)["shocks"] for s in unseen)
    assert before > 0
    assert after < 0.25 * before


def test_training_reports_one_row_per_episode():
    result = train("taxis", episodes=4, seed=0)
    assert len(result["episodes"]) == 4
    for i, row in enumerate(result["episodes"]):
        assert row["episode"] == i
        for key in ("shocks", "reward", "steps", "nearest", "distance"):
            assert key in row


def test_training_is_reproducible():
    one = train("avoid", episodes=5, seed=0)["episodes"]
    two = train("avoid", episodes=5, seed=0)["episodes"]
    assert [e["shocks"] for e in one] == [e["shocks"] for e in two]


def test_an_episode_with_learning_off_changes_nothing():
    brain = make_brain({"seed": 0})
    run_episode(brain, "avoid", 0, learning=False)
    assert abs(present(brain, odour(35))["learned"]) < 1e-9


def test_a_naive_animal_walks_to_an_unpunished_source():
    """Innate approach plus bilateral steering is enough; nothing is learned here."""
    brain = make_brain({"seed": 0})
    reached = sum(run_episode(brain, "taxis", s, learning=False)["reached"]
                  for s in range(5))
    assert reached >= 4
''',
    'tests/test_mushroom_body.py': r'''"""Kenyon cells: a sparse code, the same every time, and what dopamine does to it."""

from src import learn, make_brain, step_brain

GLOMERULI = 24


def odour(seed, n=GLOMERULI):
    import random
    rng = random.Random(seed)
    return [rng.random() ** 2 for _ in range(n)]


def present(brain, receptors, scale=10.0):
    return step_brain(brain, {"odor": [r * scale for r in receptors]})


def active(code):
    return {i for i, c in enumerate(code) if c}


def test_the_code_is_sparse():
    """A few per cent of the population, which is what makes odours separable."""
    brain = make_brain({"seed": 0})
    for seed in range(8):
        code = present(brain, odour(seed))["kenyon"]
        fraction = len(active(code)) / len(code)
        assert 0.01 <= fraction <= 0.15, fraction


def test_the_code_is_sparse_at_every_concentration():
    """Feedback inhibition sets a rank, not a fixed threshold."""
    brain = make_brain({"seed": 0})
    for scale in (0.1, 1.0, 10.0, 1000.0):
        code = present(brain, odour(9), scale)["kenyon"]
        assert 0.01 <= len(active(code)) / len(code) <= 0.15


def test_the_same_odour_always_lights_the_same_cells():
    brain = make_brain({"seed": 0})
    first = active(present(brain, odour(10))["kenyon"])
    for _ in range(5):
        assert active(present(brain, odour(10))["kenyon"]) == first


def test_concentration_does_not_change_which_cells_fire():
    """Identity is the pattern; the projection neurons have already removed the rest."""
    brain = make_brain({"seed": 0})
    weak = active(present(brain, odour(11), 1.0)["kenyon"])
    strong = active(present(brain, odour(11), 20.0)["kenyon"])
    overlap = len(weak & strong) / max(1, len(weak))
    assert overlap > 0.8, overlap


def test_two_odours_barely_share_cells():
    brain = make_brain({"seed": 0})
    a = active(present(brain, odour(12))["kenyon"])
    b = active(present(brain, odour(13))["kenyon"])
    assert len(a & b) / max(1, len(a)) < 0.4


def test_two_brains_wire_their_kenyon_cells_differently():
    """The claw map is drawn per animal. Two flies do not share a code."""
    one = active(present(make_brain({"seed": 1}), odour(14))["kenyon"])
    two = active(present(make_brain({"seed": 2}), odour(14))["kenyon"])
    assert one != two


def test_punishment_only_ever_removes_drive():
    """Depression, not potentiation: no MBON rate rises because of learning."""
    brain = make_brain({"seed": 0})
    before = present(brain, odour(15))["mbon"]
    learn(brain, present(brain, odour(15)), -1.0)
    after = present(brain, odour(15))["mbon"]
    assert all(a <= b + 1e-9 for a, b in zip(after, before))


def test_punishment_depresses_approach_and_leaves_avoid_alone():
    brain = make_brain({"seed": 0})
    before = present(brain, odour(16))["mbon"]
    for _ in range(6):
        learn(brain, present(brain, odour(16)), -1.0)
    after = present(brain, odour(16))["mbon"]
    assert after[0] < before[0] - 0.1
    assert abs(after[1] - before[1]) < 1e-9


def test_reward_depresses_avoid_and_leaves_approach_alone():
    brain = make_brain({"seed": 0})
    before = present(brain, odour(17))["mbon"]
    for _ in range(6):
        learn(brain, present(brain, odour(17)), 1.0)
    after = present(brain, odour(17))["mbon"]
    assert after[1] < before[1] - 0.1
    assert abs(after[0] - before[0]) < 1e-9


def test_learning_is_specific_to_the_odour_that_was_present():
    """A coincidence rule can only reach cells that fired, so another odour barely moves.

    Barely, not never: sparse codes drawn at random do share the occasional cell, and
    that shared cell is generalisation rather than a defect. Measured on the reference,
    eight pairings take the trained odour's approach drive to zero and leave the other
    at 0.95 -- the one cell in twenty the two codes have in common.
    """
    brain = make_brain({"seed": 0})
    trained_before = present(brain, odour(18))["mbon"][0]
    other_before = present(brain, odour(19))["mbon"][0]
    for _ in range(8):
        learn(brain, present(brain, odour(18)), -1.0)
    trained_after = present(brain, odour(18))["mbon"][0]
    other_after = present(brain, odour(19))["mbon"][0]
    assert trained_after < 0.1 * trained_before
    assert other_after > 0.8 * other_before


def test_nothing_is_learned_without_a_signal():
    brain = make_brain({"seed": 0})
    before = present(brain, odour(20))["mbon"]
    for _ in range(5):
        learn(brain, present(brain, odour(20)), 0.0)
    assert present(brain, odour(20))["mbon"] == before


def test_mbon_rates_never_go_negative():
    """A synapse can be removed. It cannot be run backwards."""
    brain = make_brain({"seed": 0})
    for _ in range(200):
        learn(brain, present(brain, odour(21)), -1.0)
    assert all(m >= 0.0 for m in present(brain, odour(21))["mbon"])
''',
    'tests/test_server.py': r'''"""The backend, as a function. No socket is opened anywhere in this file.

`handle(method, path, body)` is the whole API surface. The frozen `fly.py` wraps it in
`http.server`, and a shell around a pure function is the part not worth testing.
"""

import json

from src import handle


def get(path):
    status, content_type, body = handle("GET", path)
    return status, content_type, body


def post(path, payload=None):
    status, content_type, body = handle("POST", path,
                                        json.dumps(payload) if payload is not None else None)
    return status, content_type, body


def state():
    status, _, body = post("/api/reset", {"task": "taxis", "seed": 0, "brain": "new"})
    assert status == 200
    return json.loads(body)


def test_the_root_serves_a_page():
    status, content_type, body = get("/")
    assert status == 200
    assert "text/html" in content_type
    assert body.lstrip().lower().startswith("<!doctype html")


def test_the_page_draws_the_arena_and_the_brain():
    """What the page must show, named. A backend nobody can see is half a product."""
    _, _, page = get("/")
    for element in ("canvas", "arena", "kenyon", "ring", "mbon"):
        assert element.lower() in page.lower(), element


def test_the_page_only_calls_routes_that_exist():
    _, _, page = get("/")
    for route in ("/api/state", "/api/step", "/api/reset", "/api/train"):
        assert route in page


def test_unknown_routes_are_refused():
    status, content_type, body = get("/api/nonsense")
    assert status == 404
    assert "application/json" in content_type
    assert "error" in json.loads(body)


def test_a_body_that_is_not_json_is_refused():
    status, _, body = handle("POST", "/api/step", "{not json")
    assert status == 400
    assert "error" in json.loads(body)


def test_the_task_list_is_served():
    status, _, body = get("/api/tasks")
    assert status == 200
    tasks = json.loads(body)["tasks"]
    assert set(tasks) == {"taxis", "avoid", "choice"}


def test_state_reports_the_arena():
    data = state()
    for key in ("task", "step", "position", "heading", "sources", "shocked"):
        assert key in data, key
    assert len(data["position"]) == 2


def test_state_reports_the_brain():
    post("/api/step", {"steps": 1})
    data = json.loads(get("/api/state")[2])
    brain = data["brain"]
    for key in ("kenyon", "kenyon_cells", "projection", "ring", "mbon",
                "innate", "learned", "valence"):
        assert key in brain, key
    assert 0 < len(brain["kenyon"]) < brain["kenyon_cells"]


def test_stepping_advances_the_animal():
    before = state()
    after = json.loads(post("/api/step", {"steps": 20})[2])
    assert after["step"] > before["step"]
    assert after["position"] != before["position"]


def test_resetting_puts_the_animal_back():
    state()
    moved = json.loads(post("/api/step", {"steps": 30})[2])
    assert moved["step"] > 0
    fresh = json.loads(post("/api/reset", {"task": "taxis", "seed": 0})[2])
    assert fresh["step"] == 0


def test_a_reset_can_keep_the_brain_or_replace_it():
    """Two different things a person watching wants, and one flag between them."""
    post("/api/reset", {"task": "avoid", "seed": 0, "brain": "new"})
    post("/api/train", {"task": "avoid", "episodes": 4})
    post("/api/step", {"steps": 1})
    learned = json.loads(get("/api/state")[2])["brain"]["learned"]
    assert learned < 0.0
    post("/api/reset", {"task": "avoid", "seed": 0})
    post("/api/step", {"steps": 1})
    assert json.loads(get("/api/state")[2])["brain"]["learned"] < 0.0
    post("/api/reset", {"task": "avoid", "seed": 0, "brain": "new"})
    post("/api/step", {"steps": 1})
    assert abs(json.loads(get("/api/state")[2])["brain"]["learned"]) < 1e-9


def test_training_through_the_api_fills_in_the_history():
    post("/api/reset", {"task": "avoid", "seed": 0, "brain": "new"})
    data = json.loads(post("/api/train", {"task": "avoid", "episodes": 5})[2])
    assert len(data["history"]) >= 5
    assert all("shocks" in row for row in data["history"])


def test_switching_task_switches_the_arena():
    one = json.loads(post("/api/reset", {"task": "taxis", "seed": 0})[2])
    two = json.loads(post("/api/reset", {"task": "choice", "seed": 0})[2])
    assert one["task"] == "taxis" and two["task"] == "choice"
    assert len(two["sources"]) == 2


def test_every_response_is_the_type_it_says_it_is():
    for method, path in (("GET", "/"), ("GET", "/api/state"), ("GET", "/api/tasks")):
        status, content_type, body = handle(method, path)
        assert status == 200
        if "json" in content_type:
            json.loads(body)


def test_a_trailing_slash_is_the_same_route():
    assert get("/api/tasks")[0] == get("/api/tasks/")[0] == 200


def test_a_query_string_is_ignored():
    assert get("/api/state?t=1")[0] == 200
''',
    'tests/test_tasks.py': r'''"""The three assays, and the properties each one has to have to measure anything."""

import math

from src import make_world, step_world

TASKS = ("taxis", "avoid", "choice")
SHOCK_RADIUS = 5.0


def distance(world, name):
    for source in world["sources"]:
        if source["name"] == name:
            return math.hypot(world["position"][0] - source["position"][0],
                              world["position"][1] - source["position"][1])
    raise AssertionError(f"no source {name}")


def test_every_task_builds():
    for task in TASKS:
        world = make_world(task, 0)
        assert world["task"] == task and world["sources"]


def test_an_unknown_task_is_refused():
    try:
        make_world("fly-to-the-moon", 0)
    except ValueError:
        return
    raise AssertionError("an unknown task should raise ValueError")


def test_taxis_has_one_source_and_nothing_punished():
    world = make_world("taxis", 0)
    assert len(world["sources"]) == 1
    assert not any(s.get("punished") for s in world["sources"])


def test_the_animal_does_not_start_on_top_of_the_source():
    for task in TASKS:
        for seed in range(6):
            world = make_world(task, seed)
            assert distance(world, "A") > 2.0


def test_avoid_punishes_its_source():
    world = make_world("avoid", 0)
    assert any(s.get("punished") for s in world["sources"])


def test_avoid_leaves_somewhere_safe_to_go():
    """An assay with no safe region measures the arena, not the animal.

    With the punished source in the middle of a walled arena the animal learns the
    odour is bad, turns away, hits the wall, is reflected back through the gradient and
    is punished again -- 61 shocks an episode from a model whose valence had correctly
    reversed by tick 40.
    """
    world = make_world("avoid", 0)
    punished = [s for s in world["sources"] if s.get("punished")][0]
    corners = [(x, y) for x in (-20.0, 20.0) for y in (-20.0, 20.0)]
    far = max(math.hypot(c[0] - punished["position"][0], c[1] - punished["position"][1])
              for c in corners)
    assert far > 3.0 * SHOCK_RADIUS


def test_being_at_the_punished_source_costs():
    world = make_world("avoid", 0)
    punished = [s for s in world["sources"] if s.get("punished")][0]
    world["position"] = tuple(punished["position"])
    _, reward, _ = step_world(world, (0.0, 0.0))
    assert reward < 0.0


def test_being_far_from_everything_costs_nothing():
    world = make_world("avoid", 0)
    world["position"] = (19.0, 19.0)
    _, reward, _ = step_world(world, (0.0, 0.0))
    assert reward == 0.0


def test_choice_offers_two_odours_and_punishes_one():
    world = make_world("choice", 0)
    assert len(world["sources"]) == 2
    assert sum(bool(s.get("punished")) for s in world["sources"]) == 1


def test_the_two_odours_in_a_choice_smell_different():
    world = make_world("choice", 0)
    a, b = world["sources"]
    assert any(abs(x - y) > 1e-6 for x, y in zip(a["profile"], b["profile"]))


def test_reaching_a_safe_source_ends_the_episode():
    world = make_world("taxis", 0)
    world["position"] = world["sources"][0]["position"]
    _, reward, done = step_world(world, (0.0, 0.0))
    assert done and reward > 0.0


def test_an_episode_does_not_run_for_ever():
    world = make_world("avoid", 0)
    world["position"] = (19.0, 19.0)
    done = False
    for _ in range(2000):
        world, _, done = step_world(world, (0.0, 0.0))
        if done:
            break
    assert done
''',
}

#: Held out: the same requirements asked differently, never in the repository.
_AUDIT = {
    'audit/test_behaviour.py': r'''"""The audit set: the same animal asked differently, and never in the repository.

Every one of these restates something `tests/` already requires. They exist because an
executor is shown the source of the test it is failing and could otherwise write to
that one assertion -- so the headline number is measured on tests no agent has read.
"""

import math
import random

from src import learn, make_brain, run_episode, step_brain, train

GLOMERULI = 24


def odour(seed, n=GLOMERULI):
    rng = random.Random(seed)
    return [rng.random() ** 2 for _ in range(n)]


def present(brain, receptors, scale=10.0):
    return step_brain(brain, {"odor": [r * scale for r in receptors]})


def test_the_code_stays_sparse_over_many_odours():
    brain = make_brain({"seed": 7})
    fractions = [sum(1 for c in present(brain, odour(s))["kenyon"] if c)
                 / len(present(brain, odour(s))["kenyon"]) for s in range(40, 70)]
    assert max(fractions) <= 0.15
    assert sum(fractions) / len(fractions) >= 0.02


def test_odours_stay_separable_across_many_pairs():
    brain = make_brain({"seed": 7})
    codes = [{i for i, c in enumerate(present(brain, odour(s))["kenyon"]) if c}
             for s in range(100, 118)]
    overlaps = [len(a & b) / max(1, len(a))
                for i, a in enumerate(codes) for b in codes[i + 1:]]
    assert sum(overlaps) / len(overlaps) < 0.2


def test_the_compass_is_accurate_over_a_long_and_uneven_walk():
    brain = make_brain({"seed": 7})
    silent = [0.0] * GLOMERULI
    start = step_brain(brain, {"odor": silent})["heading"]
    total, dt = 0.0, 0.05
    for i in range(600):
        rate = math.sin(i / 37.0) * 2.0
        step_brain(brain, {"odor": silent, "angular_velocity": rate}, dt)
        total += rate * dt
    end = step_brain(brain, {"odor": silent})["heading"]
    error = math.atan2(math.sin(end - start - total), math.cos(end - start - total))
    assert abs(error) < 0.5, error


def test_punishment_is_not_forgotten_while_other_odours_come_and_go():
    brain = make_brain({"seed": 7})
    for _ in range(8):
        learn(brain, present(brain, odour(200)), -1.0)
    for _ in range(50):
        present(brain, odour(201))
    assert present(brain, odour(200))["valence"] < 0.0


def test_the_two_compartments_do_not_share_a_synapse():
    brain = make_brain({"seed": 7})
    for _ in range(8):
        learn(brain, present(brain, odour(202)), -1.0)
    punished = present(brain, odour(202))["mbon"]
    for _ in range(8):
        learn(brain, present(brain, odour(202)), 1.0)
    both = present(brain, odour(202))["mbon"]
    assert both[0] == punished[0]
    assert both[1] < punished[1]


def test_learning_generalises_no_further_than_the_code_overlaps():
    brain = make_brain({"seed": 7})
    before = [present(brain, odour(s))["valence"] for s in range(300, 310)]
    for _ in range(10):
        learn(brain, present(brain, odour(299)), -1.0)
    after = [present(brain, odour(s))["valence"] for s in range(300, 310)]
    assert sum(1 for a, b in zip(after, before) if a < 0.5 * b) <= 2


def test_a_trained_animal_takes_fewer_shocks_in_the_choice_assay():
    trained = train("choice", episodes=10, seed=0)["brain"]
    naive = make_brain({"seed": 0})
    unseen = range(2000, 2005)
    before = sum(run_episode(naive, "choice", s, learning=False)["shocks"] for s in unseen)
    after = sum(run_episode(trained, "choice", s, learning=False)["shocks"] for s in unseen)
    assert after < before


def test_an_animal_that_learnt_nothing_still_finds_food():
    """Nothing in the plasticity may be load-bearing for innate approach."""
    brain = make_brain({"seed": 7})
    reached = [run_episode(brain, "taxis", s, learning=False)["reached"] for s in range(8)]
    assert sum(reached) >= 6


def test_shocks_fall_enough_to_call_it_learning():
    rows = train("avoid", episodes=14, seed=3)["episodes"]
    shocks = [r["shocks"] for r in rows]
    assert shocks[0] > 0
    assert sum(shocks[-4:]) < sum(shocks[:4])
''',
}


def initial_files() -> Dict[str, str]:
    """Everything the run starts from. No `CONTEXT.md`: `--mode b` designs the tree."""
    files = {"spec/CONTEXT.md": _SPEC, "fly.py": _DRIVER,
             f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL}
    files.update(_TESTS)
    return files


FLY = TestSuite(name="fly", given=initial_files(), frozen=FROZEN, audit=_AUDIT)

build_tasks = FLY.build_tasks
make_runner = FLY.make_runner
suite_review = FLY.review
#: The whole suite in one interpreter -- which is the only way the shadowing
#: collision in `test_integration.py` shows up at all.
suite_failures = FLY.suite_failures
#: One test, pass or fail. No tolerance of its own -- the test owns that.
reward = reward_test
#: Exactly the audit set in the held-out tail, every driven test in the search.
HELD_OUT_FRAC = FLY.held_out_frac()


def llm_manager(complete):
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen,
                         failure=TEST_FAILURE)
