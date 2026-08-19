"""The integrand catalogue for the ERA numerical-integration task.

Standard library only, and deliberately so: this module is imported **on both
sides of the sandbox**. The host builds the problem files and holds the
reference values; the runner inside Bubblewrap rebuilds the same integrand from
the same specification and hands it to the candidate. A shared catalogue is what
makes those two the same integral rather than two integrals that agree by
inspection.

Every family here has a **closed form**. That is the whole design: a reference
produced by *another* numerical integrator would be a reference the candidate
could legitimately beat, and the score would then measure agreement with a rival
method rather than accuracy. ``tests/test_era_integrals.py`` cross-checks each
identity against mpmath at 60 digits when mpmath is installed.

The nine families are not nine variations of one difficulty. Each one breaks a
different assumption a general-purpose quadrature rule makes -- that the
integrand is bounded, that it is smooth, that it is resolved by a few hundred
nodes, that it decays, that the interval is finite, that no catastrophic
cancellation occurs. `scipy.integrate.quad` on default settings solves some of
them to machine precision and returns nonsense on others, which is what leaves
the search something to find.

The parameters are drawn per shard from a seeded ``random.Random``, so a shard
is reproducible from its index and never shares an instance with another shard.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


#: The catalogue, in a fixed order. A shard holds one instance of each.
FAMILIES: Tuple[str, ...] = (
    "beta_endpoints",
    "log_power",
    "log_oscillation",
    "lorentz_peak",
    "dirichlet_decay",
    "fresnel",
    "gamma_oscillatory",
    "gauss_oscillatory",
    "algebraic_tail",
)

#: What each family does to a quadrature rule, in the words the prompt uses.
#: The candidate is never told which problem is which family -- it receives a
#: callable and an interval -- so this is a description of the *task*, which an
#: expert would be given, and not a lookup table.
DIFFICULTIES: Dict[str, str] = {
    "beta_endpoints":
        "integrable algebraic singularities at BOTH endpoints of [0, 1]",
    "log_power":
        "a logarithmic singularity of integer power at the left endpoint",
    "log_oscillation":
        "infinitely many oscillations accumulating at the left endpoint, "
        "under an algebraic singularity",
    "lorentz_peak":
        "a needle-sharp interior peak whose width is 1e-7 to 1e-4 of the interval",
    "dirichlet_decay":
        "a barely damped oscillation on a half-infinite interval, with a "
        "removable singularity at the origin",
    "fresnel":
        "an oscillation on a half-infinite interval whose amplitude NEVER decays",
    "gamma_oscillatory":
        "an endpoint singularity multiplied by a fast oscillation under "
        "exponential damping",
    "gauss_oscillatory":
        "cancellation over the whole real line -- the answer is several orders "
        "of magnitude below the size of the integrand",
    "algebraic_tail":
        "an endpoint singularity at 0 and a heavy algebraic tail at infinity, "
        "both on the same half-infinite interval",
}

#: Draws whose exact value is smaller than this are redrawn. A relative error
#: cannot be measured against an answer that is numerically zero, and two of
#: these families (`gamma_oscillatory`, `gauss_oscillatory`) have parameter
#: regions where the integral cancels to nothing.
VALUE_FLOOR = 5e-4

#: The reference values are `math` library expressions in double precision,
#: accurate to within an ulp or two. Reporting 15 "correct digits" against a
#: reference that only has ~16 would be measuring the reference's rounding, so
#: the score saturates here.
DIGIT_CAP = 12.0


@dataclass(frozen=True)
class Problem:
    """One integral: a family, its parameters, and its limits.

    ``a``/``b`` are stored explicitly even though they follow from the family,
    because they are the two numbers the candidate actually sees and a problem
    file that does not name them would be readable only by this module.
    """

    family: str
    params: Dict[str, float]
    a: float
    b: float

    def to_dict(self) -> Dict[str, Any]:
        # The limits go over as strings: `inf` is not JSON, and Python's own
        # `Infinity` token is a dialect the runner would have to re-parse
        # anyway. `repr` of a float round-trips exactly.
        return {
            "family": self.family,
            "params": dict(self.params),
            "a": repr(self.a),
            "b": repr(self.b),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Problem":
        return Problem(
            family=str(payload["family"]),
            params={str(k): float(v) for k, v in dict(payload["params"]).items()},
            a=float(payload["a"]),
            b=float(payload["b"]),
        )

    def interval(self) -> str:
        """The interval as the prompt and the failure report show it."""
        open_left, open_right = math.isinf(self.a), math.isinf(self.b)
        left = "-inf" if open_left else f"{self.a:g}"
        right = "inf" if open_right else f"{self.b:g}"
        return f"{'(' if open_left else '['}{left}, {right}{')' if open_right else ']'}"


# --------------------------------------------------------------------------
# The integrands
# --------------------------------------------------------------------------


def _guard(a: float, b: float, body: Callable[[float], float]) -> Callable[[float], float]:
    """Wrap one family's body in the contract the candidate is promised.

    Outside ``[a, b]`` the integrand raises rather than returning something: a
    method that has drifted off its own interval is wrong, and a silent value
    there would let a transformed rule integrate a different problem and still
    score. Non-finite inputs raise for the same reason -- a candidate that
    evaluates ``f(inf)`` to probe the tail is asking a question about a limit,
    not about the integrand.

    ``OverflowError`` becomes ``inf`` because that is what the singularity *is*:
    these are integrable poles, and the candidate is told not to depend on the
    value at one.
    """

    def integrand(x: float) -> float:
        value = float(x)
        if not math.isfinite(value):
            raise ValueError("the integrand is defined at finite points only")
        if value < a or value > b:
            raise ValueError(f"{value!r} is outside the interval of integration")
        try:
            return float(body(value))
        except OverflowError:
            return math.inf

    return integrand


def _beta_endpoints(p: Dict[str, float]) -> Callable[[float], float]:
    alpha, beta = p["alpha"], p["beta"]

    def body(x: float) -> float:
        if x <= 0.0 or x >= 1.0:
            return math.inf
        return math.exp((alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log1p(-x))

    return body


def _log_power(p: Dict[str, float]) -> Callable[[float], float]:
    s, n = p["s"], int(p["n"])

    def body(x: float) -> float:
        if x <= 0.0:
            return math.inf
        return math.exp((s - 1.0) * math.log(x)) * (-math.log(x)) ** n

    return body


def _log_oscillation(p: Dict[str, float]) -> Callable[[float], float]:
    s, omega = p["s"], p["omega"]

    def body(x: float) -> float:
        if x <= 0.0:
            # No limit exists here: the cosine has infinitely many sign changes
            # in every neighbourhood of the origin. Saying so is more useful
            # than a fabricated value that a rule would silently average in.
            raise ValueError("the integrand has no limit at the left endpoint")
        log_x = math.log(x)
        return math.exp((s - 1.0) * log_x) * math.cos(omega * log_x)

    return body


def _lorentz_peak(p: Dict[str, float]) -> Callable[[float], float]:
    eps, x0 = p["eps"], p["x0"]

    def body(x: float) -> float:
        return eps / ((x - x0) ** 2 + eps * eps)

    return body


def _dirichlet_decay(p: Dict[str, float]) -> Callable[[float], float]:
    decay, freq = p["p"], p["q"]

    def body(x: float) -> float:
        if x == 0.0:
            return freq  # the removable singularity, resolved
        return math.exp(-decay * x) * math.sin(freq * x) / x

    return body


def _fresnel(p: Dict[str, float]) -> Callable[[float], float]:
    scale, phase = p["a"], int(p["cosine"])

    def body(x: float) -> float:
        argument = scale * x * x
        return math.cos(argument) if phase else math.sin(argument)

    return body


def _gamma_oscillatory(p: Dict[str, float]) -> Callable[[float], float]:
    s, b = p["s"], p["b"]

    def body(x: float) -> float:
        if x == 0.0:
            return math.inf if s < 1.0 else (1.0 if s == 1.0 else 0.0)
        damping = math.exp(-x)
        if damping == 0.0:
            return 0.0  # underflow of e^-x wins over any algebraic factor
        return math.exp((s - 1.0) * math.log(x)) * damping * math.cos(b * x)

    return body


def _gauss_oscillatory(p: Dict[str, float]) -> Callable[[float], float]:
    b, shift = p["b"], p["shift"]

    def body(x: float) -> float:
        y = x - shift
        if y * y > 700.0:
            return 0.0
        return math.exp(-y * y) * math.cos(b * x)

    return body


def _algebraic_tail(p: Dict[str, float]) -> Callable[[float], float]:
    s = p["s"]

    def body(x: float) -> float:
        if x <= 0.0:
            return math.inf
        return math.exp((s - 1.0) * math.log(x)) / (1.0 + x)

    return body


_BODIES: Dict[str, Callable[[Dict[str, float]], Callable[[float], float]]] = {
    "beta_endpoints": _beta_endpoints,
    "log_power": _log_power,
    "log_oscillation": _log_oscillation,
    "lorentz_peak": _lorentz_peak,
    "dirichlet_decay": _dirichlet_decay,
    "fresnel": _fresnel,
    "gamma_oscillatory": _gamma_oscillatory,
    "gauss_oscillatory": _gauss_oscillatory,
    "algebraic_tail": _algebraic_tail,
}


def integrand(problem: Problem) -> Callable[[float], float]:
    """The callable a candidate is handed. Pure, scalar, and deterministic."""
    try:
        build = _BODIES[problem.family]
    except KeyError:
        raise ValueError(f"unknown integrand family {problem.family!r}") from None
    return _guard(problem.a, problem.b, build(problem.params))


# --------------------------------------------------------------------------
# The reference values
# --------------------------------------------------------------------------


def exact(problem: Problem) -> float:
    """The closed form. Every identity is standard; see the module docstring.

    * ``beta_endpoints``   B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)
    * ``log_power``        int_0^1 x^{s-1} ln(1/x)^n dx = n! / s^{n+1}
    * ``log_oscillation``  int_0^1 x^{s-1} cos(w ln x) dx = s / (s^2 + w^2)
    * ``lorentz_peak``     atan((1-x0)/eps) + atan(x0/eps)
    * ``dirichlet_decay``  int_0^inf e^{-px} sin(qx)/x dx = atan(q/p)
    * ``fresnel``          int_0^inf sin(a x^2) dx = (1/2) sqrt(pi/(2a)), cosine alike
    * ``gamma_oscillatory``Gamma(s) cos(s atan b) / (1+b^2)^{s/2}
    * ``gauss_oscillatory``sqrt(pi) e^{-b^2/4} cos(b m), for the peak at x = m
    * ``algebraic_tail``   int_0^inf x^{s-1}/(1+x) dx = pi / sin(pi s)
    """
    p = problem.params
    family = problem.family
    if family == "beta_endpoints":
        alpha, beta = p["alpha"], p["beta"]
        # Three correctly-rounded `gamma` calls rather than `exp(lgamma(...))`:
        # the arguments here are all in (0.06, 0.9), nothing overflows, and the
        # logarithmic route would cost a couple of ulps in the reference itself.
        return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    if family == "log_power":
        return math.factorial(int(p["n"])) / p["s"] ** (int(p["n"]) + 1)
    if family == "log_oscillation":
        s, omega = p["s"], p["omega"]
        return s / (s * s + omega * omega)
    if family == "lorentz_peak":
        eps, x0 = p["eps"], p["x0"]
        return math.atan((1.0 - x0) / eps) + math.atan(x0 / eps)
    if family == "dirichlet_decay":
        return math.atan(p["q"] / p["p"])
    if family == "fresnel":
        return 0.5 * math.sqrt(math.pi / (2.0 * p["a"]))
    if family == "gamma_oscillatory":
        s, b = p["s"], p["b"]
        return (math.gamma(s) * math.cos(s * math.atan(b))
                / (1.0 + b * b) ** (s / 2.0))
    if family == "gauss_oscillatory":
        b, shift = p["b"], p["shift"]
        return (math.sqrt(math.pi) * math.exp(-b * b / 4.0)
                * math.cos(b * shift))
    if family == "algebraic_tail":
        s = p["s"]
        return math.pi / math.sin(math.pi * s)
    raise ValueError(f"unknown integrand family {family!r}")


# --------------------------------------------------------------------------
# Drawing a shard
# --------------------------------------------------------------------------


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _draw(family: str, rng: random.Random) -> Problem:
    inf = math.inf
    if family == "beta_endpoints":
        return Problem(family, {"alpha": rng.uniform(0.06, 0.45),
                                "beta": rng.uniform(0.06, 0.45)}, 0.0, 1.0)
    if family == "log_power":
        return Problem(family, {"s": rng.uniform(0.2, 1.0),
                                "n": float(rng.choice((2, 3, 4)))}, 0.0, 1.0)
    if family == "log_oscillation":
        return Problem(family, {"s": rng.uniform(0.3, 1.2),
                                "omega": rng.uniform(5.0, 40.0)}, 0.0, 1.0)
    if family == "lorentz_peak":
        return Problem(family, {"eps": _log_uniform(rng, 1e-7, 1e-4),
                                "x0": rng.uniform(0.15, 0.85)}, 0.0, 1.0)
    if family == "dirichlet_decay":
        return Problem(family, {"p": _log_uniform(rng, 1e-3, 5e-2),
                                "q": rng.uniform(0.5, 3.0)}, 0.0, inf)
    if family == "fresnel":
        return Problem(family, {"a": rng.uniform(0.5, 5.0),
                                "cosine": float(rng.randint(0, 1))}, 0.0, inf)
    if family == "gamma_oscillatory":
        return Problem(family, {"s": rng.uniform(0.25, 1.5),
                                "b": rng.uniform(2.0, 20.0)}, 0.0, inf)
    if family == "gauss_oscillatory":
        # The shift is not decoration. Centred at the origin this integrand is
        # *even*, and a program that maps both halves of the line onto [0, inf)
        # and doubles -- which is a plain bug -- scores 12 digits on it. The
        # first live run of this task found exactly that program, so the family
        # now carries an offset and the symmetry is gone.
        return Problem(family, {"b": rng.uniform(3.0, 5.5),
                                "shift": rng.uniform(0.3, 1.5)}, -inf, inf)
    if family == "algebraic_tail":
        return Problem(family, {"s": rng.uniform(0.15, 0.85)}, 0.0, inf)
    raise ValueError(f"unknown integrand family {family!r}")


def draw_shard(seed: int, index: int, *, families: Tuple[str, ...] = FAMILIES) -> List[Problem]:
    """One instance of every family, in an order the shard index decides.

    Shuffled because the position of a problem within a shard would otherwise
    *be* its family, and the feedback the search gets names positions. A fixed
    order would turn "problem 5 scored 2 digits" into "the Fresnel family scored
    2 digits", which is a different -- and much easier -- task than the one the
    candidate's interface describes.
    """
    # A string seed rather than a tuple: `random.Random` refuses anything but
    # None/int/float/str/bytes, and the string route hashes with SHA-512 rather
    # than `hash()`, so the draw does not move with PYTHONHASHSEED.
    rng = random.Random(f"era-integrals:{seed}:{index}")
    problems: List[Problem] = []
    for family in families:
        for _ in range(64):
            problem = _draw(family, rng)
            if abs(exact(problem)) >= VALUE_FLOOR:
                break
        else:  # pragma: no cover - 64 rejections in a row is a broken range
            raise RuntimeError(f"could not draw a usable {family!r} instance")
        problems.append(problem)
    rng.shuffle(problems)
    return problems


def digits_of(estimate: Any, truth: float) -> float:
    """Correct significant digits of ``estimate``, saturating at ``DIGIT_CAP``.

    A relative error, because the families span six orders of magnitude in the
    size of the answer and an absolute error would make the largest of them the
    only one worth optimising. Anything that is not a finite number scores zero
    rather than raising: a candidate returning ``nan`` has failed *that problem*,
    and the rest of the shard is still evidence.
    """
    try:
        value = float(estimate)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    scale = abs(truth)
    if scale == 0.0:  # pragma: no cover - VALUE_FLOOR excludes this
        return 0.0
    error = abs(value - truth) / scale
    if error <= 0.0:
        return DIGIT_CAP
    return max(0.0, min(DIGIT_CAP, -math.log10(error)))
