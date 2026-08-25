"""Measure how many correct digits NumPy and SciPy actually return.

    python -m tools.scan_numeric_precision                     # the whole sweep
    python -m tools.scan_numeric_precision --only hyp1f1,iv    # two probes
    python -m tools.scan_numeric_precision --points 200        # a quick pass

Writes ``docs/data/numeric_precision_scan.json`` and prints the ranking. The
point of the sweep is to choose the ERA targets by measurement rather than by
folklore: "everyone knows ``hyp1f1`` is bad for large negative argument" is a
claim, and a claim is not a benchmark.

How a probe is scored
---------------------
Each probe declares a **parameter distribution up front**, draws points from it,
and compares the float64 library call against mpmath. A point scores

    digits = min(CAP, -log10(|library - reference| / |reference|))

so 16 means "as good as a double can be" and 0 means "no correct digit at all".
What gets reported per function is the mean, the 10th percentile, and the share
of points under 8 digits -- that last number is the interesting one, because
these functions do not degrade smoothly. A special-function routine is usually
either right to the last ulp or wrong in the first digit, depending on which
region of the parameter space the point landed in, so a mean of 11 can mean
"uniformly a bit lossy" or "perfect on four fifths and garbage on the rest" and
only the tail statistics tell those apart.

Three rules keep the numbers about the library rather than about this file:

**The reference is independent.** mpmath shares no code with SciPy. Its own
accuracy is not assumed: every point is evaluated at ``--dps`` and again at
twice ``--dps``, and a point is **discarded** unless the two agree to
``--agree`` digits. A point where the reference has not converged can never
become evidence that SciPy is wrong -- it is dropped and counted under
``reference_rejected``.

**The distribution is declared, not tuned.** The ranges live in the probe table
below and were written before the sweep was run. Drawing a range *after*
finding where a function fails would measure the drawing.

**Degenerate points are dropped, not scored.** Poles, values below
``MIN_MAGNITUDE`` (a relative error against ~0 is not a measurement), and points
where the true value overflows or underflows float64 are rejected before
scoring -- a function cannot be blamed for returning ``inf`` where the answer
genuinely is ``inf``.

A probe that scores badly here is a *candidate*. The ERA task built from it
draws its own, larger, independently generated stress set; this sweep only
decides what is worth pointing a search at.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import scipy.special as sp


OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "data" / "numeric_precision_scan.json"

#: float64 carries about 15.95 decimal digits, so 16 is "exact as far as a
#: double can tell". Nothing is gained by distinguishing 15.9 from 16.0.
DIGIT_CAP = 16.0

#: Under this many digits, more than half of a double's mantissa is gone. It is
#: the threshold the ranking sorts on, not an error bar.
LOSS_THRESHOLD = 8.0

#: A relative error measured against a number this small is measuring the
#: reference's own underflow, not the library.
MIN_MAGNITUDE = mp.mpf("1e-280")
MAX_MAGNITUDE = mp.mpf("1e280")

DEFAULT_POINTS = 600
DEFAULT_DPS = 30
DEFAULT_AGREE = 22
SEED = 20260825


# --------------------------------------------------------------------------
# The probe table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One library entry point, its declared input range, and its reference."""

    #: Short key used by ``--only`` and in the report.
    key: str
    #: Fully qualified name of what is being measured.
    target: str
    #: The declared distribution, in words, for the report.
    domain: str
    #: ``rng -> {name: float}``.
    draw: Callable[[random.Random], Dict[str, float]]
    #: ``point -> float``. The float64 library call.
    library: Callable[[Dict[str, float]], float]
    #: ``point -> mpf``, evaluated under whatever precision mpmath is set to.
    reference: Callable[[Dict[str, float]], Any]
    #: Free-text note carried into the report.
    note: str = ""
    #: Probes whose reference costs a quadrature get fewer points.
    points_scale: float = 1.0
    #: Extra rejection rule, applied to the drawn point before any evaluation.
    reject: Optional[Callable[[Dict[str, float]], bool]] = None


def _u(rng: random.Random, low: float, high: float) -> float:
    return rng.uniform(low, high)


def _logu(rng: random.Random, low: float, high: float) -> float:
    """Log-uniform on a positive interval -- the right draw for a scale."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _near_nonpositive_integer(x: float, tol: float = 1e-6) -> bool:
    return x <= 0.5 and abs(x - round(x)) < tol


PROBES: Tuple[Probe, ...] = (
    # ---- confluent and Gauss hypergeometric -------------------------------
    Probe(
        key="hyp2f1",
        target="scipy.special.hyp2f1",
        domain="a,b,c ~ U(-30,30); z ~ U(-40,0.999)",
        draw=lambda r: {"a": _u(r, -30, 30), "b": _u(r, -30, 30),
                        "c": _u(r, -30, 30), "z": _u(r, -40, 0.999)},
        library=lambda p: float(sp.hyp2f1(p["a"], p["b"], p["c"], p["z"])),
        reference=lambda p: mp.hyp2f1(p["a"], p["b"], p["c"], p["z"]),
        reject=lambda p: _near_nonpositive_integer(p["c"]),
        note="already an ERA task; kept in the sweep as a calibration point",
    ),
    Probe(
        key="hyp1f1",
        target="scipy.special.hyp1f1",
        domain="a,b ~ U(-50,50); x ~ U(-200,200)",
        draw=lambda r: {"a": _u(r, -50, 50), "b": _u(r, -50, 50),
                        "x": _u(r, -200, 200)},
        library=lambda p: float(sp.hyp1f1(p["a"], p["b"], p["x"])),
        reference=lambda p: mp.hyp1f1(p["a"], p["b"], p["x"]),
        reject=lambda p: _near_nonpositive_integer(p["b"]),
        note="Kummer M; cancellation for x << 0 is the classic failure",
    ),
    Probe(
        key="hyperu",
        target="scipy.special.hyperu",
        domain="a,b ~ U(-20,20); x ~ logU(1e-3,100)",
        draw=lambda r: {"a": _u(r, -20, 20), "b": _u(r, -20, 20),
                        "x": _logu(r, 1e-3, 100)},
        library=lambda p: float(sp.hyperu(p["a"], p["b"], p["x"])),
        reference=lambda p: mp.hyperu(p["a"], p["b"], p["x"]),
        note="Tricomi U",
    ),
    # ---- incomplete gamma / beta ------------------------------------------
    Probe(
        key="gammainc",
        target="scipy.special.gammainc",
        domain="a ~ logU(1e-2,1e5); x ~ logU(1e-2,1e5)",
        draw=lambda r: {"a": _logu(r, 1e-2, 1e5), "x": _logu(r, 1e-2, 1e5)},
        library=lambda p: float(sp.gammainc(p["a"], p["x"])),
        reference=lambda p: mp.gammainc(p["a"], 0, p["x"], regularized=True),
        note="regularized lower P(a,x)",
    ),
    Probe(
        key="gammaincc",
        target="scipy.special.gammaincc",
        domain="a ~ logU(1e-2,1e5); x ~ logU(1e-2,1e5)",
        draw=lambda r: {"a": _logu(r, 1e-2, 1e5), "x": _logu(r, 1e-2, 1e5)},
        library=lambda p: float(sp.gammaincc(p["a"], p["x"])),
        reference=lambda p: mp.gammainc(p["a"], p["x"], mp.inf, regularized=True),
        note="regularized upper Q(a,x)",
    ),
    Probe(
        key="betainc",
        target="scipy.special.betainc",
        domain="a,b ~ logU(1e-2,1e4); x ~ U(0,1)",
        draw=lambda r: {"a": _logu(r, 1e-2, 1e4), "b": _logu(r, 1e-2, 1e4),
                        "x": _u(r, 1e-9, 1.0 - 1e-9)},
        library=lambda p: float(sp.betainc(p["a"], p["b"], p["x"])),
        reference=lambda p: mp.betainc(p["a"], p["b"], 0, p["x"], regularized=True),
        points_scale=0.6,
        note="regularized incomplete beta",
    ),
    # ---- Bessel and friends ------------------------------------------------
    Probe(
        key="jv",
        target="scipy.special.jv",
        domain="v ~ U(-60,60); x ~ logU(1e-2,500)",
        draw=lambda r: {"v": _u(r, -60, 60), "x": _logu(r, 1e-2, 500)},
        library=lambda p: float(sp.jv(p["v"], p["x"])),
        reference=lambda p: mp.besselj(p["v"], p["x"]),
    ),
    Probe(
        key="yv",
        target="scipy.special.yv",
        domain="v ~ U(-60,60); x ~ logU(1e-2,500)",
        draw=lambda r: {"v": _u(r, -60, 60), "x": _logu(r, 1e-2, 500)},
        library=lambda p: float(sp.yv(p["v"], p["x"])),
        reference=lambda p: mp.bessely(p["v"], p["x"]),
    ),
    Probe(
        key="iv",
        target="scipy.special.iv",
        domain="v ~ U(-60,60); x ~ logU(1e-2,500)",
        draw=lambda r: {"v": _u(r, -60, 60), "x": _logu(r, 1e-2, 500)},
        library=lambda p: float(sp.iv(p["v"], p["x"])),
        reference=lambda p: mp.besseli(p["v"], p["x"]),
    ),
    Probe(
        key="kv",
        target="scipy.special.kv",
        domain="v ~ U(-60,60); x ~ logU(1e-2,500)",
        draw=lambda r: {"v": _u(r, -60, 60), "x": _logu(r, 1e-2, 500)},
        library=lambda p: float(sp.kv(p["v"], p["x"])),
        reference=lambda p: mp.besselk(p["v"], p["x"]),
    ),
    Probe(
        key="struve",
        target="scipy.special.struve",
        domain="v ~ U(-15,15); x ~ logU(1e-2,100)",
        draw=lambda r: {"v": _u(r, -15, 15), "x": _logu(r, 1e-2, 100)},
        library=lambda p: float(sp.struve(p["v"], p["x"])),
        reference=lambda p: mp.struveh(p["v"], p["x"]),
        note="H_v; the series cancels catastrophically for moderate x",
    ),
    Probe(
        key="modstruve",
        target="scipy.special.modstruve",
        domain="v ~ U(-15,15); x ~ logU(1e-2,100)",
        draw=lambda r: {"v": _u(r, -15, 15), "x": _logu(r, 1e-2, 100)},
        library=lambda p: float(sp.modstruve(p["v"], p["x"])),
        reference=lambda p: mp.struvel(p["v"], p["x"]),
    ),
    Probe(
        key="spherical_jn",
        target="scipy.special.spherical_jn",
        domain="n ~ U{0..80}; x ~ logU(1e-2,300)",
        draw=lambda r: {"n": float(r.randint(0, 80)), "x": _logu(r, 1e-2, 300)},
        library=lambda p: float(sp.spherical_jn(int(p["n"]), p["x"])),
        reference=lambda p: (mp.sqrt(mp.pi / (2 * mp.mpf(p["x"])))
                             * mp.besselj(int(p["n"]) + mp.mpf("0.5"), p["x"])),
    ),
    Probe(
        key="airy_ai",
        target="scipy.special.airy[Ai]",
        domain="x ~ U(-300,50)",
        draw=lambda r: {"x": _u(r, -300, 50)},
        library=lambda p: float(sp.airy(p["x"])[0]),
        reference=lambda p: mp.airyai(p["x"]),
    ),
    Probe(
        key="airy_bi",
        target="scipy.special.airy[Bi]",
        domain="x ~ U(-300,50)",
        draw=lambda r: {"x": _u(r, -300, 50)},
        # airy() returns (Ai, Ai', Bi, Bi') -- Bi is index 2, not 1.
        library=lambda p: float(sp.airy(p["x"])[2]),
        reference=lambda p: mp.airybi(p["x"]),
    ),
    Probe(
        key="pbdv",
        target="scipy.special.pbdv[D]",
        domain="v ~ U(-20,20); x ~ U(-30,30)",
        draw=lambda r: {"v": _u(r, -20, 20), "x": _u(r, -30, 30)},
        library=lambda p: float(sp.pbdv(p["v"], p["x"])[0]),
        reference=lambda p: mp.pcfd(p["v"], p["x"]),
        points_scale=0.6,
        note="parabolic cylinder D_v(x)",
    ),
    Probe(
        key="kelvin_ber",
        target="scipy.special.kelvin[ber]",
        domain="x ~ logU(1e-2,60)",
        draw=lambda r: {"x": _logu(r, 1e-2, 60)},
        library=lambda p: float(sp.kelvin(p["x"])[0].real),
        reference=lambda p: mp.ber(0, p["x"]),
    ),
    # ---- exponential integrals ---------------------------------------------
    Probe(
        key="expi",
        target="scipy.special.expi",
        domain="x ~ U(-100,100)",
        draw=lambda r: {"x": _u(r, -100, 100)},
        library=lambda p: float(sp.expi(p["x"])),
        reference=lambda p: mp.ei(p["x"]),
    ),
    Probe(
        key="exp1",
        target="scipy.special.exp1",
        domain="x ~ logU(1e-3,100)",
        draw=lambda r: {"x": _logu(r, 1e-3, 100)},
        library=lambda p: float(sp.exp1(p["x"])),
        reference=lambda p: mp.e1(p["x"]),
    ),
    Probe(
        key="expn",
        target="scipy.special.expn",
        domain="n ~ U{1..60}; x ~ logU(1e-3,60)",
        draw=lambda r: {"n": float(r.randint(1, 60)), "x": _logu(r, 1e-3, 60)},
        library=lambda p: float(sp.expn(int(p["n"]), p["x"])),
        reference=lambda p: mp.expint(int(p["n"]), p["x"]),
    ),
    Probe(
        key="sici_si",
        target="scipy.special.sici[Si]",
        domain="x ~ U(-500,500)",
        draw=lambda r: {"x": _u(r, -500, 500)},
        library=lambda p: float(sp.sici(p["x"])[0]),
        reference=lambda p: mp.si(p["x"]),
    ),
    Probe(
        key="sici_ci",
        target="scipy.special.sici[Ci]",
        domain="x ~ logU(1e-3,500)",
        draw=lambda r: {"x": _logu(r, 1e-3, 500)},
        library=lambda p: float(sp.sici(p["x"])[1]),
        reference=lambda p: mp.ci(p["x"]),
    ),
    Probe(
        key="shichi_shi",
        target="scipy.special.shichi[Shi]",
        domain="x ~ U(-100,100)",
        draw=lambda r: {"x": _u(r, -100, 100)},
        library=lambda p: float(sp.shichi(p["x"])[0]),
        reference=lambda p: mp.shi(p["x"]),
    ),
    # ---- gamma family -------------------------------------------------------
    Probe(
        key="gammaln",
        target="scipy.special.gammaln",
        domain="x ~ U(-60,60)",
        draw=lambda r: {"x": _u(r, -60, 60)},
        library=lambda p: float(sp.gammaln(p["x"])),
        reference=lambda p: mp.log(abs(mp.gamma(p["x"]))),
        reject=lambda p: _near_nonpositive_integer(p["x"], 1e-4),
    ),
    Probe(
        key="digamma",
        target="scipy.special.digamma",
        domain="x ~ U(-60,60)",
        draw=lambda r: {"x": _u(r, -60, 60)},
        library=lambda p: float(sp.digamma(p["x"])),
        reference=lambda p: mp.digamma(p["x"]),
        reject=lambda p: _near_nonpositive_integer(p["x"], 1e-4),
    ),
    Probe(
        key="polygamma",
        target="scipy.special.polygamma",
        domain="n ~ U{1..8}; x ~ logU(1e-2,80)",
        draw=lambda r: {"n": float(r.randint(1, 8)), "x": _logu(r, 1e-2, 80)},
        library=lambda p: float(sp.polygamma(int(p["n"]), p["x"])),
        reference=lambda p: mp.polygamma(int(p["n"]), p["x"]),
    ),
    Probe(
        key="poch",
        target="scipy.special.poch",
        domain="z ~ U(-40,40); m ~ U(-20,20)",
        draw=lambda r: {"z": _u(r, -40, 40), "m": _u(r, -20, 20)},
        library=lambda p: float(sp.poch(p["z"], p["m"])),
        reference=lambda p: mp.rf(p["z"], p["m"]),
        reject=lambda p: (_near_nonpositive_integer(p["z"], 1e-4)
                          or _near_nonpositive_integer(p["z"] + p["m"], 1e-4)),
    ),
    Probe(
        key="binom",
        target="scipy.special.binom",
        domain="n ~ U(-30,120); k ~ U(-15,60)",
        draw=lambda r: {"n": _u(r, -30, 120), "k": _u(r, -15, 60)},
        library=lambda p: float(sp.binom(p["n"], p["k"])),
        reference=lambda p: mp.binomial(p["n"], p["k"]),
    ),
    # ---- error function and the normal tail ---------------------------------
    Probe(
        key="erfinv",
        target="scipy.special.erfinv",
        domain="p ~ U(-1,1)",
        draw=lambda r: {"p": _u(r, -1 + 1e-15, 1 - 1e-15)},
        library=lambda p: float(sp.erfinv(p["p"])),
        reference=lambda p: mp.erfinv(p["p"]),
    ),
    Probe(
        key="ndtri",
        target="scipy.special.ndtri",
        domain="p ~ logU(1e-300,0.5) mirrored",
        draw=lambda r: {"p": _logu(r, 1e-300, 0.5)},
        library=lambda p: float(sp.ndtri(p["p"])),
        reference=lambda p: mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p["p"]) - 1),
    ),
    Probe(
        key="log_ndtr",
        target="scipy.special.log_ndtr",
        domain="x ~ U(-300,10)",
        draw=lambda r: {"x": _u(r, -300, 10)},
        library=lambda p: float(sp.log_ndtr(p["x"])),
        reference=lambda p: mp.log(mp.ncdf(p["x"])),
    ),
    Probe(
        key="owens_t",
        target="scipy.special.owens_t",
        domain="h ~ U(-10,10); a ~ U(-20,20)",
        draw=lambda r: {"h": _u(r, -10, 10), "a": _u(r, -20, 20)},
        library=lambda p: float(sp.owens_t(p["h"], p["a"])),
        reference=lambda p: mp.quad(
            lambda t: mp.exp(-mp.mpf(p["h"]) ** 2 * (1 + t ** 2) / 2) / (1 + t ** 2),
            [0, p["a"]]) / (2 * mp.pi),
        points_scale=0.25,
        note="reference is a quadrature, so this probe runs fewer points",
    ),
    # ---- elliptic -----------------------------------------------------------
    Probe(
        key="ellipkinc",
        target="scipy.special.ellipkinc",
        domain="phi ~ U(-1.5,1.5); m ~ U(-50,1)",
        draw=lambda r: {"phi": _u(r, -1.5, 1.5), "m": _u(r, -50, 0.9999)},
        library=lambda p: float(sp.ellipkinc(p["phi"], p["m"])),
        reference=lambda p: mp.ellipf(p["phi"], p["m"]),
    ),
    Probe(
        key="ellipeinc",
        target="scipy.special.ellipeinc",
        domain="phi ~ U(-1.5,1.5); m ~ U(-50,1)",
        draw=lambda r: {"phi": _u(r, -1.5, 1.5), "m": _u(r, -50, 0.9999)},
        library=lambda p: float(sp.ellipeinc(p["phi"], p["m"])),
        reference=lambda p: mp.ellipe(p["phi"], p["m"]),
    ),
    Probe(
        key="ellipj_sn",
        target="scipy.special.ellipj[sn]",
        domain="u ~ U(-20,20); m ~ U(0,1)",
        draw=lambda r: {"u": _u(r, -20, 20), "m": _u(r, 0, 0.99999)},
        library=lambda p: float(sp.ellipj(p["u"], p["m"])[0]),
        reference=lambda p: mp.ellipfun("sn", p["u"], p["m"]),
    ),
    # ---- misc ---------------------------------------------------------------
    Probe(
        key="lambertw_k0",
        target="scipy.special.lambertw[k=0]",
        domain="x ~ U(-0.3679,50)",
        draw=lambda r: {"x": _u(r, -0.36787, 50)},
        library=lambda p: float(sp.lambertw(p["x"], 0).real),
        reference=lambda p: mp.lambertw(p["x"], 0),
    ),
    Probe(
        key="lambertw_km1",
        target="scipy.special.lambertw[k=-1]",
        domain="x ~ U(-0.3679,0)",
        draw=lambda r: {"x": _u(r, -0.36787, -1e-12)},
        library=lambda p: float(sp.lambertw(p["x"], -1).real),
        reference=lambda p: mp.lambertw(p["x"], -1),
    ),
    Probe(
        key="zeta",
        target="scipy.special.zeta",
        domain="x ~ U(-30,30) \\ {1}",
        draw=lambda r: {"x": _u(r, -30, 30)},
        library=lambda p: float(sp.zeta(p["x"])),
        reference=lambda p: mp.zeta(p["x"]),
        reject=lambda p: abs(p["x"] - 1.0) < 1e-3,
    ),
    Probe(
        key="eval_legendre",
        target="scipy.special.eval_legendre",
        domain="n ~ U{0..120}; x ~ U(-1,1)",
        draw=lambda r: {"n": float(r.randint(0, 120)), "x": _u(r, -1, 1)},
        library=lambda p: float(sp.eval_legendre(int(p["n"]), p["x"])),
        reference=lambda p: mp.legendre(int(p["n"]), p["x"]),
    ),
    Probe(
        key="elliprj",
        target="scipy.special.elliprj",
        domain="x,y,z ~ logU(1e-3,100); p ~ logU(1e-3,100)",
        draw=lambda r: {"x": _logu(r, 1e-3, 100), "y": _logu(r, 1e-3, 100),
                        "z": _logu(r, 1e-3, 100), "p": _logu(r, 1e-3, 100)},
        library=lambda p: float(sp.elliprj(p["x"], p["y"], p["z"], p["p"])),
        reference=lambda p: mp.elliprj(p["x"], p["y"], p["z"], p["p"]),
    ),
    # ---- NumPy elementary functions ----------------------------------------
    # Included so the sweep can say something about NumPy rather than assume it.
    Probe(
        key="np_sin_large",
        target="numpy.sin",
        domain="x ~ logU(1e3,1e18), random sign",
        draw=lambda r: {"x": r.choice([-1.0, 1.0]) * _logu(r, 1e3, 1e18)},
        library=lambda p: float(np.sin(p["x"])),
        reference=lambda p: mp.sin(p["x"]),
        note="argument reduction far from the origin",
    ),
    Probe(
        key="np_tan_large",
        target="numpy.tan",
        domain="x ~ logU(1e3,1e18), random sign",
        draw=lambda r: {"x": r.choice([-1.0, 1.0]) * _logu(r, 1e3, 1e18)},
        library=lambda p: float(np.tan(p["x"])),
        reference=lambda p: mp.tan(p["x"]),
    ),
    Probe(
        key="np_exp",
        target="numpy.exp",
        domain="x ~ U(-700,700)",
        draw=lambda r: {"x": _u(r, -700, 700)},
        library=lambda p: float(np.exp(p["x"])),
        reference=lambda p: mp.e ** mp.mpf(p["x"]),
    ),
    Probe(
        key="np_log1p",
        target="numpy.log1p",
        domain="x ~ logU(1e-18,1e6) mirrored",
        draw=lambda r: {"x": r.choice([-1.0, 1.0]) * _logu(r, 1e-18, 1e6)},
        library=lambda p: float(np.log1p(p["x"])),
        reference=lambda p: mp.log(1 + mp.mpf(p["x"])),
        reject=lambda p: p["x"] <= -1.0,
    ),
    Probe(
        key="np_power",
        target="numpy.power",
        domain="x ~ logU(1e-8,1e8); y ~ U(-40,40)",
        draw=lambda r: {"x": _logu(r, 1e-8, 1e8), "y": _u(r, -40, 40)},
        library=lambda p: float(np.power(p["x"], p["y"])),
        reference=lambda p: mp.mpf(p["x"]) ** mp.mpf(p["y"]),
    ),
    Probe(
        key="np_logaddexp",
        target="numpy.logaddexp",
        domain="x,y ~ U(-700,700)",
        draw=lambda r: {"x": _u(r, -700, 700), "y": _u(r, -700, 700)},
        library=lambda p: float(np.logaddexp(p["x"], p["y"])),
        reference=lambda p: mp.log(mp.e ** mp.mpf(p["x"]) + mp.e ** mp.mpf(p["y"])),
    ),
    Probe(
        key="np_hypot",
        target="numpy.hypot",
        domain="x,y ~ logU(1e-150,1e150), random sign",
        draw=lambda r: {"x": r.choice([-1.0, 1.0]) * _logu(r, 1e-150, 1e150),
                        "y": r.choice([-1.0, 1.0]) * _logu(r, 1e-150, 1e150)},
        library=lambda p: float(np.hypot(p["x"], p["y"])),
        reference=lambda p: mp.sqrt(mp.mpf(p["x"]) ** 2 + mp.mpf(p["y"]) ** 2),
    ),
    Probe(
        key="np_sinc",
        target="numpy.sinc",
        domain="x ~ U(-200,200)",
        draw=lambda r: {"x": _u(r, -200, 200)},
        library=lambda p: float(np.sinc(p["x"])),
        reference=lambda p: (mp.sin(mp.pi * mp.mpf(p["x"])) / (mp.pi * mp.mpf(p["x"]))
                             if p["x"] != 0 else mp.mpf(1)),
    ),
)

PROBES_BY_KEY = {probe.key: probe for probe in PROBES}


# --------------------------------------------------------------------------
# Scoring one probe
# --------------------------------------------------------------------------


def _correct_digits(library: float, reference: Any) -> float:
    """Correct significant digits of ``library`` against an mpf ``reference``.

    Computed in mpmath, not in float64: on a point that is nearly right, the
    difference is the quantity being measured and subtracting in double
    precision would round it away.
    """
    if library is None:
        return 0.0
    if not math.isfinite(library):
        return 0.0
    error = abs(mp.mpf(library) - reference) / abs(reference)
    if error <= 0:
        return DIGIT_CAP
    return max(0.0, min(DIGIT_CAP, float(-mp.log10(error))))


@dataclass
class ProbeResult:
    key: str
    target: str
    domain: str
    note: str
    scored: int = 0
    drawn: int = 0
    reference_rejected: int = 0
    degenerate_rejected: int = 0
    library_raised: int = 0
    digits: List[float] = field(default_factory=list)
    worst: List[Dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    def summary(self) -> Dict[str, Any]:
        values = sorted(self.digits)
        n = len(values)
        if not n:
            return {
                "key": self.key, "target": self.target, "domain": self.domain,
                "note": self.note, "scored": 0, "error": self.error or "no points scored",
            }
        mean = sum(values) / n
        return {
            "key": self.key,
            "target": self.target,
            "domain": self.domain,
            "note": self.note,
            "scored": n,
            "drawn": self.drawn,
            "mean_digits": round(mean, 3),
            "median_digits": round(values[n // 2], 3),
            "p10_digits": round(values[max(0, int(0.10 * n))], 3),
            "min_digits": round(values[0], 3),
            "frac_below_8": round(sum(1 for v in values if v < LOSS_THRESHOLD) / n, 4),
            "frac_below_1": round(sum(1 for v in values if v < 1.0) / n, 4),
            "library_raised": self.library_raised,
            "reference_rejected": self.reference_rejected,
            "degenerate_rejected": self.degenerate_rejected,
            "worst_points": self.worst,
            "seconds": round(self.seconds, 2),
        }


def run_probe(probe: Probe, *, points: int, dps: int, agree: int,
              seed: int = SEED, worst_kept: int = 5) -> ProbeResult:
    """Draw, reject, evaluate and score one probe. Pure; no I/O."""
    result = ProbeResult(key=probe.key, target=probe.target,
                         domain=probe.domain, note=probe.note)
    rng = random.Random(f"{seed}:{probe.key}")
    wanted = max(20, int(points * probe.points_scale))
    started = time.monotonic()
    detail: List[Dict[str, Any]] = []
    # Bounded so a probe whose draw is mostly rejected still terminates.
    attempts = 0
    max_attempts = wanted * 12

    while result.scored < wanted and attempts < max_attempts:
        attempts += 1
        point = probe.draw(rng)
        result.drawn += 1
        if probe.reject is not None and probe.reject(point):
            result.degenerate_rejected += 1
            continue

        # The reference, twice, at two precisions. A point survives only if the
        # two agree -- otherwise nothing here is evidence about the library.
        try:
            mp.mp.dps = dps
            low = probe.reference(point)
            mp.mp.dps = 2 * dps
            high = probe.reference(point)
        except (ValueError, ZeroDivisionError, ArithmeticError, TypeError,
                mp.libmp.libhyper.NoConvergence):
            result.reference_rejected += 1
            continue
        finally:
            mp.mp.dps = dps

        try:
            if not (mp.isfinite(low) and mp.isfinite(high)):
                result.reference_rejected += 1
                continue
            low = mp.mpf(low.real) if hasattr(low, "real") else mp.mpf(low)
            high = mp.mpf(high.real) if hasattr(high, "real") else mp.mpf(high)
            magnitude = abs(high)
            if magnitude < MIN_MAGNITUDE or magnitude > MAX_MAGNITUDE:
                result.degenerate_rejected += 1
                continue
            if abs(low - high) / magnitude > mp.mpf(10) ** (-agree):
                result.reference_rejected += 1
                continue
        except (TypeError, ValueError, ArithmeticError):
            result.reference_rejected += 1
            continue

        # The reference is now trusted to `agree` digits. Evaluate the library.
        raised = ""
        try:
            value = probe.library(point)
        except Exception as exc:  # noqa: BLE001 - any failure is a zero
            value = None
            raised = type(exc).__name__
            result.library_raised += 1

        digits = _correct_digits(value, high)
        result.digits.append(digits)
        result.scored += 1
        detail.append({
            **{key: float(val) for key, val in point.items()},
            "digits": round(digits, 3),
            "library": (None if value is None or not math.isfinite(value)
                        else repr(value)),
            "raised": raised,
        })

    mp.mp.dps = 15
    result.seconds = time.monotonic() - started
    result.worst = sorted(detail, key=lambda row: row["digits"])[:worst_kept]
    if result.scored < wanted:
        result.error = (f"only {result.scored}/{wanted} points survived "
                        f"{attempts} draws")
    return result


def _run_probe_key(payload: Tuple[str, int, int, int, int]) -> Dict[str, Any]:
    """ProcessPool entry point -- probes are independent, so they fan out."""
    key, points, dps, agree, seed = payload
    return run_probe(PROBES_BY_KEY[key], points=points, dps=dps,
                     agree=agree, seed=seed).summary()


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    """The ranking, worst first."""
    header = (f"{'target':<34} {'mean':>6} {'p10':>6} {'min':>6} "
              f"{'<8d':>7} {'<1d':>7} {'n':>5}")
    lines = [header, "-" * len(header)]
    for row in rows:
        if not row.get("scored"):
            lines.append(f"{row['target']:<34} {'--':>6}   (no points: "
                         f"{row.get('error', '')})")
            continue
        lines.append(
            f"{row['target']:<34} {row['mean_digits']:>6.2f} "
            f"{row['p10_digits']:>6.2f} {row['min_digits']:>6.2f} "
            f"{row['frac_below_8'] * 100:>6.1f}% {row['frac_below_1'] * 100:>6.1f}% "
            f"{row['scored']:>5}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS,
                        help=f"points per probe before rejection (default {DEFAULT_POINTS})")
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS,
                        help="mpmath working precision; the check runs at twice this")
    parser.add_argument("--agree", type=int, default=DEFAULT_AGREE,
                        help="digits the two reference precisions must agree to")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only", default="",
                        help="comma-separated probe keys (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--list", action="store_true", help="list probe keys and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for probe in PROBES:
            print(f"{probe.key:<16} {probe.target:<34} {probe.domain}")
        return 0

    keys = [k.strip() for k in args.only.split(",") if k.strip()] or [p.key for p in PROBES]
    unknown = [k for k in keys if k not in PROBES_BY_KEY]
    if unknown:
        raise SystemExit(f"unknown probe key(s): {', '.join(unknown)}")

    print(f"Scanning {len(keys)} probes, {args.points} points each, "
          f"reference mpmath at {args.dps}/{2 * args.dps} dps "
          f"(kept where they agree to {args.agree})")
    started = time.monotonic()
    payloads = [(key, args.points, args.dps, args.agree, args.seed) for key in keys]
    rows: List[Dict[str, Any]] = []
    if args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(_run_probe_key, payloads):
                rows.append(row)
                print(f"  done {row['target']:<34} "
                      f"mean={row.get('mean_digits', float('nan')):.2f}", flush=True)
    else:
        for payload in payloads:
            row = _run_probe_key(payload)
            rows.append(row)
            print(f"  done {row['target']:<34} "
                  f"mean={row.get('mean_digits', float('nan')):.2f}", flush=True)

    rows.sort(key=lambda r: (r.get("mean_digits", DIGIT_CAP + 1)))
    wall = time.monotonic() - started
    print()
    print(format_table(rows))
    print()
    print(f"{wall:.1f}s wall")

    payload = {
        "scan": "correct significant digits of NumPy/SciPy float64 entry points",
        "reference": {
            "library": "mpmath",
            "version": mp.__version__,
            "precisions_dps": [args.dps, 2 * args.dps],
            "kept_when_agreeing_to_digits": args.agree,
        },
        "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                     "python": sys.version.split()[0]},
        "digit_cap": DIGIT_CAP,
        "loss_threshold": LOSS_THRESHOLD,
        "points_requested": args.points,
        "seed": args.seed,
        "wall_seconds": round(wall, 2),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
