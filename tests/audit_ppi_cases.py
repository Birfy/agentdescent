"""The six inputs the PPI golden vectors are recorded on.

Shared between the regression test and the generator that writes the fixture, so
that the numbers on disk and the numbers under test come from the *same* inputs.
Duplicating the construction is how a golden file silently starts guarding a
different question than the one it was written for.

Each case pins a distinct regime rather than a distinct number. Six vectors that
all sit in the middle of the parameter space would agree with any estimator that
gets the middle right, and the middle is not where estimators break.
"""

import numpy as np

from agentdescent.audit.ppi import Stratum

__all__ = ["CASES", "build"]

#: ``name -> kwargs for build()``. The comment on each says what it would catch.
CASES = {
    # The ordinary shape: three layers, unequal weights, a verifier that is
    # generous in one direction. Catches almost any arithmetic slip.
    "stratified_generous": dict(
        weights=(0.5, 0.3, 0.2), rates=(0.8, 0.5, 0.2),
        n_per=100, n_unlab=300, bias_dir=0.3, seed=0),

    # A verifier carrying no signal. `lam` must collapse and the answer must
    # fall back to the labelled mean -- the guarantee that makes PPI safe to
    # enable. A change that lets `lam` drift up would show here first.
    "useless_verifier": dict(
        weights=(1.0,), rates=(0.5,), n_per=150, n_unlab=450,
        bias_dir=0.0, noise_f=True, seed=1),

    # A verifier that is right every time. `lam` goes to its ceiling and the
    # gain factor is large; pins the top of the range.
    "perfect_verifier": dict(
        weights=(1.0,), rates=(0.4,), n_per=120, n_unlab=880,
        bias_dir=0.0, perfect_f=True, seed=2),

    # Nothing unlabelled to borrow from. Must degenerate exactly to the
    # classical mean rather than divide by zero.
    "no_unlabelled": dict(
        weights=(1.0,), rates=(0.6,), n_per=80, n_unlab=0,
        bias_dir=0.3, seed=3),

    # Strata whose means are far apart but whose sample sizes are equal, so the
    # weights carry the whole answer. This is the case mutation A destroys.
    "weights_carry_the_answer": dict(
        weights=(0.7, 0.2, 0.1), rates=(0.9, 0.5, 0.1),
        n_per=60, n_unlab=200, bias_dir=0.25, seed=4),

    # Under `MIN_N_DOMINANT`. The warning must be present, and its wording is
    # part of the record: a caller stores the number and the reason together.
    "thin_dominant_stratum": dict(
        weights=(0.6, 0.4), rates=(0.7, 0.3),
        n_per=25, n_unlab=250, bias_dir=0.3, seed=5),
}


def build(*, weights, rates, n_per, n_unlab, bias_dir, seed,
          noise_f=False, perfect_f=False):
    """One case's strata, plus the population mean they are estimating.

    ``bias_dir`` forgives a wrong answer with that probability and never marks a
    right one down, so the verifier's error has a **sign**. Symmetric noise would
    be unbiased on balanced binary outcomes and would let a broken estimator
    record perfectly reasonable-looking vectors.
    """
    rng = np.random.default_rng(seed)
    strata, truth = [], 0.0
    for h, (w, p) in enumerate(zip(weights, rates)):
        total = n_per + n_unlab
        y = (rng.random(total) < p).astype(float)
        if noise_f:
            f = rng.random(total)
        elif perfect_f:
            f = y.copy()
        else:
            f = np.where((y == 0) & (rng.random(total) < bias_dir), 1.0, y)
        strata.append(Stratum(f"s{h}", w, f[:n_per], y[:n_per], f[n_per:]))
        truth += w * p
    return strata, truth
