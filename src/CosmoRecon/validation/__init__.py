"""
Proving that a method -- and a null test built on it -- deserves to be in the
library.

The rule this package enforces: **a reconstruction method that reports a 68%
interval must contain the truth 68% of the time,** and its counterpart for null
tests: **a test that reports p = 0.05 must be wrong 5% of the time when the null
is true.** A method that reports 68% and covers 45% is manufacturing
significance -- which is the failure mode CosmoRecon exists to expose, and would
be an embarrassing one to ship.

Both are therefore *tests*, not figures produced once for a paper.

Contents:

``nulls.py`` (done)
    Null models: small parametric universes in which a null hypothesis holds by
    construction, fitted to the same data an analysis used and able to simulate
    those data exactly as released. Lambda-CDM, flat or curved.

``calibration.py`` (done)
    Significances calibrated by simulation: rerun the whole analysis on mocks
    drawn from the fitted null model, and read the p-value off where the
    observed statistic falls among theirs. Needed because the nominal
    chi-square is miscalibrated in both directions -- measured, not supposed.

``mocks.py``
    Injection. Cosmic chronometer, BAO, supernova and growth mocks at a real
    survey's error budget from a known ``E(z)`` -- LCDM, CPL, an oscillating
    ``w``, a sign-switching Lambda -- so that the truth is known exactly and the
    recovery can be scored.

``coverage.py``
    Recovery. Over many realisations, the empirical coverage of each stated
    interval, per method and per redshift, with the binomial uncertainty on the
    coverage itself. Reported as a table a referee can read.

The cross-checks that live in the test suite rather than here follow the same
principle -- anything checkable two ways is checked two ways: an analytic GP
derivative against finite differences of the same draw, a Chebyshev expansion
against a Taylor one inside the radius of convergence, and the whole null-test
chain against a ``Reconstruction`` built analytically from a known cosmology,
where every statistic has a value known in closed form.
"""

from __future__ import annotations

from CosmoRecon.validation.nulls import LambdaCDM, NullFit, NullModel

from CosmoRecon.validation.calibration import (
    DEFAULT_N_MOCKS,
    CalibratedComparison,
    Calibration,
    calibrate,
)


__all__ = [
    # null models
    "NullModel",
    "NullFit",
    "LambdaCDM",
    # calibration
    "calibrate",
    "Calibration",
    "CalibratedComparison",
    "DEFAULT_N_MOCKS",
]
