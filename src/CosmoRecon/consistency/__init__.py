"""
Null tests of the concordance model.

Each of these forms a statistic that is a known constant if LCDM holds, and
asks the data whether it is. None of them knows which reconstructor produced
its input, which is what lets :mod:`CosmoRecon.ensemble` run the same test
through several methods and report the spread.

The shared machinery -- effective degrees of freedom, the eigenbasis
chi-square, the Hartlap correction -- lives in :mod:`.base` and is the part
these analyses most often get wrong when written by hand.

Members:

``om.py`` (done)
    ``Om`` and ``Om3``, the Sahni-Shafieloo-Starobinsky diagnostics. ``Om`` is
    constant at ``Omega_m`` in flat LCDM; ``Om3`` is exactly 1, needs neither
    ``H0`` nor ``Omega_m``, and therefore needs no extrapolation to ``z = 0``
    -- which is what ``Om`` on cosmic chronometers is quietly standing on.

``curvature.py``
    ``Ok(z)``, the Clarkson-Bassett-Lu test. Measures spatial curvature from
    ``H`` and ``D`` alone, with no dark-energy model -- and is the sharpest
    statement available on whether the curvature preferred by some CMB
    analyses is geometry or systematics.

``duality.py``
    The Etherington relation ``eta(z) = d_L / [(1 + z)^2 d_A]``, which is 1
    in any metric theory with photon conservation, plus the cosmic-opacity
    parametrisations that absorb a violation.

``litmus.py``
    The ``L(z)`` litmus test for a cosmological constant.

``growth.py``
    Growth-geometry consistency: does the measured ``f sigma_8`` match the
    growth implied by the reconstructed expansion history under GR? This is
    the S8 tension asked without a model -- as internal inconsistency rather
    than as a disagreement between two LCDM fits.

``isotropy.py``
    The cosmological principle, from BAO measured across the sky.
"""

from __future__ import annotations

from CosmoRecon.consistency.base import (
    NullTest,
    TestResult,
    effective_modes,
    significance,
)

from CosmoRecon.consistency.om import Om, Om3


__all__ = [
    # the contract
    "NullTest",
    "TestResult",
    "significance",
    "effective_modes",
    # tests
    "Om",
    "Om3",
]
