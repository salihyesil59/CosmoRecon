"""
Null tests of the concordance model.

Each of these forms a statistic that is a known constant if LCDM holds, and
asks the data whether it is. None of them knows which reconstructor produced
its input, which is what lets :mod:`CosmoRecon.ensemble` run the same test
through several methods and report the spread.

The shared machinery -- effective degrees of freedom, the eigenbasis
chi-square, the Hartlap correction -- lives in :mod:`.base` and is the part
these analyses most often get wrong when written by hand.

Planned members:

``om.py``
    ``Om(z)`` and ``Om3(z)``, the Sahni-Shafieloo-Starobinsky diagnostics.
    Constant at ``Omega_m`` in flat LCDM; its *slope* separates quintessence
    from phantom without fitting either.

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


__all__ = [
    "NullTest",
    "TestResult",
    "significance",
    "effective_modes",
]
