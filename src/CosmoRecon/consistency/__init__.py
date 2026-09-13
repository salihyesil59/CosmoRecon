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

``curvature.py`` (done)
    ``Ok(z)``, the Clarkson-Bassett-Lu test. Constant in *any* FLRW universe,
    whatever the dark energy -- so a departure from constancy falsifies
    homogeneity and isotropy, not a dark-energy model. Needs the transverse and
    radial BAO distances reconstructed jointly, and needs no calibration at all
    to ask either whether the universe is FLRW or whether it is flat.

``duality.py`` (done)
    The Etherington relation ``eta(z) = d_L / [(1 + z) D_M]``, which is 1 in
    any metric theory with photon conservation, and the opacity slope
    ``epsilon`` in ``eta ~ (1 + z)^epsilon``. The first test built from two
    different datasets, so the independence between them is declared rather
    than assumed. Neither constancy nor the slope needs a calibration; and the
    supernovae enter as ``mu - 5 log10 z``, because a reconstruction of ``mu``
    itself invents a violation in a universe that has none.

``litmus.py`` (done)
    Zunckel & Clarkson's litmus test for a cosmological constant, from
    distances alone: ``Q(z)`` is constant in flat Lambda-CDM whatever the matter
    density and the distance calibration. Curvature fools it, so
    ``CurvedLitmus`` builds the same test from a BAO release's transverse and
    radial distances together, with the curvature removed through the
    Clarkson-Bassett-Lu relation -- constant in Lambda-CDM of any curvature,
    with first derivatives only and no calibration.

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

from CosmoRecon.consistency.curvature import Curvature

from CosmoRecon.consistency.duality import Duality

from CosmoRecon.consistency.litmus import CurvedLitmus, Litmus

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
    "Curvature",
    "Duality",
    "Litmus",
    "CurvedLitmus",
]
