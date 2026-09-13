"""
Growth-geometry consistency.

``G(z; z_r)`` is constant, and equal to ``Omega_m`` once calibrated, in every
universe where GR governs the growth -- whatever the dark energy and the
curvature. Checked here: that it is exact there, from an expansion rate and
from a radial BAO distance; that it cannot see the amplitude of fluctuations,
and that :meth:`Growth.sigma8` can; that a gravitational coupling changing with
time breaks constancy, and that a constant one needs the geometric matter
density to be seen; that the null model's growth is the integrated growth; and
the refusals.

The toy universes integrate the second-order growth equation and differentiate
numerically, so the first integral the test is built from is never used to
build its input.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import Cosmography, Growth, Reconstruction, combine_independent, linear_grid
from CosmoRecon.core.errors import AlignmentError, DataError, NotResamplableError
from CosmoRecon.core.provenance import Provenance
from CosmoRecon.data import desi_dr2_bao, growth
from CosmoRecon.validation import LambdaCDM, calibrate

from tests.toy import GrowthUniverse


ANCHOR = 0.8

Z = linear_grid(0.1, 2.0, 20)
Z = Z[np.abs(Z - ANCHOR) > 0.04]

#: Widths for the tests that ask whether something is consistent: the
#: posterior needs width in the direction the signal lies in.
REALISTIC = dict(n_draws=400, sigma_Om=0.01, sigma_H0=1.0, sigma_mu0=0.05)


def calibrated(universe, source):
    """The test, calibrated for the expansion observable ``source``."""

    if source == "H":
        return dict(hubble_constant=float(universe.H0.mean()))

    return dict(hubble_distance=universe.bao_calibration)


# ============================================================
# Exactness
# ============================================================

@pytest.mark.parametrize(
    "background",
    [dict(), dict(Ok=0.1), dict(Om=0.25, Ok=-0.1), dict(w=-0.7), dict(w=-1.2, Ok=0.05)],
)
@pytest.mark.parametrize("source", ["H", "DH_over_rs"])
def test_exact_under_gr_whatever_the_expansion_history(background, source):
    """
    Constant at ``Omega_m``, from chronometer-like ``H`` and from ``D_H / r_d``,
    for Lambda and for dark energy that is not Lambda, flat and curved.
    """

    universe = GrowthUniverse(n_draws=5, **background)

    fit = universe.fit(Z)

    Om = background.get("Om", 0.3)

    curve = Growth(ANCHOR, **calibrated(universe, source)).statistic(
        fit["fsigma8"], fit[source]
    )

    assert np.allclose(curve.draws, Om, atol=1e-4)

    raw = Growth(ANCHOR).statistic(fit["fsigma8"], fit[source])

    assert np.ptp(raw.mean()) < 5e-4 * abs(raw.mean().mean())


def test_the_amplitude_is_invisible_to_the_test_and_sigma8_finds_it():
    """
    A lower ``sigma_8`` rescales ``f sigma_8`` and leaves ``G`` exactly where it
    was -- the S8 question is not a question of constancy -- while
    :meth:`Growth.sigma8`, given the geometric matter density, returns it.
    """

    low = GrowthUniverse(sigma8=0.7, n_draws=5)
    high = GrowthUniverse(sigma8=0.8, n_draws=5)

    test = Growth(ANCHOR, hubble_distance=low.bao_calibration, omega_m=0.3)

    lo, hi = low.fit(Z), high.fit(Z)

    assert np.allclose(
        test.statistic(lo["fsigma8"], lo["DH_over_rs"]).draws,
        test.statistic(hi["fsigma8"], hi["DH_over_rs"]).draws,
        rtol=1e-9,
    )

    assert np.allclose(test.sigma8(lo["fsigma8"], lo["DH_over_rs"]).draws, 0.7, atol=2e-4)
    assert np.allclose(test.sigma8(hi["fsigma8"], hi["DH_over_rs"]).draws, 0.8, atol=2e-4)


# ============================================================
# What it can and cannot see
# ============================================================

def test_a_coupling_that_changes_with_time_is_found():
    """
    ``G_eff / G = 1 + mu0 Omega_DE(a)``: the constancy test needs no
    calibration to see it, and GR with the same posterior width passes.
    """

    gr = GrowthUniverse(**REALISTIC, seed=1).fit(Z)

    assert Growth(ANCHOR).evaluate(gr["fsigma8"], gr["H"]).consistent

    modified = GrowthUniverse(mu0=0.5, **REALISTIC, seed=1).fit(Z)

    result = Growth(ANCHOR).evaluate(modified["fsigma8"], modified["H"])

    assert not result.consistent


def test_a_constant_coupling_needs_the_geometric_matter_density():
    """
    ``G_eff = 1.2 G`` at all times rescales the statistic to ``1.2 Omega_m``,
    which is constant: invisible to the constancy test, and plain once the
    geometry's ``Omega_m`` is supplied.
    """

    universe = GrowthUniverse(g_eff=1.2, n_draws=5)

    fit = universe.fit(Z)

    calibration = calibrated(universe, "DH_over_rs")

    constancy = Growth(ANCHOR, **calibration).statistic(fit["fsigma8"], fit["DH_over_rs"])

    assert np.allclose(constancy.draws, 1.2 * 0.3, atol=1e-4)

    wide = GrowthUniverse(g_eff=1.2, **REALISTIC, seed=2).fit(Z)

    assert Growth(ANCHOR).evaluate(wide["fsigma8"], wide["H"]).consistent

    against_geometry = Growth(
        ANCHOR, hubble_constant=70.0, omega_m=0.3
    ).evaluate(wide["fsigma8"], wide["H"])

    assert not against_geometry.consistent

    # And the amplitude it implies drifts with redshift.
    amplitude = Growth(ANCHOR, **calibration, omega_m=0.3).sigma8(
        fit["fsigma8"], fit["DH_over_rs"]
    )

    assert np.ptp(amplitude.mean()) > 0.05


def test_two_fits_are_combined_only_with_a_declaration():
    """
    The growth rate and the expansion history come from different surveys.
    Undeclared, they are refused; declared -- on each curve, through the
    nested arithmetic the statistic is -- they give the statistic.
    """

    universe = GrowthUniverse(n_draws=20)

    a, b = universe.fit(Z), universe.fit(Z)

    with pytest.raises(AlignmentError):
        Growth(ANCHOR).statistic(a["fsigma8"], b["H"])

    declared = Growth(ANCHOR, hubble_constant=70.0).statistic(
        a["fsigma8"].assume_independent(), b["H"].assume_independent()
    )

    assert np.allclose(declared.draws, 0.3, atol=1e-4)


# ============================================================
# The null model
# ============================================================

@pytest.mark.parametrize("Ok", [0.0, 0.1, -0.1])
def test_the_null_models_growth_is_the_integrated_growth(Ok):
    """
    Heath's integral in :class:`~CosmoRecon.validation.LambdaCDM` against a
    Runge-Kutta integration of the growth equation.
    """

    universe = GrowthUniverse(Om=0.3, Ok=Ok, n_draws=1)

    z = np.linspace(0.02, 2.5, 40)

    table = np.interp(z, universe._z, universe._fs8[0][0])

    predicted = universe.sigma8 * LambdaCDM._growth(0.3, Ok, z)

    assert np.allclose(predicted, table, rtol=2e-5)


def test_the_null_model_fits_the_growth_compilation():

    alone = LambdaCDM().fit(growth())

    assert alone.names == ("omega_m", "sigma8")
    assert 0.15 < alone.best[0] < 0.4
    assert 0.65 < alone.best[1] < 0.95

    joint = LambdaCDM().fit([growth(), desi_dr2_bao().select("DH_over_rs")])

    assert joint.names == ("omega_m", "c_over_H0_rd", "sigma8")


def test_the_calibration_runs_end_to_end_on_the_real_layout():

    grid = np.array([0.6, 0.7, 1.1, 1.3, 1.5])

    both = combine_independent(
        Cosmography("y", order=2).fit(growth(), grid=grid, n_draws=300, seed=1),
        Cosmography("y", order=2).fit(
            desi_dr2_bao().select("DH_over_rs"), grid=grid, n_draws=300, seed=1
        ),
    )

    result = calibrate(
        lambda s: Growth(0.93).statistic(s["fsigma8"], s["DH_over_rs"]),
        both,
        null=LambdaCDM(),
        marginalise_constant=True,
        n_mocks=30,
        seed=3,
    )

    assert not result.abandoned
    assert result.n_mocks == 30
    assert 0.0 < result.pte <= 1.0
    assert result.null.names == ("omega_m", "c_over_H0_rd", "sigma8")


# ============================================================
# Refusals
# ============================================================

def test_calibrations_are_checked_against_what_they_calibrate():

    fit = GrowthUniverse(n_draws=5).fit(Z)

    with pytest.raises(ValueError, match="not both"):
        Growth(ANCHOR, hubble_constant=70.0, hubble_distance=29.0)

    with pytest.raises(ValueError, match="needs a calibration"):
        Growth(ANCHOR, omega_m=0.3)

    with pytest.raises(DataError, match="hubble_distance"):
        Growth(ANCHOR, hubble_constant=70.0).statistic(fit["fsigma8"], fit["DH_over_rs"])

    with pytest.raises(DataError, match="hubble_constant"):
        Growth(ANCHOR, hubble_distance=29.0).statistic(fit["fsigma8"], fit["H"])

    with pytest.raises(ValueError, match="Omega_m"):
        Growth(ANCHOR, hubble_constant=70.0).sigma8(fit["fsigma8"], fit["H"])


def test_inputs_that_are_not_what_the_test_needs_are_refused():

    fit = GrowthUniverse(n_draws=5).fit(Z)

    transverse = Reconstruction.from_predictor(
        Z, fit["H"]._predictor, provenance=fit["H"].provenance,
        origin=fit["H"].origin, label="DM_over_rs",
    )

    with pytest.raises(DataError, match="H or D_H/r_d"):
        Growth(ANCHOR).statistic(fit["fsigma8"], transverse)

    with pytest.raises(DataError, match="signature"):
        Growth(ANCHOR).statistic(fit["H"], fit["fsigma8"])

    with pytest.raises(DataError, match="anchor"):
        Growth(0.1).statistic(fit["fsigma8"], fit["H"])

    frozen = Reconstruction.from_draws(
        Z, fit["fsigma8"].draws, provenance=Provenance(method="by hand"),
        origin=fit["H"].origin, label="fsigma8",
    )

    with pytest.raises(NotResamplableError, match="integrates"):
        Growth(ANCHOR).statistic(frozen, fit["H"])
