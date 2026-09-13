"""
The litmus tests for a cosmological constant.

Zunckel & Clarkson's flat test is zero -- or, rearranged, constant -- in every
flat Lambda-CDM universe, and says nothing about a curved one; the curved test
is constant in Lambda-CDM whatever the curvature. Checked here: both are exact
where they should be; the rearranged flat statistic is the paper's ``L(z)`` and
is blind to the calibration; curvature fools the flat test and not the curved
one; a dark energy that is not Lambda is found by both; and the refusals.

The toy universes integrate and differentiate their distances numerically, so
none of the relations the tests are built from is used to build their input.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import CurvedLitmus, Litmus, Reconstruction, linear_grid
from CosmoRecon.core.errors import AlignmentError, DataError

from tests.toy import DarkEnergyUniverse


Z = linear_grid(0.2, 2.2, 21)

#: Away from the curved test's anchor.
ANCHOR = 0.9
Z_CURVED = Z[np.abs(Z - ANCHOR) > 0.04]

#: Widths for the tests that ask whether something is consistent. ``w`` has to
#: vary between realisations for the posterior to have any width in the
#: direction a litmus test measures.
REALISTIC = dict(sigma_Om=0.01, sigma_H0=1.0, sigma_w=0.03)


# ============================================================
# Exactness
# ============================================================

@pytest.mark.parametrize("Om", [0.2, 0.3, 0.4])
@pytest.mark.parametrize("source", ["DM_over_rs", "mu_reduced"])
def test_the_flat_test_is_exact_in_flat_lcdm(Om, source):
    """
    ``Q(z)`` is constant, and equal to 1 once calibrated, in every flat
    Lambda-CDM universe -- from a BAO distance and from a supernova modulus,
    which reaches the distance through a chain rule.
    """

    universe = DarkEnergyUniverse(Om=Om, offset=0.3, n_draws=20)

    fit = universe.fit(Z)

    calibration = (
        universe.bao_calibration if source == "DM_over_rs"
        else universe.modulus_calibration
    )

    raw = Litmus().statistic(fit[source])

    assert np.ptp(raw.mean()) < 1e-4 * abs(raw.mean().mean())

    calibrated = Litmus(hubble_distance=calibration).statistic(fit[source])

    assert np.allclose(calibrated.draws, 1.0, atol=1e-4)


def test_q_is_the_papers_l_and_is_blind_to_the_calibration():
    """
    ``L(z) = 3 (1+z)^2 D'^3 (Q - 1)`` for the paper's ``L`` of a normalised
    distance, in a universe where both are far from zero -- and rescaling the
    distance by any factor rescales ``Q`` by its inverse square, which a test
    of constancy cannot see.
    """

    universe = DarkEnergyUniverse(w=-0.7, n_draws=20)

    T = universe.fit(Z)["DM_over_rs"]

    alpha = universe.bao_calibration

    Q = Litmus(hubble_distance=alpha).statistic(T)

    L = Litmus(hubble_distance=alpha).zunckel_clarkson(T)

    slope = (T.d(1) / alpha).draws

    assert np.allclose(L.draws, 3 * (1 + Z) ** 2 * slope**3 * (Q.draws - 1), atol=1e-6)
    assert np.max(np.abs(L.mean())) > 0.05

    # A distance in another calibration is a different function of redshift,
    # with its own analytic derivatives -- not the output of arithmetic.
    def rescaled_predictor(z, *, derivative=0):
        return 3.7 * T._predictor(z, derivative=derivative)

    rescaled = Reconstruction.from_predictor(
        Z, rescaled_predictor, provenance=T.provenance, origin=T.origin,
        label="DM_over_rs",
    )

    assert np.allclose(
        Litmus().statistic(rescaled).draws * 3.7**2,
        Litmus().statistic(T).draws,
        rtol=1e-9,
    )


@pytest.mark.parametrize("Ok", [-0.1, 0.0, 0.1])
def test_the_curved_test_is_exact_whatever_the_curvature(Ok):

    universe = DarkEnergyUniverse(Om=0.3, Ok=Ok, n_draws=20)

    fit = universe.fit(Z_CURVED)

    curve = CurvedLitmus(ANCHOR, hubble_distance=universe.bao_calibration).statistic(
        fit["DM_over_rs"], fit["DH_over_rs"]
    )

    assert np.allclose(curve.draws, 0.3, atol=1e-4)


# ============================================================
# What each can and cannot see
# ============================================================

def test_curvature_fools_the_flat_test_and_not_the_curved_one():
    """
    With a cosmological constant and ``Omega_k = 0.1`` the flat test reports a
    varying ``Q`` -- dynamical dark energy made entirely of curvature -- while
    the curved test stays constant.
    """

    universe = DarkEnergyUniverse(Ok=0.1, n_draws=20)

    fit = universe.fit(Z_CURVED)

    flat = Litmus(hubble_distance=universe.bao_calibration).statistic(fit["DM_over_rs"])

    assert np.ptp(flat.mean()) > 0.1

    curved = CurvedLitmus(ANCHOR, hubble_distance=universe.bao_calibration).statistic(
        fit["DM_over_rs"], fit["DH_over_rs"]
    )

    assert np.ptp(curved.mean()) < 1e-4


@pytest.mark.parametrize("w", [-0.8, -1.2])
@pytest.mark.parametrize("Ok", [0.0, 0.1])
def test_dark_energy_that_is_not_lambda_is_found(w, Ok):

    universe = DarkEnergyUniverse(Ok=Ok, w=w, n_draws=20)

    fit = universe.fit(Z_CURVED)

    curved = CurvedLitmus(ANCHOR).statistic(fit["DM_over_rs"], fit["DH_over_rs"])

    assert np.ptp(curved.mean()) > 0.01 * abs(curved.mean().mean())

    if Ok == 0.0:

        flat = Litmus().statistic(fit["DM_over_rs"])

        assert np.ptp(flat.mean()) > 0.01 * abs(flat.mean().mean())


def test_a_realistic_posterior_passes_for_lambda_and_fails_without_it():

    lam = DarkEnergyUniverse(**REALISTIC, seed=1).fit(Z_CURVED)

    assert Litmus().evaluate(lam["mu_reduced"]).consistent
    assert CurvedLitmus(ANCHOR).evaluate(lam["DM_over_rs"], lam["DH_over_rs"]).consistent

    quintessence = DarkEnergyUniverse(w=-0.7, **REALISTIC, seed=1).fit(Z_CURVED)

    assert not Litmus().evaluate(quintessence["mu_reduced"]).consistent
    assert not CurvedLitmus(ANCHOR).evaluate(
        quintessence["DM_over_rs"], quintessence["DH_over_rs"]
    ).consistent


# ============================================================
# Refusals
# ============================================================

def test_the_papers_form_needs_a_calibration():

    T = DarkEnergyUniverse(n_draws=5).fit(Z)["DM_over_rs"]

    with pytest.raises(ValueError, match="normalised"):
        Litmus().zunckel_clarkson(T)

    with pytest.raises(ValueError, match="needs a calibration"):
        CurvedLitmus(ANCHOR, omega_m=0.3)


def test_a_modulus_that_was_not_reduced_is_refused():

    m = DarkEnergyUniverse(n_draws=5).fit(Z)["mu_reduced"]

    m.label = "mu"

    with pytest.raises(DataError, match="reduced modulus"):
        Litmus().statistic(m)


def test_the_grid_must_stay_away_from_the_origin_and_the_anchor():

    fit = DarkEnergyUniverse(n_draws=5).fit(linear_grid(0.01, 2.0, 20))

    with pytest.raises(DataError, match="grid reaches"):
        Litmus().statistic(fit["DM_over_rs"])

    fit = DarkEnergyUniverse(n_draws=5).fit(np.array([0.5, ANCHOR, 1.5]))

    with pytest.raises(DataError, match="0/0"):
        CurvedLitmus(ANCHOR).statistic(fit["DM_over_rs"], fit["DH_over_rs"])

    with pytest.raises(ValueError, match="below"):
        CurvedLitmus(0.01)


def test_a_distance_that_stops_increasing_is_refused():
    """
    ``Q`` divides by ``D'^3``. A posterior whose distance turns over somewhere
    on the grid is not an expansion history, and the few draws that do it would
    dominate the statistic.
    """

    T = DarkEnergyUniverse(n_draws=50).fit(Z)["DM_over_rs"]

    class Turning:

        def __call__(self, z, *, derivative=0):

            values = T._predictor(z, derivative=derivative)

            if derivative == 1:
                values = values.copy()
                values[:5, -3:] = -1.0

            return values

    broken = Reconstruction.from_predictor(
        Z, Turning(), provenance=T.provenance, origin=T.origin, label="DM_over_rs"
    )

    with pytest.raises(DataError, match="stops increasing"):
        Litmus().statistic(broken)


def test_the_curved_test_needs_one_joint_fit():

    a = DarkEnergyUniverse(n_draws=5, seed=1).fit(Z_CURVED)
    b = DarkEnergyUniverse(n_draws=5, seed=2).fit(Z_CURVED)

    with pytest.raises(AlignmentError, match="independence claim"):
        CurvedLitmus(ANCHOR).statistic(a["DM_over_rs"], b["DH_over_rs"])
