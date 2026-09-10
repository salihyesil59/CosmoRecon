"""
The Clarkson-Bassett-Lu curvature test.

``Ok(z)`` is ``Omega_k`` in *any* FLRW universe, whatever the dark energy. So a
departure from constancy is not evidence about dark energy -- it is evidence
against homogeneity and isotropy. That makes it the strongest claim in the
library, and the one to be most careful with.

Three things are checked. That the statistic recovers a known ``Omega_k``; that
it notices an FLRW relation deliberately broken; and -- the one that matters
most -- that on the real DESI DR2 data it reports honestly that six transverse
and six radial measurements cannot support it.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import (
    C_LIGHT_KM_S,
    Cosmography,
    Curvature,
    MethodEnsemble,
    linear_grid,
)
from CosmoRecon.core.errors import AlignmentError, DataError
from CosmoRecon.data import desi_dr2_bao

from tests.toy import CurvedFLRW, bao_pair


Z = linear_grid(0.5, 2.3, 25)

#: Parameter widths for the tests that ask whether a result is *consistent*.
#:
#: The exactness tests below deliberately use a posterior with no width, so
#: that the recovered Omega_k can be compared to the truth at the
#: differentiation floor. A significance cannot be asked of such a posterior:
#: with a covariance of pure rounding error, any residual at all is infinitely
#: many sigma, and the test would be measuring the floating-point unit rather
#: than the machinery. So a consistency check gets a realistic spread.
REALISTIC = dict(sigma_Om=0.01, sigma_H0=1.0, sigma_Ok=0.01)

#: DESI DR2 publishes r_d h = 101.54 Mpc, so c / (H0 r_d) = c / (100 * 101.54).
DESI_CALIBRATION = C_LIGHT_KM_S / (100.0 * 101.54)


# ============================================================
# Recovery
# ============================================================

@pytest.mark.parametrize("omega_k", [-0.10, -0.02, 0.0, 0.02, 0.10])
def test_the_statistic_recovers_a_known_curvature(omega_k):
    """
    Fed a universe with a known ``Omega_k``, the statistic returns it -- and
    returns it *constant in redshift*, which is the property the null test
    rests on.

    The toy's transverse distance is integrated numerically and differentiated
    numerically, deliberately not through the FLRW relation
    ``D_M' = D_H sqrt(1 + Ok (H0 D_M/c)^2)``. That relation is what is being
    tested, and using it to build the input would make this circular.
    """

    universe = CurvedFLRW(Ok=omega_k)

    transverse, radial = bao_pair(universe, Z)

    curve = Curvature(
        hubble_distance=universe.calibration
    ).statistic(transverse, radial)

    assert np.allclose(curve.mean(), omega_k, atol=1e-4)

    # Constant to the toy's own differentiation floor.
    assert np.ptp(curve.mean()) < 1e-6


def test_an_flrw_universe_passes_the_test():

    universe = CurvedFLRW(Ok=0.03, **REALISTIC)

    transverse, radial = bao_pair(universe, Z)

    assert Curvature().evaluate(transverse, radial).consistent


@pytest.mark.parametrize("distortion", [0.02, 0.05])
def test_a_broken_flrw_relation_shows_up_as_a_varying_ok(distortion):
    """
    The radial distance is multiplied by ``1 + a z``, which breaks the FLRW
    relation between the two distances while leaving both perfectly smooth.
    ``Ok(z)`` then varies -- which is the signature the test exists to find,
    and which no dark-energy model can produce.
    """

    flat = CurvedFLRW(Ok=0.0)

    broken = CurvedFLRW(Ok=0.0, distort=distortion)

    def spread(universe):
        transverse, radial = bao_pair(universe, Z)
        curve = Curvature(
            hubble_distance=universe.calibration
        ).statistic(transverse, radial)
        return float(np.ptp(curve.mean()))

    assert spread(flat) < 1e-6

    assert spread(broken) > 0.02 * (distortion / 0.02)


# ============================================================
# What it needs, and what it does not
# ============================================================

def test_flatness_and_flrw_need_no_calibration():
    """
    The whole calibration -- ``H0`` and the sound horizon, neither of which
    BAO measures -- is one multiplicative constant. Constancy is unaffected by
    it, and zero times it is still zero, so two of the three questions can be
    asked without knowing it at all.
    """

    universe = CurvedFLRW(Ok=0.0, **REALISTIC)

    transverse, radial = bao_pair(universe, Z)

    # FLRW: is it constant?
    assert Curvature().evaluate(transverse, radial).consistent

    # Flat: is it zero?
    assert Curvature(omega_k=0.0).evaluate(transverse, radial).consistent

    # And a curved universe fails the flatness test but not the FLRW one.
    curved = CurvedFLRW(Ok=0.05, **REALISTIC)

    t2, r2 = bao_pair(curved, Z)

    assert Curvature().evaluate(t2, r2).consistent
    assert not Curvature(omega_k=0.0).evaluate(t2, r2).consistent


def test_a_specific_nonzero_curvature_needs_the_calibration():

    with pytest.raises(ValueError, match="needs a calibration"):
        Curvature(omega_k=0.05)

    # With it, the test is available and sharp.
    universe = CurvedFLRW(Ok=0.05, **REALISTIC)

    transverse, radial = bao_pair(universe, Z)

    test = Curvature(omega_k=0.05, hubble_distance=universe.calibration)

    assert test.evaluate(transverse, radial).consistent

    wrong = Curvature(omega_k=-0.05, hubble_distance=universe.calibration)

    assert not wrong.evaluate(transverse, radial).consistent


def test_the_two_inputs_must_come_from_one_fit():
    """
    They are correlated, at ``r`` of about ``-0.4`` in DESI DR2, and the
    statistic subtracts two quantities of similar size -- so their correlation
    sets the width of the difference. Two separate fits are refused rather
    than silently paired.
    """

    a = CurvedFLRW(Ok=0.0, seed=1)
    b = CurvedFLRW(Ok=0.0, seed=2)

    transverse, _ = bao_pair(a, Z)
    _, radial = bao_pair(b, Z)

    with pytest.raises(AlignmentError, match="independence claim"):
        Curvature().statistic(transverse, radial)


def test_the_grid_must_stay_away_from_the_origin():

    universe = CurvedFLRW(Ok=0.0)

    transverse, radial = bao_pair(universe, linear_grid(0.01, 2.0, 20))

    with pytest.raises(DataError, match="vanishes at the origin"):
        Curvature().statistic(transverse, radial)


# ============================================================
# The real answer on real data
# ============================================================

def test_desi_dr2_alone_cannot_support_the_curvature_test():
    """
    The finding, and the reason the library exists.

    Six transverse and six radial BAO measurements, one of which has to be
    *differentiated*, do not constrain ``Ok(z)``. Four nearly identical
    polynomial reconstructions of the same twelve numbers report anything from
    a fraction of a sigma to a formally infinite one -- a spurious violation of
    the Copernican principle at several sigma is available to whoever picks the
    right expansion variable.

    The method-marginalised answer is consistent with FLRW, which is the
    honest reading: this dataset cannot answer the question. Any single-method
    analysis would have reported whatever its method gave, with no way of
    telling from inside that analysis which it was.
    """

    joint = desi_dr2_bao().select("DM_over_rs", "DH_over_rs")

    grid = np.linspace(0.6, 2.25, 20)

    fit = MethodEnsemble([
        Cosmography("y"),
        Cosmography("log"),
        Cosmography("y", order=3),
        Cosmography("y", family="monomial"),
    ]).fit(joint, grid=grid, n_draws=4000, seed=4)

    comparison = fit.significance(
        lambda s: Curvature(hubble_distance=DESI_CALIBRATION).statistic(
            s["DM_over_rs"], s["DH_over_rs"]
        ),
        0.0,
        marginalise_constant=True,
        name="Ok(z) from DESI DR2 BAO alone",
    )

    sigmas = [value[3] for value in comparison.per_method.values()]

    # At least one method claims a decisive violation of FLRW, and at least
    # one sees nothing. That spread is the result.
    assert max(sigmas) > 3.0
    assert min(sigmas) < 1.0

    # And the honest answer, marginalised over the method, is consistent.
    assert comparison.marginalised[2] > 0.05, (
        "the method-marginalised curvature test claims a departure from FLRW "
        "on twelve BAO numbers, which would be an extraordinary result and is "
        "far more likely to be a bug"
    )


def test_the_joint_fit_is_what_makes_the_test_possible():
    """
    ``Ok(z)`` needs both BAO observables from one fit. Selecting them
    separately and pairing the results is refused -- correctly, since they are
    correlated at ``r ~ -0.4``.
    """

    bao = desi_dr2_bao()

    grid = np.linspace(0.6, 2.25, 15)

    joint = Cosmography("y").fit(
        bao.select("DM_over_rs", "DH_over_rs"), grid=grid, n_draws=500, seed=1
    )

    Curvature().statistic(joint["DM_over_rs"], joint["DH_over_rs"])

    separate_dm = Cosmography("y").fit(
        bao.select("DM_over_rs"), grid=grid, n_draws=500, seed=1
    )["DM_over_rs"]

    separate_dh = Cosmography("y").fit(
        bao.select("DH_over_rs"), grid=grid, n_draws=500, seed=1
    )["DH_over_rs"]

    with pytest.raises(AlignmentError):
        Curvature().statistic(separate_dm, separate_dh)
