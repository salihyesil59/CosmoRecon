"""
Distance duality.

``eta(z) = d_L / [(1 + z) D_M]`` is exactly 1 if photons are conserved and travel
on null geodesics. Checked here: that the statistic is exact; that the opacity
slope comes back with no calibration at all; that a violation growing with
redshift is found; that the calibration is asked for rather than assumed, and
the independence between two datasets declared rather than assumed.

And the finding the module is built around: a supernova modulus reconstructed as
``mu`` invents a violation in a universe that has none, and reconstructed as
``mu - 5 log10 z`` does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from CosmoRecon import (
    Cosmography,
    Duality,
    GaussianProcess,
    MethodEnsemble,
    Reconstruction,
    combine_independent,
    linear_grid,
    significance,
)
from CosmoRecon.consistency.duality import REDUCED_MODULUS
from CosmoRecon.core.errors import AlignmentError, DataError
from CosmoRecon.core.provenance import Provenance
from CosmoRecon.data import (
    Dataset,
    chronometers,
    desi_dr2_bao,
    reduced_modulus,
    union3,
)
from CosmoRecon.reconstructors.base import ReconstructionSet

from tests.toy import DistanceDuality, duality_pair


Z = linear_grid(0.5, 2.2, 25)

#: Parameter widths for the tests that ask whether something is *consistent*.
#: The exactness tests use a posterior with no width, and a significance
#: cannot be asked of one -- see the note in test_curvature.py.
REALISTIC = dict(sigma_Om=0.01, sigma_H0=1.0)

#: Inside the redshift range both the bundled supernovae (0.05 -- 2.26) and
#: DESI's transverse distances (0.51 -- 2.33) cover.
SHARED = linear_grid(0.51, 2.26, 20)


def independent_sides(epsilon=0.0, offset=0.0):
    """
    A supernova survey and a BAO survey of the same universe, as two fits
    with their own posteriors -- independent, and declared so.
    """

    supernovae = DistanceDuality(epsilon=epsilon, offset=offset, seed=1, **REALISTIC)

    bao = DistanceDuality(seed=2, **REALISTIC)

    modulus, _ = duality_pair(supernovae, Z)

    _, transverse = duality_pair(bao, Z)

    return modulus, transverse.assume_independent()


def as_fit(recon, dataset):
    """A single reconstruction as the ReconstructionSet a fit would return."""

    return ReconstructionSet(
        {recon.label: recon},
        origin=recon.origin,
        support=(0.05, 2.33),
        provenance=Provenance(
            method="toy", data=(dataset,), seed=0, n_draws=recon.n_draws
        ),
    )


# ============================================================
# Exactness
# ============================================================

def test_eta_is_exactly_one_in_a_transparent_universe():
    """
    Calibrated with the universe's own sound horizon and zero point, the
    statistic is 1 in every draw -- the two distances cancel inside each
    realisation, not on average.
    """

    universe = DistanceDuality(offset=0.37, rd=139.0)

    modulus, transverse = duality_pair(universe, Z)

    curve = Duality(
        sound_horizon=universe.rd, magnitude_offset=universe.offset
    ).statistic(modulus, transverse)

    assert np.allclose(curve.draws, 1.0, atol=1e-6)


def test_without_a_calibration_eta_is_the_calibration_constant():

    universe = DistanceDuality(offset=-0.4, rd=150.0)

    modulus, transverse = duality_pair(universe, Z)

    curve = Duality().statistic(modulus, transverse)

    assert np.allclose(curve.draws, universe.calibration, rtol=1e-6)


@pytest.mark.parametrize("epsilon", [-0.10, 0.0, 0.05, 0.20])
def test_the_opacity_slope_is_recovered_with_no_calibration(epsilon):
    """
    ``eta = K (1 + z)^epsilon`` with the level free: the slope comes back
    exactly, although the zero point differs from draw to draw and nobody
    told the statistic what the sound horizon was.
    """

    universe = DistanceDuality(
        epsilon=epsilon, offset=0.6, rd=141.0, sigma_offset=0.1,
    )

    modulus, transverse = duality_pair(universe, Z)

    slope = Duality().opacity(modulus, transverse)

    assert slope.n_z == 1
    assert np.allclose(slope.draws, epsilon, atol=1e-5)

    # A calibration only moves the level, which the slope does not see.
    calibrated = Duality(sound_horizon=141.0, magnitude_offset=0.6).opacity(
        modulus, transverse
    )

    assert np.allclose(calibrated.draws, slope.draws, atol=1e-9)


# ============================================================
# Finding a violation
# ============================================================

def test_a_transparent_universe_passes():

    modulus, transverse = independent_sides()

    assert Duality().evaluate(modulus, transverse).consistent

    slope = Duality().opacity(modulus, transverse)

    assert significance(slope, 0.0)[2] > 0.05


@pytest.mark.parametrize("epsilon", [-0.10, 0.10])
def test_an_opacity_that_grows_with_redshift_is_found(epsilon):

    modulus, transverse = independent_sides(epsilon=epsilon)

    assert not Duality().evaluate(modulus, transverse).consistent

    slope = Duality().opacity(modulus, transverse)

    mean, width = float(slope.mean()[0]), float(slope.std()[0])

    assert abs(mean - epsilon) < 3.0 * width
    assert significance(slope, 0.0)[3] > 3.0


# ============================================================
# What it needs
# ============================================================

def test_eta_equal_to_one_needs_both_halves_of_the_calibration():

    with pytest.raises(ValueError, match="needs a calibration"):
        Duality(eta=1.0)

    with pytest.raises(ValueError, match="Pass both"):
        Duality(sound_horizon=147.0)

    with pytest.raises(ValueError, match="Pass both"):
        Duality(magnitude_offset=0.0)


def test_a_wrong_zero_point_fails_eta_equal_one_and_nothing_else():
    """
    A redshift-independent offset is degenerate with the calibration. The
    constancy test cannot see it -- and should not -- while ``eta = 1`` with
    the wrong zero point fails, and with the right one passes.
    """

    modulus, transverse = independent_sides(offset=0.25)

    assert Duality().evaluate(modulus, transverse).consistent

    right = Duality(eta=1.0, sound_horizon=147.0, magnitude_offset=0.25)
    wrong = Duality(eta=1.0, sound_horizon=147.0, magnitude_offset=0.0)

    assert right.evaluate(modulus, transverse).consistent
    assert not wrong.evaluate(modulus, transverse).consistent


def test_the_two_sides_have_to_be_declared_independent():
    """
    Two fits to two datasets are two realisation streams. Pairing them by
    index is refused until the independence is said out loud -- and the claim
    can be made on the transverse distance as it comes, before the statistic
    has done any arithmetic with it.
    """

    supernovae = DistanceDuality(seed=1, **REALISTIC)
    bao = DistanceDuality(seed=2, **REALISTIC)

    modulus, _ = duality_pair(supernovae, Z)
    _, transverse = duality_pair(bao, Z)

    with pytest.raises(AlignmentError, match="independence claim"):
        Duality().statistic(modulus, transverse)

    Duality().statistic(modulus, transverse.assume_independent())


def test_a_modulus_that_was_not_reduced_is_refused():

    universe = DistanceDuality()

    modulus, transverse = duality_pair(universe, Z)

    modulus.label = "mu"

    with pytest.raises(DataError, match="singularity"):
        Duality().statistic(modulus, transverse)


def test_the_grid_must_stay_away_from_the_origin():

    universe = DistanceDuality()

    modulus, transverse = duality_pair(universe, linear_grid(0.01, 2.0, 20))

    with pytest.raises(DataError, match="0/0"):
        Duality().statistic(modulus, transverse)


def test_the_slope_refuses_a_distance_posterior_that_reaches_zero():

    universe = DistanceDuality(**REALISTIC)

    modulus, transverse = duality_pair(universe, Z)

    draws = transverse.draws.copy()
    draws[:20, 12] = -1.0

    broken = Reconstruction.from_draws(
        Z, draws, origin=transverse.origin,
        provenance=transverse.provenance, label="DM_over_rs",
    )

    with pytest.raises(DataError, match="not positive"):
        Duality().opacity(modulus, broken)


# ============================================================
# The reduced modulus
# ============================================================

def test_the_reduced_modulus_is_an_exact_transformation():

    supernovae = union3()

    reduced = reduced_modulus(supernovae)

    assert np.allclose(reduced.y, supernovae.y - 5.0 * np.log10(supernovae.z))
    assert np.array_equal(reduced.cov, supernovae.cov)

    # Still the same observations, so still refused alongside themselves.
    assert reduced.name == supernovae.name
    assert reduced.observable == REDUCED_MODULUS

    with pytest.raises(DataError, match="distance modulus"):
        reduced_modulus(chronometers())

    with pytest.raises(DataError, match="distance modulus"):
        reduced_modulus(reduced)


# ============================================================
# Two fits as one
# ============================================================

def test_combining_two_fits_does_not_invent_a_correlation():
    """
    Two fits drawn with the same seed reuse the same random numbers, and
    pairing their draws by index correlates two posteriors that share nothing.
    The combination permutes one side, which is why it is not a formality.
    """

    a = DistanceDuality(seed=5, **REALISTIC)
    b = DistanceDuality(seed=5, **REALISTIC)

    modulus, _ = duality_pair(a, Z)
    _, transverse = duality_pair(b, Z)

    index_paired = np.corrcoef(modulus.draws[:, 10], transverse.draws[:, 10])[0, 1]

    assert index_paired > 0.8

    both = combine_independent(
        as_fit(modulus, "mockSN(22)"), as_fit(transverse, "mockBAO(6)")
    )

    paired = np.corrcoef(
        both["mu_reduced"].draws[:, 10], both["DM_over_rs"].draws[:, 10]
    )[0, 1]

    assert abs(paired) < 0.2

    # One realisation stream now: no claim needed downstream, and each member
    # is still a function that regrids.
    assert both["mu_reduced"].origin == both["DM_over_rs"].origin

    Duality().statistic(both["mu_reduced"], both["DM_over_rs"])

    both["DM_over_rs"].at(linear_grid(0.6, 2.0, 7))


def test_combining_cuts_both_fits_to_the_smaller_draw_count():

    modulus, _ = duality_pair(DistanceDuality(n_draws=400, seed=1), Z)
    _, transverse = duality_pair(DistanceDuality(n_draws=250, seed=2), Z)

    both = combine_independent(
        as_fit(modulus, "mockSN(22)"), as_fit(transverse, "mockBAO(6)")
    )

    assert both["mu_reduced"].n_draws == both["DM_over_rs"].n_draws == 250


def test_combining_refuses_a_shared_dataset_or_observable():

    modulus, transverse = duality_pair(DistanceDuality(seed=1), Z)
    other_modulus, other_transverse = duality_pair(DistanceDuality(seed=2), Z)

    with pytest.raises(DataError, match="not independent"):
        combine_independent(
            as_fit(modulus, "Union3(22)"), as_fit(other_transverse, "Union3(22)")
        )

    with pytest.raises(ValueError, match="distinct functions"):
        combine_independent(
            as_fit(modulus, "mockSN(22)"), as_fit(other_modulus, "otherSN(22)")
        )


# ============================================================
# Mock surveys at the real redshifts and covariances
# ============================================================

def mock_surveys(seed, Om=0.31):
    """
    Union3-like distance moduli and DESI-DR2-like ``D_M/r_d`` from one flat
    LCDM universe -- in which distance duality holds exactly -- at the real
    releases' redshifts, with their real covariances.
    """

    supernovae = union3()
    transverse = desi_dr2_bao().select("DM_over_rs")

    def comoving(z):
        return np.array([
            integrate.quad(lambda x: 1 / np.sqrt(Om * (1 + x) ** 3 + 1 - Om), 0, zi)[0]
            for zi in z
        ])

    rng = np.random.default_rng(seed)

    mu = 5 * np.log10((1 + supernovae.z) * comoving(supernovae.z)) + 43.2
    ratio = 29.5 * comoving(transverse.z)

    def noisy(truth, cov):
        return truth + np.linalg.cholesky(cov) @ rng.normal(size=truth.size)

    return (
        Dataset(z=supernovae.z, y=noisy(mu, supernovae.cov), cov=supernovae.cov,
                observable="mu", unit="mag", name="mockSN"),
        Dataset(z=transverse.z, y=noisy(ratio, transverse.cov), cov=transverse.cov,
                observable="DM_over_rs", name="mockBAO"),
    )


@pytest.fixture(scope="module")
def mocks():

    return [mock_surveys(100 + k) for k in range(3)]


def test_reconstructing_mu_itself_invents_a_violation(mocks):
    """
    The finding the module is built around. In a universe where duality is
    exact, a third-order series fitted to ``mu`` reports a violation at
    better than ten sigma in every realisation -- the price of a logarithmic
    singularity at the origin, paid where the supernova data are sparse. The
    same series fitted to the reduced modulus sees nothing.
    """

    for supernovae, bao in mocks:

        transverse = Cosmography("y", order=3).fit(
            bao, grid=SHARED, seed=0
        )["DM_over_rs"].assume_independent()

        raw = Cosmography("y", order=3).fit(supernovae, grid=SHARED, seed=0)["mu"]

        eta_from_mu = 10.0 ** ((raw - 25.0) / 5.0) / ((1.0 + SHARED) * transverse)

        assert significance(eta_from_mu, 1.0, marginalise_constant=True)[3] > 10.0

        reduced = Cosmography("y", order=3).fit(
            reduced_modulus(supernovae), grid=SHARED, seed=0
        )[REDUCED_MODULUS]

        assert Duality().evaluate(reduced, transverse).sigma < 3.0


def test_two_ensembles_pair_their_members_by_method(mocks):
    """
    A method is a choice made once: member ``m`` on the supernovae is paired
    with member ``m`` on the BAO distances, and the mixture is a mixture of
    those pairs -- one member index per pooled draw, shared by both
    observables -- rather than two separate mixtures multiplied together.
    """

    supernovae, bao = mocks[0]

    ensemble = MethodEnsemble([
        Cosmography("y", order=3),
        Cosmography("log", order=3),
    ])

    sn_fit = ensemble.fit(reduced_modulus(supernovae), grid=SHARED, n_draws=1000, seed=1)
    bao_fit = ensemble.fit(bao, grid=SHARED, n_draws=1000, seed=1)

    both = sn_fit.with_independent(bao_fit)

    assert set(both.observables) == {REDUCED_MODULUS, "DM_over_rs"}

    mixture = both.marginalised

    assert np.array_equal(
        mixture[REDUCED_MODULUS]._predictor._member_of_draw,
        mixture["DM_over_rs"]._predictor._member_of_draw,
    )

    comparison = both.significance(
        lambda s: Duality().statistic(s[REDUCED_MODULUS], s["DM_over_rs"]),
        1.0,
        marginalise_constant=True,
    )

    assert set(comparison.per_method) == set(ensemble.names)

    different = MethodEnsemble([
        Cosmography("y", order=3),
        Cosmography("y", order=2),
    ]).fit(bao, grid=SHARED, n_draws=500, seed=1)

    with pytest.raises(ValueError, match="same set of methods"):
        sn_fit.with_independent(different)


# ============================================================
# The real answer on real data
# ============================================================

def test_union3_and_desi_dr2_give_an_opacity_of_either_sign_by_method():
    """
    Two free-order Chebyshev reconstructions of the same 22 supernova bins and
    six transverse BAO distances report opacity slopes of **opposite sign**,
    each at more than two and a half sigma -- ``+0.15 +/- 0.03`` expanded in
    ``ln(1+z)``, ``-0.07 +/- 0.03`` in ``y``. A paper using either would have
    reported a cosmic opacity, or its opposite.

    Marginalised over the method, the slope is ``-0.01 +/- 0.11``: consistent
    with a transparent universe, and far wider than either single method
    claimed. That width is the uncertainty the data actually leave.
    """

    ensemble = MethodEnsemble([
        GaussianProcess(kernel="matern"),
        Cosmography("y"),
        Cosmography("y", order=3),
        Cosmography("log"),
    ])

    both = ensemble.fit(
        reduced_modulus(union3()), grid=SHARED, n_draws=4000, seed=7
    ).with_independent(
        ensemble.fit(desi_dr2_bao().select("DM_over_rs"), grid=SHARED, n_draws=4000, seed=7)
    )

    def slope(fit):
        return Duality().opacity(fit[REDUCED_MODULUS], fit["DM_over_rs"])

    def summary(s):
        return float(s.mean()[0]), float(s.std()[0])

    in_y = summary(slope(both.members[ensemble.names[1]]))
    in_log = summary(slope(both.members[ensemble.names[3]]))

    assert in_log[0] > 0.0 > in_y[0]
    assert abs(in_log[0]) > 2.5 * in_log[1]
    assert abs(in_y[0]) > 2.5 * in_y[1]

    marginalised = summary(slope(both.marginalised))

    assert abs(marginalised[0]) < marginalised[1], (
        "the method-marginalised opacity slope excludes a transparent universe "
        "on 28 numbers, which would be an extraordinary result and is far more "
        "likely to be a bug"
    )

    assert marginalised[1] > 2.0 * max(in_y[1], in_log[1])
