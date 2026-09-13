"""
Calibration by simulation.

The nominal chi-square of a null test is miscalibrated in both directions -- a
Gaussian process almost never rejects, a free-order series rejects a true null
most of the time -- and :mod:`CosmoRecon.validation` is the repair. Checked
here: that the null model is the surveys' own fit; that a mock is the same kind
of dataset as the data, with the null true; that a fit reruns itself exactly;
that a method rejecting a true null most of the time nominally does so at the
nominal rate once calibrated; that calibration keeps the power to find a
deviation that is really there; and the refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import (
    Cosmography,
    Curvature,
    MethodEnsemble,
    Om,
    Reconstruction,
    combine_independent,
)
from CosmoRecon.core.errors import CosmoReconError, DataError, NotRefittableError
from CosmoRecon.core.provenance import Provenance
from CosmoRecon.data import (
    Dataset,
    chronometers,
    desi_dr2_bao,
    growth,
    reduced_modulus,
    union3,
)
from CosmoRecon.reconstructors.base import ReconstructionSet
from CosmoRecon.validation import LambdaCDM, calibrate


GRID = np.linspace(0.35, 1.9, 20)


def om(fit):
    """Om anchored inside the chronometers, so nothing is extrapolated."""

    return Om(z_reference=0.2).statistic(fit["H"])


def mock_chronometers(seed, *, w=-1.0, error_scale=1.0):
    """
    Chronometer-like ``H(z)`` at the real redshifts, with the real covariance
    (optionally shrunk), from flat wCDM -- Lambda-CDM at ``w = -1``.
    """

    cc = chronometers()

    cov = cc.cov * error_scale**2

    truth = 68.0 * np.sqrt(0.31 * (1 + cc.z) ** 3 + 0.69 * (1 + cc.z) ** (3 * (1 + w)))

    rng = np.random.default_rng(seed)

    return Dataset(
        z=cc.z,
        y=truth + np.linalg.cholesky(cov) @ rng.normal(size=cc.z.size),
        cov=cov,
        observable="H",
        unit="km/s/Mpc",
        name="mockCC",
    )


# ============================================================
# Null models
# ============================================================

def test_the_null_model_is_the_surveys_own_fit():
    """
    Fitted to the bundled releases, the null model returns their published
    parameters -- the same check the data loaders are held to, and the reason
    to trust the universes the mocks are drawn from.
    """

    desi = LambdaCDM().fit(desi_dr2_bao())

    assert desi.names == ("omega_m", "c_over_H0_rd")

    assert desi.best[0] == pytest.approx(0.2975, abs=0.002)
    assert 299792.458 / (100.0 * desi.best[1]) == pytest.approx(101.54, abs=0.3)

    supernovae = LambdaCDM().fit(union3())

    assert supernovae.best[0] == pytest.approx(0.356, abs=0.005)
    assert np.sqrt(supernovae.covariance[0, 0]) == pytest.approx(0.026, abs=0.003)

    # The reduced modulus is the same data, so it is the same fit.
    reduced = LambdaCDM().fit(reduced_modulus(union3()))

    assert np.allclose(reduced.best, supernovae.best, rtol=1e-4)

    curved = LambdaCDM(curved=True).fit(desi_dr2_bao())

    assert curved.names == ("omega_m", "omega_k", "c_over_H0_rd")


def test_a_null_model_refuses_what_it_cannot_predict():

    cc = chronometers()

    lensing = Dataset(
        z=cc.z, y=cc.y, cov=cc.cov, observable="convergence", unit="",
        name="not an observable the model knows",
    )

    with pytest.raises(DataError, match="convergence"):
        LambdaCDM().fit(lensing)

    # The growth rate it does know, with sigma8 as one more parameter.
    assert LambdaCDM().fit(growth()).names == ("omega_m", "sigma8")


def test_a_mock_is_the_same_kind_of_dataset_with_the_null_true():
    """
    Same class, same name, same redshifts in the same row order -- the analysis
    being calibrated cannot tell a mock from the release -- and scattered
    about the null prediction with exactly the released covariance.
    """

    bao = desi_dr2_bao()

    fitted = LambdaCDM().fit(bao)

    rng = np.random.default_rng(0)

    mocks = [fitted.simulate(rng, marginalise=False)[0] for _ in range(4000)]

    assert type(mocks[0]) is type(bao)
    assert mocks[0].name == bao.name
    assert np.array_equal(mocks[0].z, bao.z)
    assert np.array_equal(mocks[0].quantity, bao.quantity)

    values = np.array([mock.values for mock in mocks])

    (prediction,) = fitted.model.predict(fitted.best, (bao,))

    sigma = np.sqrt(np.diag(bao.cov))

    assert np.allclose(values.mean(axis=0), prediction, rtol=0.0, atol=5 * sigma / np.sqrt(4000))

    assert np.allclose(
        np.cov(values, rowvar=False), bao.cov, rtol=0.0, atol=0.1 * np.outer(sigma, sigma)
    )


def test_a_marginalised_null_draws_its_truth_from_the_fitted_posterior():

    fitted = LambdaCDM().fit(chronometers())

    rng = np.random.default_rng(1)

    draws = np.array([fitted._draw_parameters(rng) for _ in range(2000)])

    width = np.sqrt(np.diag(fitted.covariance))

    assert np.allclose(draws.mean(axis=0), fitted.best, rtol=0.0, atol=5 * width / np.sqrt(2000))
    assert np.allclose(draws.std(axis=0), width, rtol=0.1)


# ============================================================
# Refitting
# ============================================================

def test_a_fit_reruns_itself_exactly():

    data = chronometers()

    method = Cosmography("y")

    fit = method.fit(data, grid=GRID, n_draws=500, seed=3)

    assert fit.refittable
    assert fit.datasets == (data,)

    assert np.array_equal(fit.refit(fit.datasets, seed=3)["H"].draws, fit["H"].draws)

    # The configuration is the one at fit time, whatever has happened since.
    method.order = 2

    again = fit.refit(fit.datasets, seed=3)["H"]

    assert again.provenance.method == fit["H"].provenance.method
    assert np.array_equal(again.draws, fit["H"].draws)

    with pytest.raises(ValueError, match="count"):
        fit.refit((data, data))

    # A regridded set reruns onto its own grid.
    assert fit.at(np.linspace(0.4, 1.8, 7)).refit(fit.datasets, seed=3)["H"].n_z == 7


def test_a_combination_of_two_fits_reruns_both():

    grid = np.linspace(0.51, 2.26, 12)

    supernovae = Cosmography("y", order=3).fit(
        reduced_modulus(union3()), grid=grid, n_draws=500, seed=1
    )

    bao = Cosmography("y", order=3).fit(
        desi_dr2_bao().select("DM_over_rs"), grid=grid, n_draws=500, seed=1
    )

    both = combine_independent(supernovae, bao)

    assert [d.name for d in both.datasets] == ["Union3", "DESI-DR2-BAO[DM_over_rs]"]

    again = both.refit(both.datasets, seed=1)

    assert set(again) == {"mu_reduced", "DM_over_rs"}
    assert np.allclose(again["mu_reduced"].mean(), supernovae["mu_reduced"].mean(), rtol=1e-3)


def test_something_not_produced_by_a_fit_cannot_be_calibrated():

    H = Reconstruction.from_draws(
        GRID, np.ones((50, GRID.size)), provenance=Provenance(method="by hand")
    )

    hand = ReconstructionSet(
        {"H": H}, origin=H.origin, support=(0.1, 2.0), provenance=H.provenance
    )

    assert not hand.refittable

    with pytest.raises(NotRefittableError):
        hand.refit((chronometers(),))

    with pytest.raises(NotRefittableError):
        calibrate(om, hand, null=LambdaCDM(), marginalise_constant=True, n_mocks=50)


# ============================================================
# The repair
# ============================================================

def test_calibration_brings_a_biased_method_to_the_nominal_rate():
    """
    The repair, measured. In universes where Lambda-CDM holds exactly, a
    free-order Chebyshev series rejects the Om null *nominally* in most of
    them. Calibrated against mocks of a flat Lambda-CDM fitted to each
    universe's own data, it rejects about as often as a 5% test should, and
    its p-values are uniform.
    """

    nominal, calibrated = [], []

    for outer in range(15):

        fit = Cosmography("y").fit(
            mock_chronometers(700 + outer), grid=GRID, n_draws=500, seed=outer
        )

        result = calibrate(
            om, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=60, seed=outer
        )

        nominal.append(result.nominal[2])
        calibrated.append(result.pte)

    nominal, calibrated = np.array(nominal), np.array(calibrated)

    assert np.mean(nominal < 0.05) > 0.6

    assert np.mean(calibrated < 0.05) <= 0.2

    assert 0.3 < calibrated.mean() < 0.7


def test_calibration_still_finds_a_deviation_that_is_really_there():
    """
    A calibration that only ever said "consistent" would pass the test above.

    ``w = -0.6`` observed with chronometer errors a tenth of the real ones is a
    non-centrality of 44 away from the best-fitting Lambda-CDM -- a deviation
    that is unambiguously there. The calibrated test rejects it and, with no
    mock as far out as the data, says so as a bound.
    """

    data = mock_chronometers(11, w=-0.6, error_scale=0.1)

    fit = Cosmography("y", order=3).fit(data, grid=GRID, n_draws=500, seed=1)

    result = calibrate(
        om, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=100, seed=2
    )

    assert not result.consistent

    assert result.at_floor
    assert "p < " in result.summary()


def test_calibration_says_when_the_data_cannot_answer():
    """
    The same universe at the real chronometer errors is a non-centrality of
    0.44 from the best-fitting Lambda-CDM: no test can see it. The nominal test
    reports several sigma anyway -- the method's bias, read as a detection --
    and the calibrated one reports nothing.
    """

    data = mock_chronometers(11, w=-0.6)

    fit = Cosmography("y", order=3).fit(data, grid=GRID, n_draws=500, seed=1)

    result = calibrate(
        om, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=100, seed=2
    )

    assert result.nominal[3] > 3.0

    assert result.consistent


def test_the_ordering_ranks_data_and_mocks_on_equal_terms():
    """
    Without any fitting at all: when the data are just one more draw from the
    distribution the mocks come from, the leave-one-out ranking gives a uniform
    p-value. Scoring the mocks against a mean and covariance they helped
    estimate would not, and every dataset would look more extreme than it is.
    """

    from CosmoRecon.validation.calibration import _distances

    rng = np.random.default_rng(5)

    covariance = np.diag([4.0, 1.0, 0.25, 0.01, 1e-4]) + 0.1

    factor = np.linalg.cholesky(covariance)

    p_values = []

    for _ in range(600):

        draws = rng.normal(size=(61, 5)) @ factor.T + 3.0

        distance, mocks = _distances(draws[1:], draws[0])

        p_values.append((1 + np.sum(mocks >= distance)) / (mocks.size + 1))

    p_values = np.array(p_values)

    assert np.mean(p_values < 0.05) == pytest.approx(0.05, abs=0.025)
    assert p_values.mean() == pytest.approx(0.5, abs=0.04)


def test_an_ensemble_is_calibrated_member_by_member_and_as_a_mixture():
    """
    On DESI DR2 a second-order series reports a violation of FLRW at over five
    sigma, nominally. Calibrated against FLRW universes fitted to the same
    twelve numbers, it reports nothing.
    """

    bao = desi_dr2_bao().select("DM_over_rs", "DH_over_rs")

    ensemble = MethodEnsemble([Cosmography("y", order=3), Cosmography("y", order=2)])

    fit = ensemble.fit(bao, grid=np.linspace(0.6, 2.25, 15), n_draws=1000, seed=3)

    result = calibrate(
        lambda s: Curvature().statistic(s["DM_over_rs"], s["DH_over_rs"]),
        fit,
        null=LambdaCDM(curved=True),
        marginalise_constant=True,
        n_mocks=40,
        seed=4,
        name="Ok(z)",
    )

    assert set(result.per_method) == set(ensemble.names)
    assert result.marginalised.n_mocks + result.marginalised.n_failed == 40
    assert "omega_k" in result.summary()

    second_order = result.per_method[ensemble.names[1]]

    assert second_order.nominal[3] > 5.0
    assert second_order.consistent


# ============================================================
# Refusals
# ============================================================

def test_a_single_statistic_that_keeps_failing_is_abandoned_loudly():

    fit = Cosmography("y").fit(chronometers(), grid=GRID, n_draws=500, seed=1)

    calls = {"n": 0}

    def fragile(s):

        calls["n"] += 1

        if calls["n"] > 1:
            raise DataError("this analysis only works on the real data")

        return om(s)

    with pytest.raises(CosmoReconError, match="mock analyses failed"):
        calibrate(fragile, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=40, seed=1)

    with pytest.raises(ValueError, match="cannot calibrate"):
        calibrate(om, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=10)


def test_one_member_that_keeps_failing_does_not_sink_the_others():
    """
    In an ensemble a statistic undefined for one member on too many null
    universes -- an opacity slope from a posterior that reaches zero, say --
    leaves that member uncalibrated, says why, and calibrates the rest.
    """

    ensemble = MethodEnsemble([Cosmography("y", order=3), Cosmography("y", order=2)])

    fit = ensemble.fit(chronometers(), grid=GRID, n_draws=500, seed=1)

    calls: dict[str, int] = {}

    def statistic(s):

        method = s["H"].provenance.method

        calls[method] = calls.get(method, 0) + 1

        # Defined on the data, undefined for this one member on every mock.
        if "order 2" in method and calls[method] > 1:
            raise DataError("undefined for this member on mocks")

        return om(s)

    result = calibrate(
        statistic, fit, null=LambdaCDM(), marginalise_constant=True, n_mocks=40, seed=1
    )

    failing = result.per_method[ensemble.names[1]]

    assert failing.abandoned is not None
    assert np.isnan(failing.pte)
    assert "not calibrated" in result.summary()

    with pytest.raises(CosmoReconError, match="no calibrated p-value"):
        _ = failing.consistent

    assert result.per_method[ensemble.names[0]].abandoned is None
    assert result.marginalised.abandoned is None
