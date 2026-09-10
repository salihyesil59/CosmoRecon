"""
The Gaussian process, checked against the thing it approximates.

Two claims have to hold before any of this is worth using.

**The sample paths are the posterior.** They are carried in a finite spectral
basis and drawn by Matheron's rule, and neither of those is obviously exact.
:meth:`GaussianProcess.exact_posterior` computes the same posterior by
textbook conditioning, and the two are required to agree to Monte-Carlo
precision.

**The intervals mean what they say.** A reconstruction that reports 68% and
covers 45% manufactures significance, which is the failure this library exists
to expose. So the band is scored against a known truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon.core.errors import (
    DataError,
    DerivativeUnavailableError,
    EvidenceUnavailableError,
)
from CosmoRecon.core.grid import ExtrapolationWarning

from CosmoRecon.reconstructors.gp import GaussianProcess
from CosmoRecon.reconstructors.kernels import Cauchy, Matern, SquaredExponential

from tests.toy import MockData, chronometers, lcdm_H


#: The default grid, which supports one derivative throughout. Restricting
#: further -- ``nu >= 2.5`` -- is how a *second* derivative is bought, with a
#: stated prior rather than by convention; see the error raised by the free
#: kernel in ``test_free_smoothness_refuses_a_derivative_it_cannot_support``.
SMOOTH_NU = [1.5, 2.0, 2.5, 3.5, 5.0, 7.5]


@pytest.fixture(scope="module")
def data():
    return chronometers(lcdm_H, n=32)


@pytest.fixture(scope="module")
def fitted(data):

    gp = GaussianProcess(kernel="matern")

    return gp, gp.fit(data, n_draws=3000, seed=7)


# ============================================================
# Against the exact posterior
# ============================================================

@pytest.mark.parametrize("kernel", [
    Matern(nu=2.5),
    SquaredExponential(),
    Cauchy(),
])
def test_sample_paths_reproduce_the_exact_posterior(data, kernel):
    """
    The two-ways check. One hyperparameter cell, so that what is being
    compared is the *representation* and not the mixture on top of it.

    Tolerances are set from the Monte-Carlo floor: with ``n`` draws the mean
    of a Gaussian is uncertain by ``sd / sqrt(n)`` and its standard deviation
    by ``sd / sqrt(2n)``, and these are maxima over a grid of correlated
    points, so a few times the floor is the right bar. Anything systematically
    above it would be the basis, not the sampling.
    """

    n_draws = 8000

    gp = GaussianProcess(
        kernel=kernel,
        n_length=1,
        n_amplitude=1,
        length_range=(0.6, 0.6),
        amplitude_range=(60.0, 60.0),
    )

    z_star = np.linspace(0.2, 1.8, 30)

    drawn = gp.fit(data, grid=z_star, n_draws=n_draws, seed=3)["H"]

    mean, cov = gp.exact_posterior(data, z_star)

    sd = np.sqrt(np.diag(cov))

    mean_error = np.max(np.abs(drawn.mean() - mean) / sd)

    sd_error = np.max(np.abs(drawn.std() / sd - 1.0))

    assert mean_error < 5.0 / np.sqrt(n_draws), f"mean off by {mean_error:.4f} sd"

    assert sd_error < 8.0 / np.sqrt(2 * n_draws), f"width off by {100 * sd_error:.2f}%"


def test_the_basis_reports_the_accuracy_it_achieved(fitted):
    """
    The quadrature refines until it meets its tolerance, and the tolerance is
    the one the reconstructor was configured with -- not whatever it happened
    to reach.
    """

    gp, fit = fitted

    assert fit["H"]._predictor.quadrature_error <= gp.tol


# ============================================================
# Recovery
# ============================================================

def test_the_band_contains_the_truth(fitted):
    """
    Injection and recovery. The truth is an exact LCDM expansion history; the
    68% band has to contain it across the range the data support.
    """

    _, fit = fitted

    H = fit["H"]

    lo, hi = H.interval(0.68)

    covered = ((lcdm_H(H.z) >= lo) & (lcdm_H(H.z) <= hi)).mean()

    assert covered > 0.6, f"68% band covers the truth at only {100 * covered:.0f}%"


def test_pointwise_coverage_is_honest_over_many_realisations():
    """
    The claim an error bar makes, tested as a frequency.

    Each realisation is a fresh noise draw around the same truth; over enough
    of them the 68% interval has to contain the truth about 68% of the time.
    A method reporting 68% and covering 45% is manufacturing significance --
    exactly what this library is built to catch -- so it would be poor form to
    ship one.

    The bar is deliberately loose at both ends. Too low is dishonest; much too
    high is a different failure, an interval so wide it says nothing.
    """

    z_test = np.linspace(0.3, 1.7, 8)

    truth = lcdm_H(z_test)

    hits = 0
    total = 0

    for seed in range(12):

        mock = chronometers(lcdm_H, n=32, seed=100 + seed)

        gp = GaussianProcess(kernel=Matern(nu=SMOOTH_NU), n_length=12, n_amplitude=8)

        H = gp.fit(mock, grid=z_test, n_draws=1200, seed=seed)["H"]

        lo, hi = H.interval(0.68)

        hits += int(((truth >= lo) & (truth <= hi)).sum())

        total += z_test.size

    coverage = hits / total

    assert 0.55 < coverage < 0.95, f"68% intervals cover {100 * coverage:.0f}%"


# ============================================================
# The determinism clause
# ============================================================

def test_a_draw_is_the_same_function_on_every_grid(fitted):
    """
    The contract the whole library rests on. Draw ``k`` evaluated on two
    grids has to be one function sampled twice, not two functions -- otherwise
    ``at()`` silently resamples and nothing downstream can be combined.

    The comparison is to machine precision rather than bit-exact, and the
    distinction is worth stating. Evaluating on a wider grid changes the shape
    of the matrix product that sums several hundred basis terms, so BLAS
    blocks it differently and the last bits move -- by a few tens of units in
    the last place, a relative 1e-15. That is the arithmetic rounding
    differently, not the sampler drawing again, and the two failures look
    nothing alike: a resampled path would differ in the third significant
    figure, not the fifteenth.
    """

    _, fit = fitted

    H = fit["H"]

    z_probe = np.array([0.31, 0.77, 1.42])

    first = H.at(z_probe).draws

    second = H.at(np.concatenate([[0.2], z_probe, [1.7]])).draws[:, 1:4]

    assert np.allclose(first, second, rtol=1e-12, atol=0.0)


def test_a_refit_with_the_same_seed_is_the_same_fit(data):

    a = GaussianProcess(kernel=Matern(nu=2.5)).fit(data, n_draws=200, seed=5)["H"]
    b = GaussianProcess(kernel=Matern(nu=2.5)).fit(data, n_draws=200, seed=5)["H"]

    assert np.array_equal(a.draws, b.draws)


# ============================================================
# Smoothness
# ============================================================

def test_free_smoothness_refuses_a_derivative_it_cannot_support(fitted):
    """
    The default grid runs from Matern-3/2 upwards, so a first derivative
    exists across the whole prior and ``w(z)`` is available without asking for
    anything. A *second* is not: Matern-3/2 has none, the posterior on ``nu``
    reaches it, and the cosmographic quantities built from ``H''`` therefore
    do not exist on part of the posterior they would be reported for.

    The library says so, and says how much of the posterior it is speaking
    for. No other GP reconstruction code makes this check, which is why jerk
    parameters reconstructed with a Matern-3/2 kernel are published routinely.
    """

    _, fit = fitted

    assert fit["H"].d(1).mean().shape == fit["H"].mean().shape

    with pytest.raises(DerivativeUnavailableError, match="of this posterior"):
        _ = fit["H"].d(2).draws


def test_restricting_the_prior_buys_the_derivative(data):
    """
    ``nu >= 3/2`` throughout gives a first derivative and stops there; the
    second has to be bought separately, by excluding the processes that do not
    have one.
    """

    gp = GaussianProcess(kernel=Matern(nu=SMOOTH_NU))

    H = gp.fit(data, n_draws=800, seed=4)["H"]

    dH = H.d(1)

    assert dH.draws.shape == (H.n_draws, H.n_z)

    # H(z) is increasing over this range, so its derivative should be too.
    assert np.all(dH.mean() > 0.0)

    # But two derivatives were not bought, and are not given.
    with pytest.raises(DerivativeUnavailableError):
        _ = H.d(2).draws


def test_analytic_kernels_differentiate_to_any_order(data):

    H = GaussianProcess(kernel=SquaredExponential()).fit(
        data, n_draws=400, seed=4
    )["H"]

    assert H._predictor.max_derivative is None

    assert np.isfinite(H.d(3).mean()).all()


def test_the_analytic_derivative_matches_finite_differences(data):
    """
    Third cross-check: the closed-form phase shift that produces a derivative
    in this basis, against differencing the same draws.
    """

    gp = GaussianProcess(kernel=Matern(nu=SMOOTH_NU))

    H = gp.fit(data, n_draws=600, seed=4)["H"]

    inner = slice(3, -3)

    assert np.allclose(
        H.d(1).mean()[inner],
        H.d(1, numerical=True).mean()[inner],
        rtol=5e-3,
    )


# ============================================================
# The thesis
# ============================================================

def test_marginalising_hyperparameters_is_wider_than_optimising_them(data):
    """
    The reason this module exists.

    The standard recipe maximises the marginal likelihood over amplitude and
    length scale and reconstructs at those values. Doing that discards a real
    uncertainty, and the interval it reports is correspondingly too narrow.
    Here the same data are fitted both ways and the difference is measured.
    """

    gp = GaussianProcess(kernel=Matern(nu=2.5))

    grid = np.linspace(0.2, 1.9, 25)

    marginalised = gp.fit(data, grid=grid, n_draws=3000, seed=6)["H"]

    # The cell the standard recipe would have stopped at.
    best = gp._cells[int(np.argmax(gp._log_weights))]

    pinned = GaussianProcess(
        kernel=Matern(nu=2.5),
        n_length=1,
        n_amplitude=1,
        length_range=(best["length_scale"],) * 2,
        amplitude_range=(best["amplitude"],) * 2,
    ).fit(data, grid=grid, n_draws=3000, seed=6)["H"]

    ratio = marginalised.std() / pinned.std()

    assert np.median(ratio) > 1.02, (
        "marginalising the hyperparameters produced no more uncertainty than "
        f"fixing them (median width ratio {np.median(ratio):.3f}) -- either "
        "the posterior is degenerate or the marginalisation is not happening"
    )


def test_evidence_is_finite_and_comparable_across_kernels(data):

    values = {}

    for kernel in (Matern(nu=2.5), SquaredExponential(), Cauchy()):

        gp = GaussianProcess(kernel=kernel)

        gp.fit(data, n_draws=100, seed=8)

        values[gp.describe()] = gp.log_evidence

    assert all(np.isfinite(v) for v in values.values())

    # Different kernels are different models of the same data, so they should
    # not all agree to the last decimal.
    assert np.ptp(list(values.values())) > 0.01


def test_evidence_before_a_fit_is_an_error():

    with pytest.raises(RuntimeError, match="Nothing has been fitted"):
        _ = GaussianProcess().log_evidence


def test_a_method_without_an_evidence_says_so():
    """
    The base class refuses rather than returning ``nan``, so an
    evidence-weighted ensemble cannot silently shrink to the members that
    happen to define one.
    """

    from CosmoRecon.reconstructors.base import Reconstructor

    class Nameless(Reconstructor):
        def describe(self):
            return "toy"

        def hyperparameters(self):
            return {}

        def _fit(self, data, **kwargs):
            raise NotImplementedError

    with pytest.raises(EvidenceUnavailableError, match="evidence-weighted"):
        _ = Nameless().log_evidence


# ============================================================
# Provenance, support and bad input
# ============================================================

def test_the_fit_records_what_it_did(fitted):

    _, fit = fitted

    assert fit.provenance.method == "GP(Matern, nu free)"
    assert fit.provenance.data == ("H(32)",)
    assert fit.provenance.seed == 7
    assert fit.provenance.hyperparameters["kernel"] == "Matern"


def test_support_comes_from_the_data_not_the_grid(data, fitted):

    _, fit = fitted

    assert fit.support == pytest.approx((data.z.min(), data.z.max()))


def test_extrapolation_is_flagged(fitted):

    _, fit = fitted

    with pytest.warns(ExtrapolationWarning, match="outside the range"):
        _ = fit["H"].at(np.array([5.0])).draws


@pytest.mark.parametrize("bad, match", [
    (MockData([0.1, 0.2], [70.0, 80.0], [1.0, 1.0]), "not enough"),
    (MockData([0.1, 0.2, 0.3], [70.0, 80.0, 90.0], [1.0, 1.0]), "uncertainties for"),
])
def test_malformed_data_is_refused(bad, match):

    with pytest.raises(DataError, match=match):
        GaussianProcess().fit(bad)


def test_datasets_measuring_different_things_cannot_be_merged():

    a = chronometers(lcdm_H, n=10, seed=1)
    b = chronometers(lcdm_H, n=10, seed=2)
    b.observable = "fsigma8"

    with pytest.raises(DataError, match="different quantities"):
        GaussianProcess().fit([a, b])
