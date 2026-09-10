"""
Cosmography: convergence, composition, and the claim about bases.

Three things are checked here that the cosmographic literature mostly does
not check at all.

**That the series converges where it is fitted.** A ``z``-series fitted past
``z = 1`` is outside its radius of convergence, and the coefficients read off
it are not the function's. That has to raise.

**That the chain rule is right at high order.** Cosmography is about second,
third and fourth derivatives, taken with respect to ``z`` while the series is
in ``y(z)``. Faa di Bruno through the Bell polynomials is easy to get subtly
wrong, so it is checked against finite differences of the same draws.

**That Chebyshev and Taylor are the same model.** The module docstring claims
they span the same functions and differ only through the prior. If that is
true, forcing the prior flat has to make them coincide -- so it is tested
rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon.core.errors import ConvergenceError, DataError

from CosmoRecon.reconstructors.cosmography import Cosmography
from CosmoRecon.reconstructors.series import (
    LogRedshift,
    Redshift,
    YRedshift,
    bell_polynomials,
    get_variable,
)

from tests.toy import chronometers, lcdm_H


GRID = np.linspace(0.2, 1.9, 25)


@pytest.fixture(scope="module")
def data():
    return chronometers(lcdm_H, n=32, z_max=2.0)


@pytest.fixture(scope="module")
def low_z_data():
    """Data entirely inside the z-series radius of convergence."""

    return chronometers(lcdm_H, n=25, z_max=0.9, seed=3)


# ============================================================
# Recovery
# ============================================================

def test_the_band_contains_the_truth(data):

    H = Cosmography().fit(data, grid=GRID, n_draws=3000, seed=1)["H"]

    lo, hi = H.interval(0.68)

    truth = lcdm_H(GRID)

    assert ((truth >= lo) & (truth <= hi)).mean() > 0.6


@pytest.mark.parametrize("variable", ["y", "log"])
def test_every_convergent_variable_recovers_the_truth(data, variable):

    H = Cosmography(variable).fit(data, grid=GRID, n_draws=2000, seed=1)["H"]

    at_one = H.at(1.0)

    assert abs(float(at_one.mean()[0]) - lcdm_H(1.0)) < 3 * float(at_one.std()[0])


# ============================================================
# Convergence
# ============================================================

def test_a_z_series_past_its_radius_is_refused(data):
    """
    The check the cosmographic literature is missing. Data reaching z = 2 are
    outside |z| < 1, so a Taylor series in z fitted to them does not have the
    function's coefficients -- and no width of error bar repairs that.
    """

    with pytest.raises(ConvergenceError, match="outside the range"):
        Cosmography("z").fit(data, n_draws=100, seed=1)


def test_a_z_series_inside_its_radius_is_allowed(low_z_data):

    H = Cosmography("z").fit(low_z_data, grid=np.linspace(0.1, 0.8, 15),
                             n_draws=800, seed=1)["H"]

    at_half = H.at(0.5)

    assert abs(float(at_half.mean()[0]) - lcdm_H(0.5)) < 3 * float(at_half.std()[0])


def test_the_default_variable_converges_where_the_data_are(data):
    """``y`` exists precisely so that this works where ``z`` does not."""

    Cosmography("y").fit(data, n_draws=100, seed=1)

    assert YRedshift().convergence_range()[1] > 1e3
    assert Redshift().convergence_range()[1] == 1.0


def test_strict_can_be_turned_off_for_reproduction(data):
    """
    Reproducing someone else's analysis is a real need, so the guard can be
    lowered to a warning -- deliberately, and never by default.
    """

    with pytest.warns(RuntimeWarning, match="outside the range"):
        Cosmography("z", strict=False).fit(data, n_draws=100, seed=1)


def test_evaluating_past_the_radius_is_refused(low_z_data):

    H = Cosmography("z").fit(low_z_data, n_draws=200, seed=1)["H"]

    with pytest.raises(ConvergenceError, match="converges for"):
        _ = H.at(3.0).draws


def test_an_order_the_data_cannot_support_is_refused():

    tiny = chronometers(lcdm_H, n=5, seed=2)

    with pytest.raises(DataError, match="prior wearing a polynomial"):
        Cosmography(order=[8, 9]).fit(tiny, n_draws=50, seed=1)


# ============================================================
# The claim about bases
# ============================================================

@pytest.mark.parametrize("order", [3, 5])
def test_chebyshev_and_monomial_are_the_same_model(data, order):
    """
    The module docstring claims a Chebyshev series and a monomial series of
    the same degree in the same variable span the same functions, and differ
    here only through the prior. Force the prior flat and they must coincide.

    The tolerance is the Monte-Carlo floor, loosened at order 5 because the
    monomial design matrix is genuinely ill-conditioned there -- which is the
    other half of the claim, and the reason Chebyshev is the default.
    """

    n_draws = 6000

    results = {}

    for family in ("chebyshev", "monomial"):

        results[family] = Cosmography(
            order=order,
            family=family,
            n_scale=1,
            scale_range=(1e6, 1e6),      # effectively flat
        ).fit(data, grid=GRID, n_draws=n_draws, seed=2)["H"]

    mean_gap = np.max(
        np.abs(results["chebyshev"].mean() - results["monomial"].mean())
        / results["chebyshev"].std()
    )

    assert mean_gap < 8.0 / np.sqrt(n_draws), f"differ by {mean_gap:.4f} sd"


def test_the_prior_is_what_makes_the_two_bases_differ(data):
    """The converse: with the default prior they are not the same fit."""

    widths = {}

    for family in ("chebyshev", "monomial"):

        widths[family] = Cosmography(order=5, family=family).fit(
            data, grid=GRID, n_draws=4000, seed=2
        )["H"].std()

    ratio = widths["chebyshev"] / widths["monomial"]

    assert np.max(np.abs(ratio - 1.0)) > 0.02


# ============================================================
# The chain rule
# ============================================================

@pytest.mark.parametrize("variable", ["y", "log", "z"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_faa_di_bruno_matches_finite_differences(variable, order):
    """
    The composition ``d^n/dz^n f(x(z))`` against differencing the same draws.

    Two independent routes to one number: the analytic path goes through the
    Bell polynomials, the numerical one never touches them. This is the test
    that would catch a wrong binomial coefficient in the recursion, which is
    otherwise invisible -- a wrong third derivative still looks like a curve.
    """

    mock = chronometers(lcdm_H, n=30, z_max=0.9, seed=5)

    grid = np.linspace(0.25, 0.75, 40)

    H = Cosmography(variable, order=5).fit(
        mock, grid=grid, n_draws=400, seed=1
    )["H"]

    analytic = H.d(order).mean()

    numeric = H.d(order, numerical=True).mean()

    inner = slice(order + 2, -(order + 2))

    scale = np.max(np.abs(analytic[inner]))

    assert np.allclose(
        analytic[inner], numeric[inner], rtol=0.0, atol=2e-2 * scale
    )


def test_bell_polynomials_match_the_chain_rule_written_out():
    """
    ``B[1][1] = x'``, ``B[2][1] = x''``, ``B[2][2] = x'^2``,
    ``B[3][1] = x'''``, ``B[3][2] = 3 x' x''``, ``B[3][3] = x'^3``.

    The low orders of Faa di Bruno, checked against the form anyone would
    derive by hand -- so that the recursion is anchored to something known
    before it is trusted at orders nobody writes out.
    """

    x1 = np.array([2.0])
    x2 = np.array([-3.0])
    x3 = np.array([5.0])

    B = bell_polynomials([x1, x2, x3], 3)

    assert B[1][1] == pytest.approx(x1)
    assert B[2][1] == pytest.approx(x2)
    assert B[2][2] == pytest.approx(x1**2)
    assert B[3][1] == pytest.approx(x3)
    assert B[3][2] == pytest.approx(3 * x1 * x2)
    assert B[3][3] == pytest.approx(x1**3)


@pytest.mark.parametrize("variable", [Redshift(), YRedshift(), LogRedshift()])
def test_variable_derivatives_match_finite_differences(variable):
    """Each variable's closed-form derivatives, checked numerically."""

    z = np.linspace(0.3, 1.5, 9)

    h = 1e-4

    for order, analytic in enumerate(variable.derivatives(z, 3), start=1):

        def evaluate(zz, k=order - 1):
            return variable.derivatives(zz, max(k, 1))[k - 1] if k else variable(zz)

        numeric = (evaluate(z + h) - evaluate(z - h)) / (2 * h)

        assert np.allclose(analytic, numeric, rtol=1e-5)


# ============================================================
# Order marginalisation
# ============================================================

def test_marginalising_the_order_is_wider_than_pinning_it(data):
    """
    The same argument the Gaussian process makes about its kernel, in the
    place where the evidence happens to be available in closed form: picking
    an order and quoting an interval conditioned on it understates what the
    data leave open.
    """

    marginalised = Cosmography().fit(data, grid=GRID, n_draws=4000, seed=3)["H"]

    reference = Cosmography()
    reference.fit(data, grid=GRID, n_draws=10, seed=3)

    best_order = reference._cells[int(np.argmax(reference._log_weights))][0]

    pinned = Cosmography(order=best_order).fit(
        data, grid=GRID, n_draws=4000, seed=3
    )["H"]

    ratio = marginalised.std() / pinned.std()

    assert np.median(ratio) > 1.01, (
        f"marginalising the order changed nothing (median ratio "
        f"{np.median(ratio):.3f}); the evidence is presumably concentrated on "
        f"order {best_order}, which would make this test uninformative"
    )


def test_evidence_is_finite_and_prefers_a_sensible_order(data):

    fit = Cosmography()

    fit.fit(data, n_draws=100, seed=1)

    assert np.isfinite(fit.log_evidence)

    weights = np.exp(fit._log_weights - fit._log_weights.max())

    orders = np.array([cell[0] for cell in fit._cells])

    favoured = orders[int(np.argmax(weights))]

    # LCDM's H(z) over this range is smooth and gently curved; a very high
    # order would mean the evidence is not doing its job.
    assert 1 <= favoured <= 5


# ============================================================
# Pade
# ============================================================

@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (2, 2)])
def test_pade_recovers_the_truth_and_reports_its_acceptance(data, m, n):

    fit = Cosmography(pade=(m, n))

    H = fit.fit(data, grid=GRID, n_draws=1500, seed=1)["H"]

    at_one = H.at(1.0)

    assert abs(float(at_one.mean()[0]) - lcdm_H(1.0)) < 3 * float(at_one.std()[0])

    # Every accepted draw is pole-free in the fitted range by construction,
    # and the fraction that had to be thrown away is reported rather than
    # hidden inside a wide quantile.
    assert 0.2 <= fit.acceptance <= 1.0


def test_a_posterior_of_mostly_singular_approximants_is_refused(data, monkeypatch):
    """
    When most of the Pade posterior has a pole inside the observable range,
    the fit is not a reconstruction with large errors -- it is a posterior
    over singular functions, and it is refused rather than widened.

    The threshold is raised here rather than reaching for a denominator order
    that fails naturally, because on real data the acceptance rate is *not*
    monotonic in the denominator degree: measured on this mock, Pade[2/1]
    accepts 88%, Pade[1/3] accepts 32% and Pade[2/4] accepts 93%. Picking an
    order and asserting it fails would be asserting an accident. What is
    worth testing is the logic -- that a low rate is caught rather than
    quietly delivered.
    """

    from CosmoRecon.reconstructors import cosmography as module

    monkeypatch.setattr(module, "_MIN_ACCEPTANCE", 0.999)

    with pytest.raises(ConvergenceError, match="pole"):
        Cosmography(pade=(2, 2)).fit(data, n_draws=400, seed=1)


def test_acceptance_is_reported_for_every_pade_fit(data):
    """
    The rate is part of the result, not a diagnostic printed once. A reader
    of a Pade reconstruction should be able to see how much of the posterior
    had to be thrown away to produce it.
    """

    rates = {}

    for m, n in [(2, 1), (1, 3), (2, 4)]:

        fit = Cosmography(pade=(m, n))

        fit.fit(data, n_draws=400, seed=1)

        rates[(m, n)] = fit.acceptance

    assert all(0.0 < rate <= 1.0 for rate in rates.values())

    # And it genuinely varies with the shape of the approximant, which is why
    # it is worth reporting at all.
    assert max(rates.values()) - min(rates.values()) > 0.1


def test_pade_needs_a_denominator():

    with pytest.raises(ValueError, match="n >= 1"):
        Cosmography(pade=(3, 0))


def test_pade_differentiates(data):

    H = Cosmography(pade=(2, 1)).fit(data, grid=GRID, n_draws=800, seed=1)["H"]

    analytic = H.d(1).mean()

    numeric = H.d(1, numerical=True).mean()

    inner = slice(3, -3)

    assert np.allclose(
        analytic[inner], numeric[inner],
        rtol=0.0, atol=2e-2 * np.max(np.abs(analytic[inner])),
    )


# ============================================================
# Contract
# ============================================================

def test_a_draw_is_the_same_function_on_every_grid(data):

    H = Cosmography().fit(data, grid=GRID, n_draws=500, seed=1)["H"]

    probe = np.array([0.4, 0.9, 1.5])

    first = H.at(probe).draws

    second = H.at(np.concatenate([[0.3], probe, [1.7]])).draws[:, 1:4]

    assert np.allclose(first, second, rtol=1e-12, atol=0.0)


def test_the_fit_records_what_it_did(data):

    fit = Cosmography(pade=(2, 1)).fit(data, n_draws=200, seed=9)

    assert fit.provenance.method == "Pade[2/1](y = z/(1+z))"
    assert fit.provenance.seed == 9
    assert fit.provenance.hyperparameters["family"] == "chebyshev"


@pytest.mark.parametrize("name, cls", [
    ("y", YRedshift), ("z", Redshift), ("log", LogRedshift),
])
def test_variables_resolve_by_name(name, cls):

    assert isinstance(get_variable(name), cls)


def test_unknown_variables_are_refused():

    with pytest.raises(ValueError, match="Unknown expansion variable"):
        get_variable("scale_factor")
