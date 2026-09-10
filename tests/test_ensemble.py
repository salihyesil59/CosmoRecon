"""
The ensemble layer: pooling methods, and reporting how much they disagree.

Two things have to hold.

**The mixture and the budget have to be the same object seen twice.** The
variance budget is computed analytically from each member's mean and variance;
the method-marginalised posterior is built by pooling draws. Nothing forces
them to agree, so their agreement is a real check on both.

**A significance has to be attributable.** The same statistic, evaluated under
each member alone and under the mixture, is what turns "the method matters"
from an assertion into a number.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoRecon import (
    Cosmography,
    ExtrapolationWarning,
    GaussianProcess,
    MethodEnsemble,
)
from CosmoRecon.core.errors import (
    ConvergenceError,
    DerivativeUnavailableError,
    EvidenceUnavailableError,
)
from CosmoRecon.reconstructors.base import Reconstructor

from tests.toy import chronometers, lcdm_H


GRID = np.linspace(0.15, 1.90, 25)


def cpl_H(z, H0=70.0, Om=0.3, w0=-0.7, wa=-1.2):
    """A universe that is genuinely not LCDM, to be detected or missed."""

    z = np.asarray(z, dtype=float)

    de = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))

    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * de)


def five_methods() -> MethodEnsemble:

    return MethodEnsemble([
        GaussianProcess(kernel="matern"),
        GaussianProcess(kernel="squared_exponential"),
        Cosmography("y"),
        Cosmography("log"),
        Cosmography(pade=(2, 1)),
    ])


@pytest.fixture(scope="module")
def lcdm_fit():

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        return five_methods().fit(
            chronometers(lcdm_H, n=32, z_max=2.0, seed=11),
            grid=GRID,
            n_draws=4000,
            seed=4,
        )


@pytest.fixture(scope="module")
def cpl_fit():
    """
    A CPL universe measured at percent-level precision -- roughly where BAO
    already is, and where the statistical errors are small enough that the
    disagreement between methods is the dominant term.
    """

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        return five_methods().fit(
            chronometers(cpl_H, n=40, z_max=2.0, sigma_frac=0.02, seed=17),
            grid=GRID,
            n_draws=3000,
            seed=4,
        )


# ============================================================
# The mixture is the budget
# ============================================================

def test_the_pooled_draws_reproduce_the_variance_budget(lcdm_fit):
    """
    The law of total variance, computed from per-method moments, against the
    variance of the pooled draws themselves.

    These are two independent routes: one sums weighted variances and
    between-method scatter analytically, the other takes ``var()`` of a pile
    of realisations. If the pooling weights, the draw counts or the
    within-member sampling were wrong, they would part company.
    """

    budget = lcdm_fit.budget()

    ratio = lcdm_fit.marginalised.var() / budget.total

    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)
    assert np.all(np.abs(ratio - 1.0) < 0.10)


def test_the_pooled_mean_is_the_weighted_member_mean(lcdm_fit):

    budget = lcdm_fit.budget()

    weighted = sum(
        weight * budget.means[name] for name, weight in budget.weights.items()
    )

    gap = np.max(
        np.abs(lcdm_fit.marginalised.mean() - weighted) / lcdm_fit.marginalised.std()
    )

    assert gap < 0.1


def test_the_marginalised_posterior_is_a_full_reconstruction(lcdm_fit):
    """
    Not a summary. It regrids and it differentiates, because each pooled draw
    still knows which member's function it is -- so a null test built on the
    mixture has every guarantee a null test built on one method has.
    """

    mixture = lcdm_fit.marginalised

    assert mixture.resamplable

    probe = np.array([0.4, 1.1])

    first = mixture.at(probe).draws

    second = mixture.at(np.concatenate([[0.3], probe, [1.5]])).draws[:, 1:3]

    assert np.allclose(first, second, rtol=1e-12, atol=0.0)


def test_the_mixture_is_only_as_differentiable_as_its_roughest_member(lcdm_fit):
    """
    The free-nu Gaussian process in this ensemble reaches down to Matern-3/2,
    which has a first derivative and no second -- so the mixture has a first
    and no second, and the refusal names the member responsible rather than
    leaving the user to guess.

    The polynomial members are analytic and would happily hand back a fourth
    derivative. That they do not is the point: a mixture is as rough as its
    roughest constituent, and pretending otherwise would report a jerk
    parameter that four fifths of the posterior does not have.
    """

    assert lcdm_fit.marginalised.d(1).mean().shape == (GRID.size,)

    with pytest.raises(DerivativeUnavailableError, match="Member 'GP"):
        _ = lcdm_fit.marginalised.d(2).draws


# ============================================================
# Significance, before and after
# ============================================================

def test_the_same_data_support_wildly_different_significances(cpl_fit):
    """
    The library's thesis as a single number.

    Forty measurements of a CPL universe at 2% precision. Every method agrees
    on ``H(1.0)`` to well inside two units. Yet the significance of the
    deviation from a LCDM reference ranges over more than an order of
    magnitude across the five, because the methods distribute their
    (small) uncertainties across redshift differently.

    Note what this is and is not. The injected truth genuinely does deviate,
    so the methods reporting a large significance are directionally right and
    the flexible ones are under-powered -- this is not a demonstration that
    Pade invents detections. It is a demonstration that an analyst who had
    chosen one method could report a decisive detection and an analyst who
    had chosen another could report nothing, from the same data, with neither
    able to tell from their own analysis which they were.
    """

    reference = lcdm_H(GRID, H0=70.0, Om=0.3)

    comparison = cpl_fit.significance(
        lambda H: H - reference, 0.0, name="H(z) - LCDM"
    )

    sigmas = [value[3] for value in comparison.per_method.values()]

    assert max(sigmas) - min(sigmas) > 10.0, (
        f"methods spanned only {max(sigmas) - min(sigmas):.1f} sigma; the "
        "scenario is no longer illustrative"
    )

    # And the marginalised answer is not larger than the loudest member: a
    # mixture is wider than its parts, so it cannot be more confident.
    assert comparison.marginalised[3] <= max(sigmas) + 1e-9

    assert comparison.deflation > 1.0


def test_every_member_and_the_mixture_are_reported(cpl_fit):

    reference = lcdm_H(GRID, H0=70.0, Om=0.3)

    comparison = cpl_fit.significance(lambda H: H - reference, 0.0)

    assert set(comparison.per_method) == set(cpl_fit.members)

    name, strongest = comparison.best_single

    assert strongest == max(v[3] for v in comparison.per_method.values())
    assert name in cpl_fit.members

    assert "method-marginalised" in comparison.summary()


def test_method_marginalisation_repairs_a_miscalibrated_member():
    """
    The strongest argument for the whole library, measured rather than
    asserted.

    A rigid parametric family cannot contain the truth, and its posterior
    covers only the uncertainty in its own coefficients -- not the error it
    makes by being the wrong shape. So its interval is confidently too narrow.
    Measured over 24 realisations of LCDM chronometers, the nominal 68%
    interval of a Pade[2/1] fit contains the truth about 45% of the time. That
    is not a bug in the fit; the posterior is exactly right for the model. The
    model is wrong, and nothing inside a single-method analysis can see it.

    Pooling across methods sees it, because the between-method scatter is
    precisely the missing term. The marginalised interval comes back to
    roughly its nominal coverage.

    This is the failure mode this library exists to expose, found in one of
    its own reconstructors and repaired by its own ensemble layer.
    """

    z_test = np.linspace(0.3, 1.7, 8)

    truth = lcdm_H(z_test)

    covered = {"-- method-marginalised": 0}

    total = 0

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        for seed in range(24):

            mock = chronometers(lcdm_H, n=32, seed=200 + seed)

            fit = five_methods().fit(
                mock, grid=z_test, n_draws=1200, seed=seed
            )

            lo, hi = fit.marginalised.interval(0.68)

            covered["-- method-marginalised"] += int(
                ((truth >= lo) & (truth <= hi)).sum()
            )

            for name, curve in fit.members.items():

                lo, hi = curve.interval(0.68)

                covered[name] = covered.get(name, 0) + int(
                    ((truth >= lo) & (truth <= hi)).sum()
                )

            total += z_test.size

    rates = {name: count / total for name, count in covered.items()}

    pade = next(name for name in rates if name.startswith("Pade"))

    mixture = rates["-- method-marginalised"]

    # The pathology is real and reproducible.
    assert rates[pade] < 0.60, (
        f"Pade covered {100 * rates[pade]:.0f}%, so the miscalibration this "
        "test is about is not present and the test proves nothing"
    )

    # And pooling repairs it.
    assert mixture > rates[pade] + 0.10
    assert 0.58 < mixture < 0.85, f"mixture covered {100 * mixture:.0f}%"


# ============================================================
# Weighting
# ============================================================

def test_equal_weighting_is_the_default(lcdm_fit):

    weights = lcdm_fit.weights

    assert len(set(np.round(list(weights.values()), 12))) == 1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_evidence_weighting_uses_the_evidence():

    ensemble = MethodEnsemble(
        [Cosmography("y"), Cosmography("log")], weights="evidence"
    )

    fit = ensemble.fit(
        chronometers(lcdm_H, n=32, seed=11), grid=GRID, n_draws=400, seed=1
    )

    weights = fit.weights

    assert sum(weights.values()) == pytest.approx(1.0)

    # Two different models of the same data should not weigh identically.
    assert len(set(np.round(list(weights.values()), 6))) == 2


def test_evidence_weighting_refuses_a_member_without_one():
    """
    Silently averaging over the subset that happens to define an evidence
    would report a spread across fewer methods than were asked for -- which is
    the number the caller came for.
    """

    class Neural(Reconstructor):

        provides_evidence = False

        def describe(self):
            return "toy neural"

        def hyperparameters(self):
            return {}

        def _fit(self, data, **kwargs):
            raise NotImplementedError

    ensemble = MethodEnsemble(
        [Cosmography("y"), Neural()], weights="evidence"
    )

    with pytest.raises(EvidenceUnavailableError, match="do not"):
        ensemble._resolve_weights()


# ============================================================
# Refusals
# ============================================================

def test_a_member_that_cannot_fit_is_not_silently_dropped():
    """
    ``Cosmography("z")`` cannot fit data reaching z = 2, and an ensemble that
    quietly continued with four members would report their spread as though it
    were across five.
    """

    ensemble = MethodEnsemble([Cosmography("y"), Cosmography("z")])

    with pytest.raises(ConvergenceError, match="cannot be silently dropped"):
        ensemble.fit(
            chronometers(lcdm_H, n=32, z_max=2.0, seed=11), n_draws=100, seed=1
        )


def test_one_member_is_not_an_ensemble():

    with pytest.raises(ValueError, match="at least two methods"):
        MethodEnsemble([Cosmography("y")])


def test_members_must_be_distinguishable():
    """
    Two identically configured members would appear as one row in a budget,
    and the second would silently overwrite the first.
    """

    with pytest.raises(ValueError, match="share a name"):
        MethodEnsemble([Cosmography("y"), Cosmography("y")])


def test_an_unknown_weighting_is_refused():

    with pytest.raises(ValueError, match="Unknown weighting"):
        MethodEnsemble([Cosmography("y"), Cosmography("log")], weights="aic")


# ============================================================
# Reproducibility
# ============================================================

def test_the_same_seed_gives_the_same_ensemble():

    data = chronometers(lcdm_H, n=32, seed=11)

    def run():
        return MethodEnsemble([Cosmography("y"), Cosmography("log")]).fit(
            data, grid=GRID, n_draws=300, seed=2
        )

    first, second = run(), run()

    assert np.array_equal(first.marginalised.draws, second.marginalised.draws)

    for name in first.members:
        assert np.array_equal(
            first.members[name].draws, second.members[name].draws
        )
