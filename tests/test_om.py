"""
The Om family, checked against the papers that defined it.

Three kinds of check.

**Exactness.** In a flat LCDM universe ``Om(z)`` is ``Omega_m`` and ``Om3`` is
1, identically -- not approximately, not on average. Every draw individually
has to give those values, because the cancellation is algebraic. A machinery
that summarised before combining would return the right mean and the wrong
width; here both come out at machine precision.

**Against a published figure.** Figure 1 of Shafieloo, Sahni & Starobinsky
(2012) plots ``Om3(0.2, 0.35, z3)`` against ``z3 - z2`` for quintessence,
LCDM and phantom. Reproducing it quantitatively is a check on the whole
implementation against something outside this repository.

**That the correlations matter.** ``Om3`` at different ``z3`` all share
``H(z1)`` and ``H(z2)``. Propagating marginal error bars as though the three
were independent -- which is what a curve-with-error-bars representation
forces -- misstates the width by factors of two to seven, in both directions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoRecon import GaussianProcess, Om, Om3, linear_grid
from CosmoRecon.core.errors import DataError
from CosmoRecon.core.grid import ExtrapolationWarning
from CosmoRecon.data import chronometers as real_chronometers

from tests.toy import FlatCPL, FlatLCDM, reconstruction


def sharp(**kwargs):
    """A toy posterior with negligible parameter width: a single universe."""

    tiny = dict(
        sigma_H0=1e-9, sigma_Om=1e-9, sigma_w0=1e-9, sigma_wa=1e-9, n_draws=200
    )

    if "w0" in kwargs:
        return FlatCPL(**{**tiny, **kwargs})

    return FlatLCDM(**{k: v for k, v in {**tiny, **kwargs}.items()
                       if k not in ("sigma_w0", "sigma_wa")})


# ============================================================
# Exactness
# ============================================================

def test_om_is_omega_m_exactly_in_flat_lcdm():
    """
    Not approximately. The dark-energy term cancels algebraically, so each
    draw gives its own ``Omega_m`` and the reconstructed ``Om(z)`` inherits
    that parameter's posterior unchanged -- same mean, same width, zero slope.
    """

    z = linear_grid(0.05, 1.5, 40)

    predictor = FlatLCDM()

    curve = Om().statistic(reconstruction(predictor, z))

    assert np.allclose(curve.mean(), predictor.Om.mean(), atol=1e-12)
    assert np.allclose(curve.std(), predictor.Om.std(ddof=1), atol=1e-12)

    assert np.ptp(curve.mean()) < 1e-14


def test_om3_is_one_exactly_in_flat_lcdm():
    """
    And with *zero width*, which is the sharper statement. Om3 is a ratio in
    which ``H0``, ``Omega_m`` and the whole shape of the posterior cancel
    within each realisation, so every draw returns 1 and the spread is
    floating-point noise rather than a posterior.
    """

    z = linear_grid(0.6, 1.5, 30)

    curve = Om3(z1=0.2, z2=0.5).statistic(reconstruction(FlatLCDM(), z))

    assert np.allclose(curve.mean(), 1.0, atol=1e-12)
    assert np.all(curve.std() < 1e-12)


def test_a_consistent_universe_is_reported_as_consistent():

    z = linear_grid(0.6, 1.5, 30)

    result = Om3(z1=0.2, z2=0.5).evaluate(reconstruction(FlatLCDM(), z))

    assert result.consistent
    assert result.pte == 1.0


# ============================================================
# Against the published figure
# ============================================================

def test_om3_reproduces_figure_one_of_the_defining_paper():
    """
    Shafieloo, Sahni & Starobinsky (2012), Phys. Rev. D 86, 103527, Fig. 1:
    ``Om3(0.2, 0.35, z3)`` against ``z3 - z2``, with ``Omega_m = 0.27``.

    The published curves reach roughly 1.1 for quintessence at ``w = -0.9``
    and 0.9 for phantom at ``w = -1.1`` by a separation of 2, with LCDM pinned
    at unity throughout. Both branches grow monotonically away from 1 with
    separation, which is the property that makes the diagnostic useful.
    """

    z3 = linear_grid(0.40, 2.35, 30)

    test = Om3(z1=0.2, z2=0.35)

    def curve(predictor):
        return test.statistic(reconstruction(predictor, z3)).mean()

    quintessence = curve(sharp(w0=-0.9, wa=0.0, Om=0.27))
    lcdm = curve(sharp(Om=0.27))
    phantom = curve(sharp(w0=-1.1, wa=0.0, Om=0.27))

    assert np.allclose(lcdm, 1.0, atol=1e-12)

    # Above for quintessence, below for phantom, everywhere.
    assert np.all(quintessence > 1.0)
    assert np.all(phantom < 1.0)

    # Monotone in the separation, as the figure shows.
    assert np.all(np.diff(quintessence) > 0.0)
    assert np.all(np.diff(phantom) < 0.0)

    # And quantitatively where the figure puts them at a separation of 2.
    assert quintessence[-1] == pytest.approx(1.11, abs=0.02)
    assert phantom[-1] == pytest.approx(0.90, abs=0.02)


def test_om_sits_above_omega_m_for_quintessence_and_below_for_phantom():
    """
    The other half of the Sahni-Shafieloo-Starobinsky statement, and the one
    that makes a *measured* Om worth plotting: which side of the constant it
    falls on is the sign of ``w + 1``.
    """

    z = linear_grid(0.05, 1.5, 20)

    def curve(predictor):
        return Om().statistic(reconstruction(predictor, z)).mean()

    assert np.all(curve(sharp(w0=-0.9, wa=0.0, Om=0.27)) > 0.27)
    assert np.all(curve(sharp(w0=-1.1, wa=0.0, Om=0.27)) < 0.27)


def test_a_specified_omega_m_makes_the_test_stronger():
    """
    ``Om(omega_m=...)`` tests the value as well as the flatness, which a
    free-constant version cannot: a phantom universe whose Om is constant to
    the eye still sits at the wrong level.
    """

    z = linear_grid(0.05, 1.5, 20)

    H = reconstruction(FlatLCDM(Om=0.34, sigma_Om=0.01), z)

    assert Om().evaluate(H).consistent                    # flat, so it passes

    assert not Om(omega_m=0.27).evaluate(H).consistent    # but at 0.34, not 0.27


# ============================================================
# What Om3 does not need
# ============================================================

def test_om_needs_an_extrapolation_to_zero_and_om3_does_not():
    """
    The practical difference between the two diagnostics.

    ``Om(z)`` is anchored at ``z = 0``; the lowest cosmic chronometer sits at
    ``z = 0.07``, so it is standing on an extrapolation and the library says
    so. ``Om3`` is anchored at two redshifts the user picks, and with both
    inside the data it touches nothing that was not measured.
    """

    data = real_chronometers()

    grid = np.linspace(0.45, 1.90, 20)

    H = GaussianProcess(kernel="matern").fit(
        data, grid=grid, n_draws=400, seed=3
    )["H"]

    with pytest.warns(ExtrapolationWarning):
        _ = Om().statistic(H).draws

    with warnings.catch_warnings():

        warnings.simplefilter("error", ExtrapolationWarning)

        _ = Om3(z1=0.15, z2=0.35).statistic(H).draws


# ============================================================
# The correlations are the point
# ============================================================

def test_independent_error_propagation_gets_the_width_badly_wrong():
    """
    ``Om3(z1, z2, z3)`` shares ``H(z1)`` and ``H(z2)`` across its whole
    length, and is a ratio of two differences, so it is neither weakly
    correlated nor Gaussian.

    Propagating marginal error bars as though the three redshifts were
    independent -- the only thing a curve-with-error-bars representation can
    do -- misstates the width by a factor of two in the middle of the range
    and several at the ends, in *both* directions. There is no fudge factor
    that repairs it; the correlations have to be carried, which is what the
    draws do.
    """

    grid = np.linspace(0.45, 1.90, 25)

    z1, z2 = 0.15, 0.35

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        H = GaussianProcess(kernel="matern").fit(
            real_chronometers(), grid=grid, n_draws=4000, seed=3
        )["H"]

        exact = Om3(z1=z1, z2=z2).statistic(H).std()

    h1, s1 = float(H.at(z1).mean()[0]), float(H.at(z1).std()[0])
    h2, s2 = float(H.at(z2).mean()[0]), float(H.at(z2).std()[0])
    h3, s3 = H.mean(), H.std()

    def om3_of(a, b, c):

        numerator = (b**2 / a**2 - 1.0) / ((1 + z2) ** 3 - (1 + z1) ** 3)
        denominator = (c**2 / a**2 - 1.0) / ((1 + grid) ** 3 - (1 + z1) ** 3)

        return numerator / denominator

    base = om3_of(h1, h2, h3)

    eps = 1e-5

    naive = np.sqrt(
        ((om3_of(h1 * (1 + eps), h2, h3) - base) / eps * s1 / h1) ** 2
        + ((om3_of(h1, h2 * (1 + eps), h3) - base) / eps * s2 / h2) ** 2
        + ((om3_of(h1, h2, h3 * (1 + eps)) - base) / eps * s3 / h3) ** 2
    )

    ratio = exact / naive

    # Too wide in the middle, too narrow at the ends: not a rescaling.
    assert np.min(ratio) < 0.7
    assert np.max(ratio) > 2.0


# ============================================================
# Refusals
# ============================================================

def test_a_grid_touching_its_own_reference_is_refused():
    """At ``z = z_ref`` a two-point Om is 0/0, and useless around it."""

    z = linear_grid(0.0, 1.5, 20)

    with pytest.raises(DataError, match="0/0"):
        Om().statistic(reconstruction(FlatLCDM(), z))


def test_om3_refuses_a_grid_touching_its_reference():

    z = linear_grid(0.2, 1.5, 20)          # starts exactly at z1

    with pytest.raises(DataError, match="0/0"):
        Om3(z1=0.2, z2=0.5).statistic(reconstruction(FlatLCDM(), z))


def test_om3_refuses_two_coincident_anchors():

    with pytest.raises(ValueError, match="same redshift"):
        Om3(z1=0.3, z2=0.3)


def test_omega_m_only_means_omega_m_at_zero():
    """
    Anchored anywhere else the diagnostic is ``Omega_m / h^2(z_ref)``, so
    testing it against a matter density would be testing the wrong number.
    """

    with pytest.raises(ValueError, match="only equal to Omega_m"):
        Om(z_reference=0.5, omega_m=0.3)


# ============================================================
# On real data
# ============================================================

def test_both_diagnostics_run_on_the_real_chronometers():

    grid = np.linspace(0.45, 1.90, 20)

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        H = GaussianProcess(kernel="matern").fit(
            real_chronometers(), grid=grid, n_draws=1500, seed=3
        )["H"]

        om = Om().evaluate(H)

        om3 = Om3(z1=0.15, z2=0.35).evaluate(H)

    # 32 differential ages cannot see a DESI-scale deviation, and a test that
    # said otherwise would be measuring its own assumptions.
    assert om.consistent
    assert om3.consistent

    # The effective degrees of freedom are the rank of the statistic's own
    # covariance, computed rather than taken from the grid. For a
    # Gaussian-process posterior -- which is genuinely high-dimensional --
    # that rank can reach the grid size, and does here. It is a measurement
    # either way, and it is what the chi-square is scored against.
    assert 0 < om3.n_eff <= grid.size
    assert 0.0 <= om3.pte <= 1.0
