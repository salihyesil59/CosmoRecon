"""
The core contract: draws, alignment, derivatives, and the significance
machinery.

These are the tests that have to pass before any reconstruction method is
worth writing, because every method's output is only as trustworthy as the
object it is handed back in.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import (
    AlignmentError,
    DerivativeUnavailableError,
    GridMismatchError,
    NotResamplableError,
    linear_grid,
)

from CosmoRecon.consistency import significance
from CosmoRecon.core.errors import InsufficientDrawsError
from CosmoRecon.core.provenance import CONSTANT_ORIGIN
from CosmoRecon.core.reconstruction import Reconstruction
from CosmoRecon.ensemble import total_variance

from tests.toy import FlatCPL, FlatLCDM, reconstruction


Z = linear_grid(0.01, 1.5, 60)


@pytest.fixture
def H():
    return reconstruction(FlatLCDM(), Z)


# ============================================================
# Representation
# ============================================================

def test_draws_have_the_documented_shape(H):

    assert H.draws.shape == (H.n_draws, H.n_z)
    assert H.mean().shape == (H.n_z,)
    assert H.cov().shape == (H.n_z, H.n_z)


def test_grid_is_not_writable_through_the_property(H):

    with pytest.raises(ValueError):
        H.z[0] = 99.0


def test_quantiles_refuse_a_tail_the_draws_cannot_support():

    thin = reconstruction(FlatLCDM(n_draws=100), Z)

    thin.quantile(0.16)                       # 16 draws in the tail: fine

    with pytest.raises(InsufficientDrawsError):
        thin.quantile(0.001)                  # 0.1 draws in the tail: not


# ============================================================
# The point of the whole design
# ============================================================

def test_correlations_cancel_exactly_in_a_derived_quantity(H):
    """
    ``Om(z) = [E(z)^2 - 1] / [(1+z)^3 - 1]`` is constant at ``Omega_m`` in
    flat LCDM.

    Getting that back with *zero* scatter across redshift is the whole
    argument for carrying draws: the H0 uncertainty cancels between ``H(z)``
    and ``H(0)`` draw by draw, and the Omega_m uncertainty is the only thing
    left. Propagate diagonal error bars through the same expression and the
    result is neither constant nor correctly sized.
    """

    E = H / H.at(0.0)

    Om = (E**2 - 1.0) / ((1.0 + Z) ** 3 - 1.0)

    truth_mean = H._predictor.Om.mean()
    truth_std = H._predictor.Om.std(ddof=1)

    assert np.allclose(Om.mean(), truth_mean, atol=1e-10)
    assert np.allclose(Om.std(), truth_std, atol=1e-10)

    # And the curve is flat to machine precision, not merely flat-ish.
    assert np.ptp(Om.mean()) < 1e-12


def test_a_one_point_reconstruction_broadcasts(H):
    """
    ``H.at(0.0)`` is a scalar-valued posterior and broadcasts across the
    grid -- and, because it is draw-aligned with ``H``, dividing by it
    removes the H0 uncertainty rather than adding to it.
    """

    ratio = H / H.at(0.0)

    assert ratio.n_z == H.n_z

    fractional_H = H.std() / H.mean()
    fractional_ratio = ratio.std() / ratio.mean()

    # H carries ~1.4% from H0 alone; E(z) should carry far less near z = 0,
    # which only happens if the two H0 factors cancelled draw by draw.
    assert fractional_ratio[0] < 0.1 * fractional_H[0]


# ============================================================
# Alignment
# ============================================================

def test_same_fit_combines_and_keeps_its_origin(H):

    assert (H + H).origin == H.origin
    assert (H * 2.0).origin == H.origin


def test_different_fits_refuse_to_combine(H):

    other = reconstruction(FlatLCDM(seed=1), Z, label="H2")

    with pytest.raises(AlignmentError, match="independence claim"):
        H + other


def test_independence_has_to_be_declared(H):

    other = reconstruction(FlatLCDM(seed=1), Z, label="H2")

    combined = H + other.assume_independent()

    assert combined.n_draws == min(H.n_draws, other.n_draws)
    assert combined.origin == H.origin        # H is the anchor


def test_constants_combine_with_anything(H):

    c = Reconstruction.constant(Z, 1.0)

    assert c.origin == CONSTANT_ORIGIN
    assert (H * c).origin == H.origin


def test_grids_must_match(H):

    with pytest.raises(GridMismatchError, match="common_support"):
        H + H.at(linear_grid(0.01, 1.5, 61))


# ============================================================
# Derivatives
# ============================================================

def test_analytic_derivative_matches_finite_differences(H):

    analytic = H.d(1)
    numeric = H.d(1, numerical=True)

    # Away from the edges, where np.gradient is second-order accurate.
    inner = slice(2, -2)

    assert np.allclose(
        analytic.mean()[inner],
        numeric.mean()[inner],
        rtol=2e-3,
    )


def test_second_derivative_reaches_the_method_not_a_derivative_of_a_derivative(H):

    with pytest.raises(DerivativeUnavailableError):
        _ = H.d(1).d(1).draws      # toy model stops at order 1


def test_derived_quantities_have_no_analytic_derivative(H):

    ratio = H / H.at(0.0)

    with pytest.raises(DerivativeUnavailableError, match="numerical=True"):
        ratio.d(1)

    ratio.d(1, numerical=True)     # opt-in works


def test_derived_quantities_cannot_be_regridded(H):

    ratio = H / H.at(0.0)

    with pytest.raises(NotResamplableError, match="operands"):
        ratio.at(linear_grid(0.1, 1.0, 20))


# ============================================================
# Provenance
# ============================================================

def test_provenance_composes_through_arithmetic(H):

    Om = ((H / H.at(0.0)) ** 2 - 1.0) / ((1.0 + Z) ** 3 - 1.0)

    assert Om.provenance.is_derived
    assert "H" in Om.provenance.expression
    assert Om.provenance.methods() == ("toy",)
    assert Om.provenance.datasets() == ("mock(30)",)


def test_provenance_records_both_methods_when_two_fits_meet(H):

    other = reconstruction(FlatLCDM(seed=1), Z, method="other toy")

    mixed = H - other.assume_independent()

    assert set(mixed.provenance.methods()) == {"toy", "other toy"}


# ============================================================
# Significance
# ============================================================

def test_exact_lcdm_shows_no_deviation():
    """An exactly constant Om(z) has no resolved modes and no deviation."""

    H = reconstruction(FlatLCDM(), Z)

    Om = ((H / H.at(0.0)) ** 2 - 1.0) / ((1.0 + Z) ** 3 - 1.0)

    chi2, n_eff, pte, sigma = significance(
        Om, float(H._predictor.Om.mean()), marginalise_constant=True
    )

    assert n_eff == 0
    assert pte == 1.0
    assert sigma == 0.0


def test_injected_dynamical_dark_energy_is_detected():
    """
    The test that matters: a CPL universe must fail the Om(z) null test, and
    fail it against an honest number of degrees of freedom.

    The dof bound is a check on the eigenbasis machinery, and it is a sharp
    one here because the answer is known in advance. ``Om(z)`` built from this
    toy depends on exactly three parameters that change its *shape* --
    ``Omega_m``, ``w0``, ``wa``, with ``H0`` cancelling between ``H(z)`` and
    ``H(0)`` -- so the covariance is rank three up to the curvature of a
    non-linear map, and its spectrum falls off a cliff after the third mode
    (1, 8e-2, 7e-3, then 1e-5 and below). Recovering that from a 60-point grid
    is the routine measuring the rank rather than assuming the grid.

    Note what this is *not*: a claim that smooth curves always carry a handful
    of numbers. A Gaussian-process posterior is genuinely high-dimensional and
    reports a much larger dof on the same grid, correctly.
    """

    H = reconstruction(FlatCPL(w0=-0.8, wa=-0.8), Z)

    Om = ((H / H.at(0.0)) ** 2 - 1.0) / ((1.0 + Z) ** 3 - 1.0)

    chi2, n_eff, pte, sigma = significance(
        Om, float(Om.mean().mean()), marginalise_constant=True
    )

    assert 0 < n_eff < 10, f"a smooth curve should not carry {n_eff} modes"
    assert sigma > 3.0, f"CPL(-0.8, -0.8) went undetected at {sigma:.2f} sigma"


def test_significance_refuses_a_rank_starved_covariance():
    """
    A rough curve resolves as many modes as it has draws to resolve them
    with, and at that point the covariance estimate -- not the data -- sets
    the chi-square. Refusing is the only honest answer.
    """

    rng = np.random.default_rng(7)

    rough = Reconstruction.from_draws(
        np.arange(30.0),
        rng.normal(size=(8, 30)),
        provenance=reconstruction(FlatLCDM(n_draws=8), Z).provenance,
    )

    with pytest.raises(ValueError, match="rank-starved"):
        significance(rough, 0.0)


# ============================================================
# Variance budget
# ============================================================

def test_law_of_total_variance_holds_against_a_brute_force_pool():
    """
    The identity the budget is built on, checked by pooling the draws and
    taking their variance directly.
    """

    rng = np.random.default_rng(3)

    pools = {
        name: rng.normal(loc, scale, size=(6000, 1))
        for name, (loc, scale) in {
            "a": (70.0, 1.0),
            "b": (71.5, 1.4),
            "c": (69.2, 0.8),
        }.items()
    }

    budget = total_variance(
        means={k: v.mean(axis=0) for k, v in pools.items()},
        variances={k: v.var(axis=0, ddof=1) for k, v in pools.items()},
        weights=dict.fromkeys(pools, 1.0),
        z=np.array([0.0]),
    )

    pooled = np.concatenate(list(pools.values()), axis=0)

    assert np.isclose(budget.total[0], pooled.var(axis=0, ddof=1)[0], rtol=2e-3)

    # And the split is not trivial: three methods disagreeing by ~1 km/s/Mpc
    # on an interval of ~1 km/s/Mpc contribute real methodological variance.
    assert 0.2 < budget.method_fraction()[0] < 0.8


def test_a_budget_needs_more_than_one_method():

    with pytest.raises(ValueError, match="at least two methods"):
        total_variance(
            means={"a": np.zeros(3)},
            variances={"a": np.ones(3)},
            weights={"a": 1.0},
            z=np.arange(3.0),
        )
