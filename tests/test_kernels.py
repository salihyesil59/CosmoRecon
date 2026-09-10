"""
Kernels, checked against themselves.

Each kernel states a covariance and, separately, a spectral density. Those two
statements are equivalent by Bochner's theorem, and they are written here as
independent pieces of code -- so rebuilding the covariance from the spectral
quadrature and comparing it against the closed form is a genuine cross-check,
not a tautology. Getting it wrong would mean building sample paths from a
different prior than the marginal likelihood was computed under, silently, with
nothing about the output looking unusual.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon.reconstructors.kernels import (
    Cauchy,
    Matern,
    RationalQuadratic,
    SquaredExponential,
    get_kernel,
)


KERNEL_CASES = [
    (SquaredExponential(), {}),
    (Matern(), {"nu": 1.0}),
    (Matern(), {"nu": 1.5}),
    (Matern(), {"nu": 2.5}),
    (Matern(), {"nu": 7.5}),
    (RationalQuadratic(), {"alpha": 0.75}),
    (RationalQuadratic(), {"alpha": 5.0}),
    (Cauchy(), {}),
]


@pytest.mark.parametrize("kernel, shape", KERNEL_CASES)
def test_spectral_quadrature_reproduces_the_covariance(kernel, shape):
    """
    ``k(r) = sum_m w_m cos(omega_m r)`` -- the two halves of each kernel have
    to agree, or sample paths come from a different prior than the marginal
    likelihood was computed under.

    The covariance and the spectral density are written as independent pieces
    of code, so this is a real cross-check rather than a tautology. It also
    checks the arithmetic of the rational quadratic's dual form, which is the
    one derivation here with room to go wrong quietly.
    """

    length_scale = 0.37

    r = np.linspace(0.0, 4.0 * length_scale, 40)

    omega, weights, error = kernel.spectral_quadrature(
        length_scale, r.max(), **shape
    )

    approx = (weights[None, :] * np.cos(np.outer(r, omega))).sum(axis=1)

    exact = kernel.covariance(r, length_scale, **shape)

    assert np.allclose(approx, exact, atol=3e-4)

    # And the routine's own report of how well it did has to be honest.
    assert error <= 3e-4
    assert error >= np.max(np.abs(approx - exact)) - 1e-12


@pytest.mark.parametrize("kernel, shape", KERNEL_CASES)
def test_quadrature_weights_normalise_to_the_variance(kernel, shape):
    """``k(0) = 1``, exactly, whatever the tail truncation did."""

    _, weights, _ = kernel.spectral_quadrature(0.37, 1.5, **shape)

    assert np.isclose(weights.sum(), 1.0, atol=1e-14)
    assert np.all(weights >= 0.0)


def test_a_spectral_tail_too_heavy_to_represent_is_refused():
    """
    Matern-1/2 cannot be carried in a truncated sample-path basis at any
    affordable size, and saying so is better than approximating it quietly --
    a path from the wrong kernel looks exactly like a path from the right one.
    """

    with pytest.raises(ValueError, match="spectral tail is too heavy"):
        Matern().spectral_quadrature(0.37, 2.0, nu=0.5)


@pytest.mark.parametrize("kernel, shape", KERNEL_CASES)
def test_covariance_is_one_at_zero_separation(kernel, shape):

    assert np.isclose(kernel.covariance(0.0, 1.0, **shape), 1.0)


def test_matern_tends_to_the_squared_exponential():
    """Large nu is the Gaussian limit, and both halves should show it."""

    r = np.linspace(0.0, 2.0, 30)

    assert np.allclose(
        Matern().covariance(r, 1.0, nu=200.0),
        SquaredExponential().covariance(r, 1.0),
        atol=5e-3,
    )


def test_rational_quadratic_tends_to_the_squared_exponential():

    r = np.linspace(0.0, 2.0, 30)

    assert np.allclose(
        RationalQuadratic().covariance(r, 1.0, alpha=5000.0),
        SquaredExponential().covariance(r, 1.0),
        atol=5e-3,
    )


# ============================================================
# Smoothness
# ============================================================

@pytest.mark.parametrize(
    "nu, expected",
    [(0.5, 0), (1.0, 0), (1.5, 1), (2.0, 1), (2.5, 2), (3.5, 3), (5.0, 4)],
)
def test_matern_smoothness_limit(nu, expected):
    """
    A Matern-nu process is m times differentiable iff m < nu.

    The consequence worth stating: nu = 1.5, the most common fixed choice in
    the literature, supports H'(z) and not H''(z) -- so w(z) is defined under
    it and the cosmographic jerk is not.
    """

    assert Matern().max_derivative(nu=nu) == expected


@pytest.mark.parametrize("kernel, shape", [
    (SquaredExponential(), {}),
    (RationalQuadratic(), {"alpha": 1.0}),
    (Cauchy(), {}),
])
def test_analytic_kernels_have_no_derivative_limit(kernel, shape):

    assert kernel.max_derivative(**shape) is None


# ============================================================
# Configuration
# ============================================================

def test_named_kernels_come_free_not_pinned():
    """
    ``"matern"`` marginalises nu. Pinning it is available and has to be
    asked for -- which is the library's whole argument, expressed as a
    default.
    """

    assert get_kernel("matern").nu is None
    assert len(get_kernel("matern").shape_grid()["nu"]) > 1

    assert Matern(nu=1.5).shape_grid()["nu"].tolist() == [1.5]

    # And a restricted grid -- how a derivative is bought, explicitly.
    assert Matern(nu=[2.5, 1.5]).shape_grid()["nu"].tolist() == [1.5, 2.5]


def test_unknown_kernel_names_are_refused():

    with pytest.raises(ValueError, match="Unknown kernel"):
        get_kernel("gaussian-ish")
