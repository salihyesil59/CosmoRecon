"""
Redshift grids, and the arithmetic of bringing two of them together.

A grid is a small thing that causes large errors. Two failure modes recur in
the reconstruction literature and both are addressed here rather than left to
each caller:

**Extrapolation that looks like a measurement.** A Gaussian process outside
the redshift range of its data returns to the prior mean with growing error
bars, and a plot of it looks like a reconstruction. :func:`support_of` and
:func:`common_support` make the honest range explicit, and
:func:`check_within_support` turns silently-extrapolated output into a
warning.

**Grids that are too fine to be honest.** A thousand points across a
reconstruction supported by thirty-two chronometer measurements are a thousand
strongly correlated numbers, and a chi-square that treats them as a thousand
degrees of freedom will report a detection made entirely of correlation. The
grid does not cause that -- :mod:`CosmoRecon.consistency` computes an
effective number of degrees of freedom -- but :func:`suggested_n` says what a
defensible resolution looks like for a given dataset.
"""

from __future__ import annotations

import warnings

import numpy as np

from CosmoRecon.typing import Array, Redshift


__all__ = [
    "linear_grid",
    "log_grid",
    "support_of",
    "common_support",
    "check_within_support",
    "suggested_n",
    "ExtrapolationWarning",
]


class ExtrapolationWarning(UserWarning):
    """
    Issued when a reconstruction is evaluated outside the range of the data
    that constrained it.

    A warning rather than an error: extrapolating on purpose is a legitimate
    thing to do -- forecasting, or showing where a method loses its grip --
    and the caller who means it can silence this. Doing it by accident is the
    case worth catching.
    """


# ============================================================
# Building grids
# ============================================================

def linear_grid(
    z_min: float,
    z_max: float,
    n: int = 200,
) -> Array:
    """
    Evenly spaced redshifts, endpoints included.

    The default for anything plotted against ``z``.
    """

    _check_range(z_min, z_max, n)

    return np.linspace(z_min, z_max, n, dtype=float)


def log_grid(
    z_min: float,
    z_max: float,
    n: int = 200,
) -> Array:
    """
    Evenly spaced in ``log(1 + z)``, endpoints included.

    The right default when the quantity varies over decades of scale factor
    -- a growth history, or anything reaching towards recombination -- where
    a linear grid spends most of its points where nothing happens.
    """

    _check_range(z_min, z_max, n)

    return np.expm1(
        np.linspace(np.log1p(z_min), np.log1p(z_max), n, dtype=float)
    )


def _check_range(z_min: float, z_max: float, n: int) -> None:

    if not np.isfinite([z_min, z_max]).all():

        raise ValueError("Grid endpoints must be finite.")

    if z_min < -1.0:

        raise ValueError(f"Redshift below -1 is not a redshift: {z_min}.")

    if z_max <= z_min:

        raise ValueError(
            f"Grid needs z_max > z_min, got ({z_min}, {z_max})."
        )

    if n < 2:

        raise ValueError(f"A grid needs at least two points, got {n}.")


# ============================================================
# Support
# ============================================================

def support_of(*z_data: Redshift) -> tuple[float, float]:
    """
    The redshift range over which data actually constrain a reconstruction:
    the smallest interval containing every measurement redshift given.
    """

    if not z_data:

        raise ValueError("No data redshifts given.")

    stacked = np.concatenate(
        [np.atleast_1d(np.asarray(z, dtype=float)).ravel() for z in z_data]
    )

    if stacked.size == 0:

        raise ValueError("No data redshifts given.")

    return float(stacked.min()), float(stacked.max())


def common_support(
    *supports: tuple[float, float],
) -> tuple[float, float]:
    """
    The overlap of several supports -- where a joint statement is defensible.

    A distance-duality test built from supernovae reaching ``z = 2.3`` and
    BAO reaching ``z = 0.5`` is a statement about ``z <= 0.5``, whatever the
    grid says. Raises if the supports do not overlap at all, because the
    answer in that case is not a narrower range but a different analysis.
    """

    if not supports:

        raise ValueError("No supports given.")

    lo = max(s[0] for s in supports)
    hi = min(s[1] for s in supports)

    if hi <= lo:

        raise ValueError(
            f"These datasets do not overlap in redshift: the ranges "
            f"{list(supports)} have no common interval. There is no grid on "
            "which a joint test of them is defined."
        )

    return lo, hi


def check_within_support(
    z: Redshift,
    support: tuple[float, float],
    *,
    what: str = "reconstruction",
) -> None:
    """
    Warn if ``z`` reaches outside ``support``.

    Called by the reconstructors on every evaluation, so that a curve drawn
    beyond its data says so once rather than never.
    """

    grid = np.atleast_1d(np.asarray(z, dtype=float))

    lo, hi = support

    below = float(grid.min())
    above = float(grid.max())

    if below < lo or above > hi:

        warnings.warn(
            f"{what} evaluated on [{below:.4g}, {above:.4g}], outside the "
            f"range [{lo:.4g}, {hi:.4g}] where data constrain it. Beyond "
            "that range the result is the prior, not a measurement.",
            ExtrapolationWarning,
            stacklevel=3,
        )


# ============================================================
# Resolution
# ============================================================

def suggested_n(n_data: int, *, per_datum: int = 8, cap: int = 400) -> int:
    """
    A defensible number of grid points for a reconstruction from ``n_data``
    measurements.

    There is no theorem here, and none is claimed: the grid does not add
    information, so this only picks a resolution fine enough to draw the
    curve smoothly and coarse enough that nobody mistakes it for a sample
    size. Significances are computed against the effective degrees of freedom
    (see :mod:`CosmoRecon.consistency`), never against ``n``.
    """

    if n_data < 2:

        raise ValueError(f"Cannot reconstruct from {n_data} points.")

    return int(min(cap, max(50, per_datum * n_data)))
