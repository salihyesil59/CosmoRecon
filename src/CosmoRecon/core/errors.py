"""
The exception hierarchy.

Most of these exist to turn a *silent* statistical mistake into a loud one.
That is the theme: a library whose output is an error bar has to fail rather
than quietly return a number that looks fine and is wrong.

Everything derives from :class:`CosmoReconError`, so a caller wrapping a whole
pipeline can catch one thing.
"""

from __future__ import annotations


__all__ = [
    "CosmoReconError",
    "AlignmentError",
    "ConvergenceError",
    "GridMismatchError",
    "DerivativeUnavailableError",
    "NotResamplableError",
    "NotRefittableError",
    "EvidenceUnavailableError",
    "InsufficientDrawsError",
    "DataError",
    "OptionalDependencyError",
]


class CosmoReconError(Exception):
    """Base class for every error this library raises."""


# ============================================================
# Combining reconstructions
# ============================================================

class AlignmentError(CosmoReconError):
    """
    Two reconstructions from *different* fits were combined without saying
    how they relate.

    Draws from one fit share a realisation index: draw ``k`` of ``H`` and
    draw ``k`` of ``D`` describe the same universe, so arithmetic between
    them carries their correlation exactly. Draws from two separate fits
    share nothing, and pairing them by index would invent a correlation
    structure that nobody chose -- usually a spuriously *tight* one, because
    the spurious correlation is positive.

    There is no safe default here, only a physical claim the user has to
    make: call ``assume_independent()`` on one of them if the two fits really
    do use disjoint data, or refit them together if they do not.
    """


class GridMismatchError(CosmoReconError):
    """
    Two reconstructions were combined on incompatible redshift grids.

    Fixed with :meth:`~CosmoRecon.core.reconstruction.Reconstruction.at`,
    which re-evaluates one of them on the other's grid. This is the
    redshift-mismatch problem that every distance-duality analysis has to
    solve; here it is a method call rather than an assumption.
    """


# ============================================================
# Limits of a particular reconstruction
# ============================================================

class DerivativeUnavailableError(CosmoReconError):
    """
    A derivative was requested at an order the underlying method cannot
    provide analytically, with the numerical fallback disabled.

    Differentiating a reconstruction numerically is legitimate, but it is a
    different estimator with different -- generally understated --
    uncertainties, so it never happens implicitly.
    """


class NotResamplableError(CosmoReconError):
    """
    ``at()`` was called on a reconstruction that exists only on its own grid.

    The result of arithmetic between reconstructions holds draws, not a
    predictor: ``H**2 * D.d(1)**2`` cannot be re-evaluated at a new redshift
    because the operation was applied to samples, not to functions. Move the
    ``at()`` call to the *operands*, before the arithmetic.
    """


class NotRefittableError(CosmoReconError):
    """
    A result was asked to rerun its analysis on new data and does not know how.

    A fit remembers the method, the data and the grid it came from, so that a
    significance can be calibrated by rerunning exactly the same analysis on
    simulated data. Something built outside a fit -- a reconstruction wrapped
    around a hand-written predictor, or a set assembled by hand -- has no
    analysis to rerun, and its significance can only be the nominal one.
    """


class ConvergenceError(CosmoReconError):
    """
    A series expansion was used outside the region where it converges.

    Distinct from :class:`~CosmoRecon.core.grid.ExtrapolationWarning`, and the
    difference is the reason this is an error and that is a warning.
    Extrapolating past the data is a weak statement -- the answer is the prior,
    which is a defensible thing to look at. A truncated series outside its
    radius of convergence is not a weak statement about the function; it is not
    a statement about the function at all, and adding terms makes it worse
    rather than better.

    The case that matters in practice: the Taylor series of a cosmological
    distance in ``z`` has radius of convergence ``|z| = 1``, because of the
    singularity at ``z = -1``. Fitting one to supernovae reaching ``z = 2`` and
    reading off a jerk parameter is the single most common error in the
    cosmographic literature, and it is what the ``y = z / (1 + z)`` variable
    was introduced to fix.
    """


class EvidenceUnavailableError(CosmoReconError):
    """
    A Bayesian evidence was requested from a method that does not define one
    (a neural reconstruction, a genetic-algorithm fit).

    Raised rather than returned as ``nan`` so that an evidence-weighted
    ensemble cannot silently drop a member.
    """


class InsufficientDrawsError(CosmoReconError):
    """
    Too few draws to support the statistic being asked for.

    A 99.7% interval estimated from 100 draws is noise; the quantile
    machinery says so instead of returning it.
    """


# ============================================================
# Inputs and environment
# ============================================================

class DataError(CosmoReconError):
    """A dataset is malformed, inconsistent, or missing what a fit needs."""


class OptionalDependencyError(CosmoReconError):
    """
    A feature was used whose optional dependency is not installed.

    The message names the extra to install, e.g. ``pip install
    'cosmorecon[nested]'``.
    """

    def __init__(
        self,
        feature: str,
        package: str,
        extra: str,
    ) -> None:

        super().__init__(
            f"{feature} needs {package!r}, which is not installed. "
            f"Install it with:  pip install 'cosmorecon[{extra}]'"
        )

        self.feature = feature
        self.package = package
        self.extra = extra
