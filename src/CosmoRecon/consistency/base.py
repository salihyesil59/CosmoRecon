"""
Null tests: what every consistency check has in common.

A null test takes reconstructions, forms a statistic that is a known constant
if the concordance model holds, and asks how far the data are from that
constant. The forming is easy -- one line of arithmetic on
:class:`~CosmoRecon.core.reconstruction.Reconstruction` objects. Turning the
answer into a significance is where these analyses go wrong, in one specific
way, and this module exists to get that part right once.

**The trap.** A reconstruction evaluated on two hundred redshifts is not two
hundred measurements. It is a smooth function pinned by perhaps thirty data
points, so its values at neighbouring redshifts are almost the same number.
A chi-square that divides by the pointwise error bar and counts two hundred
degrees of freedom will report several sigma of deviation from data that
contain none. The deviation is real; the degrees of freedom are not.

**The fix.** Compute the chi-square in the eigenbasis of the statistic's own
covariance, keep only the modes the draws actually resolve, and report that
count as the effective number of degrees of freedom. On a smooth
reconstruction it is typically a handful, not the grid size -- which is the
honest statement about how much independent information a null test of this
kind can carry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import stats

from CosmoRecon.typing import Array

from CosmoRecon.core.provenance import Provenance
from CosmoRecon.core.reconstruction import Reconstruction


__all__ = ["NullTest", "TestResult", "significance", "effective_modes"]


#: Eigenvalues below this fraction of the largest are treated as unresolved
#: and dropped. They are the directions in which the draws carry no
#: information, and inverting them manufactures chi-square out of Monte Carlo
#: noise.
_EIGEN_TOL = 1e-8


# ============================================================
# Result
# ============================================================

@dataclass(frozen=True, slots=True)
class TestResult:
    """
    The outcome of one null test.

    Carries the curve as well as the number, because the number alone hides
    the thing a reader wants: *where* in redshift the deviation sits, and
    whether it looks like a feature or like a slope.
    """

    #: ``"Om"``, ``"Ok"``, ``"distance duality"``.
    name: str

    #: The statistic as a function of redshift, with its full posterior.
    curve: Reconstruction

    #: What the statistic equals if the concordance model holds -- ``0`` for
    #: the litmus test, ``1`` for distance duality, a constant (unknown but
    #: redshift-independent) for ``Om``.
    null_value: float | Array

    #: Chi-square of the deviation, computed in the resolved eigenbasis.
    chi2: float

    #: Effective degrees of freedom: how many independent directions the
    #: statistic actually has. Almost never the grid size.
    n_eff: int

    #: Probability of a deviation this large or larger under the null.
    pte: float

    #: The same, as a two-sided Gaussian-equivalent number of sigma. Quoted
    #: because everyone quotes it; :attr:`pte` is the honest one.
    sigma: float

    #: Where the statistic came from, all the way back to the data.
    provenance: Provenance

    #: The redshift range over which every input was a measurement rather
    #: than a prior.
    support: tuple[float, float] | None = None

    # ---------------------------------------------------------

    @property
    def consistent(self) -> bool:
        """Whether the null survives at the conventional 5% level."""

        return self.pte >= 0.05

    def summary(self) -> str:
        """One line, for a log or a table."""

        verdict = "consistent" if self.consistent else "DEVIATION"

        return (
            f"{self.name}: chi2 = {self.chi2:.2f} / {self.n_eff} eff. dof, "
            f"p = {self.pte:.3g} ({self.sigma:.2f} sigma) -- {verdict}"
        )

    def __str__(self) -> str:

        return self.summary()


# ============================================================
# Significance
# ============================================================

def effective_modes(
    covariance: Array,
    *,
    tol: float = _EIGEN_TOL,
    scale: float | None = None,
) -> tuple[Array, Array]:
    """
    The resolved eigenmodes of a covariance matrix.

    Returns ``(eigenvalues, eigenvectors)`` keeping only directions whose
    eigenvalue exceeds ``tol * scale``. Those are the directions in which the
    posterior actually varies; the rest are the smoothness of the
    reconstruction, and inverting them turns a smooth curve into a spurious
    detection.

    ``scale`` defaults to this matrix's own largest eigenvalue, which is the
    right reference for a covariance as it comes off a reconstruction. It has
    to be passed explicitly once the matrix has been *projected* -- if the
    statistic is very nearly constant, projecting out the constant direction
    leaves floating-point noise, whose own largest eigenvalue is also noise,
    and a threshold relative to it would admit every rounding error as a
    resolved mode. :func:`significance` therefore passes the scale of the
    covariance before projection.
    """

    values, vectors = np.linalg.eigh(
        0.5 * (covariance + covariance.T),
    )

    if values.size == 0 or values.max() <= 0.0:

        raise ValueError(
            "The statistic has no positive variance anywhere -- its "
            "covariance is singular. That usually means the draws are "
            "degenerate (a one-draw reconstruction, or a constant)."
        )

    reference = float(values.max()) if scale is None else float(scale)

    keep = values > tol * reference

    return values[keep], vectors[:, keep]


def significance(
    curve: Reconstruction,
    null_value: float | Array,
    *,
    marginalise_constant: bool = False,
    tol: float = _EIGEN_TOL,
) -> tuple[float, int, float, float]:
    """
    How far ``curve`` sits from ``null_value``.

    Returns ``(chi2, n_eff, pte, sigma)``.

    ``marginalise_constant`` is for the tests whose null is *"some constant,
    we do not care which"* -- ``Om(z)`` is constant at ``Omega_m`` in flat
    LCDM, and the test is about the flatness of the curve, not about the
    value it is flat at. Setting it projects out the direction in which the
    whole curve shifts up or down, and drops one effective degree of freedom
    to pay for it.

    The chi-square uses the covariance estimated from the draws, so it
    inherits that estimate's noise. The Hartlap-style correction below undoes
    the resulting bias in the inverse; without it, a covariance built from a
    number of draws comparable to the number of retained modes gives a
    chi-square biased high, and a detection that is an artefact of the sample
    size.
    """

    delta = curve.mean() - np.asarray(null_value, dtype=float)

    cov = curve.cov()

    # Fixed before any projection, so that projecting out a direction cannot
    # redefine what counts as a resolved mode. See :func:`effective_modes`.
    scale = float(np.linalg.eigvalsh(0.5 * (cov + cov.T)).max())

    if scale <= 0.0:

        raise ValueError(
            "The statistic has no variance -- there is nothing to test it "
            "against. A one-draw reconstruction will do this."
        )

    if marginalise_constant:

        ones = np.ones_like(delta)

        # Project out the uniform direction from both the residual and the
        # covariance: what is left is the shape of the curve.
        weight = ones / np.sqrt(ones @ ones)

        projector = np.eye(delta.size) - np.outer(weight, weight)

        delta = projector @ delta

        cov = projector @ cov @ projector.T

    values, vectors = effective_modes(cov, tol=tol, scale=scale)

    n_eff = int(values.size)

    if n_eff == 0:

        # Every direction fell below the tolerance: the statistic is constant
        # to machine precision, which happens when it was built out of terms
        # that cancel exactly. There is no deviation to report and no
        # chi-square to report it with.
        return 0.0, 0, 1.0, 0.0

    amplitudes = vectors.T @ delta

    chi2 = float(np.sum(amplitudes**2 / values))

    # Hartlap: the inverse of a covariance estimated from N draws is biased
    # by (N - 1) / (N - p - 2) in p dimensions. Applied only where it is
    # defined; a fit with fewer draws than modes has no business quoting a
    # significance at all.
    n_draws = curve.n_draws

    if n_draws - n_eff - 2 <= 0:

        raise ValueError(
            f"{n_draws} draws cannot support a chi-square over {n_eff} "
            "resolved modes -- the covariance estimate is rank-starved and "
            "the significance would be an artefact of it. Refit with more "
            "draws, or coarsen the grid."
        )

    chi2 *= (n_draws - n_eff - 2) / (n_draws - 1)

    pte = float(stats.chi2.sf(chi2, df=n_eff))

    sigma = float(stats.norm.isf(0.5 * pte)) if pte > 0.0 else np.inf

    return chi2, n_eff, pte, sigma


# ============================================================
# The contract
# ============================================================

class NullTest(ABC):
    """
    Base class for the consistency tests.

    A subclass supplies a name, the value the statistic takes under the null,
    and the arithmetic that builds the statistic. Everything about
    significance, effective degrees of freedom and provenance is handled here,
    so that two tests written months apart report numbers that mean the same
    thing.
    """

    #: Whether the null is *"constant, value unknown"* rather than a specific
    #: number -- see ``marginalise_constant`` in :func:`significance`.
    null_is_free_constant: bool = False

    # ---------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """The test's name, as it should appear in a table."""

    @property
    @abstractmethod
    def null_value(self) -> float:
        """What the statistic equals if the concordance model holds."""

    @abstractmethod
    def statistic(self, *inputs, **kwargs) -> Reconstruction:
        """
        Build the statistic from reconstructions.

        Written as ordinary arithmetic on
        :class:`~CosmoRecon.core.reconstruction.Reconstruction` objects, which
        is what makes the covariance between the inputs propagate exactly
        rather than approximately. Do not summarise anything here.
        """

    # ---------------------------------------------------------

    def evaluate(
        self,
        *inputs,
        support: tuple[float, float] | None = None,
        **kwargs,
    ) -> TestResult:
        """
        Build the statistic and test it against the null.
        """

        curve = self.statistic(*inputs, **kwargs)

        chi2, n_eff, pte, sigma = significance(
            curve,
            self.null_value,
            marginalise_constant=self.null_is_free_constant,
        )

        return TestResult(
            name=self.name,
            curve=curve,
            null_value=self.null_value,
            chi2=chi2,
            n_eff=n_eff,
            pte=pte,
            sigma=sigma,
            provenance=curve.provenance,
            support=support,
        )

    def __repr__(self) -> str:

        return f"<{type(self).__name__}: {self.name} (null = {self.null_value})>"
