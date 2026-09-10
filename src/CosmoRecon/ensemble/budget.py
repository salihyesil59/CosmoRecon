"""
How much of the error bar is a measurement, and how much is a choice.

This is the library's thesis reduced to one identity. For an ensemble of
reconstruction methods with weights ``w_m``, the law of total variance splits
the spread of the answer in two:

    Var_total(z)  =  sum_m w_m Var_m(z)          <- statistical
                  +  sum_m w_m [mu_m(z) - mu(z)]^2   <- methodological

The first term is what the data do not determine. The second is what the
*analyst* did not determine -- the part that would move if the same data went
through a Chebyshev expansion instead of a Matern kernel. Published
reconstructions quote the first and omit the second, which is why the
literature contains kernel-dependence papers rather than kernel-marginalised
results.

:meth:`VarianceBudget.method_fraction` is the number this library exists to
produce: the fraction of the quoted uncertainty that is methodology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CosmoRecon.typing import Array


__all__ = ["VarianceBudget", "total_variance"]


@dataclass(frozen=True, slots=True)
class VarianceBudget:
    """
    The two halves of an ensemble's uncertainty, redshift by redshift.
    """

    #: The grid these are defined on.
    z: Array

    #: Weighted mean of the within-method variances. What the data leave
    #: undetermined once a method is fixed.
    statistical: Array

    #: Weighted variance of the between-method means. What the choice of
    #: method contributes on top.
    methodological: Array

    #: Their sum -- the variance of the method-marginalised posterior.
    total: Array

    #: Method name -> weight, as used. Kept so a figure can say whether the
    #: split was equal-weighted or evidence-weighted.
    weights: dict[str, float]

    #: Method name -> its own mean curve, for the per-method overlay that
    #: makes a budget plot readable.
    means: dict[str, Array]

    # ---------------------------------------------------------

    def method_fraction(self) -> Array:
        """
        The share of the total variance that is methodological, in ``[0, 1]``.

        Read it as: *this fraction of my error bar would go away if the
        community agreed on one reconstruction method -- and would not go
        away by taking more data.*
        """

        with np.errstate(invalid="ignore", divide="ignore"):

            frac = np.where(
                self.total > 0.0,
                self.methodological / self.total,
                0.0,
            )

        return np.asarray(frac, dtype=float)

    def inflation(self) -> Array:
        """
        How much wider the honest error bar is than the single-method one:
        ``sqrt(total / statistical)``.

        The number to put in an abstract. A value of 1.6 means a published
        interval from any one method is 60% too tight.
        """

        with np.errstate(invalid="ignore", divide="ignore"):

            ratio = np.where(
                self.statistical > 0.0,
                np.sqrt(self.total / self.statistical),
                np.nan,
            )

        return np.asarray(ratio, dtype=float)

    # ---------------------------------------------------------

    def summary(self) -> str:
        """One line: the range of the method share across redshift."""

        frac = self.method_fraction()

        worst = int(np.argmax(frac))

        return (
            f"method variance is {100 * frac.mean():.0f}% of the total on "
            f"average, peaking at {100 * frac[worst]:.0f}% at z = "
            f"{self.z[worst]:.3g}; error bars inflate by a factor "
            f"{np.nanmedian(self.inflation()):.2f} (median)"
        )

    def __str__(self) -> str:

        return self.summary()


# ============================================================

def total_variance(
    means: dict[str, Array],
    variances: dict[str, Array],
    weights: dict[str, float],
    z: Array,
) -> VarianceBudget:
    """
    Apply the law of total variance to a set of per-method summaries.

    Kept as a free function so it can be used on anything with a mean and a
    variance per method -- including a null test's significance curve, not
    only a reconstruction.
    """

    names = list(means)

    if set(names) != set(variances) or set(names) != set(weights):

        raise ValueError(
            "means, variances and weights must cover the same methods; got "
            f"{sorted(means)}, {sorted(variances)}, {sorted(weights)}."
        )

    if len(names) < 2:

        raise ValueError(
            "A method-variance budget needs at least two methods -- with one "
            "member there is nothing to disagree, and the methodological "
            "term is zero by construction rather than by measurement."
        )

    w = np.array([weights[n] for n in names], dtype=float)

    if np.any(w < 0.0):

        raise ValueError("Weights must be non-negative.")

    if not np.isfinite(w).all() or w.sum() <= 0.0:

        raise ValueError("Weights must be finite and sum to something positive.")

    w = w / w.sum()

    mu = np.stack([np.asarray(means[n], dtype=float) for n in names])

    var = np.stack([np.asarray(variances[n], dtype=float) for n in names])

    pooled_mean = np.tensordot(w, mu, axes=(0, 0))

    statistical = np.tensordot(w, var, axes=(0, 0))

    methodological = np.tensordot(w, (mu - pooled_mean) ** 2, axes=(0, 0))

    return VarianceBudget(
        z=np.asarray(z, dtype=float),
        statistical=statistical,
        methodological=methodological,
        total=statistical + methodological,
        weights={n: float(wi) for n, wi in zip(names, w, strict=True)},
        means={n: np.asarray(means[n], dtype=float) for n in names},
    )
