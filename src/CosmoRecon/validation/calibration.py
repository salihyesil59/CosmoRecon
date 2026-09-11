"""
Significance, calibrated by simulation.

The nominal significance of a null test --
:func:`~CosmoRecon.consistency.base.significance` -- is a chi-square of the
statistic's posterior mean against its posterior covariance, with the degrees
of freedom counted from that covariance. It is a clean number to compute and the
wrong one to trust, and the error is not small. On mock surveys in which the
null holds **exactly**, at the real releases' redshifts and covariances:

=========================================  ============================
analysis                                   nominally beyond p = 0.05
=========================================  ============================
Om(z), Gaussian process                    0% of realisations
Om(z), Chebyshev in ``y``                  87%
distance duality, Chebyshev in ``ln(1+z)``  95%
curvature, Chebyshev in ``y``              88%
=========================================  ============================

Two different things go wrong, in opposite directions.

**A Gaussian process is wider than its own sampling error** in every direction
its prior still dominates -- the posterior variance exceeds the variance of the
posterior mean across data realisations by a median factor of a hundred or
more -- and those directions are counted as degrees of freedom, so any real
deviation is diluted towards nothing.

**A series with its order left free is biased.** With the truth Lambda-CDM its
posterior mean sits two to four of its own standard deviations away in some
directions. Measured against the *true* sampling covariance, the bias alone
takes it from 6-9% false positives to 64-100%. That is why no better covariance
repairs the test: bias is not variance.

The repair, and why it has two parts
------------------------------------

**The reference distribution.** Fit a null model to the same data
(:mod:`CosmoRecon.validation.nulls`), draw mock datasets from it with the null
true, rerun the whole analysis on each -- the same methods, settings and grid,
through ``refit`` -- and read the p-value off where the data fall among the
mocks.

**The ordering.** Something has to say which realisation is "more extreme",
and the nominal chi-square is the wrong thing to say it. Its largest terms come
from directions in which the posterior is narrowest, which for a biased method
are exactly the directions its bias dominates -- so under the null it is huge
and wildly variable, and a real signal barely moves it. Used as the ordering,
it gives a test with the right size and almost no power: a ``w = -0.6``
universe observed with chronometer errors a tenth of the real ones, where an
oracle likelihood-ratio test has a non-centrality of 44, is detected in 0-15%
of realisations.

So the ordering here is the distance of the statistic's posterior mean from
**the null mocks' own mean, in the null mocks' own covariance**. The mean
subtracts the method's bias at the null; the covariance weighs every direction
by how much the estimate actually moves between realisations of the null, not
by how narrow the posterior claims to be. Each mock's distance is computed from
the *other* mocks, so the data and the mocks are ranked on equal terms. Same
universes, same methods: 90-100% detection, and 0-5% false positives when the
null is true.

End to end, twenty to sixty realisations each of the Om, duality and curvature
analyses, every one calibrated against 50-100 mocks of a null model fitted to
its own data, give 0-8% false positives at ``p < 0.05``, where the nominal test
gave anywhere from 0% (a Gaussian process) to 97% (a free-order series). The
calibrated p-values average 0.40-0.68: close to uniform for the Gaussian
processes, slightly conservative for the series. Fixing the null parameters at
their best fit instead of drawing them from their posterior changes nothing,
because either way the null model sits on the data by construction.

What a calibrated test will then tell you, and the nominal one would not, is
when the data cannot answer. At the real chronometer errors that ``w = -0.6``
universe is a non-centrality of 0.44 away from the best-fitting Lambda-CDM --
no test can see it -- while a third-order series reports it nominally at
4.7 sigma.

What it costs, and what it cannot do
------------------------------------

- **Refits.** ``n_mocks`` of them, of every member of an ensemble -- seconds
  for a series, minutes for a Gaussian process.
- **Mocks enough to estimate a covariance.** The ordering needs the spread of
  the null mocks in as many dimensions as the statistic has redshifts, so the
  mocks must comfortably outnumber the grid points. Too few, and the
  calibration is abandoned rather than reported.
- **A floor.** The smallest calibrated p-value is ``1 / (n_mocks + 1)``: two
  hundred mocks cannot say more than 2.8 sigma, a thousand not more than 3.3.
  At the floor the result is a bound and is reported as one.
- **A model.** The significance is the significance *at* the fitted null
  model, whose name and parameters travel with the result. The mock truths are
  drawn from its parameter posterior, so the calibration does not assume the
  null universe is known better than the data say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from CosmoRecon.typing import Array

from CosmoRecon.core.errors import CosmoReconError, NotRefittableError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import significance

from CosmoRecon.validation.nulls import NullFit, NullModel


__all__ = ["Calibration", "CalibratedComparison", "calibrate", "DEFAULT_N_MOCKS"]


#: Mocks per calibration by default: enough to estimate the null spread of a
#: statistic on a few tens of redshifts, and a floor of 1/201 -- 2.8 sigma --
#: beyond which the answer is a bound.
DEFAULT_N_MOCKS = 200

#: The fraction of mock analyses allowed to fail before a calibration is
#: abandoned. A method that cannot run on a fifth of the universes in which the
#: null holds is not being calibrated by the rest of them.
_MAX_FAILED_FRACTION = 0.05

#: Successful mocks needed beyond the statistic's number of redshifts before
#: its null covariance is worth inverting.
_MOCK_MARGIN = 10

#: Directions in which the null mocks' covariance falls below this fraction of
#: its largest eigenvalue are unresolved and left out of the distance -- the
#: constant direction a marginalised test projects away, and rounding.
_MODE_TOL = 1e-10

_SINGLE = "statistic"

_MIXTURE = "method-marginalised"

#: What a mock analysis may raise without it being a bug: a refusal from a
#: reconstructor or a statistic.
_FAILURES = (CosmoReconError, ValueError, np.linalg.LinAlgError)


# ============================================================
# Results
# ============================================================

@dataclass(frozen=True)
class Calibration:
    """
    One statistic's significance: nominal, and calibrated by simulation.
    """

    name: str

    #: ``(chi2, n_eff, pte, sigma)`` as
    #: :func:`~CosmoRecon.consistency.base.significance` reports it.
    nominal: tuple[float, int, float, float]

    #: The calibrated probability of a statistic at least this far from the
    #: null mocks, with the null true.
    pte: float

    #: The same, as a two-sided Gaussian-equivalent number of sigma.
    sigma: float

    #: No mock was as far out as the data, so :attr:`pte` is an upper bound
    #: and :attr:`sigma` a lower one.
    at_floor: bool

    #: Mock analyses that ran.
    n_mocks: int

    #: Mock analyses that raised, and are not in :attr:`mocks`.
    n_failed: int

    #: The null model the mocks were drawn from, fitted to the data.
    null: NullFit

    #: The data's distance from the null mocks, in their covariance.
    observed: float

    #: Each mock's distance from the other mocks.
    mocks: Array

    #: Why there is no calibrated p-value, if there is none. :attr:`pte` and
    #: :attr:`sigma` are then ``nan``.
    abandoned: str | None = None

    # ---------------------------------------------------------

    @property
    def consistent(self) -> bool:
        """
        Whether the null survives at the conventional 5% level.

        Raises for an abandoned calibration rather than answering: the nominal
        verdict is the one this exists to replace, and there is no other.
        """

        if self.abandoned is not None:

            raise CosmoReconError(
                f"{self.name} has no calibrated p-value: {self.abandoned}"
            )

        return self.pte >= 0.05

    def summary(self) -> str:

        if self.abandoned is not None:

            return (
                f"{self.name}: nominally {self.nominal[3]:.2f} sigma "
                f"(p = {self.nominal[2]:.3g}); not calibrated -- {self.abandoned}"
            )

        bound = "<" if self.at_floor else "="

        more = ">" if self.at_floor else ""

        failed = f", {self.n_failed} failed" if self.n_failed else ""

        return (
            f"{self.name}: nominally {self.nominal[3]:.2f} sigma "
            f"(p = {self.nominal[2]:.3g}); calibrated p {bound} {self.pte:.3g} "
            f"({more}{self.sigma:.2f} sigma) against {self.n_mocks} mocks of "
            f"{self.null.model.name}{failed}"
        )

    def __str__(self) -> str:

        return self.summary()


@dataclass(frozen=True)
class CalibratedComparison:
    """
    A statistic's significance under every member of an ensemble and under the
    mixture, each nominal and calibrated.
    """

    name: str

    per_method: dict[str, Calibration]

    marginalised: Calibration

    null: NullFit

    def summary(self) -> str:

        def line(label: str, c: Calibration) -> str:

            if c.abandoned is not None:

                return (
                    f"   {label:36s} nominal {c.nominal[3]:5.2f} sigma   "
                    f"not calibrated ({c.n_mocks} mocks ran, {c.n_failed} failed)"
                )

            calibrated = f"{'>' if c.at_floor else ' '}{c.sigma:4.2f}"

            return (
                f"   {label:36s} nominal {c.nominal[3]:5.2f} sigma   "
                f"calibrated {calibrated} sigma  "
                f"(p {'<' if c.at_floor else '='} {c.pte:.3g})"
            )

        lines = [
            f"{self.name}, calibrated against {self.marginalised.n_mocks} "
            f"mocks of {self.null.summary()}"
        ]

        ordered = sorted(
            self.per_method.items(),
            key=lambda item: np.inf if item[1].abandoned else item[1].pte,
        )

        for method, c in ordered:
            lines.append(line(method, c))

        lines.append(line(f"-- {_MIXTURE}", self.marginalised))

        return "\n".join(lines)

    def __str__(self) -> str:

        return self.summary()


# ============================================================
# The ordering
# ============================================================

def _summary_vector(curve: Reconstruction, marginalise_constant: bool) -> Array:
    """
    The statistic's posterior mean, with the uniform level removed when the
    null is "some constant".
    """

    mean = np.atleast_1d(np.asarray(curve.mean(), dtype=float))

    if not np.all(np.isfinite(mean)):

        raise ValueError("The statistic's posterior mean is not finite.")

    return mean - mean.mean() if marginalise_constant else mean


def _distance(reference: Array, point: Array) -> float:
    """Squared Mahalanobis distance of ``point`` from the rows of ``reference``."""

    centre = reference.mean(axis=0)

    cov = np.atleast_2d(np.cov(reference, rowvar=False))

    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))

    largest = float(values.max())

    if largest <= 0.0:
        return 0.0

    keep = values > _MODE_TOL * largest

    amplitudes = vectors[:, keep].T @ (point - centre)

    return float(np.sum(amplitudes**2 / values[keep]))


def _distances(mocks: Array, observed: Array) -> tuple[float, Array]:
    """
    The data's distance from the null mocks, and each mock's from the others.

    Leave-one-out for the mocks, so that data and mocks are measured on the
    same terms: a mock scored against a mean and covariance it helped estimate
    would sit closer to them than any new realisation could, and the data would
    look more extreme than they are.
    """

    observed_distance = _distance(mocks, observed)

    mock_distances = np.array([
        _distance(np.delete(mocks, j, axis=0), mocks[j])
        for j in range(mocks.shape[0])
    ])

    return observed_distance, mock_distances


# ============================================================
# Calibration
# ============================================================

def calibrate(
    statistic: Callable,
    fit,
    null_value: float = 0.0,
    *,
    null: NullModel,
    marginalise_constant: bool = False,
    n_mocks: int = DEFAULT_N_MOCKS,
    seed: int | None = None,
    marginalise_null: bool = True,
    name: str = "statistic",
) -> Calibration | CalibratedComparison:
    """
    A null test's significance, calibrated against simulations in which the
    null holds.

    >>> fit = MethodEnsemble([...]).fit(bao.select("DM_over_rs", "DH_over_rs"))
    ...                                                        # doctest: +SKIP
    >>> result = calibrate(                                    # doctest: +SKIP
    ...     lambda s: Curvature().statistic(s["DM_over_rs"], s["DH_over_rs"]),
    ...     fit, 0.0,
    ...     null=LambdaCDM(curved=True),
    ...     marginalise_constant=True,
    ... )
    >>> print(result.summary())                                # doctest: +SKIP

    Parameters
    ----------
    statistic
        The same callable :meth:`~CosmoRecon.ensemble.method.EnsembleFit.significance`
        takes: a whole :class:`~CosmoRecon.reconstructors.base.ReconstructionSet`
        in, the statistic out.
    fit
        A :class:`~CosmoRecon.reconstructors.base.ReconstructionSet` or an
        :class:`~CosmoRecon.ensemble.method.EnsembleFit` that came out of a fit
        and so knows how to rerun it. An ensemble gives a calibration per
        member and for the mixture, from the same mocks.
    null_value, marginalise_constant
        As for the nominal test, which is reported alongside. The calibrated
        test does not use ``null_value``: the null mocks' own mean takes its
        place, which is how the method's bias at the null is taken out.
        ``marginalise_constant`` removes the uniform level from every
        posterior mean before they are compared.
    null
        The null model: fitted to ``fit.datasets``, then simulated.
    n_mocks
        Mock analyses to run. Must comfortably exceed the number of redshifts
        the statistic is evaluated on, and sets the floor on the p-value; see
        the module docstring.
    seed
        Makes the whole calibration reproducible.
    marginalise_null
        Draw each mock's truth from the null model's parameter posterior
        (default) rather than fixing it at the best fit.
    name
        For the summary.
    """

    if n_mocks < 20:

        raise ValueError(
            f"{n_mocks} mocks cannot calibrate anything: the smallest p-value "
            f"they can report is 1/{n_mocks + 1}. Use at least a few hundred "
            "for a number anyone should act on."
        )

    if not getattr(fit, "refittable", False):

        raise NotRefittableError(
            "Calibration reruns the analysis on simulated data, and this "
            "object does not know how it was produced. Pass the result of a "
            "Reconstructor.fit, a MethodEnsemble.fit, or a combination of "
            "those, rather than something assembled by hand."
        )

    null_fit = null.fit(fit.datasets)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31))

    ensemble = hasattr(fit, "members") and hasattr(fit, "marginalised")

    def part(result, key: str):

        if not ensemble:
            return result

        return result.marginalised if key == _MIXTURE else result.members[key]

    keys = [*fit.members, _MIXTURE] if ensemble else [_SINGLE]

    # The data themselves must go through: a statistic that cannot be computed
    # on the observations has nothing to calibrate.
    nominal: dict[str, tuple] = {}

    observed: dict[str, Array] = {}

    for key in keys:

        curve = statistic(part(fit, key))

        nominal[key] = significance(
            curve, null_value, marginalise_constant=marginalise_constant
        )

        observed[key] = _summary_vector(curve, marginalise_constant)

    samples: dict[str, list[Array]] = {key: [] for key in keys}

    failures = dict.fromkeys(keys, 0)

    abandoned: dict[str, str] = {}

    allowed = max(1, int(_MAX_FAILED_FRACTION * n_mocks))

    def give_up(key: str, reason: str, error: BaseException | None = None) -> None:

        abandoned[key] = reason

        if not ensemble:

            raise CosmoReconError(reason) from error

    def fail(key: str, error: BaseException, attempted: int) -> None:

        failures[key] += 1

        if failures[key] > allowed and key not in abandoned:

            give_up(
                key,
                f"{failures[key]} of the first {attempted} mock analyses "
                f"failed, more than the {100 * _MAX_FAILED_FRACTION:.0f}% a "
                "calibration can absorb: the statistic is undefined on too many "
                "universes in which the null holds for the rest of them to "
                f"calibrate it. The most recent failure: {error}",
                error,
            )

    for index in range(n_mocks):

        live = [key for key in keys if key not in abandoned]

        if not live:
            break

        rng = np.random.default_rng([seed, index])

        mock = null_fit.simulate(rng, marginalise=marginalise_null)

        try:
            result = fit.refit(mock, seed=int(rng.integers(2**31 - 1)))

        except _FAILURES as error:

            for key in live:
                fail(key, error, index + 1)

            continue

        # Per statistic, so that one member unable to compute it on some
        # universe does not throw away what every other member made of it.
        for key in live:

            try:
                vector = _summary_vector(statistic(part(result, key)), marginalise_constant)

            except _FAILURES as error:
                fail(key, error, index + 1)

            else:
                samples[key].append(vector)

    def calibrated(key: str, label: str) -> Calibration:

        mocks = np.asarray(samples[key], dtype=float)

        needed = observed[key].size + _MOCK_MARGIN

        if key not in abandoned and mocks.shape[0] < needed:

            give_up(
                key,
                f"only {mocks.shape[0]} mock analyses ran, for a statistic on "
                f"{observed[key].size} redshift(s): the spread of the null mocks "
                f"in that many dimensions needs at least {needed}. Run more "
                "mocks, or evaluate the statistic on a coarser grid.",
            )

        if key in abandoned:

            return Calibration(
                name=label,
                nominal=nominal[key],
                pte=float("nan"),
                sigma=float("nan"),
                at_floor=False,
                n_mocks=int(mocks.shape[0]),
                n_failed=failures[key],
                null=null_fit,
                observed=float("nan"),
                mocks=np.array([]),
                abandoned=abandoned[key],
            )

        distance, mock_distances = _distances(mocks, observed[key])

        count = int(np.sum(mock_distances >= distance))

        pte = (1.0 + count) / (mock_distances.size + 1.0)

        return Calibration(
            name=label,
            nominal=nominal[key],
            pte=float(pte),
            sigma=float(stats.norm.isf(0.5 * pte)),
            at_floor=count == 0,
            n_mocks=int(mock_distances.size),
            n_failed=failures[key],
            null=null_fit,
            observed=distance,
            mocks=mock_distances,
        )

    if not ensemble:
        return calibrated(_SINGLE, name)

    return CalibratedComparison(
        name=name,
        per_method={method: calibrated(method, method) for method in fit.members},
        marginalised=calibrated(_MIXTURE, _MIXTURE),
        null=null_fit,
    )
