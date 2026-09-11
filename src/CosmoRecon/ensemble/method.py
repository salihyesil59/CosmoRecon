"""
Running one question through several methods, and reporting the disagreement.

Everything below is possible because :mod:`CosmoRecon.consistency` and
:mod:`CosmoRecon.inverse` speak only
:class:`~CosmoRecon.core.reconstruction.Reconstruction` and never ask which
reconstructor produced it. Swap the reconstructor, keep the analysis, difference
the answers.

Two outputs, and the second is the one nothing else produces.

**A method-marginalised posterior.** The members' draws pooled by weight. It is
a full :class:`~CosmoRecon.core.reconstruction.Reconstruction`, not a summary:
it can be regridded, differentiated analytically and fed to a null test, and
each of its draws is still one member's realisation of a function, so the
guarantees the whole library rests on survive the mixture.

**A significance, before and after.** The same statistic evaluated under each
member alone and under the mixture. Published reconstructions quote the first
kind of number. The gap between them is how much of a detection was the method.

On weighting. Equal is the default. Bayesian evidence compares models of the
same data under the same likelihood, and two reconstruction methods with
different function-space priors are that only loosely -- a Chebyshev series and
a Matern process are not competing hypotheses about the universe, they are
competing descriptions of a curve. Evidence weighting is available for the
cases where the members really are comparable, and it refuses outright when a
member has no evidence, rather than quietly averaging over the subset that
happens to define one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from CosmoRecon.typing import Array, Redshift

from CosmoRecon.core.errors import CosmoReconError, GridMismatchError
from CosmoRecon.core.provenance import Provenance, new_origin
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import significance as _significance

from CosmoRecon.ensemble.budget import VarianceBudget, total_variance

from CosmoRecon.reconstructors.base import (
    DEFAULT_N_DRAWS,
    ReconstructionSet,
    Reconstructor,
    combine_independent,
)


__all__ = ["MethodEnsemble", "EnsembleFit", "SignificanceComparison"]


# ============================================================
# The mixture predictor
# ============================================================

class _MixturePaths:
    """
    A pooled set of draws that still knows which function each one is.

    Draw ``k`` of the mixture is row ``row_of_draw[k]`` of member
    ``member_of_draw[k]``, and both arrays are fixed at fit time -- so the
    mixture satisfies the determinism clause of the
    :class:`~CosmoRecon.core.reconstruction.Predictor` contract for the same
    reason its members do, and can be regridded and differentiated like any
    other reconstruction.
    """

    __slots__ = ("_predictors", "_names", "_member_of_draw", "_row_of_draw")

    def __init__(
        self,
        predictors: list,
        names: list[str],
        member_of_draw: Array,
        row_of_draw: Array,
    ) -> None:

        self._predictors = predictors
        self._names = names
        self._member_of_draw = member_of_draw
        self._row_of_draw = row_of_draw

    def __call__(self, z: Array, *, derivative: int = 0) -> Array:

        z = np.atleast_1d(np.asarray(z, dtype=float))

        out = np.empty((self._member_of_draw.size, z.size), dtype=float)

        for index in np.unique(self._member_of_draw):

            rows = np.flatnonzero(self._member_of_draw == index)

            try:
                values = self._predictors[index](z, derivative=derivative)

            except CosmoReconError as error:

                # A mixture is only as differentiable as its roughest member,
                # and which member refused is the useful half of the message.
                raise type(error)(
                    f"Member {self._names[index]!r} of this ensemble cannot "
                    f"do it: {error}"
                ) from error

            out[rows] = values[self._row_of_draw[rows]]

        return out


# ============================================================
# Significance, before and after
# ============================================================

@dataclass(frozen=True, slots=True)
class SignificanceComparison:
    """
    One statistic's significance under each method alone, and under the
    mixture.

    The gap is the point. A deviation that is four sigma under one kernel and
    two under the method-marginalised posterior was, to that extent, a
    property of the kernel.
    """

    #: Method name -> ``(chi2, n_eff, pte, sigma)``.
    per_method: dict[str, tuple[float, int, float, float]]

    #: The same, for the method-marginalised posterior.
    marginalised: tuple[float, int, float, float]

    #: What the statistic is called, for the summary line.
    name: str = "statistic"

    # ---------------------------------------------------------

    @property
    def best_single(self) -> tuple[str, float]:
        """
        The member reporting the largest significance, and how large.

        Named ``best`` in the sense a reader of the literature would mean it:
        the number a paper would have quoted had it chosen that method.
        """

        name = max(self.per_method, key=lambda k: self.per_method[k][3])

        return name, self.per_method[name][3]

    @property
    def deflation(self) -> float:
        """
        How much the strongest single-method significance shrinks once the
        method is marginalised over.

        Greater than one means at least one member was reporting a detection
        the ensemble does not support.
        """

        _, strongest = self.best_single

        after = self.marginalised[3]

        if after <= 0.0:
            return float("inf") if strongest > 0.0 else 1.0

        return float(strongest / after)

    def summary(self) -> str:

        name, strongest = self.best_single

        after = self.marginalised[3]

        lines = [f"{self.name}:"]

        for method, (_, n_eff, pte, sigma) in sorted(
            self.per_method.items(), key=lambda item: -item[1][3]
        ):
            lines.append(
                f"   {method:24s} {sigma:5.2f} sigma  (p = {pte:.3g},"
                f" {n_eff} eff. dof)"
            )

        lines.append(
            f"   {'-- method-marginalised':24s} {after:5.2f} sigma  "
            f"(p = {self.marginalised[2]:.3g}, {self.marginalised[1]} eff. dof)"
        )

        lines.append(
            f"   strongest single method ({name}) reports {strongest:.2f} "
            f"sigma; marginalising the method leaves {after:.2f}"
        )

        return "\n".join(lines)

    def __str__(self) -> str:

        return self.summary()


# ============================================================
# The result of an ensemble fit
# ============================================================

class EnsembleFit:
    """
    What every member made of the same data, and what their disagreement adds
    up to.

    A member's fit is a :class:`~CosmoRecon.reconstructors.base.ReconstructionSet`,
    not a single curve, because a joint fit produces several correlated
    observables and dropping all but one of them would be exactly the silent
    error this library exists to prevent. So does the pooled result: the
    mixture assigns one ``(member, realisation)`` pair per draw and applies it
    to *every* observable, which is what keeps ``D_M/r_d`` and ``D_H/r_d``
    aligned across the mixture as they were within each member.
    """

    __slots__ = ("_sets", "_weights", "_grid", "_observables", "_marginalised")

    def __init__(
        self,
        sets: Mapping[str, "ReconstructionSet"],
        weights: Mapping[str, float],
        grid: Array,
        observables: tuple[str, ...],
        marginalised: "ReconstructionSet",
    ) -> None:

        self._sets = dict(sets)
        self._weights = dict(weights)
        self._grid = grid
        self._observables = observables
        self._marginalised = marginalised

    # ---------------------------------------------------------

    @property
    def members(self) -> dict[str, "ReconstructionSet"]:
        """Each method's own fit, on the shared grid."""

        return dict(self._sets)

    @property
    def weights(self) -> dict[str, float]:

        return dict(self._weights)

    @property
    def observables(self) -> tuple[str, ...]:
        """What was reconstructed -- one name, or several from a joint fit."""

        return self._observables

    @property
    def marginalised(self) -> "ReconstructionSet":
        """
        The method-marginalised posterior: every member's draws, pooled by
        weight.

        A full set of reconstructions, not a summary. Each regrids, each
        differentiates, and a null test built on them is a null test that does
        not depend on which method was picked. In a joint fit the observables
        stay aligned with each other, because one draw of the mixture is one
        realisation of one member.
        """

        return self._marginalised

    # ---------------------------------------------------------

    def _one(self, observable: str | None) -> str:

        if observable is not None:

            if observable not in self._observables:

                raise KeyError(
                    f"This ensemble reconstructed {list(self._observables)}, "
                    f"not {observable!r}."
                )

            return observable

        if len(self._observables) == 1:
            return self._observables[0]

        raise ValueError(
            f"This is a joint fit of {list(self._observables)}, so a budget "
            "has to name which one it is for. There is no single number "
            "covering both."
        )

    def curves(self, observable: str | None = None) -> dict[str, Reconstruction]:
        """One observable's reconstruction from each method."""

        name = self._one(observable)

        return {method: fit[name] for method, fit in self._sets.items()}

    def budget(self, observable: str | None = None) -> VarianceBudget:
        """
        The law-of-total-variance split across the members, for one
        observable.

        See :mod:`CosmoRecon.ensemble.budget`.
        """

        curves = self.curves(observable)

        return total_variance(
            means={name: r.mean() for name, r in curves.items()},
            variances={name: r.var() for name, r in curves.items()},
            weights=self._weights,
            z=self._grid,
        )

    def significance(
        self,
        statistic: Callable[["ReconstructionSet"], Reconstruction],
        null_value: float,
        *,
        marginalise_constant: bool = False,
        name: str = "statistic",
    ) -> SignificanceComparison:
        """
        Evaluate a null test under every member and under the mixture.

        ``statistic`` takes a whole
        :class:`~CosmoRecon.reconstructors.base.ReconstructionSet` and returns
        the ``z``-dependent quantity to test -- so a test needing two
        correlated observables is written the same way as one needing a single
        curve:

        >>> fit.significance(                                    # doctest: +SKIP
        ...     lambda s: Om().statistic(s["H"]),
        ...     null_value=0.3,
        ...     marginalise_constant=True,
        ... )
        >>> fit.significance(                                    # doctest: +SKIP
        ...     lambda s: Curvature().statistic(s["DM_over_rs"], s["DH_over_rs"]),
        ...     null_value=0.0,
        ...     marginalise_constant=True,
        ... )

        The same callable is applied to each member's own fit and to the
        pooled one, so nothing about the test changes between the two numbers
        except which posterior it was evaluated on. That is what makes the
        difference attributable to the method rather than to the analysis.
        """

        per_method = {
            method: _significance(
                statistic(fit),
                null_value,
                marginalise_constant=marginalise_constant,
            )
            for method, fit in self._sets.items()
        }

        pooled = _significance(
            statistic(self._marginalised),
            null_value,
            marginalise_constant=marginalise_constant,
        )

        return SignificanceComparison(
            per_method=per_method,
            marginalised=pooled,
            name=name,
        )

    def with_independent(self, other: "EnsembleFit") -> "EnsembleFit":
        """
        This ensemble fit and another over **different, independent data**,
        as one.

        >>> sn = ensemble.fit(reduced_modulus(union3()), grid=grid)
        ...                                                        # doctest: +SKIP
        >>> bao = ensemble.fit(desi.select("DM_over_rs"), grid=grid)
        ...                                                        # doctest: +SKIP
        >>> both = sn.with_independent(bao)                        # doctest: +SKIP
        >>> both.significance(                                     # doctest: +SKIP
        ...     lambda s: Duality().statistic(s["mu_reduced"], s["DM_over_rs"]),
        ...     1.0, marginalise_constant=True,
        ... )

        What a two-dataset null test needs from an ensemble, and why it is not
        just the two marginalised posteriors side by side.

        **A method is a choice made once.** An analyst who reconstructs the
        supernovae with a Matern process reconstructs the BAO distances with
        one too. So member ``m`` of the result is member ``m`` of this fit
        paired with member ``m`` of the other -- see
        :func:`~CosmoRecon.reconstructors.base.combine_independent` -- and its
        significance is the number that analyst would have quoted.

        **The mixture is a mixture of those pairs**, not the product of two
        separate mixtures. Pooling each side on its own would pair a Gaussian
        process on one dataset with a polynomial on the other in most draws,
        and the "method-marginalised" number would then describe analyses
        nobody runs. Pooling the pairs keeps the before-and-after comparison
        between like and like.

        Weights multiply, since the two datasets are independent: equal
        weights stay equal, and evidence weights combine the way evidences of
        independent data do.
        """

        if set(self._sets) != set(other._sets):

            raise ValueError(
                f"The two ensembles have different members "
                f"({sorted(self._sets)} and {sorted(other._sets)}). Members are "
                "paired by method, so both datasets have to be reconstructed "
                "by the same set of methods."
            )

        if self._grid.size != other._grid.size or not np.allclose(
            self._grid, other._grid
        ):

            raise GridMismatchError(
                "The two ensembles were fitted on different grids. Fit both "
                "on one grid inside the redshift range the datasets share -- "
                "which is also the only range a joint statement about them "
                "means anything on; see core.grid.common_support."
            )

        names = list(self._sets)

        sets = {
            name: combine_independent(self._sets[name], other._sets[name])
            for name in names
        }

        raw = {name: self._weights[name] * other._weights[name] for name in names}

        total = sum(raw.values())

        weights = {name: value / total for name, value in raw.items()}

        observables = tuple(sorted(set(self._observables) | set(other._observables)))

        pooled = self._marginalised.provenance

        marginalised = _pool(
            sets,
            weights,
            observables,
            n_draws=min(pooled.n_draws, other._marginalised.provenance.n_draws),
            seed=int(pooled.seed or 0),
            method=pooled.method,
        )

        return EnsembleFit(
            sets=sets,
            weights=weights,
            grid=self._grid,
            observables=observables,
            marginalised=marginalised,
        )

    # ---------------------------------------------------------

    def __len__(self) -> int:

        return len(self._sets)

    def __repr__(self) -> str:

        return (
            f"<EnsembleFit {len(self)} methods of "
            f"{list(self._observables)} on {self._grid.size} redshifts>"
        )


# ============================================================
# The ensemble
# ============================================================

class MethodEnsemble:
    """
    Several reconstructors, fitted to one dataset, compared.

    >>> ensemble = MethodEnsemble([                              # doctest: +SKIP
    ...     GaussianProcess(kernel="matern"),
    ...     GaussianProcess(kernel="squared_exponential"),
    ...     Cosmography("y"),
    ...     Cosmography(pade=(2, 1)),
    ... ])
    >>> fit = ensemble.fit(chronometers())                       # doctest: +SKIP
    >>> print(fit.budget().summary())                            # doctest: +SKIP

    Parameters
    ----------
    members
        The reconstructors. Two at least -- with one there is nothing to
        disagree and the methodological term is zero by construction rather
        than by measurement.
    weights
        ``"equal"`` (default) or ``"evidence"``. See the module docstring for
        why equal is the default; evidence weighting raises if any member does
        not define one.
    names
        Optional labels. Defaults to each member's ``describe()``, which is
        what appears in captions and is required to distinguish two
        configurations of the same class.
    """

    def __init__(
        self,
        members: Sequence[Reconstructor],
        *,
        weights: str = "equal",
        names: Sequence[str] | None = None,
    ) -> None:

        members = list(members)

        if len(members) < 2:

            raise ValueError(
                f"An ensemble needs at least two methods, got {len(members)}. "
                "With one member the method-variance term is zero because "
                "there is nothing to compare, not because the methods agree."
            )

        if str(weights) not in ("equal", "evidence"):

            raise ValueError(
                f"Unknown weighting {weights!r}; use 'equal' or 'evidence'."
            )

        self.members = members

        self.weighting = str(weights)

        if names is None:
            names = [member.describe() for member in members]

        names = list(names)

        if len(set(names)) != len(names):

            raise ValueError(
                f"Two members share a name: {names}. Names are what tells "
                "one configuration from another in a budget or a caption, so "
                "they have to be distinct -- pass names= explicitly."
            )

        self.names = names

    # ---------------------------------------------------------

    def describe(self) -> str:

        return f"ensemble of {len(self.members)} ({self.weighting}-weighted)"

    def fit(
        self,
        data,
        *,
        grid: Redshift | None = None,
        n_draws: int = DEFAULT_N_DRAWS,
        seed: int | None = None,
    ) -> EnsembleFit:
        """
        Fit every member to the same data, on the same grid, and pool them.

        A member that cannot fit these data raises rather than being skipped.
        Dropping it would change the budget silently, and the budget is the
        number the caller came for -- a spread across four methods reported as
        though it were across five is exactly the kind of quiet error this
        library exists to prevent.
        """

        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31))

        fits = []

        for name, member in zip(self.names, self.members, strict=True):

            try:
                fits.append(member.fit(data, grid=grid, n_draws=n_draws, seed=seed))

            except CosmoReconError as error:

                raise type(error)(
                    f"Ensemble member {name!r} could not fit these data, and "
                    "a member cannot be silently dropped -- the spread across "
                    "the others would then be reported as though it were "
                    f"across all of them. The member said: {error}"
                ) from error

        observables = {tuple(sorted(fit)) for fit in fits}

        if len(observables) != 1:

            raise ValueError(
                f"Members produced different observables: {observables}. "
                "An ensemble compares methods on one question."
            )

        names = tuple(next(iter(observables)))

        sets = dict(zip(self.names, fits, strict=True))

        weights = self._resolve_weights()

        grid_array = np.asarray(fits[0][names[0]].z, dtype=float)

        marginalised = _pool(
            sets, weights, names, n_draws=n_draws, seed=seed, method=self.describe()
        )

        return EnsembleFit(
            sets=sets,
            weights=weights,
            grid=grid_array,
            observables=names,
            marginalised=marginalised,
        )

    # ---------------------------------------------------------

    def _resolve_weights(self) -> dict[str, float]:

        if self.weighting == "equal":

            return {name: 1.0 / len(self.names) for name in self.names}

        missing = [
            name
            for name, member in zip(self.names, self.members, strict=True)
            if not member.provides_evidence
        ]

        if missing:

            from CosmoRecon.core.errors import EvidenceUnavailableError

            raise EvidenceUnavailableError(
                f"Evidence weighting was asked for, but {missing} do not "
                "define a Bayesian evidence. Weighting the rest would report "
                "a spread across fewer methods than were requested. Use "
                "weights='equal', or drop those members explicitly."
            )

        log_evidence = np.array(
            [member.log_evidence for member in self.members], dtype=float
        )

        weights = np.exp(log_evidence - logsumexp(log_evidence))

        return dict(zip(self.names, weights.tolist(), strict=True))

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return f"<MethodEnsemble {self.names} ({self.weighting}-weighted)>"


# ============================================================
# Pooling
# ============================================================

def _pool(
    sets: dict[str, "ReconstructionSet"],
    weights: dict[str, float],
    observables: tuple[str, ...],
    *,
    n_draws: int,
    seed: int,
    method: str,
) -> "ReconstructionSet":
    """
    Build the method-marginalised posterior.

    Each member contributes a share of the pooled draws proportional to its
    weight, sampled without replacement from its own so that no realisation
    appears twice.

    The ``(member, realisation)`` assignment is chosen **once** and applied to
    every observable. That is what keeps a joint fit's observables aligned
    across the mixture: draw ``k`` of the pooled ``D_M/r_d`` and draw ``k`` of
    the pooled ``D_H/r_d`` are the same realisation of the same member, so
    their correlation survives pooling exactly as it survived the fit.
    Assigning them separately would destroy it and nothing downstream would
    notice.

    The pooled curves keep a predictor -- they dispatch per draw to the member
    that produced them -- so the mixture is regriddable and differentiable
    rather than a fixed table of numbers.
    """

    rng = np.random.default_rng([seed, 0xC05E])

    names = list(sets)

    shares = np.array([weights[name] for name in names], dtype=float)

    counts = np.floor(shares * n_draws).astype(int)

    # Hand the rounding remainder to the heaviest members, so the pooled size
    # is exactly what was asked for.
    for index in np.argsort(-shares)[: n_draws - int(counts.sum())]:
        counts[index] += 1

    member_of_draw = np.concatenate(
        [np.full(count, index) for index, count in enumerate(counts)]
    )

    available = [sets[name][observables[0]].n_draws for name in names]

    row_of_draw = np.concatenate([
        rng.choice(size, size=count, replace=False)
        if count <= size
        else rng.integers(0, size, size=count)
        for size, count in zip(available, counts, strict=True)
    ])

    origin = new_origin()

    members = {}

    for observable in observables:

        curves = [sets[name][observable] for name in names]

        provenance = Provenance(
            method=method,
            data=curves[0].provenance.datasets(),
            hyperparameters={
                "members": names,
                "weights": {n: float(weights[n]) for n in names},
            },
            seed=seed,
            n_draws=int(member_of_draw.size),
            parents=tuple(curve.provenance for curve in curves),
        )

        predictors = [curve._predictor for curve in curves]

        if any(predictor is None for predictor in predictors):

            # Some member holds draws rather than a function. The mixture then
            # can only do the same, and says so through the usual channel.
            draws = np.concatenate([
                curve.draws[row_of_draw[member_of_draw == index]]
                for index, curve in enumerate(curves)
            ])

            members[observable] = Reconstruction.from_draws(
                curves[0].z,
                draws,
                provenance=provenance,
                origin=origin,
                label=observable,
                unit=curves[0].unit,
            )

        else:

            members[observable] = Reconstruction.from_predictor(
                curves[0].z,
                _MixturePaths(predictors, names, member_of_draw, row_of_draw),
                provenance=provenance,
                origin=origin,
                label=observable,
                unit=curves[0].unit,
            )

    first = sets[names[0]]

    return ReconstructionSet(
        members,
        origin=origin,
        support=first.support,
        provenance=Provenance(
            method=method,
            data=first.provenance.datasets(),
            seed=seed,
            n_draws=int(member_of_draw.size),
        ),
    )
