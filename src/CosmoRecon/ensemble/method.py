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

from CosmoRecon.core.errors import CosmoReconError
from CosmoRecon.core.provenance import Provenance, new_origin
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import significance as _significance

from CosmoRecon.ensemble.budget import VarianceBudget, total_variance

from CosmoRecon.reconstructors.base import DEFAULT_N_DRAWS, Reconstructor


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
    """

    __slots__ = ("_members", "_weights", "_grid", "_observable", "_marginalised")

    def __init__(
        self,
        members: Mapping[str, Reconstruction],
        weights: Mapping[str, float],
        grid: Array,
        observable: str,
        marginalised: Reconstruction,
    ) -> None:

        self._members = dict(members)
        self._weights = dict(weights)
        self._grid = grid
        self._observable = observable
        self._marginalised = marginalised

    # ---------------------------------------------------------

    @property
    def members(self) -> dict[str, Reconstruction]:
        """Each method's own reconstruction, on the shared grid."""

        return dict(self._members)

    @property
    def weights(self) -> dict[str, float]:

        return dict(self._weights)

    @property
    def observable(self) -> str:

        return self._observable

    @property
    def marginalised(self) -> Reconstruction:
        """
        The method-marginalised posterior: every member's draws, pooled by
        weight.

        A full reconstruction, not a summary. It regrids, it differentiates,
        and a null test built on it is a null test that does not depend on
        which method was picked.
        """

        return self._marginalised

    # ---------------------------------------------------------

    def budget(self) -> VarianceBudget:
        """
        The law-of-total-variance split across the members.

        See :mod:`CosmoRecon.ensemble.budget`.
        """

        return total_variance(
            means={name: r.mean() for name, r in self._members.items()},
            variances={name: r.var() for name, r in self._members.items()},
            weights=self._weights,
            z=self._grid,
        )

    def significance(
        self,
        statistic: Callable[[Reconstruction], Reconstruction],
        null_value: float,
        *,
        marginalise_constant: bool = False,
        name: str = "statistic",
    ) -> SignificanceComparison:
        """
        Evaluate a null test under every member and under the mixture.

        ``statistic`` takes a reconstruction and returns the ``z``-dependent
        quantity to test -- ordinary arithmetic, exactly as it would be
        written for a single method:

        >>> fit.significance(                                    # doctest: +SKIP
        ...     lambda H: ((H / H.at(0.0)) ** 2 - 1) / ((1 + z) ** 3 - 1),
        ...     null_value=0.3,
        ...     marginalise_constant=True,
        ... )

        The same callable is applied to each member's own reconstruction and
        to the pooled one, so nothing about the test changes between the two
        numbers except which posterior it was evaluated on. That is what makes
        the difference attributable to the method rather than to the analysis.
        """

        per_method = {
            method: _significance(
                statistic(curve),
                null_value,
                marginalise_constant=marginalise_constant,
            )
            for method, curve in self._members.items()
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

    # ---------------------------------------------------------

    def __len__(self) -> int:

        return len(self._members)

    def __repr__(self) -> str:

        return (
            f"<EnsembleFit {len(self)} methods of {self._observable} "
            f"on {self._grid.size} redshifts>"
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

        observable = next(iter(observables))[0]

        members = {
            name: fit[observable]
            for name, fit in zip(self.names, fits, strict=True)
        }

        weights = self._resolve_weights()

        grid_array = next(iter(members.values())).z

        marginalised = self._pool(members, weights, n_draws, seed, observable)

        return EnsembleFit(
            members=members,
            weights=weights,
            grid=np.asarray(grid_array, dtype=float),
            observable=observable,
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

    def _pool(
        self,
        members: dict[str, Reconstruction],
        weights: dict[str, float],
        n_draws: int,
        seed: int,
        observable: str,
    ) -> Reconstruction:
        """
        Build the method-marginalised posterior.

        Each member contributes a share of the pooled draws proportional to
        its weight, sampled without replacement from its own so that no
        realisation appears twice. The pooled object keeps a predictor -- it
        dispatches per draw to the member that produced it -- so the mixture
        is regriddable and differentiable rather than a fixed table of
        numbers.
        """

        rng = np.random.default_rng([seed, 0xC05E])

        names = list(members)

        shares = np.array([weights[name] for name in names], dtype=float)

        counts = np.floor(shares * n_draws).astype(int)

        # Hand the rounding remainder to the heaviest members, so the pooled
        # size is exactly what was asked for.
        for index in np.argsort(-shares)[: n_draws - int(counts.sum())]:
            counts[index] += 1

        member_of_draw = np.concatenate(
            [np.full(count, index) for index, count in enumerate(counts)]
        )

        row_of_draw = np.concatenate(
            [
                rng.choice(members[name].n_draws, size=count, replace=False)
                if count <= members[name].n_draws
                else rng.integers(0, members[name].n_draws, size=count)
                for name, count in zip(names, counts, strict=True)
            ]
        )

        predictors = [members[name]._predictor for name in names]

        provenance = Provenance(
            method=self.describe(),
            data=members[names[0]].provenance.data,
            hyperparameters={
                "members": names,
                "weights": {name: float(weights[name]) for name in names},
            },
            seed=seed,
            n_draws=int(member_of_draw.size),
            parents=tuple(members[name].provenance for name in names),
        )

        if any(predictor is None for predictor in predictors):

            # Some member holds draws rather than a function. The mixture then
            # can only do the same, and says so through the usual channel.
            draws = np.concatenate(
                [
                    members[name].draws[row_of_draw[member_of_draw == index]]
                    for index, name in enumerate(names)
                ]
            )

            return Reconstruction.from_draws(
                members[names[0]].z,
                draws,
                provenance=provenance,
                origin=new_origin(),
                label=observable,
                unit=members[names[0]].unit,
            )

        return Reconstruction.from_predictor(
            members[names[0]].z,
            _MixturePaths(predictors, names, member_of_draw, row_of_draw),
            provenance=provenance,
            origin=new_origin(),
            label=observable,
            unit=members[names[0]].unit,
        )

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return f"<MethodEnsemble {self.names} ({self.weighting}-weighted)>"
