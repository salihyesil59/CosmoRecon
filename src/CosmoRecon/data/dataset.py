"""
Dataset containers.

Small, frozen, and opinionated about two things a plain array of numbers
cannot express.

**What may not be combined.** Two supernova compilations built from
overlapping objects are not two independent measurements, and a reconstruction
that treats them as such reports an interval that is too tight by a factor
nobody can work out afterwards. Every dataset here declares what it excludes,
and :func:`check_combination` refuses the pairing rather than leaving it to be
noticed in review.

**What a set of numbers actually measures.** A BAO release is not a
measurement of one function of redshift: it is ``D_M/r_d`` and ``D_H/r_d``
together, correlated, at shared redshifts. Handing all thirteen numbers to a
reconstructor as though they traced one curve would fit a function through two
different quantities, so :class:`MultiObservableDataset` refuses that and says
what to do instead.

There are two right answers, and which one is right depends on the question.
``select("DM_over_rs")`` gives one observable with its own covariance block --
one function, fittable by any method, with its correlation to the others
dropped. ``select("DM_over_rs", "DH_over_rs")`` keeps that correlation and
gives something a **joint** reconstruction can consume, which is what the
curvature and distance-duality tests need in order to be defined at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from CosmoRecon.typing import Array

from CosmoRecon.core.errors import DataError


__all__ = [
    "Dataset",
    "MultiObservableDataset",
    "check_combination",
]


# ============================================================
# One observable
# ============================================================

@dataclass(frozen=True, slots=True)
class Dataset:
    """
    Measurements of one quantity as a function of redshift, with their
    covariance.

    This is what a reconstructor consumes. The attribute names are the ones
    :func:`~CosmoRecon.reconstructors.base.unpack_dataset` looks for, so a
    user's own arrays wrapped in this class work everywhere the bundled data
    do.
    """

    #: Measurement redshifts, ascending.
    z: Array

    #: The measured values.
    y: Array

    #: Full covariance, ``(n, n)``. Diagonal where the release published only
    #: uncertainties -- never invented off-diagonal terms.
    cov: Array

    #: What is being measured: ``"H"``, ``"mu"``, ``"fsigma8"``,
    #: ``"DM_over_rs"``. Becomes the name of the reconstruction.
    observable: str

    #: Physical unit, for plot axes. Empty for a dimensionless quantity.
    unit: str = ""

    #: Short identifier, e.g. ``"CC(Favale2023)"``. Goes into provenance.
    name: str = ""

    #: Where the numbers come from, in a form that could go in a bibliography.
    reference: str = ""

    #: Anything a user has to know to interpret the result -- an arbitrary
    #: zero point, a fiducial assumed in the reduction, a known
    #: model-dependence. Printed by :meth:`describe`.
    note: str = ""

    #: Names of datasets this one may not be combined with, because they
    #: measure the same objects.
    excludes: tuple[str, ...] = ()

    # ---------------------------------------------------------

    def __post_init__(self) -> None:

        n = self.z.size

        if self.y.size != n:

            raise DataError(
                f"{self.name or 'dataset'}: {n} redshifts and {self.y.size} "
                "values do not pair up."
            )

        if self.cov.shape != (n, n):

            raise DataError(
                f"{self.name or 'dataset'}: covariance has shape "
                f"{self.cov.shape}, expected ({n}, {n})."
            )

        if not np.allclose(self.cov, self.cov.T, rtol=1e-10, atol=0.0):

            raise DataError(f"{self.name or 'dataset'}: covariance is not symmetric.")

        smallest = float(np.linalg.eigvalsh(self.cov).min())

        if smallest <= 0.0:

            raise DataError(
                f"{self.name or 'dataset'}: covariance is not positive "
                f"definite (smallest eigenvalue {smallest:.3g}). A likelihood "
                "built on it is not a likelihood."
            )

    # ---------------------------------------------------------

    @property
    def sigma(self) -> Array:
        """Marginal uncertainties -- the square root of the diagonal."""

        return np.sqrt(np.diag(self.cov))

    def __len__(self) -> int:

        return int(self.z.size)

    def describe(self) -> str:
        """A few lines a user can read before trusting the numbers."""

        lines = [
            f"{self.name}: {len(self)} measurements of {self.observable}"
            f" over z = {self.z.min():.4g} to {self.z.max():.4g}",
            f"  reference: {self.reference}",
        ]

        off_diagonal = np.abs(self.cov - np.diag(np.diag(self.cov))).max()

        lines.append(
            "  covariance: full" if off_diagonal > 0.0 else "  covariance: diagonal"
        )

        if self.excludes:
            lines.append(f"  excludes: {', '.join(self.excludes)}")

        if self.note:
            lines.append(f"  note: {self.note}")

        return "\n".join(lines)

    def __repr__(self) -> str:

        return (
            f"<Dataset {self.name} {self.observable} n={len(self)} "
            f"z=[{self.z.min():.4g}, {self.z.max():.4g}]>"
        )


# ============================================================
# Several observables at once
# ============================================================

@dataclass(frozen=True, slots=True)
class MultiObservableDataset:
    """
    A release measuring more than one quantity, with the correlations between
    them.

    A BAO release is the case that matters: ``D_M/r_d`` and ``D_H/r_d`` at the
    same redshifts, correlated at the 0.2--0.5 level within a tracer. The
    correlation is the reason the two cannot simply be split into two files
    and treated separately -- and the reason this class exists rather than a
    pair of :class:`Dataset` objects.

    It deliberately cannot be handed to a reconstructor that fits one scalar
    function: thirteen numbers that are alternately a transverse distance and
    a Hubble distance do not trace one curve. :meth:`select` narrows it to one
    observable, or to several with the covariance between them kept -- and a
    reconstructor declaring ``supports_joint`` takes the latter and fits both
    functions from one realisation, so their correlation survives into
    everything built from them.
    """

    z: Array

    values: Array

    cov: Array

    #: Which quantity each entry is, parallel to :attr:`z`.
    quantity: np.ndarray

    #: Unit per quantity name. Usually all dimensionless for BAO.
    units: dict[str, str] = field(default_factory=dict)

    name: str = ""

    reference: str = ""

    note: str = ""

    excludes: tuple[str, ...] = ()

    # ---------------------------------------------------------

    def __post_init__(self) -> None:

        n = self.z.size

        if self.values.size != n or self.quantity.size != n:

            raise DataError(
                f"{self.name or 'dataset'}: {n} redshifts, "
                f"{self.values.size} values and {self.quantity.size} labels "
                "do not pair up."
            )

        if self.cov.shape != (n, n):

            raise DataError(
                f"{self.name or 'dataset'}: covariance has shape "
                f"{self.cov.shape}, expected ({n}, {n})."
            )

    # ---------------------------------------------------------

    @property
    def y(self):
        """
        Refuses, on purpose.

        A reconstructor reaching for ``.y`` is about to fit one function
        through several different quantities. Raising here is what turns that
        into a message instead of a curve.
        """

        raise DataError(
            f"{self.name or 'This dataset'} measures "
            f"{sorted(self.quantities())} together, with the correlations "
            "between them, so it is not one function of redshift and cannot "
            "be reconstructed as one.\n"
            "\n"
            "There are two right answers. Narrow it to one observable -- "
            f".select('{self.quantities()[0]}') -- which carries that "
            "quantity's own covariance block and drops its correlation with "
            "the others; any method can fit that. Or keep the correlation -- "
            f".select({', '.join(repr(q) for q in self.quantities()[:2])}) -- "
            "and hand the result to a reconstructor that declares "
            "supports_joint, which fits both functions from one realisation. "
            "Cosmography does; the Gaussian process does not yet.\n"
            "\n"
            "The second is what the curvature and distance-duality tests need: "
            "they are built from both observables and are not defined unless "
            "the correlation between them survives."
        )

    def quantities(self) -> tuple[str, ...]:
        """The distinct quantities present, in the order they first appear."""

        seen: dict[str, None] = {}

        for label in self.quantity:
            seen.setdefault(str(label), None)

        return tuple(seen)

    def select(self, *quantities: str):
        """
        Some of the observables, with the covariance between them.

        One name gives a :class:`Dataset` -- a single function of redshift,
        which is what a reconstructor fits. Several give another
        :class:`MultiObservableDataset` carrying the block of the covariance
        that couples them, for a **joint** reconstruction.

        The distinction matters. Selecting one observable drops its
        correlation with the others, which is unavoidable when reconstructing
        a single curve and is stated in the note attached to the result.
        Selecting two keeps it -- and the null tests that need both, the
        curvature test and distance duality, are only defined if it is kept.
        """

        if not quantities:

            raise DataError(
                f"Nothing selected. {self.name or 'This dataset'} measures "
                f"{sorted(self.quantities())}."
            )

        missing = [q for q in quantities if q not in self.quantities()]

        if missing:

            raise DataError(
                f"{self.name or 'This dataset'} has no {missing[0]!r}; it "
                f"measures {sorted(self.quantities())}."
            )

        labels = np.asarray(self.quantity).astype(str)

        index = np.flatnonzero(np.isin(labels, list(quantities)))

        dropped = sorted(set(self.quantities()) - set(quantities))

        if len(quantities) > 1:

            note = (
                f"{self.name} also measures {dropped}, correlated with these; "
                "that cross-covariance is not carried here."
                if dropped
                else ""
            )

            return MultiObservableDataset(
                z=np.asarray(self.z)[index],
                values=np.asarray(self.values)[index],
                cov=np.asarray(self.cov)[np.ix_(index, index)],
                quantity=labels[index],
                units=dict(self.units),
                name=f"{self.name}[{'+'.join(quantities)}]",
                reference=self.reference,
                note=(note + " " + self.note).strip(),
                excludes=self.excludes,
            )

        quantity = quantities[0]

        note = (
            f"{self.name} also measures {dropped}, correlated with this; that "
            "cross-covariance is not carried here."
            if dropped
            else ""
        )

        return Dataset(
            z=np.asarray(self.z)[index],
            y=np.asarray(self.values)[index],
            cov=np.asarray(self.cov)[np.ix_(index, index)],
            observable=quantity,
            unit=self.units.get(quantity, ""),
            name=f"{self.name}[{quantity}]",
            reference=self.reference,
            note=(note + " " + self.note).strip(),
            excludes=self.excludes,
        )

    def __len__(self) -> int:

        return int(self.z.size)

    def describe(self) -> str:

        lines = [
            f"{self.name}: {len(self)} measurements over z = "
            f"{self.z.min():.4g} to {self.z.max():.4g}",
            f"  quantities: {', '.join(self.quantities())}",
            f"  reference: {self.reference}",
        ]

        if self.excludes:
            lines.append(f"  excludes: {', '.join(self.excludes)}")

        if self.note:
            lines.append(f"  note: {self.note}")

        return "\n".join(lines)

    def __repr__(self) -> str:

        return (
            f"<MultiObservableDataset {self.name} "
            f"{list(self.quantities())} n={len(self)}>"
        )


# ============================================================
# Combining
# ============================================================

def check_combination(*datasets) -> None:
    """
    Refuse a set of datasets that may not be used together.

    Overlapping compilations are the trap: two supernova samples sharing most
    of their objects, or two BAO releases built from the same galaxies, are
    not two independent measurements. Fitting both doubles their weight and
    narrows the interval by roughly the square root of that, which looks like
    a better measurement and is not one.

    Called by :func:`~CosmoRecon.data.loader.load`, and worth calling by hand
    whenever datasets are assembled some other way.
    """

    names = [getattr(d, "name", "") for d in datasets]

    for index, dataset in enumerate(datasets):

        for other_index, other in enumerate(datasets):

            if index >= other_index:
                continue

            forbidden = set(getattr(dataset, "excludes", ()))

            if names[other_index] in forbidden or names[index] in set(
                getattr(other, "excludes", ())
            ):

                raise DataError(
                    f"{names[index]} and {names[other_index]} may not be "
                    "combined: they are built from overlapping observations, "
                    "so using both counts the same measurements twice and "
                    "reports an interval that is too tight. Choose one."
                )
