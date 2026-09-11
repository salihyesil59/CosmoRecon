"""
What every reconstruction method has to provide.

The contract is deliberately narrow. A reconstructor takes data and returns a
:class:`ReconstructionSet` -- the several functions one fit produces, sharing
one realisation index. It does not know what a null test is, and no null test
knows what it is. That separation is what lets
:mod:`CosmoRecon.ensemble` swap one method for another and ask what changed,
which is the question the library was built to answer.

Subclasses implement :meth:`Reconstructor._fit`. The public :meth:`Reconstructor.fit`
around it is a template method that does the parts every method must get right
and none should re-implement: seeding the draw stream, recording provenance,
and pinning the redshift range over which the result is a measurement rather
than a prior.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Mapping

import numpy as np
from scipy import linalg

from CosmoRecon.typing import Array, Redshift

from CosmoRecon.core.errors import (
    DataError,
    EvidenceUnavailableError,
    NotRefittableError,
)
from CosmoRecon.core.grid import (
    common_support,
    linear_grid,
    suggested_n,
    support_of,
)
from CosmoRecon.core.provenance import Origin, Provenance, new_origin
from CosmoRecon.core.reconstruction import Reconstruction


__all__ = [
    "Reconstructor",
    "ReconstructionSet",
    "combine_independent",
    "unpack_dataset",
    "unpack_joint",
]


#: Draws taken by default. Large enough that a 95% interval is stable and a
#: 99% one is reportable (see the tail check in ``Reconstruction.quantile``),
#: small enough that a four-member ensemble still fits in memory on a laptop.
DEFAULT_N_DRAWS = 2000


# ============================================================
# The output of one fit
# ============================================================

class ReconstructionSet(Mapping[str, Reconstruction]):
    """
    The functions one fit produced, all mutually aligned.

    A Gaussian process fitted to cosmic chronometers yields ``H``; integrating
    each draw yields ``D_C``; those two are not independent, and here they
    cannot accidentally be treated as if they were, because they carry the
    same :data:`~CosmoRecon.core.provenance.Origin`.

    Behaves as a read-only mapping, and exposes its members as attributes as
    well, so both of these read naturally:

    >>> fit["H"]                # doctest: +SKIP
    >>> fit.H                   # doctest: +SKIP
    """

    __slots__ = (
        "_members",
        "_origin",
        "_support",
        "_provenance",
        "_recipe",
        "_datasets",
    )

    def __init__(
        self,
        members: Mapping[str, Reconstruction],
        *,
        origin: Origin,
        support: tuple[float, float],
        provenance: Provenance,
        recipe: Callable[[tuple, int | None], "ReconstructionSet"] | None = None,
        datasets: tuple | None = None,
    ) -> None:

        if not members:

            raise ValueError("A fit must produce at least one function.")

        for name, r in members.items():

            if r.origin != origin:

                raise ValueError(
                    f"{name!r} carries origin {r.origin}, but this fit is "
                    f"origin {origin}. Every function from one fit must "
                    "share the realisation index -- build them from the same "
                    "draws rather than fitting twice."
                )

        self._members = dict(members)

        self._origin = origin

        self._support = support

        self._provenance = provenance

        self._recipe = recipe

        self._datasets = None if datasets is None else tuple(datasets)

    # ---------------------------------------------------------

    def __getitem__(self, key: str) -> Reconstruction:

        return self._members[key]

    def __iter__(self) -> Iterator[str]:

        return iter(self._members)

    def __len__(self) -> int:

        return len(self._members)

    def __getattr__(self, name: str) -> Reconstruction:

        try:
            return self._members[name]

        except KeyError:

            raise AttributeError(
                f"This fit produced {sorted(self._members)}, not {name!r}."
            ) from None

    # ---------------------------------------------------------

    @property
    def origin(self) -> Origin:

        return self._origin

    @property
    def support(self) -> tuple[float, float]:
        """
        The redshift range the data actually constrain.

        Outside it the reconstruction returns the prior, and every consumer
        in the library treats a curve there as decoration rather than as a
        measurement.
        """

        return self._support

    @property
    def provenance(self) -> Provenance:

        return self._provenance

    @property
    def datasets(self) -> tuple | None:
        """
        The datasets this set was fitted to, in the order :meth:`refit` takes
        them -- or ``None`` for a set that did not come out of a fit.
        """

        return self._datasets

    @property
    def refittable(self) -> bool:

        return self._recipe is not None

    def refit(self, datasets, *, seed: int | None = None) -> "ReconstructionSet":
        """
        The same analysis -- the same method with the same settings, on the
        same grid -- run on different data.

        What calibration by simulation is built on
        (:func:`CosmoRecon.validation.calibrate`): draw datasets in which a null
        hypothesis holds, refit each exactly as the real data were fitted, and
        see what the statistic does. ``datasets`` must match :attr:`datasets`
        in number and order. The method is a copy taken at fit time, so
        nothing done to the original reconstructor since can leak in.
        """

        if self._recipe is None:

            raise NotRefittableError(
                "This set was not produced by a fit, so there is no analysis "
                "to rerun on new data. Fit a Reconstructor (or combine fits "
                "with combine_independent) to get a set that remembers how it "
                "was made."
            )

        datasets = tuple(datasets) if isinstance(datasets, (list, tuple)) else (datasets,)

        if len(datasets) != len(self._datasets):

            raise ValueError(
                f"This set was fitted to {len(self._datasets)} dataset(s) and "
                f"was handed {len(datasets)} to refit. They are matched by "
                "position, so the count has to agree."
            )

        return self._recipe(datasets, seed)

    # ---------------------------------------------------------

    def at(self, z: Redshift) -> "ReconstructionSet":
        """
        The whole set on a new grid, still mutually aligned.

        Alignment survives because each member's predictor is deterministic:
        see the contract on :class:`~CosmoRecon.core.reconstruction.Predictor`.
        A refit of the result is regridded the same way.
        """

        recipe = None

        if self._recipe is not None:

            inner = self._recipe

            def recipe(datasets, seed):
                return inner(datasets, seed).at(z)

        return ReconstructionSet(
            {name: r.at(z) for name, r in self._members.items()},
            origin=self._origin,
            support=self._support,
            provenance=self._provenance,
            recipe=recipe,
            datasets=self._datasets,
        )

    def __repr__(self) -> str:

        lo, hi = self._support

        return (
            f"<ReconstructionSet {sorted(self._members)} "
            f"origin={self._origin} support=[{lo:.4g}, {hi:.4g}] "
            f"| {self._provenance.describe()}>"
        )


# ============================================================
# Two independent fits, as one
# ============================================================

class _RowSelection:
    """
    A predictor whose draws are a fixed selection of another predictor's rows.

    The selection is fixed at construction, so the wrapper satisfies the
    determinism clause of the
    :class:`~CosmoRecon.core.reconstruction.Predictor` contract whenever the
    inner predictor does -- and a reconstruction built on it can still be
    regridded and differentiated analytically.
    """

    __slots__ = ("_inner", "_rows")

    def __init__(self, inner, rows: np.ndarray) -> None:

        self._inner = inner
        self._rows = rows

    def __call__(self, z: Array, *, derivative: int = 0) -> Array:

        return self._inner(z, derivative=derivative)[self._rows]


def _set_draws(fit: ReconstructionSet) -> int:

    return int(next(iter(fit.values())).n_draws)


def _reindexed(
    r: Reconstruction,
    rows: np.ndarray,
    origin: Origin,
) -> Reconstruction:

    provenance = r.provenance.with_draws(int(rows.size))

    if r.resamplable:

        return Reconstruction(
            r.z,
            predictor=_RowSelection(r._predictor, rows),
            origin=origin,
            provenance=provenance,
            label=r.label,
            unit=r.unit,
        )

    return Reconstruction(
        r.z,
        draws=r.draws[rows],
        origin=origin,
        provenance=provenance,
        label=r.label,
        unit=r.unit,
    )


def combine_independent(
    first: ReconstructionSet,
    second: ReconstructionSet,
) -> ReconstructionSet:
    """
    Two fits that share no data, as one set on one realisation index.

    >>> both = combine_independent(                             # doctest: +SKIP
    ...     GaussianProcess().fit(reduced_modulus(union3()), grid=grid),
    ...     GaussianProcess().fit(desi.select("DM_over_rs"), grid=grid),
    ... )
    >>> Duality().evaluate(both["mu_reduced"], both["DM_over_rs"])
    ...                                                         # doctest: +SKIP

    The name is the claim, and calling it is making the claim in the source,
    where a reader can disagree with it -- exactly as
    :meth:`~CosmoRecon.core.reconstruction.Reconstruction.assume_independent`
    does for a single curve. The difference is scope: this declares it once
    for every function either fit produced, and the result is an ordinary
    :class:`ReconstructionSet` that a null test, an ensemble or a later
    ``at()`` can use without knowing that it was ever two fits.

    How the realisations are paired. The first set keeps its draw order; the
    second's rows are permuted by a fixed permutation before being paired
    with them. Permuting is not a formality: two fits drawn with the same seed
    reuse the same underlying random numbers, and pairing their draws by
    index would correlate two posteriors that share nothing. The permutation
    is seeded from the two fits' own seeds, so the result is reproducible.
    If the fits hold different numbers of draws, both are cut to the smaller.

    Refused:

    - **The same observable in both.** Two posteriors for one function is a
      question for :mod:`CosmoRecon.ensemble`, not a union.
    - **The same dataset in both.** Whatever else is true of two fits to one
      dataset, they are not independent, and pairing them as though they
      were narrows every interval built from both.
    - **Supports that do not overlap.** The combined set is a measurement only
      where both are, and if that range is empty there is no joint statement
      to make.
    """

    shared = sorted(set(first) & set(second))

    if shared:

        raise ValueError(
            f"Both fits produced {shared}. A union needs distinct functions; "
            "two posteriors for the same function are a comparison between "
            "methods or datasets, which is what MethodEnsemble is for."
        )

    overlap = sorted(
        set(first.provenance.datasets()) & set(second.provenance.datasets())
    )

    if overlap:

        raise DataError(
            f"Both fits used {overlap}, so they are not independent: pairing "
            "their draws as though they were would count those measurements "
            "twice and report an interval that is too tight. Fit the "
            "observables jointly instead."
        )

    support = common_support(first.support, second.support)

    n_first, n_second = _set_draws(first), _set_draws(second)

    n = min(n_first, n_second)

    rng = np.random.default_rng([
        int(first.provenance.seed or 0),
        int(second.provenance.seed or 0),
        n,
        0x1DE,
    ])

    rows_first = np.arange(n)

    rows_second = rng.permutation(n_second)[:n]

    origin = new_origin()

    members = {
        name: _reindexed(r, rows_first, origin) for name, r in first.items()
    }

    members.update(
        {name: _reindexed(r, rows_second, origin) for name, r in second.items()}
    )

    provenance = first.provenance.derive(
        f"independent({', '.join(first)}; {', '.join(second)})",
        second.provenance,
        n_draws=n,
    )

    recipe = None

    datasets = None

    if first.refittable and second.refittable:

        split = len(first.datasets)

        datasets = first.datasets + second.datasets

        def recipe(new, seed):
            return combine_independent(
                first.refit(new[:split], seed=seed),
                second.refit(new[split:], seed=None if seed is None else seed + 7919),
            )

    return ReconstructionSet(
        members,
        origin=origin,
        support=support,
        provenance=provenance,
        recipe=recipe,
        datasets=datasets,
    )


# ============================================================
# The contract
# ============================================================

class Reconstructor(ABC):
    """
    Base class for every non-parametric method in the library.

    Implement :meth:`_fit`, declare :attr:`provides_evidence`, and give
    :meth:`describe` something a figure caption can use.
    """

    #: Whether this method defines a Bayesian evidence. ``False`` for a
    #: neural reconstruction or a genetic-algorithm fit -- and the ensemble
    #: layer checks this rather than discovering a ``nan`` halfway through
    #: an evidence-weighted average.
    provides_evidence: bool = False

    #: Whether :meth:`_fit` returns predictors that can be re-evaluated at
    #: arbitrary redshift. Everything in the library so far does; a method
    #: that only produces draws on a grid would set this ``False`` and lose
    #: access to ``at()`` and analytic derivatives.
    resamplable: bool = True

    #: Whether this method can reconstruct **several correlated observables at
    #: once**, sharing one realisation index.
    #:
    #: A BAO release measures ``D_M/r_d`` and ``D_H/r_d`` together, correlated
    #: at ``r`` of about -0.4 within each tracer, and the null tests that need
    #: both -- the curvature test, distance duality -- are only defined if that
    #: correlation is carried. A method that fits one scalar function at a time
    #: cannot do it, and says so rather than fitting the two separately and
    #: leaving the user to combine them as though they were independent.
    supports_joint: bool = False

    # ---------------------------------------------------------

    @abstractmethod
    def describe(self) -> str:
        """
        This method, spelled the way it should appear in a caption:
        ``"GP(Matern, nu free)"``, ``"Chebyshev(order=5)"``.

        Goes into every :class:`~CosmoRecon.core.provenance.Provenance` this
        reconstructor produces, so it has to distinguish two configurations
        of the same class -- that distinction is the library's subject.
        """

    @abstractmethod
    def hyperparameters(self) -> dict[str, object]:
        """
        Everything that would change the answer if changed.

        Kernel, expansion order, node count, prior ranges. Not thread counts
        or progress bars: this dictionary is evidence, and padding it with
        performance knobs makes it useless for telling two fits apart.
        """

    @abstractmethod
    def _fit(
        self,
        data,
        *,
        grid: Array,
        n_draws: int,
        rng: np.random.Generator,
        provenance: Provenance,
        origin: Origin,
    ) -> Mapping[str, Reconstruction]:
        """
        The method itself.

        Build every returned :class:`~CosmoRecon.core.reconstruction.Reconstruction`
        with ``origin=origin`` and ``provenance=provenance``, and derive them
        all from one set of draws so that the realisation index means the
        same thing across the set.
        """

    # ---------------------------------------------------------

    def fit(
        self,
        data,
        *,
        grid: Redshift | None = None,
        n_draws: int = DEFAULT_N_DRAWS,
        seed: int | None = None,
    ) -> ReconstructionSet:
        """
        Fit ``data`` and return the functions it constrains.

        The bookkeeping here is not optional decoration, which is why it
        lives in the base class rather than in each method:

        - **The seed is recorded.** A reconstruction whose draws cannot be
          regenerated cannot be checked by a referee.
        - **The support is measured from the data**, not from the grid, so a
          curve plotted past the last measurement is flagged by
          :func:`~CosmoRecon.core.grid.check_within_support` rather than
          passing as signal.
        - **The grid defaults to a resolution proportional to the data**,
          because a grid finer than that adds points and no information.
        """

        z_data = _data_redshifts(data)

        support = support_of(z_data)

        if grid is None:

            grid_array = linear_grid(
                support[0],
                support[1],
                suggested_n(z_data.size),
            )

        else:

            grid_array = np.atleast_1d(np.asarray(grid, dtype=float))

        if seed is None:

            seed = int(np.random.SeedSequence().entropy % (2**31))

        rng = np.random.default_rng(seed)

        origin = new_origin()

        provenance = Provenance(
            method=self.describe(),
            data=_data_names(data),
            hyperparameters=dict(self.hyperparameters()),
            seed=seed,
            n_draws=n_draws,
        )

        # Taken before _fit touches any state, so a later refit is this
        # configuration and nothing that happened to the instance afterwards.
        template = copy.deepcopy(self)

        members = self._fit(
            data,
            grid=grid_array,
            n_draws=n_draws,
            rng=rng,
            provenance=provenance,
            origin=origin,
        )

        if isinstance(data, (list, tuple)):
            datasets = tuple(data)
            wrap = list
        else:
            datasets = (data,)
            wrap = _only

        def recipe(new_datasets, new_seed):
            return copy.deepcopy(template).fit(
                wrap(new_datasets), grid=grid_array, n_draws=n_draws, seed=new_seed
            )

        return ReconstructionSet(
            members,
            origin=origin,
            support=support,
            provenance=provenance,
            recipe=recipe,
            datasets=datasets,
        )

    # ---------------------------------------------------------

    @property
    def log_evidence(self) -> float:
        """
        The log Bayesian evidence of the last fit.

        Raises :class:`~CosmoRecon.core.errors.EvidenceUnavailableError` for
        methods that do not define one. Raising rather than returning ``nan``
        is deliberate: an evidence-weighted ensemble that silently dropped a
        member would report a spread across a smaller set of methods than the
        caller asked for, which is exactly the number the caller came for.
        """

        raise EvidenceUnavailableError(
            f"{self.describe()} does not define a Bayesian evidence, so it "
            "cannot take part in an evidence-weighted ensemble. Use "
            "weights='equal', or drop this member explicitly."
        )

    def __repr__(self) -> str:

        return f"<{type(self).__name__}: {self.describe()}>"


def _only(datasets: tuple):
    """The single dataset of a one-dataset fit."""

    (dataset,) = datasets

    return dataset


# ============================================================
# Data introspection
# ============================================================

def _data_redshifts(data) -> Array:
    """
    Every measurement redshift in ``data``, however it was packaged.

    Accepts a single dataset, a sequence of them, or anything exposing ``z``.
    Kept permissive here and strict in :mod:`CosmoRecon.data`, so that a user
    experimenting with their own arrays is not blocked at the door.
    """

    if hasattr(data, "z"):

        return np.atleast_1d(np.asarray(data.z, dtype=float)).ravel()

    if isinstance(data, (list, tuple)):

        parts = [_data_redshifts(item) for item in data]

        return np.concatenate(parts)

    raise TypeError(
        f"Cannot find measurement redshifts on {type(data).__name__}. A "
        "dataset needs a .z attribute, or pass a sequence of datasets."
    )


def _data_names(data) -> tuple[str, ...]:
    """Dataset names with their sizes, for the provenance record."""

    if hasattr(data, "z"):

        n = np.atleast_1d(np.asarray(data.z)).size

        name = getattr(data, "name", type(data).__name__)

        return (f"{name}({n})",)

    if isinstance(data, (list, tuple)):

        return tuple(name for item in data for name in _data_names(item))

    return ()


# ============================================================
# Dataset unpacking
# ============================================================

def unpack_dataset(data) -> tuple[Array, Array, Array, str]:
    """
    Pull ``(z, y, covariance, label)`` out of whatever was passed.

    Accepts a single dataset or a sequence of them. A sequence is
    concatenated with a **block-diagonal** covariance, which asserts that the
    datasets are mutually independent -- true for cosmic chronometers and
    supernovae, false for a BAO measurement and a full-shape analysis of the
    same galaxies. The assertion is the caller's; this only makes it explicit.
    """

    if isinstance(data, (list, tuple)):

        parts = [unpack_dataset(item) for item in data]

        z = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])

        cov = linalg.block_diag(*[p[2] for p in parts])

        labels = {p[3] for p in parts}

        if len(labels) > 1:

            raise DataError(
                f"These datasets measure different quantities ({sorted(labels)}) "
                "and cannot be reconstructed as one function. Fit them "
                "separately, or convert them to a common observable first."
            )

        return z, y, cov, parts[0][3]

    z = _require(data, ("z",), "measurement redshifts")

    y = _require(data, ("y", "H", "value", "values"), "measured values")

    if z.size != y.size:

        raise DataError(
            f"{z.size} redshifts and {y.size} values do not pair up."
        )

    if z.size < 3:

        raise DataError(
            f"{z.size} measurements are not enough to reconstruct a function. "
            "A Gaussian process fitted to two points is its prior."
        )

    cov = getattr(data, "cov", None)

    if cov is None:
        cov = getattr(data, "covariance", None)

    if cov is None:

        sigma = _require(data, ("sigma", "err", "error", "dy"), "uncertainties")

        if sigma.size != z.size:

            raise DataError(
                f"{sigma.size} uncertainties for {z.size} measurements."
            )

        cov = np.diag(sigma.astype(float) ** 2)

    else:
        cov = np.atleast_2d(np.asarray(cov, dtype=float))

    if cov.shape != (z.size, z.size):

        raise DataError(
            f"Covariance has shape {cov.shape}, expected "
            f"({z.size}, {z.size})."
        )

    label = str(
        getattr(data, "observable", None)
        or getattr(data, "label", None)
        or "f"
    )

    order = np.argsort(z)

    return z[order], y[order], cov[np.ix_(order, order)], label


def _require(data, names: tuple[str, ...], what: str) -> Array:

    for name in names:

        value = getattr(data, name, None)

        if value is not None:

            return np.atleast_1d(np.asarray(value, dtype=float)).ravel()

    raise DataError(
        f"Cannot find {what} on {type(data).__name__}: looked for "
        f"{' , '.join(names)}."
    )


def unpack_joint(data) -> tuple[Array, Array, Array, np.ndarray]:
    """
    ``(z, values, covariance, quantity)`` for a dataset measuring one or
    several observables together.

    The single-observable case comes back as one quantity repeated, so a
    reconstructor written against this signature handles both without
    branching -- a joint fit of one function is just a fit.

    Row order is **preserved**, unlike :func:`unpack_dataset`, because the
    covariance's rows are in the release's own order and reordering one
    without the other is the kind of mistake that produces a plausible curve
    and a wrong interval.
    """

    quantity = getattr(data, "quantity", None)

    if quantity is None:

        z, y, cov, label = unpack_dataset(data)

        return z, y, cov, np.array([label] * z.size)

    z = np.atleast_1d(np.asarray(data.z, dtype=float)).ravel()

    values = np.atleast_1d(np.asarray(data.values, dtype=float)).ravel()

    cov = np.atleast_2d(np.asarray(data.cov, dtype=float))

    quantity = np.asarray(quantity)

    if not (z.size == values.size == quantity.size):

        raise DataError(
            f"{getattr(data, 'name', 'dataset')}: {z.size} redshifts, "
            f"{values.size} values and {quantity.size} labels do not pair up."
        )

    if cov.shape != (z.size, z.size):

        raise DataError(
            f"{getattr(data, 'name', 'dataset')}: covariance has shape "
            f"{cov.shape}, expected ({z.size}, {z.size})."
        )

    for label in np.unique(quantity):

        count = int(np.sum(quantity == label))

        if count < 3:

            raise DataError(
                f"{label!r} has {count} measurement(s), which is not enough "
                "to reconstruct a function of redshift. Select the "
                "observables that are, or fit a model instead."
            )

    return z, values, cov, quantity
