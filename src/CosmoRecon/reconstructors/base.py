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

from abc import ABC, abstractmethod
from typing import Iterator, Mapping

import numpy as np
from scipy import linalg

from CosmoRecon.typing import Array, Redshift

from CosmoRecon.core.errors import DataError, EvidenceUnavailableError
from CosmoRecon.core.grid import linear_grid, suggested_n, support_of
from CosmoRecon.core.provenance import Origin, Provenance, new_origin
from CosmoRecon.core.reconstruction import Reconstruction


__all__ = ["Reconstructor", "ReconstructionSet", "unpack_dataset"]


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

    __slots__ = ("_members", "_origin", "_support", "_provenance")

    def __init__(
        self,
        members: Mapping[str, Reconstruction],
        *,
        origin: Origin,
        support: tuple[float, float],
        provenance: Provenance,
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

    # ---------------------------------------------------------

    def at(self, z: Redshift) -> "ReconstructionSet":
        """
        The whole set on a new grid, still mutually aligned.

        Alignment survives because each member's predictor is deterministic:
        see the contract on :class:`~CosmoRecon.core.reconstruction.Predictor`.
        """

        return ReconstructionSet(
            {name: r.at(z) for name, r in self._members.items()},
            origin=self._origin,
            support=self._support,
            provenance=self._provenance,
        )

    def __repr__(self) -> str:

        lo, hi = self._support

        return (
            f"<ReconstructionSet {sorted(self._members)} "
            f"origin={self._origin} support=[{lo:.4g}, {hi:.4g}] "
            f"| {self._provenance.describe()}>"
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

        members = self._fit(
            data,
            grid=grid_array,
            n_draws=n_draws,
            rng=rng,
            provenance=provenance,
            origin=origin,
        )

        return ReconstructionSet(
            members,
            origin=origin,
            support=support,
            provenance=provenance,
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
