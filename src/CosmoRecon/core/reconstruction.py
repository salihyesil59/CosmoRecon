"""
The spine: a posterior over functions.

A :class:`Reconstruction` is not a curve with error bars. It is a set of
*draws* -- whole realisations of a function -- sharing one realisation index,
so that draw ``k`` of ``H`` and draw ``k`` of ``D`` describe the same
universe. Every operation is applied per draw, which is what makes

>>> Ok = (H**2 * Dp**2 - C_LIGHT**2) / (H0**2 * D**2)      # doctest: +SKIP

correct rather than approximately correct: the correlation between ``H`` and
``D``, the non-Gaussianity of the ratio, and the covariance between adjacent
redshifts all survive, because none of them was ever summarised away.

Three invariants hold this together, and each is enforced rather than
documented and hoped for.

**Alignment.** Draws from one fit share an :data:`~CosmoRecon.core.provenance.Origin`.
Combining two different origins raises :class:`AlignmentError` unless the user
states the independence claim explicitly with :meth:`Reconstruction.assume_independent`.

**Determinism.** A reconstruction that carries a *predictor* can be
re-evaluated at any redshift and any derivative order, and the predictor is
required to be deterministic: draw ``k`` at ``z`` and draw ``k`` at ``z'`` must
come from the same realisation of the function. Without that, ``at()`` would
silently resample and alignment would evaporate at the first regrid.

**Honesty about derivatives.** ``d(n)`` asks the predictor for an analytic
derivative. Where the method has none, it raises rather than quietly finite-
differencing, because a numerical derivative of a posterior is a different
estimator with tighter-looking and wronger intervals.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from CosmoRecon.typing import Array, DrawArray, Redshift

from CosmoRecon.core.errors import (
    AlignmentError,
    DerivativeUnavailableError,
    GridMismatchError,
    InsufficientDrawsError,
    NotResamplableError,
)

from CosmoRecon.core.provenance import (
    CONSTANT_ORIGIN,
    Origin,
    Provenance,
    new_origin,
)


__all__ = ["Predictor", "Reconstruction"]


#: How close two grids must be to count as the same one. Grids are built by
#: the same routines from the same endpoints, so genuine equality differs
#: only by floating-point noise; anything larger is a real mismatch and the
#: user wants to hear about it.
_GRID_RTOL = 1e-12
_GRID_ATOL = 1e-14

#: Draws required in the tail before a quantile is reported rather than
#: refused. Five is the point below which the estimate is dominated by which
#: draws happened to land there.
_MIN_TAIL_DRAWS = 5


# ============================================================
# The predictor contract
# ============================================================

@runtime_checkable
class Predictor(Protocol):
    """
    A function-space posterior that can be evaluated anywhere.

    Implemented by each reconstructor. The contract is short and the second
    clause is the load-bearing one:

    1. ``predictor(z, derivative=n)`` returns shape ``(n_draws, len(z))``.
    2. It is **deterministic**. Two calls with different ``z`` must evaluate
       the *same* set of realisations, in the same order. A GP that draws
       fresh samples on each call satisfies clause 1 and destroys the
       library; see ``reconstructors/gp.py``, which fixes a set of pathwise
       samples at fit time and evaluates those.
    3. ``derivative`` is with respect to ``z``. Raising
       :class:`DerivativeUnavailableError` for orders the method cannot do
       analytically is correct behaviour, not a failure.
    """

    def __call__(
        self,
        z: Array,
        *,
        derivative: int = 0,
    ) -> DrawArray:
        ...


# ============================================================
# Reconstruction
# ============================================================

class Reconstruction:
    """
    A posterior over a function of redshift.

    Not normally constructed by hand -- a
    :class:`~CosmoRecon.reconstructors.base.Reconstructor` returns these, and
    arithmetic between them returns more of them. The two classmethods
    :meth:`from_predictor` and :meth:`from_draws` are the entry points for
    people writing a new reconstructor.
    """

    __slots__ = (
        "_z",
        "_draws",
        "_predictor",
        "_analytic",
        "_origin",
        "_independent",
        "provenance",
        "label",
        "unit",
    )

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def __init__(
        self,
        z: Array,
        *,
        draws: DrawArray | None = None,
        predictor: Predictor | None = None,
        analytic: tuple[Array, Array] | None = None,
        origin: Origin,
        provenance: Provenance,
        label: str = "",
        unit: str = "",
        independent: bool = False,
    ) -> None:

        if draws is None and predictor is None:

            raise ValueError(
                "A Reconstruction needs draws, a predictor, or both."
            )

        self._z = np.ascontiguousarray(z, dtype=float)

        if self._z.ndim != 1:

            raise ValueError(
                f"Redshift grid must be 1-d, got shape {self._z.shape}."
            )

        self._draws = None if draws is None else np.asarray(draws, dtype=float)

        if self._draws is not None:

            if self._draws.ndim != 2:

                raise ValueError(
                    "Draws must have shape (n_draws, n_z), got "
                    f"{self._draws.shape}."
                )

            if self._draws.shape[1] != self._z.size:

                raise ValueError(
                    f"Draws have {self._draws.shape[1]} redshifts but the "
                    f"grid has {self._z.size}."
                )

        self._predictor = predictor

        #: Exact ``(mean, covariance)`` on :attr:`z` where the method has
        #: them in closed form -- a GP, or any linear-in-parameters
        #: expansion. Used only to answer :meth:`mean` and :meth:`cov`
        #: without Monte Carlo noise; the draws remain the representation.
        self._analytic = analytic

        self._origin = origin

        self._independent = independent

        self.provenance = provenance

        self.label = label

        self.unit = unit

    # ---------------------------------------------------------

    @classmethod
    def from_predictor(
        cls,
        z: Redshift,
        predictor: Predictor,
        *,
        provenance: Provenance,
        origin: Origin | None = None,
        analytic: tuple[Array, Array] | None = None,
        label: str = "",
        unit: str = "",
    ) -> "Reconstruction":
        """
        Build from a fitted method that can be re-evaluated.

        This is what a reconstructor returns. The result supports
        :meth:`at` and analytic :meth:`d`.
        """

        return cls(
            np.atleast_1d(np.asarray(z, dtype=float)),
            predictor=predictor,
            analytic=analytic,
            origin=new_origin() if origin is None else origin,
            provenance=provenance,
            label=label,
            unit=unit,
        )

    @classmethod
    def from_draws(
        cls,
        z: Redshift,
        draws: DrawArray,
        *,
        provenance: Provenance,
        origin: Origin | None = None,
        label: str = "",
        unit: str = "",
    ) -> "Reconstruction":
        """
        Build from draws on a fixed grid.

        Use this when the realisations exist but the function behind them
        does not -- the output of arithmetic, or of a method that produces
        samples on a grid and nothing else. The result cannot be regridded
        or differentiated analytically, and says so.
        """

        return cls(
            np.atleast_1d(np.asarray(z, dtype=float)),
            draws=draws,
            origin=new_origin() if origin is None else origin,
            provenance=provenance,
            label=label,
            unit=unit,
        )

    @classmethod
    def constant(
        cls,
        z: Redshift,
        value: float,
        *,
        n_draws: int = 1,
        label: str = "",
        unit: str = "",
    ) -> "Reconstruction":
        """
        A known number promoted onto a grid.

        Carries :data:`CONSTANT_ORIGIN`, so it combines with anything: a
        constant is correlated with everything and with nothing, and there is
        no independence claim to make about it.
        """

        grid = np.atleast_1d(np.asarray(z, dtype=float))

        return cls(
            grid,
            draws=np.full((n_draws, grid.size), float(value)),
            origin=CONSTANT_ORIGIN,
            provenance=Provenance(
                method="constant",
                expression=repr(value),
                n_draws=n_draws,
            ),
            label=label,
            unit=unit,
        )

    # ---------------------------------------------------------
    # The representation
    # ---------------------------------------------------------

    @property
    def z(self) -> Array:
        """The evaluation grid. Read-only."""

        view = self._z.view()
        view.flags.writeable = False

        return view

    @property
    def draws(self) -> DrawArray:
        """
        The realisations, shape ``(n_draws, n_z)``.

        Materialised from the predictor on first access and cached.
        """

        if self._draws is None:

            self._draws = np.asarray(
                self._predictor(self._z, derivative=0),
                dtype=float,
            )

        return self._draws

    @property
    def n_draws(self) -> int:

        return self.draws.shape[0]

    @property
    def n_z(self) -> int:

        return self._z.size

    @property
    def origin(self) -> Origin:
        """Which fit these draws came from. See :mod:`.provenance`."""

        return self._origin

    @property
    def resamplable(self) -> bool:
        """Whether :meth:`at` and analytic :meth:`d` are available."""

        return self._predictor is not None

    # ---------------------------------------------------------
    # Summaries -- never the representation
    # ---------------------------------------------------------

    def mean(self) -> Array:
        """Posterior mean. Exact where the method provides closed-form moments."""

        if self._analytic is not None:
            return self._analytic[0].copy()

        return self.draws.mean(axis=0)

    def cov(self) -> Array:
        """
        Posterior covariance *across redshift*, shape ``(n_z, n_z)``.

        The off-diagonal is the whole point: a null test evaluated on a fine
        grid has far fewer independent points than it has grid points, and
        anything that reports a significance needs this matrix rather than
        the diagonal.
        """

        if self._analytic is not None:
            return self._analytic[1].copy()

        return np.cov(self.draws, rowvar=False, ddof=1)

    def var(self) -> Array:

        if self._analytic is not None:
            return np.diag(self._analytic[1]).copy()

        return self.draws.var(axis=0, ddof=1)

    def std(self) -> Array:

        return np.sqrt(self.var())

    def quantile(self, q: float | Sequence[float]) -> Array:
        """
        Posterior quantiles, straight from the draws.

        No Gaussian assumption anywhere: the ratio in a distance-duality test
        is skewed, and this reports the skew instead of symmetrising it away.
        """

        qs = np.atleast_1d(np.asarray(q, dtype=float))

        if np.any((qs < 0.0) | (qs > 1.0)):

            raise ValueError("Quantiles must lie in [0, 1].")

        tail = float(np.min(np.minimum(qs, 1.0 - qs)))

        if tail * self.n_draws < _MIN_TAIL_DRAWS:

            raise InsufficientDrawsError(
                f"Quantile {qs.min():.4g} from {self.n_draws} draws puts "
                f"only {tail * self.n_draws:.1f} draws in the tail; at least "
                f"{_MIN_TAIL_DRAWS} are needed for the estimate to mean "
                f"anything. Refit with more draws."
            )

        return np.quantile(self.draws, qs, axis=0)

    def interval(self, level: float = 0.68) -> tuple[Array, Array]:
        """
        Equal-tailed credible interval at ``level``, as ``(lower, upper)``.
        """

        half = 0.5 * (1.0 - level)

        bounds = self.quantile([half, 1.0 - half])

        return bounds[0], bounds[1]

    # ---------------------------------------------------------
    # Re-evaluation and differentiation
    # ---------------------------------------------------------

    def at(self, z: Redshift) -> "Reconstruction":
        """
        The same posterior, on a different grid.

        This is how two reconstructions built from datasets with different
        redshifts are brought together -- the distance-duality relation needs
        ``d_L`` and ``d_A`` at the same ``z``, and no two catalogues agree on
        one. Because the predictor is deterministic, draw ``k`` here is the
        same realisation as draw ``k`` on the original grid, so the result
        stays aligned with its siblings.
        """

        if self._predictor is None:

            raise NotResamplableError(
                f"{self._describe()} holds draws on a fixed grid and cannot "
                "be re-evaluated -- it came out of arithmetic, or out of a "
                "method that only produces samples. Call .at() on the "
                "operands instead, before combining them."
            )

        grid = np.atleast_1d(np.asarray(z, dtype=float))

        return Reconstruction(
            grid,
            predictor=self._predictor,
            origin=self._origin,
            provenance=self.provenance,
            label=self.label,
            unit=self.unit,
            independent=self._independent,
        )

    def d(
        self,
        order: int = 1,
        *,
        numerical: bool = False,
    ) -> "Reconstruction":
        """
        The ``order``-th derivative with respect to redshift.

        Analytic by default: a Gaussian process differentiates into another
        Gaussian process, a Chebyshev expansion term by term, a spline
        exactly. Where the method cannot, this raises -- pass
        ``numerical=True`` to opt in to finite differences on the grid, with
        the understanding that it is a *different* estimator whose intervals
        are not the analytic ones.
        """

        if order < 0:

            raise ValueError("Derivative order must be non-negative.")

        if order == 0:

            return self

        prime = "'" * order

        if self._predictor is not None and not numerical:

            derivative_predictor = _DerivativePredictor(self._predictor, order)

            return Reconstruction(
                self._z,
                predictor=derivative_predictor,
                origin=self._origin,
                provenance=self.provenance.derive(
                    f"d{order}/dz{order} {self.label or 'f'}",
                ),
                label=f"{self.label}{prime}" if self.label else "",
                unit=f"{self.unit}/z" if self.unit else "",
                independent=self._independent,
            )

        if not numerical:

            raise DerivativeUnavailableError(
                f"{self._describe()} has no analytic derivative. Pass "
                "numerical=True to finite-difference on the grid instead -- "
                "but note that the result is a different estimator, and its "
                "interval is not the posterior on the derivative."
            )

        draws = self.draws

        for _ in range(order):

            draws = np.gradient(draws, self._z, axis=1, edge_order=2)

        return Reconstruction.from_draws(
            self._z,
            draws,
            origin=self._origin,
            provenance=self.provenance.derive(
                f"numerical d{order}/dz{order} {self.label or 'f'}",
            ),
            label=f"{self.label}{prime}" if self.label else "",
            unit=f"{self.unit}/z" if self.unit else "",
        )

    # ---------------------------------------------------------
    # Independence
    # ---------------------------------------------------------

    def assume_independent(self) -> "Reconstruction":
        """
        Declare that this reconstruction shares no data with whatever it is
        about to be combined with.

        Required because there is no safe default. Pairing draw ``k`` of one
        fit with draw ``k`` of another is only meaningful if the two are
        genuinely independent; if they are not -- two GPs over overlapping
        supernova compilations, say -- the pairing invents a correlation and
        the resulting interval is too tight.

        Saying it out loud is the point: the claim belongs in the source,
        where a reader can disagree with it. The draws of a reconstruction
        marked this way are permuted before combination, so that no accident
        of ordering can leak back in.
        """

        return Reconstruction(
            self._z,
            draws=self._draws,
            predictor=self._predictor,
            analytic=self._analytic,
            origin=self._origin,
            provenance=self.provenance,
            label=self.label,
            unit=self.unit,
            independent=True,
        )

    # ---------------------------------------------------------
    # Arithmetic
    # ---------------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Make numpy's ufuncs work draw-wise.

        ``np.log(H)``, ``np.sqrt(1 + Om)`` and ``H1 - H2`` all route through
        here, so the alignment and grid checks cannot be sidestepped by
        reaching for numpy instead of an operator.
        """

        if method != "__call__" or kwargs.get("out") is not None:

            return NotImplemented

        recons = [item for item in inputs if isinstance(item, Reconstruction)]

        grid, origin, aligned = _align(recons)

        lookup = {id(r): arr for r, arr in zip(recons, aligned, strict=True)}

        operands = []

        for item in inputs:

            if isinstance(item, Reconstruction):
                operands.append(lookup[id(item)])

            else:
                operands.append(np.asarray(item, dtype=float))

        result = ufunc(*operands, **kwargs)

        symbols = [_operand_name(item) for item in inputs]

        expression = f"{ufunc.__name__}({', '.join(symbols)})"

        return Reconstruction.from_draws(
            grid,
            result,
            origin=origin,
            provenance=_merge_provenance(recons, expression, result.shape[0]),
            label=_as_label(expression),
            unit="",
        )

    def _binary(self, other, op: Callable, symbol: str, flip: bool = False):

        recons = [self]

        if isinstance(other, Reconstruction):
            recons.append(other)

        grid, origin, aligned = _align(recons)

        left = aligned[0]

        if isinstance(other, Reconstruction):
            right = aligned[1]

        else:
            right = np.asarray(other, dtype=float)

        if flip:
            left, right = right, left

        result = op(left, right)

        names = [
            _operand_name(item)
            for item in ((other, self) if flip else (self, other))
        ]

        expression = f"({names[0]} {symbol} {names[1]})"

        return Reconstruction.from_draws(
            grid,
            result,
            origin=origin,
            provenance=_merge_provenance(recons, expression, result.shape[0]),
            label=_as_label(expression),
        )

    def __add__(self, other):
        return self._binary(other, np.add, "+")

    def __radd__(self, other):
        return self._binary(other, np.add, "+", flip=True)

    def __sub__(self, other):
        return self._binary(other, np.subtract, "-")

    def __rsub__(self, other):
        return self._binary(other, np.subtract, "-", flip=True)

    def __mul__(self, other):
        return self._binary(other, np.multiply, "*")

    def __rmul__(self, other):
        return self._binary(other, np.multiply, "*", flip=True)

    def __truediv__(self, other):
        return self._binary(other, np.divide, "/")

    def __rtruediv__(self, other):
        return self._binary(other, np.divide, "/", flip=True)

    def __pow__(self, other):
        return self._binary(other, np.power, "**")

    def __rpow__(self, other):
        return self._binary(other, np.power, "**", flip=True)

    def __neg__(self):
        return self._binary(-1.0, np.multiply, "*")

    def __abs__(self):
        return np.abs(self)

    # ---------------------------------------------------------

    def _describe(self) -> str:

        name = self.label or "Reconstruction"

        return f"{name} ({self.provenance.describe()})"

    def __repr__(self) -> str:

        kind = "predictor" if self.resamplable else "draws"

        span = (
            f"z=[{self._z[0]:.4g}, {self._z[-1]:.4g}]"
            if self.n_z > 1
            else f"z={self._z[0]:.4g}"
        )

        return (
            f"<Reconstruction {self.label or '?'} "
            f"[{kind}] {span} n_z={self.n_z} "
            f"n_draws={self.n_draws} origin={self._origin} "
            f"| {self.provenance.describe()}>"
        )


# ============================================================
# Derivative predictor
# ============================================================

class _DerivativePredictor:
    """
    Wraps a predictor so that ``derivative=m`` on the wrapper means
    ``derivative=m + order`` on the original.

    Keeping the wrapper thin is what lets ``H.d(1).d(1)`` reach the
    method's analytic second derivative rather than differentiating a
    derivative numerically.
    """

    __slots__ = ("_inner", "_order")

    def __init__(self, inner: Predictor, order: int) -> None:

        self._inner = inner
        self._order = order

    def __call__(self, z: Array, *, derivative: int = 0) -> DrawArray:

        return self._inner(z, derivative=derivative + self._order)


# ============================================================
# Alignment
# ============================================================

def _align(
    recons: Sequence[Reconstruction],
) -> tuple[Array, Origin, list[DrawArray]]:
    """
    Bring several reconstructions onto one grid and one realisation index.

    Returns the shared grid, the origin the result should carry, and the draw
    arrays in the same order as the input, each truncated to the common draw
    count and permuted where independence was declared.

    The rules, in the order they are checked:

    1. **Grids must already match.** Regridding silently would hide the
       redshift-mismatch problem that :meth:`Reconstruction.at` exists to
       make visible.
    2. **At most one origin may be unmarked.** Any additional distinct origin
       must have declared :meth:`Reconstruction.assume_independent`. The
       unmarked one is the *anchor*: its draw order is preserved, so the
       result stays aligned with the rest of that fit.
    3. **Independent members are permuted.** Their index order carries no
       information, and permuting guarantees that none leaks in.
    """

    if not recons:

        raise ValueError("Nothing to align.")

    # A one-point reconstruction is a scalar-valued posterior -- H0, the
    # curvature, a growth normalisation -- and broadcasts against a grid the
    # way a number does. Keeping it a Reconstruction rather than an array is
    # what preserves its correlation with the curve it normalises: in
    # ``H / H.at(0.0)`` the H0 uncertainty cancels draw by draw, exactly, and
    # E(z) comes out far tighter than either factor. That cancellation is the
    # reason the object exists.
    extended = [r for r in recons if r.n_z > 1]

    grid = extended[0].z if extended else recons[0].z

    for other in extended[1:]:

        if other.n_z != grid.size or not np.allclose(
            other.z, grid, rtol=_GRID_RTOL, atol=_GRID_ATOL
        ):

            span = (
                f"[{other.z[0]:.4g}, {other.z[-1]:.4g}] ({other.n_z} points)"
            )

            raise GridMismatchError(
                f"Cannot combine {extended[0]._describe()} on the grid "
                f"[{grid[0]:.4g}, {grid[-1]:.4g}] ({grid.size} points) with "
                f"{other._describe()} on {span}. Put them on one grid first "
                "with .at(z) -- which is also the right place to decide "
                "which redshift range a joint statement about them is "
                "actually defined on; see core.grid.common_support."
            )

    # -- origins -------------------------------------------------

    anchors: dict[Origin, Reconstruction] = {}

    for r in recons:

        if r.origin == CONSTANT_ORIGIN or r._independent:
            continue

        anchors.setdefault(r.origin, r)

    if len(anchors) > 1:

        listed = ", ".join(
            f"{r._describe()} [origin {o}]" for o, r in anchors.items()
        )

        raise AlignmentError(
            "These reconstructions come from different fits and were "
            f"combined without an independence claim: {listed}. Draw k of "
            "one is not draw k of the other, so pairing them by index would "
            "invent a correlation. If the fits genuinely share no data, say "
            "so with .assume_independent(); if they do share data, fit them "
            "together instead."
        )

    anchor_origin = next(iter(anchors), CONSTANT_ORIGIN)

    if anchor_origin != CONSTANT_ORIGIN:

        # There is an anchor fit, and its draw order is preserved below, so
        # the result stays aligned with everything else from that fit.
        result_origin = anchor_origin

    elif all(r.origin == CONSTANT_ORIGIN for r in recons):

        # Constants all the way down: still a constant.
        result_origin = CONSTANT_ORIGIN

    else:

        # Every contributing fit declared itself independent, so the result
        # is a realisation stream of its own, aligned with nothing.
        result_origin = new_origin()

    # -- draw counts ---------------------------------------------

    counts = [r.n_draws for r in recons]

    n_common = min(counts)

    same_origin = [
        r.n_draws for r in recons if r.origin == anchor_origin and not r._independent
    ]

    if same_origin and len(set(same_origin)) > 1:

        raise AlignmentError(
            "Reconstructions from one fit disagree on their number of draws "
            f"({sorted(set(same_origin))}). That should be impossible and "
            "means the realisation index no longer lines up; refit rather "
            "than truncating."
        )

    # -- assemble ------------------------------------------------

    aligned: list[DrawArray] = []

    for r in recons:

        draws = r.draws

        if r._independent and r.origin != CONSTANT_ORIGIN:

            rng = np.random.default_rng([int(r.origin), int(n_common)])

            draws = draws[rng.permutation(draws.shape[0])]

        if draws.shape[0] == 1 and n_common > 1:

            # A constant, or a degenerate one-draw curve: broadcast rather
            # than truncate everything else down to it.
            draws = np.repeat(draws, n_common, axis=0)

        if draws.shape[1] == 1 and grid.size > 1:

            # The scalar-valued case described above.
            draws = np.repeat(draws, grid.size, axis=1)

        aligned.append(draws[:n_common])

    return np.array(grid, dtype=float), result_origin, aligned


def _merge_provenance(
    recons: Sequence[Reconstruction],
    expression: str,
    n_draws: int,
) -> Provenance:
    """Compose the provenance of a derived quantity from its operands."""

    if not recons:

        return Provenance(expression=expression, n_draws=n_draws)

    first, *rest = (r.provenance for r in recons)

    return first.derive(
        expression,
        *rest,
        n_draws=n_draws,
    )


#: How long a derived label is allowed to get before it stops being useful in
#: a legend and starts being a wall of parentheses.
_MAX_LABEL = 48


def _operand_name(value: object) -> str:
    """
    An operand as it should appear inside an expression string.

    A derived reconstruction carries its own expression as its label, so
    nesting composes: ``H`` and ``H`` give ``(H / H)``, and squaring that
    gives ``((H / H) ** 2)`` rather than ``(f ** 2)``. The expression is how
    a reader of a figure caption works out what was plotted, so it is worth
    the bookkeeping.
    """

    if isinstance(value, Reconstruction):

        return value.label or value.provenance.expression or "f"

    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return f"{float(arr):.6g}"

    return f"array({arr.shape[-1]})"


def _as_label(expression: str) -> str:
    """The expression, trimmed to something a legend can hold."""

    if len(expression) <= _MAX_LABEL:
        return expression

    return expression[: _MAX_LABEL - 3] + "..."
