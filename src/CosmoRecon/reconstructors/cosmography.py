"""
Cosmographic reconstruction: series expansions of an observable.

The oldest model-independent method there is, and the one with the sharpest
failure mode. Three things are done differently here from what the literature
does.

**The series has to converge where the data are.** A cosmological distance has
a singularity at ``z = -1``, so its Taylor series in ``z`` has radius of
convergence ``|z| = 1`` -- and beyond that radius, keeping more terms makes the
truncation worse rather than better. Supernova compilations reach ``z ~ 2.3``.
Fitting a ``z``-series to them and reading off a jerk parameter is not an
approximation with a large error bar; it is not an approximation at all. This
module refuses, names the redshift at which the fit left the radius, and points
at the variable that fixes it -- ``y = z / (1 + z)``, which is the default here
and which exists for exactly this reason (Cattoen & Visser 2007).

**The order is marginalised, not chosen.** Cosmographic analyses pick an order
by habit or by an information criterion applied after the fact, and then quote
an interval conditioned on that choice. The model here is linear in its
coefficients, so the evidence for each order is available in closed form, and
the order can be carried as a posterior like any other unknown. That is the
same argument this library makes about a Gaussian process's kernel, in the
place where it happens to be cheap.

**Two of the three usual "different" expansions are the same expansion.** It is
standard to compare Taylor, Chebyshev and Pade as three alternatives. But a
degree-``N`` monomial series and a degree-``N`` Chebyshev series in the same
variable span *the same space of functions* -- they are two coordinate systems
on one model, and with a flat prior they give the identical posterior. They
differ here only through the prior on their coefficients, and through
conditioning, where Chebyshev wins by a wide margin and is the default. The
genuinely distinct choices are the expansion *variable* and going rational
(Pade).

Implementation
--------------

For a polynomial family the model is linear and Gaussian, so there is no
approximation anywhere: the coefficient posterior is exact in closed form, and
a draw is a coefficient vector. Evaluating it at a new redshift is a matrix
product, and the derivative to any order is the basis differentiated in closed
form composed with the change of variable through Faa di Bruno -- see
:mod:`CosmoRecon.reconstructors.series`.

A Pade approximant is built from the drawn coefficients of the underlying
series, one draw at a time, which is how the cosmographic literature builds
them and which propagates the uncertainty exactly. The evidence reported for a
Pade fit is therefore the evidence of the series it was re-expanded from,
stated rather than implied. Draws whose denominator has a root inside the
fitted range are rejected and redrawn: an approximant with a pole in the
observable range is not an expansion history, and the acceptance rate is
reported so that a low one is visible rather than silent.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P
from scipy import linalg
from scipy.special import logsumexp

from CosmoRecon.typing import Array

from CosmoRecon.core.errors import ConvergenceError, DataError
from CosmoRecon.core.grid import check_within_support
from CosmoRecon.core.provenance import Origin, Provenance
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.reconstructors.base import Reconstructor, unpack_dataset
from CosmoRecon.reconstructors.series import (
    ExpansionVariable,
    compose_derivatives,
    design_matrix,
    get_variable,
)


__all__ = ["Cosmography"]


_JITTER = 1e-10

#: Orders marginalised over when none is given. Starts at 1 because a linear
#: fit is a legitimate member of the comparison, and stops at 6 because beyond
#: that the evidence for a cosmographic series is invariably falling and the
#: extra coefficients are prior.
DEFAULT_ORDER_GRID = (1, 2, 3, 4, 5, 6)

#: A Pade fit below this acceptance rate is refused: the posterior is mostly
#: approximants with a pole in the observable range, which means the orders
#: asked for are not supported by the data.
_MIN_ACCEPTANCE = 0.2


# ============================================================
# The predictor
# ============================================================

class _SeriesPaths:
    """
    A fixed set of coefficient vectors, evaluated as functions.

    Deterministic by construction -- the coefficients *are* the draws, so
    draw ``k`` is the same function at every redshift it is asked about, which
    is what the :class:`~CosmoRecon.core.reconstruction.Predictor` contract
    requires. Nothing here is sampled at evaluation time.
    """

    __slots__ = (
        "_coefficients",
        "_degree_of_draw",
        "_variable",
        "_family",
        "_slope",
        "_offset",
        "_column_scale",
        "_pade",
        "_support",
        "_convergence",
        "_label",
        "max_derivative",
        "acceptance",
    )

    def __init__(
        self,
        coefficients: Array,          # (n_draws, n_terms), zero-padded
        degree_of_draw: Array,        # (n_draws,)
        variable: ExpansionVariable,
        family: str,
        slope: float,
        offset: float,
        column_scale: Array,          # (n_terms,)
        support: tuple[float, float],
        convergence: tuple[float, float],
        label: str,
        pade: tuple[Array, Array] | None = None,   # (P, Q) coefficients
        acceptance: float = 1.0,
    ) -> None:

        self._coefficients = coefficients
        self._degree_of_draw = degree_of_draw
        self._variable = variable
        self._family = family

        #: The affine map from the expansion variable onto [-1, 1] over the
        #: fitted range. Chebyshev needs it; the monomials merely benefit.
        self._slope = slope
        self._offset = offset

        #: Columns are normalised to unit RMS over the data before fitting, so
        #: that one prior width means the same thing for every term. Undone
        #: here rather than folded into the coefficients, so that the stored
        #: numbers stay the ones the prior was written on.
        self._column_scale = column_scale

        self._support = support
        self._convergence = convergence
        self._label = label

        self._pade = pade

        #: A polynomial and a rational function are both analytic, so every
        #: derivative exists. Unlike a Gaussian process, there is nothing to
        #: gate here.
        self.max_derivative = None

        #: Fraction of Pade draws that had no pole in the fitted range. 1.0
        #: for a polynomial fit.
        self.acceptance = acceptance

    # ---------------------------------------------------------

    def __call__(self, z: Array, *, derivative: int = 0) -> Array:

        z = np.atleast_1d(np.asarray(z, dtype=float))

        lo, hi = self._convergence

        outside = z[(z < lo) | (z > hi)]

        if outside.size:

            raise ConvergenceError(
                f"A series in {self._variable.name} converges for "
                f"{lo:g} <= z <= {hi:g}, and evaluation was asked for at "
                f"z = {outside.min():.4g} to {outside.max():.4g}. Outside "
                "that range the truncated series does not approximate the "
                f"function -- {self._variable.why()}."
            )

        check_within_support(
            z, self._support, what=f"cosmographic reconstruction of {self._label}"
        )

        u = self._slope * self._variable(z) + self._offset

        if self._pade is not None:
            values = self._pade_derivatives(u, derivative)

        else:
            values = self._polynomial_derivatives(u, derivative)

        if derivative == 0:
            return values[0]

        # d/dz of the affine map times the variable's own derivatives.
        inner = [
            self._slope * d
            for d in self._variable.derivatives(z, derivative)
        ]

        return compose_derivatives(values, inner, derivative)

    # ---------------------------------------------------------

    def _polynomial_derivatives(self, u: Array, order: int) -> list[Array]:
        """
        ``[f(u)]`` for order 0, else ``[f'(u), ..., f^(order)(u)]``.

        Padding is harmless: a draw fitted at degree three carries zeros above
        it, and a zero coefficient contributes nothing at any derivative
        order.
        """

        degree = self._coefficients.shape[1] - 1

        scaled = self._coefficients / self._column_scale[None, :]

        if order == 0:

            return [scaled @ design_matrix(u, degree, self._family)]

        return [
            scaled @ design_matrix(u, degree, self._family, derivative=k)
            for k in range(1, order + 1)
        ]

    def _pade_derivatives(self, u: Array, order: int) -> list[Array]:
        """
        The same for ``f = P / Q``, by the recursion

            ``f^(n) = [P^(n) - sum_k C(n,k) Q^(k) f^(n-k)] / Q``

        which needs only polynomial derivatives of the numerator and
        denominator and works at any order.
        """

        p_coefficients, q_coefficients = self._pade

        # (n_draws, n_z) for each derivative order of P and Q.
        def poly_derivatives(coefficients: Array, n: int) -> list[Array]:

            out = []

            for k in range(n + 1):

                c = (
                    coefficients
                    if k == 0
                    else P.polyder(coefficients, m=k, axis=0)
                )

                out.append(
                    P.polyval(u, c)
                    if c.size
                    else np.zeros((coefficients.shape[1], u.size))
                )

            return out

        p_values = poly_derivatives(p_coefficients, order)
        q_values = poly_derivatives(q_coefficients, order)

        if np.any(np.abs(q_values[0]) < 1e-12 * np.max(np.abs(q_values[0]))):

            raise ConvergenceError(
                "A Pade denominator vanishes inside the requested redshift "
                "range for at least one draw, so the approximant has a pole "
                "there and its value is meaningless. Reduce the denominator "
                "order, or evaluate over a narrower range."
            )

        f = [p_values[0] / q_values[0]]

        for n in range(1, order + 1):

            total = p_values[n].copy()

            for k in range(1, n + 1):

                total -= math.comb(n, k) * q_values[k] * f[n - k]

            f.append(total / q_values[0])

        return f if order == 0 else f[1:]

    def __repr__(self) -> str:

        kind = "Pade" if self._pade is not None else self._family

        return f"<_SeriesPaths {kind} in {self._variable.name}>"


# ============================================================
# The reconstructor
# ============================================================

class Cosmography(Reconstructor):
    """
    Series reconstruction, with the order marginalised and the radius of
    convergence enforced.

    >>> Cosmography().fit(chronometers)                       # doctest: +SKIP
    >>> Cosmography("z").fit(supernovae)                      # doctest: +SKIP
    ConvergenceError: ... reaches z = 2.26, outside |z| < 1 ...
    >>> Cosmography(pade=(2, 1)).fit(chronometers)            # doctest: +SKIP

    Parameters
    ----------
    variable
        ``"y"`` (default), ``"z"`` or ``"log"``, or an
        :class:`~CosmoRecon.reconstructors.series.ExpansionVariable`. This is
        the choice that matters: it decides where the series converges. The
        default is the one that converges everywhere data exist.
    order
        ``None`` marginalises over :data:`DEFAULT_ORDER_GRID`; an integer pins;
        a sequence marginalises over exactly that set. Ignored when ``pade``
        is given, which pins the order to ``m + n``.
    family
        ``"chebyshev"`` (default) or ``"monomial"``. The same function space
        either way -- see the module docstring -- but Chebyshev is far better
        conditioned and its coefficient prior is the more sensible one.
    pade
        ``(m, n)`` to re-expand the fitted series as a Pade approximant of
        numerator degree ``m`` and denominator degree ``n``.
    strict
        Whether leaving the radius of convergence is an error. ``False``
        downgrades it to a warning, which exists so that a published analysis
        can be reproduced, not so that a new one can be written.
    n_scale, scale_range
        The grid for the coefficient prior width, marginalised like everything
        else. The default range is set from the scatter and level of the data.
    """

    provides_evidence = True

    def __init__(
        self,
        variable: ExpansionVariable | str = "y",
        *,
        order: int | Sequence[int] | None = None,
        family: str = "chebyshev",
        pade: tuple[int, int] | None = None,
        strict: bool = True,
        n_scale: int = 20,
        scale_range: tuple[float, float] | None = None,
    ) -> None:

        self.variable = get_variable(variable)

        self.family = str(family).lower()

        if self.family not in ("chebyshev", "monomial"):

            raise ValueError(
                f"Unknown family {family!r}; use 'chebyshev' or 'monomial'."
            )

        if pade is not None:

            m, n = int(pade[0]), int(pade[1])

            if m < 0 or n < 1:

                raise ValueError(
                    f"A Pade approximant needs m >= 0 and n >= 1, got {pade}. "
                    "With n = 0 it is a polynomial -- ask for one."
                )

            pade = (m, n)

        self.pade = pade

        self.order = order

        self.strict = bool(strict)

        self.n_scale = int(n_scale)

        self.scale_range = scale_range

        self._log_evidence: float | None = None

        self._cells: list[tuple[int, float]] | None = None

        self._log_weights: Array | None = None

        self._acceptance: float | None = None

    # ---------------------------------------------------------

    def describe(self) -> str:

        if self.pade is not None:

            return f"Pade[{self.pade[0]}/{self.pade[1]}]({self.variable.name})"

        family = self.family.capitalize()

        if self.order is None:
            order = "order free"

        elif np.isscalar(self.order):
            order = f"order {int(self.order)}"

        else:
            grid = sorted(int(o) for o in self.order)
            order = f"order in [{grid[0]}, {grid[-1]}]"

        return f"{family}({self.variable.name}, {order})"

    def hyperparameters(self) -> dict[str, object]:

        return {
            "variable": type(self.variable).__name__,
            "family": self.family,
            "order": self._order_grid(),
            "pade": self.pade,
            "strict": self.strict,
            "n_scale": self.n_scale,
            "scale_range": self.scale_range,
        }

    @property
    def log_evidence(self) -> float:
        """
        Log marginal likelihood of the fitted series, marginalised over order
        and prior width.

        For a Pade fit this is the evidence of the **series it was re-expanded
        from**, not of the rational function. The two are different models of
        the data and this number describes the first. Said out loud because a
        Pade evidence quoted without that caveat would be wrong in a way an
        evidence-weighted ensemble would happily act on.
        """

        if self._log_evidence is None:

            raise RuntimeError("Nothing has been fitted yet.")

        return self._log_evidence

    @property
    def acceptance(self) -> float:
        """
        Fraction of Pade draws with no pole inside the fitted range. ``1.0``
        for a polynomial fit.
        """

        if self._acceptance is None:

            raise RuntimeError("Nothing has been fitted yet.")

        return self._acceptance

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    def _order_grid(self) -> list[int]:

        if self.pade is not None:
            return [self.pade[0] + self.pade[1]]

        if self.order is None:
            return list(DEFAULT_ORDER_GRID)

        if np.isscalar(self.order):
            return [int(self.order)]

        return sorted({int(o) for o in self.order})

    def _check_convergence(self, z: Array) -> None:
        """
        Refuse a fit whose data leave the radius of convergence.

        The check is on the *data*, not on the plotting grid, because that is
        where the damage is done: coefficients fitted to points outside the
        radius are not the coefficients of the function's expansion, and every
        cosmographic parameter read off them is meaningless however narrow the
        grid it is later plotted on.
        """

        lo, hi = self.variable.convergence_range()

        worst = float(np.max(z))

        if worst <= hi and float(np.min(z)) >= lo:
            return

        message = (
            f"The data reach z = {worst:.4g}, outside the range "
            f"{lo:g} <= z <= {hi:g} where a series in "
            f"{self.variable.name} converges: {self.variable.why()}. "
            "Fitting here does not give a poorly determined expansion, it "
            "gives coefficients that are not the function's. Use "
            "variable='y' (the default), which converges over every redshift "
            "anyone has data at, or restrict the data."
        )

        if self.strict:
            raise ConvergenceError(message)

        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=3)

    # ---------------------------------------------------------
    # The fit
    # ---------------------------------------------------------

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

        z, y, cov, label = unpack_dataset(data)

        self._check_convergence(z)

        orders = [o for o in self._order_grid() if o + 1 <= z.size - 1]

        if not orders:

            raise DataError(
                f"{z.size} measurements cannot support any of the requested "
                f"orders {self._order_grid()}. A series with as many "
                "coefficients as there are data points is the prior wearing a "
                "polynomial."
            )

        degree = max(orders)

        # The affine map onto [-1, 1] over the range the data span in the
        # expansion variable. Chebyshev is only orthogonal there, and the
        # monomials are merely far better behaved.
        x_data = self.variable(z)

        x_lo, x_hi = float(x_data.min()), float(x_data.max())

        if x_hi <= x_lo:

            raise DataError("All measurements are at the same redshift.")

        slope = 2.0 / (x_hi - x_lo)

        offset = -(x_hi + x_lo) / (x_hi - x_lo)

        u_data = slope * x_data + offset

        full = design_matrix(u_data, degree, self.family)      # (degree+1, n)

        # Unit RMS per column, so that one prior width means the same thing
        # for the constant term and for the highest one.
        column_scale = np.sqrt(np.mean(full**2, axis=1))

        column_scale[column_scale <= 0.0] = 1.0

        design = full / column_scale[:, None]

        # -- the pieces every cell reuses -------------------------

        cov = cov + _JITTER * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])

        chol = linalg.cho_factor(cov, lower=True)

        log_det_cov = 2.0 * float(np.sum(np.log(np.diag(chol[0]))))

        cov_inv_y = linalg.cho_solve(chol, y)

        cov_inv_design = linalg.cho_solve(chol, design.T)       # (n, degree+1)

        A_full = design @ cov_inv_design                        # (p, p)

        b_full = design @ cov_inv_y                             # (p,)

        y_cov_y = float(y @ cov_inv_y)

        scales = self._scale_grid(y)

        cells = list(itertools.product(orders, scales))

        log_like = np.empty(len(cells))

        posteriors: list[tuple[Array, Array]] = []

        two_pi = z.size * math.log(2.0 * math.pi)

        for index, (order, scale) in enumerate(cells):

            p = order + 1

            A = A_full[:p, :p]
            b = b_full[:p]

            # Matrix determinant lemma on M = s^2 Phi^T Phi + C: everything
            # stays (order + 1) square, so the whole grid costs nothing.
            G = np.eye(p) + scale**2 * A

            sign, log_det_G = np.linalg.slogdet(G)

            if sign <= 0:

                log_like[index] = -np.inf
                posteriors.append((np.zeros(p), np.zeros((p, p))))
                continue

            quadratic = y_cov_y - scale**2 * float(b @ np.linalg.solve(G, b))

            log_like[index] = -0.5 * (
                quadratic + log_det_cov + log_det_G + two_pi
            )

            precision = A + np.eye(p) / scale**2

            covariance = np.linalg.inv(precision)

            posteriors.append((covariance @ b, covariance))

        if not np.isfinite(log_like).any():

            raise DataError(
                "No order and prior width gave a finite likelihood, which "
                "usually means the data covariance is not positive definite."
            )

        self._log_evidence = float(logsumexp(log_like) - math.log(len(cells)))

        self._cells = cells

        self._log_weights = log_like

        weights = np.exp(log_like - log_like.max())
        weights /= weights.sum()

        # -- draw ------------------------------------------------

        coefficients, degree_of_draw, pade, acceptance = self._draw(
            weights, cells, posteriors, degree, n_draws, rng,
            u_range=(float(u_data.min()), float(u_data.max())),
            column_scale=column_scale,
        )

        self._acceptance = acceptance

        paths = _SeriesPaths(
            coefficients=coefficients,
            degree_of_draw=degree_of_draw,
            variable=self.variable,
            family=self.family,
            slope=slope,
            offset=offset,
            column_scale=column_scale,
            support=(float(z.min()), float(z.max())),
            convergence=self.variable.convergence_range(),
            label=label,
            pade=pade,
            acceptance=acceptance,
        )

        return {
            label: Reconstruction.from_predictor(
                grid,
                paths,
                provenance=provenance,
                origin=origin,
                label=label,
                unit=str(getattr(data, "unit", "")),
            )
        }

    # ---------------------------------------------------------

    def _scale_grid(self, y: Array) -> Array:
        """
        Prior widths for the coefficients, log-uniform.

        The columns are unit-RMS, so a coefficient is the contribution of its
        basis function to the observable in the observable's own units. The
        range therefore runs from a small fraction of the data's level to well
        above it, and the level has to include the mean: a quantity of 100
        with a scatter of 1 still needs a constant term near 100.
        """

        level = abs(float(np.mean(y))) + float(np.std(y, ddof=1))

        if level <= 0.0:

            raise DataError("The measurements have no scale to fit.")

        lo, hi = self.scale_range or (0.05 * level, 20.0 * level)

        return np.geomspace(lo, hi, self.n_scale)

    def _draw(
        self,
        weights: Array,
        cells: list[tuple[int, float]],
        posteriors: list[tuple[Array, Array]],
        degree: int,
        n_draws: int,
        rng: np.random.Generator,
        *,
        u_range: tuple[float, float],
        column_scale: Array,
    ) -> tuple[Array, Array, tuple[Array, Array] | None, float]:
        """
        Coefficient draws, and their Pade re-expansion where one was asked for.
        """

        if self.pade is None:

            cell_index = rng.choice(len(cells), size=n_draws, p=weights)

            return (
                *self._coefficients_for(cell_index, cells, posteriors, degree, rng),
                None,
                1.0,
            )

        # Rejection sampling: an approximant with a pole in the fitted range
        # is not an expansion history, so those draws are replaced rather than
        # kept and hidden inside a wide quantile. Replacing keeps the draw
        # count and the realisation index intact, which dropping would not.
        probe = np.linspace(u_range[0], u_range[1], 200)

        kept_coefficients = []
        kept_degrees = []
        kept_p = []
        kept_q = []

        attempted = 0
        accepted = 0

        for _ in range(20):

            wanted = n_draws - accepted

            if wanted <= 0:
                break

            batch = max(wanted * 2, 256)

            cell_index = rng.choice(len(cells), size=batch, p=weights)

            coefficients, degrees = self._coefficients_for(
                cell_index, cells, posteriors, degree, rng
            )

            p_coeff, q_coeff = self._to_pade(coefficients, column_scale)

            # A real root inside the range shows up as a sign change.
            q_on_probe = P.polyval(probe, q_coeff)             # (batch, n_probe)

            good = ~np.any(
                np.sign(q_on_probe[:, :-1]) != np.sign(q_on_probe[:, 1:]),
                axis=1,
            )

            attempted += batch
            accepted += int(good.sum())

            kept_coefficients.append(coefficients[good])
            kept_degrees.append(degrees[good])
            kept_p.append(p_coeff[:, good])
            kept_q.append(q_coeff[:, good])

        rate = accepted / attempted if attempted else 0.0

        if accepted < n_draws:

            raise ConvergenceError(
                f"Only {accepted} of {attempted} Pade[{self.pade[0]}/"
                f"{self.pade[1]}] draws had no pole inside the fitted range "
                f"(acceptance {100 * rate:.1f}%), short of the {n_draws} "
                "asked for. The data do not support a denominator of this "
                "order: most of the posterior is approximants with a "
                "singularity where the observations are. Reduce the "
                "denominator degree, or use a polynomial."
            )

        if rate < _MIN_ACCEPTANCE:

            raise ConvergenceError(
                f"Pade[{self.pade[0]}/{self.pade[1]}] accepted only "
                f"{100 * rate:.1f}% of draws -- the rest had a pole inside "
                "the fitted range. A posterior that mostly consists of "
                "singular approximants is not a reconstruction of an "
                "expansion history. Reduce the denominator degree."
            )

        coefficients = np.concatenate(kept_coefficients)[:n_draws]
        degrees = np.concatenate(kept_degrees)[:n_draws]

        p_coeff = np.concatenate(kept_p, axis=1)[:, :n_draws]
        q_coeff = np.concatenate(kept_q, axis=1)[:, :n_draws]

        return coefficients, degrees, (p_coeff, q_coeff), rate

    def _coefficients_for(
        self,
        cell_index: Array,
        cells: list[tuple[int, float]],
        posteriors: list[tuple[Array, Array]],
        degree: int,
        rng: np.random.Generator,
    ) -> tuple[Array, Array]:
        """
        Draw coefficients from each selected cell's exact Gaussian posterior.

        Zero-padded to the largest degree on the grid, so that draws taken at
        different orders live in one array and stay index-aligned.
        """

        out = np.zeros((cell_index.size, degree + 1))

        degrees = np.empty(cell_index.size, dtype=int)

        for index in np.unique(cell_index):

            rows = np.flatnonzero(cell_index == index)

            order, _ = cells[index]

            mean, covariance = posteriors[index]

            p = order + 1

            chol = np.linalg.cholesky(
                covariance + _JITTER * np.trace(covariance) / p * np.eye(p)
            )

            out[np.ix_(rows, np.arange(p))] = (
                mean[None, :] + rng.standard_normal((rows.size, p)) @ chol.T
            )

            degrees[rows] = order

        return out, degrees

    def _to_pade(
        self,
        coefficients: Array,
        column_scale: Array,
    ) -> tuple[Array, Array]:
        """
        Turn each draw's series coefficients into a Pade numerator and
        denominator, in the scaled variable ``u``.

        The standard construction: match the rational function's own series to
        the given one through order ``m + n``. Done per draw, so the
        uncertainty passes through the non-linear map exactly rather than by
        linearising it.
        """

        m, n = self.pade

        # Back to monomial coefficients in u: undo the column normalisation,
        # then the Chebyshev basis if that is what was fitted.
        series = coefficients / column_scale[None, :]

        if self.family == "chebyshev":

            series = np.stack(
                [C.cheb2poly(row) for row in series],
            )

        # Pad or trim to exactly m + n + 1 terms.
        c = np.zeros((series.shape[0], m + n + 1))

        width = min(series.shape[1], m + n + 1)

        c[:, :width] = series[:, :width]

        # Denominator: sum_j b_j c[m + i - j] = -c[m + i], i = 1..n, b_0 = 1.
        matrix = np.zeros((series.shape[0], n, n))

        rhs = np.zeros((series.shape[0], n))

        for i in range(1, n + 1):

            for j in range(1, n + 1):

                k = m + i - j

                if 0 <= k <= m + n:
                    matrix[:, i - 1, j - 1] = c[:, k]

            rhs[:, i - 1] = -c[:, m + i]

        b = np.linalg.solve(
            matrix + _JITTER * np.eye(n)[None, :, :],
            rhs[:, :, None],
        )[:, :, 0]

        q = np.concatenate([np.ones((series.shape[0], 1)), b], axis=1)

        # Numerator: a_i = sum_{j=0}^{min(i, n)} b_j c[i - j].
        a = np.zeros((series.shape[0], m + 1))

        for i in range(m + 1):

            for j in range(min(i, n) + 1):

                a[:, i] += q[:, j] * c[:, i - j]

        # numpy.polynomial wants the coefficient axis first.
        return a.T, q.T

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return f"<Cosmography {self.describe()}>"
