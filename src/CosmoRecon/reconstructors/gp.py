"""
Gaussian-process reconstruction.

The method this library's field runs on, with the two habits that make its
published error bars untrustworthy removed.

**Hyperparameters are marginalised, not optimised.** The standard recipe
maximises the marginal likelihood over the kernel amplitude and length scale
and then reconstructs at those values, as if they were known. They are not:
with thirty-odd cosmic chronometer measurements the length scale is poorly
determined, and conditioning on its best-fit value discards that ignorance and
reports an interval narrower than the data support. Here the hyperparameters
carry a posterior, and every sample path is drawn at hyperparameters drawn
from it.

**Smoothness is inferred, not assumed.** ``Matern(nu=1.5)`` and its
squared-exponential cousin are chosen by convention, and the convention is
never reported as an assumption -- even though kernel choice moves
reconstructions of ``H(z)`` by about as much as changing the cosmological
model does. The default kernel here marginalises ``nu``.

There is a third thing this module does that is not standard at all. A
Matern-``nu`` process is differentiable only ``ceil(nu) - 1`` times, so a
Matern-3/2 GP -- the most common single choice in the literature -- has no
second derivative, and the cosmographic quantities built from ``H''`` do not
exist under it. Numbers can still be produced, because any finite
representation of a sample path is smooth; they are not a posterior on
anything. Asking for a derivative past the limit raises here.

Implementation
--------------

Sample paths are carried in a fixed spectral basis, which is what satisfies
the determinism clause of the
:class:`~CosmoRecon.core.reconstruction.Predictor` contract: the frequencies,
phases and weights are set at fit time, so draw ``k`` is the *same function*
however many times and at whatever redshifts it is later evaluated. A GP that
re-samples on each call would satisfy every other requirement and quietly
destroy the alignment the whole library rests on.

The basis is a **quadrature** on the kernel's spectral density, not a sample
from it. Random Fourier features -- the usual construction -- converge as
``1 / sqrt(M)``, which in practice leaves several percent of error in the
*width* of the reconstructed band at any affordable feature count; measured
against the exact posterior, 8192 random features still misstate the standard
deviation by about 5%. An error bar is this library's output, so several
percent of it being a representation artefact is not acceptable. In one
dimension the spectral density is a one-dimensional integral, and quadrature
on it converges geometrically instead: a few hundred nodes reach a part in
``10^4`` or better, and the achieved error is *measured* at fit time rather
than assumed.

Within that basis the posterior is drawn by Matheron's rule -- prior sample,
then a data-dependent correction requiring only an ``n x n`` solve -- so the
cost is set by the number of measurements, not by the size of the basis. The
whole construction is checked against the exact posterior in the test suite;
:meth:`GaussianProcess.exact_posterior` is what it is checked against.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping

import numpy as np
from scipy import linalg
from scipy.special import logsumexp

from CosmoRecon.typing import Array, Redshift

from CosmoRecon.core.errors import DataError, DerivativeUnavailableError
from CosmoRecon.core.grid import check_within_support
from CosmoRecon.core.provenance import Origin, Provenance
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.reconstructors.base import Reconstructor, unpack_dataset
from CosmoRecon.reconstructors.kernels import Kernel, get_kernel


__all__ = ["GaussianProcess"]


#: Added to the diagonal before a Cholesky, relative to its mean. Data
#: covariances arrive from published papers and are occasionally singular to
#: working precision; this keeps a fit from dying on someone else's rounding.
_JITTER = 1e-10

#: The quantile of the smoothness posterior used to decide whether a
#: derivative exists. Taking the minimum drawn ``nu`` would let one draw in
#: two thousand veto a derivative the posterior overwhelmingly supports;
#: taking the median would let a third of the posterior be differentiated
#: when it cannot be. One percent is the point at which "the posterior does
#: not support this" stops being an overstatement.
_SMOOTHNESS_QUANTILE = 0.01


# ============================================================
# The predictor
# ============================================================

class _SpectralPaths:
    """
    A fixed set of sample paths, evaluable anywhere, differentiable
    analytically.

    Each path is a finite sum ``sum_m c_m cos(omega_m z + phi_m)`` plus a
    constant, so its ``n``-th derivative is
    ``sum_m c_m omega_m^n cos(omega_m z + phi_m + n pi / 2)`` -- closed form,
    at any order, at any redshift, and at no extra cost.

    The frequencies come from a quadrature on the kernel's spectral density
    rather than from sampling it, so the basis reproduces the kernel to a
    tolerance that was *checked* at fit time rather than hoped for. They are
    also fixed once, which is what satisfies the determinism clause of the
    :class:`~CosmoRecon.core.reconstruction.Predictor` contract: draw ``k`` is
    the same function on every grid it is ever evaluated on.

    That the *representation* is infinitely differentiable says nothing about
    the *process*. Paths of a rough kernel are approximated here by smooth
    functions, and differentiating them past the kernel's own smoothness gives
    a number with no posterior interpretation. :attr:`max_derivative` records
    that limit, and it is enforced before any such number is produced.
    """

    __slots__ = (
        "_omega",
        "_phase",
        "_amplitude",
        "_cell_of_draw",
        "_weights",
        "_mean_scale",
        "_support",
        "_label",
        "max_derivative",
        "quadrature_error",
        "_limit_of_draw",
    )

    def __init__(
        self,
        omega: Array,              # (n_cells, n_features)
        phase: Array,              # (n_cells, n_features)
        amplitude: Array,          # (n_cells, n_features), zero where padded
        cell_of_draw: Array,       # (n_draws,)
        weights: Array,            # (n_draws, n_features + 1)
        mean_scale: float,
        support: tuple[float, float],
        label: str,
        max_derivative: int | None,
        quadrature_error: float,
        limit_of_draw: Array,
    ) -> None:

        self._omega = omega
        self._phase = phase
        self._amplitude = amplitude
        self._cell_of_draw = cell_of_draw
        self._weights = weights
        self._mean_scale = mean_scale
        self._support = support
        self._label = label

        self.max_derivative = max_derivative

        #: Per draw, the highest derivative that draw's kernel supports. Kept
        #: so that a refusal can say how much of the posterior it is speaking
        #: for, rather than only that it refused.
        self._limit_of_draw = limit_of_draw

        #: The worst discrepancy between the quadrature basis and the exact
        #: kernel, over the separations the data span. Carried so it can be
        #: reported rather than assumed.
        self.quadrature_error = quadrature_error

    # ---------------------------------------------------------

    def __call__(self, z: Array, *, derivative: int = 0) -> Array:

        if self.max_derivative is not None and derivative > self.max_derivative:

            supported = float(np.mean(self._limit_of_draw >= derivative))

            raise DerivativeUnavailableError(
                f"Order {derivative} was asked for, and only "
                f"{100 * supported:.1f}% of this posterior has a derivative "
                f"that high. A Matern process with smoothness nu is "
                "differentiable ceil(nu) - 1 times, and the fitted posterior "
                "on nu reaches below that here -- so the quantity being built "
                "does not exist on part of the posterior it would be reported "
                "for. A number can be produced; it would describe the "
                "sample-path basis rather than the function. "
                "The fix is a prior, chosen deliberately rather than by "
                "convention: restrict the kernel to processes that have the "
                "derivative, e.g. Matern(nu=[1.5, 2.0, 2.5, 3.5, 5.0, 7.5]) "
                "for one derivative or nu >= 2.5 for two, or use the squared "
                "exponential, which has all of them. Restricting has to "
                "happen before the fit, because dropping the rough draws "
                "afterwards would break the realisation index that every "
                "later combination depends on."
            )

        z = np.atleast_1d(np.asarray(z, dtype=float))

        check_within_support(
            z, self._support, what=f"GP reconstruction of {self._label}"
        )

        out = np.empty((self._weights.shape[0], z.size), dtype=float)

        offset = 0.5 * math.pi * derivative

        for cell in np.unique(self._cell_of_draw):

            rows = np.flatnonzero(self._cell_of_draw == cell)

            omega = self._omega[cell][:, None]

            # (n_features, n_z): the basis, differentiated in closed form.
            # Padded features carry zero amplitude and drop out here.
            design = (
                self._amplitude[cell][:, None]
                * omega**derivative
                * np.cos(omega * z[None, :] + self._phase[cell][:, None] + offset)
            )

            out[rows] = self._weights[rows, :-1] @ design

            if derivative == 0:

                # The constant feature: a weak prior on the overall level, so
                # that the reconstruction is not pulled towards zero outside
                # the data. Its derivative is zero, hence the guard.
                out[rows] += self._weights[rows, -1:] * self._mean_scale

        return out


# ============================================================
# The reconstructor
# ============================================================

class GaussianProcess(Reconstructor):
    """
    Gaussian-process reconstruction with the hyperparameters marginalised.

    >>> gp = GaussianProcess(kernel="matern")            # doctest: +SKIP
    >>> fit = gp.fit(chronometers)                       # doctest: +SKIP
    >>> H = fit["H"]                                     # doctest: +SKIP
    >>> w = ...                                          # doctest: +SKIP

    Parameters
    ----------
    kernel
        A :class:`~CosmoRecon.reconstructors.kernels.Kernel` or its name.
        Named kernels come in their *free* form -- ``"matern"`` marginalises
        ``nu``. Pin it deliberately with ``Matern(nu=1.5)``.
    n_nodes, max_nodes, tol
        The sample-path basis. Each hyperparameter cell gets a quadrature on
        its own spectral density, refined from ``n_nodes`` up to ``max_nodes``
        until it reproduces that cell's kernel to ``tol`` over the separations
        the data span. Cells that cannot get there raise rather than
        approximate quietly.

        The defaults are not tuning knobs to reach for. ``tol = 3e-4`` puts
        the basis four orders of magnitude below any statistical uncertainty
        a reconstruction will quote, and every kernel in the default grids
        meets it inside ``max_nodes``.
    n_length, n_amplitude
        Grid resolution for the two universal hyperparameters, both
        log-uniform. The grid is the default because it needs no dependency
        and, in three dimensions or fewer, resolves the posterior better than
        a short chain would.
    length_range, amplitude_range
        Override the automatic ranges, as ``(lo, hi)``. The defaults are set
        from the data: length scales from a twentieth of the redshift span to
        twice it, amplitudes from a tenth to ten times the scatter of the
        measurements.
    """

    provides_evidence = True

    def __init__(
        self,
        kernel: Kernel | str = "matern",
        *,
        n_nodes: int = 256,
        max_nodes: int = 4096,
        tol: float = 3e-4,
        n_length: int = 24,
        n_amplitude: int = 16,
        length_range: tuple[float, float] | None = None,
        amplitude_range: tuple[float, float] | None = None,
    ) -> None:

        self.kernel = get_kernel(kernel)

        self.n_nodes = int(n_nodes)

        self.max_nodes = int(max_nodes)

        self.tol = float(tol)

        self.n_length = int(n_length)

        self.n_amplitude = int(n_amplitude)

        self.length_range = length_range

        self.amplitude_range = amplitude_range

        self._log_evidence: float | None = None

        self._cells: list[dict] | None = None

        self._log_weights: Array | None = None

    # ---------------------------------------------------------

    def describe(self) -> str:

        return self.kernel.describe()

    def hyperparameters(self) -> dict[str, object]:

        return {
            "kernel": type(self.kernel).__name__,
            "shape": {
                name: values.tolist()
                for name, values in self.kernel.shape_grid().items()
            },
            "n_nodes": self.n_nodes,
            "tol": self.tol,
            "n_length": self.n_length,
            "n_amplitude": self.n_amplitude,
            "length_range": self.length_range,
            "amplitude_range": self.amplitude_range,
        }

    @property
    def log_evidence(self) -> float:
        """
        Log marginal likelihood, marginalised over the hyperparameter grid.

        A quadrature on the grid this class defines, under log-uniform priors
        on amplitude and length scale and a uniform prior on the shape
        parameter -- so it is comparable between two kernels fitted to the
        same data with the same ranges, and not comparable to an evidence
        computed under different priors. Ensemble weighting uses it; nothing
        else should.
        """

        if self._log_evidence is None:

            raise RuntimeError("Nothing has been fitted yet.")

        return self._log_evidence

    # ---------------------------------------------------------
    # The grid
    # ---------------------------------------------------------

    def _build_grid(self, z: Array, y: Array) -> list[dict]:
        """
        Every hyperparameter cell, as a flat list.

        Kept flat rather than nested so that the posterior over it is an
        ordinary categorical distribution -- which is what makes drawing
        sample paths at hyperparameters drawn from the posterior a single
        line rather than a nested loop.
        """

        span = float(z.max() - z.min())

        if span <= 0.0:

            raise DataError("All measurements are at the same redshift.")

        lo, hi = self.length_range or (span / 20.0, 2.0 * span)

        lengths = np.geomspace(lo, hi, self.n_length)

        scatter = float(np.std(y, ddof=1))

        if scatter <= 0.0:

            raise DataError("The measurements have no scatter to fit.")

        alo, ahi = self.amplitude_range or (0.1 * scatter, 10.0 * scatter)

        amplitudes = np.geomspace(alo, ahi, self.n_amplitude)

        shape_grid = self.kernel.shape_grid()

        shape_names = tuple(shape_grid)

        shape_values = [
            dict(zip(shape_names, combo, strict=True))
            for combo in itertools.product(*shape_grid.values())
        ]

        return [
            {"length_scale": float(ell), "amplitude": float(amp), **shape}
            for shape in shape_values
            for ell in lengths
            for amp in amplitudes
        ]

    def _log_likelihoods(
        self,
        cells: list[dict],
        z: Array,
        y: Array,
        cov: Array,
        mean_scale: float,
    ) -> Array:
        """
        Exact GP log marginal likelihood for every cell.

        Exact, not random-feature: the weights on the hyperparameter
        posterior should not inherit the sampling approximation, and an
        ``n x n`` Cholesky per cell costs nothing at these data volumes.

        The kernel matrix depends only on ``(length_scale, shape)``, so it is
        built once per distinct pair and reused across amplitudes -- which is
        where most of the time would otherwise go, ``scipy.special.kv`` being
        the expensive part of a Matern.
        """

        r = np.abs(z[:, None] - z[None, :])

        n = z.size

        log_two_pi = n * math.log(2.0 * math.pi)

        # The mean is marginalised by carrying it as a very broad constant
        # component of the covariance, so it never appears as a parameter.
        base = cov + mean_scale**2

        out = np.empty(len(cells), dtype=float)

        cache: dict[tuple, Array] = {}

        for index, cell in enumerate(cells):

            shape = {k: v for k, v in cell.items()
                     if k not in ("length_scale", "amplitude")}

            key = (cell["length_scale"], *sorted(shape.items()))

            unit = cache.get(key)

            if unit is None:

                unit = self.kernel.covariance(r, cell["length_scale"], **shape)

                cache[key] = unit

            total = cell["amplitude"] ** 2 * unit + base

            total[np.diag_indices(n)] += _JITTER * np.trace(total) / n

            try:
                factor = linalg.cho_factor(total, lower=True)

            except linalg.LinAlgError:

                out[index] = -np.inf
                continue

            alpha = linalg.cho_solve(factor, y)

            log_det = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))

            out[index] = -0.5 * (float(y @ alpha) + log_det + log_two_pi)

        return out

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

        # A weak prior on the overall level, carried as one extra constant
        # feature. Without it the posterior decays towards zero outside the
        # data, which for H(z) is not ignorance but a claim.
        mean_scale = 10.0 * max(float(np.std(y, ddof=1)), abs(float(np.mean(y))))

        cells = self._build_grid(z, y)

        log_like = self._log_likelihoods(cells, z, y, cov, mean_scale)

        if not np.isfinite(log_like).any():

            raise DataError(
                "No hyperparameter cell gave a finite likelihood. That "
                "usually means the data covariance is not positive definite."
            )

        # Log-uniform priors on amplitude and length scale, uniform on the
        # shape parameter, are exactly uniform weights on this grid -- which
        # is why the grid is built in geomspace.
        self._log_evidence = float(logsumexp(log_like) - math.log(len(cells)))

        weights = np.exp(log_like - log_like.max())
        weights /= weights.sum()

        self._cells = cells
        self._log_weights = log_like

        draw_cells = rng.choice(len(cells), size=n_draws, p=weights)

        paths = self._sample_paths(
            draw_cells, cells, z, y, cov, mean_scale, rng,
            support=(float(z.min()), float(z.max())),
            label=label,
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

    def _sample_paths(
        self,
        draw_cells: Array,
        cells: list[dict],
        z: Array,
        y: Array,
        cov: Array,
        mean_scale: float,
        rng: np.random.Generator,
        *,
        support: tuple[float, float],
        label: str,
    ) -> _SpectralPaths:
        """
        Draw the sample paths, by Matheron's rule in the spectral basis.

        For a model ``f = Phi w`` with ``w ~ N(0, I)`` and noise covariance
        ``C``, a draw from the posterior is

            ``w_post = w_prior + Phi^T (Phi Phi^T + C)^-1 (y - Phi w_prior - e)``

        with ``w_prior ~ N(0, I)`` and ``e ~ N(0, C)``. It is exact for this
        model, and the only linear system is ``n x n`` -- the cost is set by
        how many measurements there are, not by how many features carry the
        path. Forming the posterior over the features directly would instead
        be a dense factorisation of a matrix the size of the basis, per cell.

        The basis itself is built per cell by
        :meth:`~CosmoRecon.reconstructors.kernels.Kernel.spectral_quadrature`,
        which refines until it reproduces that cell's kernel to a checked
        tolerance. Different cells need different numbers of nodes -- a rough
        Matern needs many more than a squared exponential -- so the arrays are
        sized by the largest and the unused entries carry zero amplitude.
        """

        used = np.unique(draw_cells)

        r_max = float(z.max() - z.min())

        # Build each cell's basis first: the feature count is whatever the
        # tolerance demands, and is not known until it has been reached.
        bases: dict[int, tuple[Array, Array, Array]] = {}

        worst_error = 0.0

        for cell_index in used:

            cell = cells[cell_index]

            shape = {k: v for k, v in cell.items()
                     if k not in ("length_scale", "amplitude")}

            omega, quad, error = self.kernel.spectral_quadrature(
                cell["length_scale"],
                r_max,
                tol=self.tol,
                n_start=self.n_nodes,
                n_max=self.max_nodes,
                **shape,
            )

            worst_error = max(worst_error, error)

            # One cosine and one sine per node, written as two cosines with a
            # quarter-turn between them so that the whole basis differentiates
            # by a single phase shift.
            root = cell["amplitude"] * np.sqrt(quad)

            bases[int(cell_index)] = (
                np.concatenate([omega, omega]),
                np.concatenate([np.zeros_like(omega),
                                np.full_like(omega, -0.5 * math.pi)]),
                np.concatenate([root, root]),
            )

        n_features = max(b[0].size for b in bases.values())

        omega_of_cell = np.zeros((len(cells), n_features))
        phase_of_cell = np.zeros((len(cells), n_features))
        amp_of_cell = np.zeros((len(cells), n_features))

        weights = np.zeros((draw_cells.size, n_features + 1))

        noise_chol = linalg.cholesky(
            cov + _JITTER * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0]),
            lower=True,
        )

        smoothness: list[int | None] = []

        for cell_index in used:

            cell = cells[cell_index]

            shape = {k: v for k, v in cell.items()
                     if k not in ("length_scale", "amplitude")}

            smoothness.append(self.kernel.max_derivative(**shape))

            omega, phase, amp = bases[int(cell_index)]

            width = omega.size

            omega_of_cell[cell_index, :width] = omega
            phase_of_cell[cell_index, :width] = phase
            amp_of_cell[cell_index, :width] = amp

            rows = np.flatnonzero(draw_cells == cell_index)

            # Design at the data, including the constant feature.
            design = np.zeros((z.size, n_features + 1))

            design[:, :width] = amp[None, :] * np.cos(
                z[:, None] * omega[None, :] + phase[None, :]
            )

            design[:, n_features] = mean_scale

            prior = rng.standard_normal((rows.size, n_features + 1))

            noise = rng.standard_normal((rows.size, z.size)) @ noise_chol.T

            residual = y[None, :] - prior @ design.T - noise

            gram = design @ design.T + cov

            gram[np.diag_indices(z.size)] += _JITTER * np.trace(gram) / z.size

            factor = linalg.cho_factor(gram, lower=True)

            alpha = linalg.cho_solve(factor, residual.T).T

            weights[rows] = prior + alpha @ design

        # The posterior's own answer on how many derivatives exist.
        #
        # A cell reporting None is analytic, so it is entered as a sentinel
        # far above any order anyone will ask for, and a mixture that is
        # entirely analytic comes back out as None. Taking a low quantile
        # rather than the minimum keeps one improbable rough draw from
        # vetoing a derivative the posterior otherwise supports; see the note
        # on _SMOOTHNESS_QUANTILE.
        unlimited = 1 << 20

        per_cell = np.full(len(cells), unlimited, dtype=np.int64)

        for cell_index, smooth in zip(used, smoothness, strict=True):

            if smooth is not None:
                per_cell[cell_index] = smooth

        limit = int(
            np.quantile(
                per_cell[draw_cells].astype(float),
                _SMOOTHNESS_QUANTILE,
                method="lower",
            )
        )

        return _SpectralPaths(
            omega=omega_of_cell,
            phase=phase_of_cell,
            amplitude=amp_of_cell,
            cell_of_draw=draw_cells,
            weights=weights,
            mean_scale=mean_scale,
            support=support,
            label=label,
            max_derivative=None if limit >= unlimited else limit,
            quadrature_error=worst_error,
            limit_of_draw=per_cell[draw_cells],
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def exact_posterior(
        self,
        data,
        z_star: Redshift,
    ) -> tuple[Array, Array]:
        """
        The exact GP posterior mean and covariance at ``z_star``, marginalised
        over the same hyperparameter grid the draws were taken from.

        Nothing in the library uses this; the test suite does. It is the
        second way of computing what :meth:`fit` computes the first way, and
        the agreement between them is what says the random-feature basis is
        fine enough. Anything that can be checked two ways is checked two
        ways.
        """

        z, y, cov, _ = unpack_dataset(data)

        z_star = np.atleast_1d(np.asarray(z_star, dtype=float))

        mean_scale = 10.0 * max(float(np.std(y, ddof=1)), abs(float(np.mean(y))))

        cells = self._build_grid(z, y)

        log_like = self._log_likelihoods(cells, z, y, cov, mean_scale)

        weights = np.exp(log_like - log_like.max())
        weights /= weights.sum()

        # This is a validation routine, and an exact posterior per cell over a
        # fine grid is the one genuinely expensive thing in the module. Cells
        # are visited heaviest-first and stopped once the remainder cannot
        # move the answer, which turns a sweep over the whole grid into a
        # sweep over the part of it the data actually support.
        order = np.argsort(weights)[::-1]

        keep = order[: 1 + int(np.searchsorted(np.cumsum(weights[order]), 1.0 - 1e-6))]

        r_dd = np.abs(z[:, None] - z[None, :])
        r_sd = np.abs(z_star[:, None] - z[None, :])
        r_ss = np.abs(z_star[:, None] - z_star[None, :])

        mean = np.zeros(z_star.size)

        second = np.zeros((z_star.size, z_star.size))

        for index in keep:

            weight = weights[index]

            cell = cells[index]

            shape = {k: v for k, v in cell.items()
                     if k not in ("length_scale", "amplitude")}

            amp2 = cell["amplitude"] ** 2

            K_dd = amp2 * self.kernel.covariance(r_dd, cell["length_scale"], **shape) \
                + cov + mean_scale**2
            K_sd = amp2 * self.kernel.covariance(r_sd, cell["length_scale"], **shape) \
                + mean_scale**2
            K_ss = amp2 * self.kernel.covariance(r_ss, cell["length_scale"], **shape) \
                + mean_scale**2

            K_dd[np.diag_indices(z.size)] += _JITTER * np.trace(K_dd) / z.size

            factor = linalg.cho_factor(K_dd, lower=True)

            m = K_sd @ linalg.cho_solve(factor, y)

            C = K_ss - K_sd @ linalg.cho_solve(factor, K_sd.T)

            mean += weight * m

            second += weight * (C + np.outer(m, m))

        return mean, second - np.outer(mean, mean)

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return f"<GaussianProcess {self.kernel!r} tol={self.tol:g}>"
