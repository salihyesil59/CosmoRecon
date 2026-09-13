"""
Null models: small parametric universes in which a null hypothesis holds by
construction.

A null test asks how often a statistic this extreme would arise **if the
hypothesis were true**. The nominal chi-square in
:func:`~CosmoRecon.consistency.base.significance` answers a different question
-- how far a posterior mean sits from the null in units of the posterior width
-- and the two agree only when the reconstruction is unbiased and its posterior
width is its sampling error. Measured on mocks where the null holds exactly,
neither is true for any method in the library: a Gaussian process's posterior
is wider than its sampling error, and a series with its order left free is
biased by two to four of its own standard deviations in some directions. So the
only way to know how an analysis behaves under the null is to make the null
true and run the analysis.

That needs a universe to make it true in. A null model is the smallest family
that satisfies the hypothesis exactly and can predict every observable the
analysis consumed -- flat Lambda-CDM for the Om diagnostics, Lambda-CDM with
curvature for the Clarkson-Bassett-Lu test (FLRW, whatever the curvature),
Lambda-CDM distances for distance duality (which it satisfies automatically,
since both distances come from one metric). It is fitted to the **same data**
the analysis used, and mock truths are drawn from its parameter posterior
rather than fixed at the best fit, so the calibration does not pretend to know
the null universe better than the data do.

This is a model dependence, and it is stated rather than hidden: a calibrated
significance is the significance *at* the fitted null model, and every result
built on one carries that model's name and parameters.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg, optimize

from CosmoRecon.typing import Array

from CosmoRecon.core.errors import DataError


__all__ = ["NullModel", "NullFit", "LambdaCDM"]


#: Observables measured as a ratio to the sound horizon.
_BAO = frozenset({"DM_over_rs", "DH_over_rs", "DV_over_rs"})

#: Supernova distance moduli, raw or reduced.
_MODULI = frozenset({"mu", "mu_reduced"})

#: The growth rate of structure.
_GROWTH = "fsigma8"

#: Points in ``u = a^(5/2)`` for the growth integral. The integrand is smooth in
#: ``u`` all the way to ``a = 0``, and the trapezoid rule on this many points is
#: good to a part in a million against an ODE integration.
_GROWTH_POINTS = 20001

#: Attempts at drawing a physical parameter set before giving up.
_MAX_REDRAWS = 1000


class _Unphysical(Exception):
    """A parameter set for which the model has no expansion history."""


# ============================================================
# Data layout
# ============================================================

@dataclass(frozen=True, slots=True)
class _Layout:
    """One dataset's rows, in the dataset's own order."""

    z: Array
    values: Array
    cov: Array
    quantity: np.ndarray
    field: str


def _layout(dataset) -> _Layout:
    """
    Rows in the order the dataset stores them.

    Deliberately not :func:`~CosmoRecon.reconstructors.base.unpack_dataset`,
    which sorts by redshift: a simulated dataset is written back field by
    field, and sorting on the way in without unsorting on the way out would
    pair each value with the wrong row of the covariance.
    """

    quantity = getattr(dataset, "quantity", None)

    z = np.asarray(dataset.z, dtype=float)

    if quantity is not None:

        return _Layout(
            z=z,
            values=np.asarray(dataset.values, dtype=float),
            cov=np.asarray(dataset.cov, dtype=float),
            quantity=np.asarray(quantity).astype(str),
            field="values",
        )

    return _Layout(
        z=z,
        values=np.asarray(dataset.y, dtype=float),
        cov=np.asarray(dataset.cov, dtype=float),
        quantity=np.full(z.size, str(dataset.observable)),
        field="y",
    )


# ============================================================
# The contract
# ============================================================

class NullModel(ABC):
    """
    A parametric family in which a null hypothesis holds exactly.

    Subclasses say which parameters a given set of datasets needs, predict
    those datasets' values from a parameter vector, and give a starting point
    and bounds. Fitting, the parameter posterior and simulation are shared.
    """

    #: How the model is named in a result, e.g. ``"flat LCDM"``.
    name: str = "null model"

    @abstractmethod
    def parameter_names(self, datasets: Sequence) -> tuple[str, ...]:
        """The parameters needed to predict every row of ``datasets``."""

    @abstractmethod
    def predict(self, parameters: Array, datasets: Sequence) -> list[Array]:
        """
        Predicted values for each dataset, in that dataset's row order.

        Raises ``_Unphysical`` for a parameter set with no expansion history.
        """

    @abstractmethod
    def start(self, datasets: Sequence) -> Array:
        """A starting point for the fit."""

    @abstractmethod
    def bounds(self, datasets: Sequence) -> tuple[Array, Array]:
        """Lower and upper bounds on every parameter."""

    # ---------------------------------------------------------

    def fit(self, datasets) -> "NullFit":
        """
        Generalised least squares against every dataset's own covariance.

        The datasets are taken to be mutually independent -- the same
        assumption :func:`~CosmoRecon.reconstructors.base.combine_independent`
        makes in the analysis being calibrated, and for the same reason. The
        parameter posterior is the Laplace approximation at the best fit,
        which is what the simulations draw from.
        """

        datasets = _as_tuple(datasets)

        names = self.parameter_names(datasets)

        layouts = [_layout(d) for d in datasets]

        factors = [linalg.cholesky(lay.cov, lower=True) for lay in layouts]

        size = sum(lay.values.size for lay in layouts)

        def residual(parameters):

            try:
                predictions = self.predict(parameters, datasets)

            except _Unphysical:
                return np.full(size, 1e6)

            return np.concatenate([
                linalg.solve_triangular(factor, lay.values - prediction, lower=True)
                for factor, lay, prediction in zip(factors, layouts, predictions, strict=True)
            ])

        lower, upper = self.bounds(datasets)

        start = np.clip(self.start(datasets), lower, upper)

        solution = optimize.least_squares(
            residual, start, bounds=(lower, upper), x_scale="jac"
        )

        if not solution.success:

            raise DataError(
                f"Fitting {self.name} to {[getattr(d, 'name', '?') for d in datasets]} "
                f"did not converge: {solution.message}"
            )

        jacobian = solution.jac

        covariance = np.linalg.pinv(jacobian.T @ jacobian)

        return NullFit(
            model=self,
            datasets=datasets,
            names=names,
            best=solution.x,
            covariance=covariance,
            chi2=float(2.0 * solution.cost),
            dof=int(size - len(names)),
        )


def _as_tuple(datasets) -> tuple:

    if isinstance(datasets, (list, tuple)):
        return tuple(datasets)

    return (datasets,)


# ============================================================
# A fitted null model
# ============================================================

@dataclass(frozen=True)
class NullFit:
    """
    A null model fitted to particular data, ready to simulate them.
    """

    model: NullModel

    datasets: tuple

    names: tuple[str, ...]

    #: Best-fit parameters, in the order of :attr:`names`.
    best: Array

    #: Laplace-approximation covariance of the parameters.
    covariance: Array

    chi2: float

    dof: int

    # ---------------------------------------------------------

    def _draw_parameters(self, rng: np.random.Generator) -> Array:

        lower, upper = self.model.bounds(self.datasets)

        for _ in range(_MAX_REDRAWS):

            draw = rng.multivariate_normal(self.best, self.covariance)

            if np.any(draw < lower) or np.any(draw > upper):
                continue

            try:
                self.model.predict(draw, self.datasets)

            except _Unphysical:
                continue

            return draw

        raise DataError(
            f"Could not draw a physical parameter set for {self.model.name} "
            f"from its fitted posterior in {_MAX_REDRAWS} attempts. The fit "
            "is probably sitting on a boundary of its parameter space, which "
            "means the null model does not describe these data."
        )

    def simulate(
        self,
        rng: np.random.Generator,
        *,
        marginalise: bool = True,
    ) -> tuple:
        """
        One mock realisation of every dataset, with the null true.

        The truth is drawn from the parameter posterior when ``marginalise`` is
        set, the default, and is the best fit otherwise; the noise is drawn
        from each dataset's released covariance. The mocks are the datasets'
        own classes with only their values replaced, so the analysis being
        calibrated cannot tell them from the real thing.
        """

        parameters = self._draw_parameters(rng) if marginalise else self.best

        predictions = self.model.predict(parameters, self.datasets)

        out = []

        for dataset, prediction in zip(self.datasets, predictions, strict=True):

            lay = _layout(dataset)

            noise = linalg.cholesky(lay.cov, lower=True) @ rng.standard_normal(lay.values.size)

            out.append(dataclasses.replace(dataset, **{lay.field: prediction + noise}))

        return tuple(out)

    def summary(self) -> str:

        spread = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

        parameters = ", ".join(
            f"{name} = {value:.4g} +/- {width:.2g}"
            for name, value, width in zip(self.names, self.best, spread, strict=True)
        )

        return f"{self.model.name}: {parameters}; chi2 = {self.chi2:.1f} / {self.dof}"

    def __str__(self) -> str:

        return self.summary()


# ============================================================
# Lambda-CDM
# ============================================================

class LambdaCDM(NullModel):
    """
    Lambda-CDM, flat or with curvature, predicting whatever the data measure.

    >>> LambdaCDM().fit(chronometers())                        # doctest: +SKIP
    >>> LambdaCDM(curved=True).fit(desi.select("DM_over_rs", "DH_over_rs"))
    ...                                                        # doctest: +SKIP

    The parameter vector is built from the observables present, so the model
    never carries a parameter the data cannot see: ``omega_m`` always,
    ``omega_k`` if ``curved``, ``H0`` for expansion-rate data,
    ``c_over_H0_rd`` for anything measured against the sound horizon, and
    ``mu_offset`` -- the supernova zero point, absolute magnitude and ``H0``
    together -- for distance moduli, and ``sigma8`` for the growth rate.
    Radiation is neglected, which at ``z < 3`` is a part in ten thousand.

    The growth rate is GR's, from Heath's integral
    ``delta ~ E(a) int_0^a da' / (a' E)^3`` -- exact for matter, curvature and a
    cosmological constant, and checked in the test suite against an ODE
    integration of the growth equation.

    Which null it is. Flat, it is the null of the Om diagnostics. With
    curvature, it is an FLRW universe, the null of the curvature test --
    though only one member of that family, since dark energy is fixed to a
    cosmological constant. For distance duality either will do: both distances
    come from one metric, so ``eta = 1`` holds by construction. For the
    growth-geometry test it is GR with a cosmological constant -- one member
    of a null that allows any expansion history.
    """

    #: Observables this model can predict.
    SUPPORTED = frozenset({"H", _GROWTH}) | _BAO | _MODULI

    def __init__(self, *, curved: bool = False) -> None:

        self.curved = bool(curved)

        self.name = "LCDM with curvature" if self.curved else "flat LCDM"

    # ---------------------------------------------------------

    def _present(self, datasets: Sequence) -> set[str]:

        present = {str(q) for d in datasets for q in _layout(d).quantity}

        unsupported = sorted(present - self.SUPPORTED)

        if unsupported:

            raise DataError(
                f"{self.name} cannot predict {unsupported}; it knows "
                f"{sorted(self.SUPPORTED)}."
            )

        return present

    def parameter_names(self, datasets: Sequence) -> tuple[str, ...]:

        present = self._present(datasets)

        names = ["omega_m"]

        if self.curved:
            names.append("omega_k")

        if "H" in present:
            names.append("H0")

        if present & _BAO:
            names.append("c_over_H0_rd")

        if present & _MODULI:
            names.append("mu_offset")

        if _GROWTH in present:
            names.append("sigma8")

        return tuple(names)

    def bounds(self, datasets: Sequence) -> tuple[Array, Array]:

        table = {
            "omega_m": (0.01, 1.0),
            "omega_k": (-0.8, 0.8),
            "H0": (10.0, 300.0),
            "c_over_H0_rd": (1.0, 200.0),
            "mu_offset": (0.0, 100.0),
            "sigma8": (0.05, 3.0),
        }

        names = self.parameter_names(datasets)

        return (
            np.array([table[n][0] for n in names]),
            np.array([table[n][1] for n in names]),
        )

    # ---------------------------------------------------------

    def _background(self, omega_m: float, omega_k: float, z_max: float):
        """``E(z)`` and the dimensionless transverse distance on a fine grid."""

        grid = np.linspace(0.0, 1.02 * z_max, 4001)

        E2 = omega_m * (1 + grid) ** 3 + omega_k * (1 + grid) ** 2 + (1 - omega_m - omega_k)

        if np.any(E2 <= 0.0):
            raise _Unphysical

        E = np.sqrt(E2)

        chi = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (1.0 / E[1:] + 1.0 / E[:-1]) * np.diff(grid)),
        ])

        if abs(omega_k) < 1e-10:
            transverse = chi

        elif omega_k > 0.0:
            root = np.sqrt(omega_k)
            transverse = np.sinh(root * chi) / root

        else:
            root = np.sqrt(-omega_k)
            transverse = np.sin(root * chi) / root

        if np.any(transverse[1:] <= 0.0):
            raise _Unphysical

        return grid, E, transverse

    @staticmethod
    def _growth(omega_m: float, omega_k: float, grid: Array) -> Array:
        """
        ``f sigma_8 / sigma_8(0)`` on ``grid``, from Heath's integral.

        With ``u = a^(5/2)`` the integral is
        ``J(a) = (2/5) int_0^u du' / (Omega_m + Omega_k a + Omega_L a^3)^(3/2)``,
        smooth down to ``a = 0``; then ``delta = E J`` and
        ``f = d ln E / d ln a + 1 / (a^2 E^3 J)``.
        """

        omega_l = 1.0 - omega_m - omega_k

        u = np.linspace(0.0, 1.0, _GROWTH_POINTS)

        a = u**0.4

        base = omega_m + omega_k * a + omega_l * a**3

        if np.any(base <= 0.0):
            raise _Unphysical

        integrand = 0.4 / base**1.5

        J = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u)),
        ])

        scale = 1.0 / (1.0 + grid)

        J_grid = np.interp(scale**2.5, u, J)

        E2 = omega_m / scale**3 + omega_k / scale**2 + omega_l

        E = np.sqrt(E2)

        log_slope = -0.5 * (3.0 * omega_m / scale**3 + 2.0 * omega_k / scale**2) / E2

        delta = E * J_grid

        f = log_slope + 1.0 / (scale**2 * E**3 * J_grid)

        return f * delta / J[-1]

    def predict(self, parameters: Array, datasets: Sequence) -> list[Array]:

        names = self.parameter_names(datasets)

        p = dict(zip(names, np.asarray(parameters, dtype=float), strict=True))

        layouts = [_layout(d) for d in datasets]

        z_max = max(float(lay.z.max()) for lay in layouts)

        grid, E_grid, D_grid = self._background(
            p["omega_m"], p.get("omega_k", 0.0), z_max
        )

        growth_grid = (
            self._growth(p["omega_m"], p.get("omega_k", 0.0), grid)
            if "sigma8" in p else None
        )

        out = []

        for lay in layouts:

            z = lay.z

            E = np.interp(z, grid, E_grid)
            D = np.interp(z, grid, D_grid)

            prediction = np.empty(z.size)

            for label in np.unique(lay.quantity):

                rows = lay.quantity == label

                if label == "H":
                    value = p["H0"] * E[rows]

                elif label == "DH_over_rs":
                    value = p["c_over_H0_rd"] / E[rows]

                elif label == "DM_over_rs":
                    value = p["c_over_H0_rd"] * D[rows]

                elif label == "DV_over_rs":
                    value = p["c_over_H0_rd"] * (z[rows] * D[rows] ** 2 / E[rows]) ** (1.0 / 3.0)

                elif label == "mu":
                    value = 5.0 * np.log10((1 + z[rows]) * D[rows]) + p["mu_offset"]

                elif label == _GROWTH:
                    value = p["sigma8"] * np.interp(z[rows], grid, growth_grid)

                else:  # mu_reduced
                    value = 5.0 * np.log10((1 + z[rows]) * D[rows] / z[rows]) + p["mu_offset"]

                prediction[rows] = value

            out.append(prediction)

        return out

    def start(self, datasets: Sequence) -> Array:
        """
        ``omega_m = 0.3``, flat, and each calibration solved for at that
        background -- the nuisance parameters enter linearly or as an offset,
        so a good start for them is a median ratio or difference.
        """

        names = self.parameter_names(datasets)

        layouts = [_layout(d) for d in datasets]

        z_max = max(float(lay.z.max()) for lay in layouts)

        grid, E_grid, D_grid = self._background(0.3, 0.0, z_max)

        growth_grid = self._growth(0.3, 0.0, grid)

        ratios: dict[str, list[float]] = {
            "H0": [], "c_over_H0_rd": [], "mu_offset": [], "sigma8": [],
        }

        for lay in layouts:

            E = np.interp(lay.z, grid, E_grid)
            D = np.interp(lay.z, grid, D_grid)
            F = np.interp(lay.z, grid, growth_grid)

            for i, label in enumerate(lay.quantity):

                v, z = lay.values[i], lay.z[i]

                if label == "H":
                    ratios["H0"].append(v / E[i])
                elif label == "DH_over_rs":
                    ratios["c_over_H0_rd"].append(v * E[i])
                elif label == "DM_over_rs":
                    ratios["c_over_H0_rd"].append(v / D[i])
                elif label == "DV_over_rs":
                    ratios["c_over_H0_rd"].append(v / (z * D[i] ** 2 / E[i]) ** (1.0 / 3.0))
                elif label == "mu":
                    ratios["mu_offset"].append(v - 5.0 * np.log10((1 + z) * D[i]))
                elif label == _GROWTH:
                    ratios["sigma8"].append(v / F[i])
                else:
                    ratios["mu_offset"].append(v - 5.0 * np.log10((1 + z) * D[i] / z))

        values = {"omega_m": 0.3, "omega_k": 0.0}

        values.update({k: float(np.median(v)) for k, v in ratios.items() if v})

        return np.array([values[n] for n in names])

    def __repr__(self) -> str:

        return f"<LambdaCDM {'curved' if self.curved else 'flat'}>"
