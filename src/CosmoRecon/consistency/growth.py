"""
Growth-geometry consistency: does the growth of structure follow from the
expansion history under general relativity?

In GR, with matter the only component that clusters and growth independent of
scale on the scales redshift-space distortions measure, the linear density
contrast obeys

    ``delta'' + (2 + H'/H) delta' = (3/2) Omega_m0 H_0^2 delta / (a^3 H^2)``

with primes ``d/d ln a`` -- whatever the dark energy does to ``H``, and
whatever the curvature. ``sigma_8(a)`` is proportional to ``delta``, so ``f
sigma_8 = d sigma_8 / d ln a``, and multiplying through by ``a^2 H`` gives the
equation a first integral:

    ``Psi(z) = a H d/d ln a [a^2 H f sigma_8] = (3/2) Omega_m0 H_0^2 sigma_8(z)``.

``Psi`` needs the expansion rate, the growth rate, and one derivative of each.
``sigma_8(z)`` needs its value today, which is what redshift-space distortions
do not measure; but its *change* between two redshifts is an integral of the
data,

    ``sigma_8(z) - sigma_8(z_r) = Sigma(z) = - int_{z_r}^z f sigma_8 / (1 + z') dz'``,

so anchoring at a reference redshift removes it:

    ``G(z; z_r) = [Psi(z) - Psi(z_r)] / [(3/2) Sigma(z)] = Omega_m0 H_0^2``

at every redshift. That is :class:`Growth`. No ``sigma_8``, no ``H_0``, no
second derivative, and no dark-energy model: an expansion history
reconstructed from chronometers or from BAO, and the growth rate reconstructed
from redshift-space distortions, either agree with this or GR does not hold as
assumed.

The relation, the anchored statistic and the null model's growth were checked
numerically before being written here -- against an ODE integration of the
growth equation, for flat and curved universes, for ``w = -0.7``, and for a
gravitational coupling that varies with time, which breaks it. The test suite
repeats those checks on universes whose growth is integrated rather than built
from any of these relations.

What it measures, and what it does not
--------------------------------------

The constant is ``Omega_m0 H_0^2`` in the units of the expansion rate
supplied: ``Omega_m h^2 x 10^4`` for ``H`` in km/s/Mpc, ``Omega_m / A^2`` for
the inverse of ``D_H / r_d``, with ``A = c / (H_0 r_d)``. Supplying the
calibration makes it ``Omega_m``.

**Constancy** tests the *time dependence* of gravity. A gravitational coupling
``G_eff / G`` that changes with redshift makes ``G(z)`` change with it. A
coupling that is different from Newton's but constant does not: it rescales
``G(z)`` to ``(G_eff / G) Omega_m``, which is a perfectly constant function.

**The value** tests that. The same ``Omega_m`` read from the geometry --
:class:`~CosmoRecon.consistency.litmus.CurvedLitmus` reads ``Omega_m / A^2``
from the same BAO release -- and from the growth should agree; passing it as
``omega_m`` turns the test into "``G(z)`` equals the geometric value".

**The amplitude** -- the S8 question -- is invisible to both, because a
``sigma_8`` that is ten per cent lower rescales ``f sigma_8`` and leaves ``G``
unchanged. :meth:`Growth.sigma8` answers it the other way round: given the
geometric ``Omega_m / A^2``, ``Psi(z)`` *is* ``sigma_8(z)``, and adding back the
integral from today gives ``sigma_8(0)`` at every redshift -- constant under
GR, and its value is the amplitude today. No Lambda-CDM anywhere, no Boltzmann
code: only GR, and the assumption that matter alone clusters.

**The data are not model-independent from the raw measurement up.** Each
survey in a growth compilation converted its redshift-space distortion signal
into ``f sigma_8`` under a fiducial cosmology; see
:func:`CosmoRecon.data.growth`. And a growth compilation and a BAO release that
observe overlapping volumes are not independent to arbitrary precision, so the
claim that they are is made in the source, where it can be disputed.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DataError, NotResamplableError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import NullTest


__all__ = ["Growth"]


#: Expansion observables the statistic can be built from, and how.
_RATE = "H"
_RADIAL = "DH_over_rs"

#: Labels that are certainly not a growth rate -- the arguments swapped.
_NOT_GROWTH = frozenset({_RATE, _RADIAL, "DM_over_rs", "DV_over_rs", "mu", "mu_reduced"})

#: Gauss-Legendre nodes per integral. The integrand is a smooth
#: reconstruction divided by ``1 + z``; 24 nodes integrate a polynomial of
#: degree 47 exactly.
_NODES = 24

#: How close a grid point may come to the reference redshift, where the
#: anchored statistic is 0/0.
_MIN_SEPARATION = 1e-3


# ============================================================
# Pieces
# ============================================================

def _expansion_rate(
    expansion: Reconstruction,
) -> tuple[Reconstruction, Reconstruction]:
    """
    ``(h, h')`` up to one constant factor, from ``H`` or from ``D_H / r_d``.

    ``D_H / r_d = c / (H r_d)``, so its inverse is the expansion rate in units
    of ``c / r_d``, and its derivative follows from the reconstruction's own.
    """

    if expansion.label == _RATE:
        return expansion, expansion.d(1)

    if expansion.label == _RADIAL:
        return 1.0 / expansion, -expansion.d(1) / expansion**2

    raise DataError(
        f"The expansion history has to be H or D_H/r_d, and this is "
        f"{expansion.label or 'unlabelled'!r}. A transverse distance gives the "
        "expansion rate only through a derivative and an assumption about "
        "curvature, which this test should not be quietly making."
    )


def _psi(fsigma8: Reconstruction, h: Reconstruction, h1: Reconstruction):
    """``Psi = a H d/d ln a [a^2 H f sigma_8]``, written in redshift."""

    z = fsigma8.z

    product = (h1 * fsigma8 + h * fsigma8.d(1)) * (1.0 + z) - 2.0 * h * fsigma8

    return -h * product / (1.0 + z) ** 3


def _integral(fsigma8: Reconstruction, lower: float) -> Reconstruction:
    """
    ``int_lower^z f sigma_8 / (1 + z') dz'`` on the grid of ``fsigma8``, per
    draw, by Gauss-Legendre quadrature on the reconstruction itself.

    The nodes are evaluated with :meth:`Reconstruction.at`, so draw ``k`` of
    the integral is the integral of draw ``k`` -- and it stays in the same
    realisation frame as everything else built from that fit.
    """

    z = fsigma8.z

    x, w = np.polynomial.legendre.leggauss(_NODES)

    half = 0.5 * (z - lower)

    nodes = 0.5 * (z + lower)[:, None] + half[:, None] * x[None, :]

    try:
        values = fsigma8.at(nodes.ravel()).draws

    except NotResamplableError:

        raise NotResamplableError(
            "The growth test integrates f sigma_8 between redshifts, and this "
            "reconstruction holds draws on a fixed grid only -- it came out of "
            "arithmetic, or out of a method that only samples. Pass the "
            "reconstruction the fit returned, before any arithmetic on it."
        ) from None

    values = values.reshape(values.shape[0], z.size, _NODES)

    draws = np.sum(values / (1.0 + nodes) * w, axis=2) * half

    return fsigma8.with_draws(
        z,
        draws,
        expression=f"int_{lower:g}^z fsigma8 / (1+z) dz",
        label="",
    )


def _check_growth_rate(fsigma8: Reconstruction) -> None:

    if fsigma8.label in _NOT_GROWTH:

        raise DataError(
            f"The first argument is {fsigma8.label!r}, which is not a growth "
            "rate. The signature is (fsigma8, expansion)."
        )


# ============================================================
# The test
# ============================================================

class Growth(NullTest):
    """
    Growth-geometry consistency under GR: ``G(z; z_r)`` equals
    ``Omega_m0 H_0^2`` at every redshift, whatever the expansion history.

    >>> both = combine_independent(                             # doctest: +SKIP
    ...     Cosmography("y").fit(growth(), grid=grid),
    ...     Cosmography("y").fit(desi.select("DH_over_rs"), grid=grid),
    ... )
    >>> Growth(z_reference=0.93).evaluate(both["fsigma8"], both["DH_over_rs"])
    ...                                                        # doctest: +SKIP

    The two inputs come from different datasets, so the independence between
    them has to be declared -- with
    :func:`~CosmoRecon.reconstructors.base.combine_independent`, or
    :meth:`~CosmoRecon.ensemble.method.EnsembleFit.with_independent` for two
    ensembles -- and the growth rate has to be the reconstruction the fit
    returned, because the test integrates it between redshifts.

    Parameters
    ----------
    z_reference
        The anchor. The grid must stay away from it -- the statistic is 0/0
        there -- and it is best placed where both inputs are well measured.
    hubble_constant
        ``H_0``, in the unit of an ``H`` reconstruction. Supplying it makes the
        statistic ``Omega_m``.
    hubble_distance
        ``c / (H_0 r_d)``, for a ``D_H / r_d`` reconstruction. Supplying it
        makes the statistic ``Omega_m``.
    omega_m
        Test against this matter density instead of against "some constant"
        -- a geometric ``Omega_m``, so that a constant but non-Newtonian
        coupling is seen. Needs a calibration, and :meth:`sigma8` needs it.
    """

    def __init__(
        self,
        z_reference: float,
        *,
        hubble_constant: float | None = None,
        hubble_distance: float | None = None,
        omega_m: float | None = None,
    ) -> None:

        if hubble_constant is not None and hubble_distance is not None:

            raise ValueError(
                "Pass hubble_constant for an H reconstruction or "
                "hubble_distance for D_H/r_d, not both: the expansion rate "
                "arrives in one unit."
            )

        for label, value in (
            ("hubble_constant", hubble_constant),
            ("hubble_distance", hubble_distance),
            ("omega_m", omega_m),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{label} must be positive, got {value}.")

        if omega_m is not None and hubble_constant is None and hubble_distance is None:

            raise ValueError(
                f"Testing against Omega_m = {omega_m:g} needs a calibration: the "
                "statistic is Omega_m H0^2 in the unit of the expansion rate, "
                "and neither chronometers nor BAO fix that unit to Omega_m. "
                "Pass hubble_constant or hubble_distance, or test constancy, "
                "which needs neither."
            )

        self.z_reference = float(z_reference)

        self.hubble_constant = None if hubble_constant is None else float(hubble_constant)

        self.hubble_distance = None if hubble_distance is None else float(hubble_distance)

        self.omega_m = None if omega_m is None else float(omega_m)

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        base = f"growth G(z; {self.z_reference:g})"

        return base if self.omega_m is None else f"{base} vs {self.omega_m:g}"

    @property
    def null_value(self) -> float:

        return 0.0 if self.omega_m is None else self.omega_m

    @property
    def null_is_free_constant(self) -> bool:

        return self.omega_m is None

    # ---------------------------------------------------------

    def _scale(self, expansion: Reconstruction) -> float:
        """
        The factor that turns ``Omega_m0 H_0^2``, in the unit of the expansion
        rate supplied, into ``Omega_m`` -- 1 with no calibration.
        """

        if self.hubble_constant is not None:

            if expansion.label != _RATE:

                raise DataError(
                    "hubble_constant calibrates an H reconstruction, and this "
                    f"is {expansion.label!r}. For D_H/r_d pass hubble_distance."
                )

            return 1.0 / self.hubble_constant**2

        if self.hubble_distance is not None:

            if expansion.label != _RADIAL:

                raise DataError(
                    "hubble_distance calibrates a D_H/r_d reconstruction, and "
                    f"this is {expansion.label!r}. For H pass hubble_constant."
                )

            return self.hubble_distance**2

        return 1.0

    def statistic(
        self,
        fsigma8: Reconstruction,
        expansion: Reconstruction,
    ) -> Reconstruction:
        """
        ``G(z; z_r)`` from ``f sigma_8`` and ``H`` or ``D_H / r_d``.
        """

        _check_growth_rate(fsigma8)

        scale = self._scale(expansion)

        z = fsigma8.z

        gap = np.abs(z - self.z_reference)

        if np.any(gap < _MIN_SEPARATION):

            raise DataError(
                f"The grid reaches z = {z[np.argmin(gap)]:.6g}, within "
                f"{_MIN_SEPARATION:g} of the reference redshift "
                f"{self.z_reference:g}, where this statistic is 0/0. Evaluate "
                "on a grid that stays away from its own anchor."
            )

        # sigma_8(z) - sigma_8(z_r). First, because it is the step that needs
        # the growth rate as the fit returned it.
        change = -_integral(fsigma8, self.z_reference)

        h, h1 = _expansion_rate(expansion)

        anchor_h, anchor_h1 = _expansion_rate(expansion.at(self.z_reference))

        psi = _psi(fsigma8, h, h1)

        anchor = _psi(fsigma8.at(self.z_reference), anchor_h, anchor_h1)

        curve = (psi - anchor) / (1.5 * change)

        if scale != 1.0:
            curve = curve * scale

        curve.label = self.name

        return curve

    def sigma8(
        self,
        fsigma8: Reconstruction,
        expansion: Reconstruction,
    ) -> Reconstruction:
        """
        ``sigma_8`` today, at every redshift of the grid: constant under GR.

        ``Psi(z) / [(3/2) Omega_m0 H_0^2]`` is ``sigma_8(z)``, and adding
        ``int_0^z f sigma_8 / (1 + z') dz'`` carries it back to today. Needs the
        matter density and its calibration -- in practice the geometric
        ``Omega_m / A^2`` of the same BAO release -- because the growth rate
        alone knows ``sigma_8`` only up to that factor.

        Not anchored, so ``z_reference`` plays no part; the integral runs from
        ``z = 0``, which asks the growth reconstruction for its value below the
        lowest measurement.
        """

        _check_growth_rate(fsigma8)

        if self.omega_m is None:

            raise ValueError(
                "sigma_8 needs Omega_m, with its calibration: the growth "
                "equation fixes Omega_m H0^2 sigma_8, not sigma_8. Pass "
                "omega_m together with hubble_constant or hubble_distance."
            )

        scale = self._scale(expansion)

        since = _integral(fsigma8, 0.0)

        h, h1 = _expansion_rate(expansion)

        today = _psi(fsigma8, h, h1) * (scale / (1.5 * self.omega_m)) + since

        today.label = "sigma8(0)"

        return today
