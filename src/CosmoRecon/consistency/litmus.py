"""
The litmus test for a cosmological constant, from distances.

Zunckel & Clarkson (2008), Phys. Rev. Lett. 101, 181301, "A litmus test for
Lambda". With ``D = (H_0 / c) d_L / (1 + z)`` the dimensionless transverse
comoving distance, a flat Lambda-CDM universe has (their Eq. 5)

    ``Omega_m = (1 - D'^2) / { [(1+z)^3 - 1] D'^2 }``

at every redshift, and differentiating (their Eq. 6)

    ``L(z) = zeta D'' + 3 (1+z)^2 D' (1 - D'^2) = 0``,  ``zeta = 2 [(1+z)^3 - 1]``

for **every** flat Lambda-CDM model, whatever ``Omega_m``. Where ``Om(z)``
needs the expansion rate, this needs only a distance-redshift relation -- a
supernova compilation with no ``H(z)`` data at all -- and pays for it with a
second derivative.

Both equations, and their curved generalisation (Eq. 7), were checked
symbolically against a derivation from the Friedmann equation before being
written here. The test suite checks them numerically, on universes whose
distances are integrated and differentiated on a grid rather than built from
any of these relations.

What it needs, and what it does not
-----------------------------------

``L(z)`` as written needs ``D`` normalised by ``c / H_0`` -- that is what the
``1 - D'^2`` term compares against -- and no distance data provide that. A
supernova compilation knows its distances up to a zero point, a BAO release up
to ``c / (H_0 r_d)``. Rearranged instead as

    ``Q(z) = [zeta D'' + 3 (1+z)^2 D'] / [3 (1+z)^2 D'^3]``,
    ``L(z) = 3 (1+z)^2 D'^3 [Q(z) - 1]``,

the same condition reads ``Q = 1`` for a normalised distance and ``Q = 1 /
alpha^2`` for one known only as ``alpha D``: a constant either way. And a
constant ``Q`` is exactly flat Lambda-CDM -- the differential equation it
imposes, ``E^2 - [(1+z)^3 - 1] (E^2)' / [3 (1+z)^2] = const``, has the solution
``E^2 = c + K [(1+z)^3 - 1]`` and nothing else. So :class:`Litmus` tests whether
``Q`` is constant, which is the whole flat test and needs no calibration;
supplying one turns it into ``Q = 1``, and :meth:`Litmus.zunckel_clarkson`
returns ``L(z)`` itself.

Curvature, and why there are two classes
----------------------------------------

In a curved Lambda-CDM universe the flat test fails: ``Omega_k = +-0.1`` moves
``Q`` by order unity, which is a detection of dynamical dark energy made
entirely of curvature. Zunckel & Clarkson remove ``Omega_k`` with the
Clarkson-Bassett-Lu relation ``Omega_k = (h^2 D'^2 - 1) / D^2``, which needs
``h = H / H_0`` alongside the distance.

A BAO release measures exactly that pair, as ``T = D_M / r_d = A D`` and
``R = D_H / r_d = A / h``, with one unknown ``A = c / (H_0 r_d)``. Curved
Lambda-CDM has ``h^2 = 1 + Omega_m [(1+z)^3 - 1] + Omega_k [(1+z)^2 - 1]``, so
anchoring at a reference redshift ``z_r`` removes the 1, taking ``Omega_k`` from
the Clarkson-Bassett-Lu relation removes the curvature, and

    ``O(z; z_r) = { R^-2(z) - R^-2(z_r) - kappa(z) [(1+z)^2 - (1+z_r)^2] } /
                 [(1+z)^3 - (1+z_r)^3]``,   ``kappa = [(T'/R)^2 - 1] / T^2``

equals ``Omega_m / A^2`` at every redshift in Lambda-CDM with any curvature.
That is :class:`CurvedLitmus`. It is Zunckel & Clarkson's idea -- curvature
removed through the Clarkson-Bassett-Lu relation -- written as a constant
rather than as a vanishing derivative, which removes both the second
derivative and the need to know ``H_0``. Testing constancy needs no
calibration; with ``A`` supplied the statistic is ``Omega_m`` itself.

It assumes FLRW, through ``kappa``. Whether the universe is FLRW is the
question :class:`~CosmoRecon.consistency.curvature.Curvature` asks of the same
two functions.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DataError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import NullTest
from CosmoRecon.consistency.duality import REDUCED_MODULUS


__all__ = ["Litmus", "CurvedLitmus"]


#: ``ln(10) / 5``: a distance modulus is ``5 log10`` of a distance.
_MAGNITUDE = np.log(10.0) / 5.0

#: Redshift below which the statistics are refused. The distance and its
#: derivatives are divided by, and the reduced modulus's chain rule carries a
#: ``1 / z``; both degrade long before the origin.
_MIN_REDSHIFT = 0.05

#: How close a grid point may come to the curved test's reference redshift,
#: where its numerator and denominator both vanish.
_MIN_SEPARATION = 1e-3


def _distance_and_derivatives(
    source: Reconstruction,
) -> tuple[Reconstruction, Reconstruction, Reconstruction]:
    """
    ``(D, D', D'')`` up to one constant factor, from a distance or a reduced
    distance modulus.

    For a reduced modulus ``m = mu - 5 log10 z`` the distance is
    ``D ~ z 10^(m/5) / (1+z)``, and its derivatives follow by the chain rule
    from ``m'`` and ``m''`` -- asked of the reconstruction, so they are the
    method's analytic derivatives rather than finite differences.
    """

    if source.label == "mu":

        raise DataError(
            "This is a reconstruction of the distance modulus itself, which "
            "carries a logarithmic singularity at the origin that no series "
            "represents -- see CosmoRecon.consistency.duality. Reconstruct "
            "the reduced modulus mu - 5 log10(z) with "
            "CosmoRecon.data.reduced_modulus instead."
        )

    if source.label != REDUCED_MODULUS:
        return source, source.d(1), source.d(2)

    z = source.z

    # ln D = ln[z / (1+z)] + (ln 10 / 5) m, up to a constant.
    slope = 1.0 / (z * (1.0 + z)) + _MAGNITUDE * source.d(1)

    curvature = (
        slope**2
        - (1.0 + 2.0 * z) / (z * (1.0 + z)) ** 2
        + _MAGNITUDE * source.d(2)
    )

    distance = (z / (1.0 + z)) * np.exp(_MAGNITUDE * (source - 25.0))

    return distance, distance * slope, distance * curvature


def _check_grid(z: np.ndarray) -> None:

    if np.any(z < _MIN_REDSHIFT):

        raise DataError(
            f"The grid reaches z = {z.min():.4g}. The distance and its "
            "derivatives enter these statistics as divisors, so below "
            f"z = {_MIN_REDSHIFT:g} they are dominated by that division "
            "rather than by the data."
        )


# ============================================================
# Flat
# ============================================================

class Litmus(NullTest):
    """
    Zunckel & Clarkson's flat litmus test: ``Q(z)`` is constant in flat
    Lambda-CDM, whatever ``Omega_m`` and whatever the distance calibration.

    >>> fit = Cosmography("y").fit(reduced_modulus(union3()), grid=grid)
    ...                                                        # doctest: +SKIP
    >>> Litmus().evaluate(fit["mu_reduced"])                   # doctest: +SKIP

    The input is a reconstruction of a distance -- ``D_M / r_d``, ``D_M`` in any
    unit -- or of a reduced distance modulus, from which the distance and its
    derivatives are built. Either way the method has to supply a second
    derivative: a series always can, a Gaussian process only if its kernel is
    smooth enough, and a Matern posterior reaching below ``nu = 5/2`` refuses.

    Parameters
    ----------
    hubble_distance
        ``alpha`` in "the supplied distance is ``alpha`` times ``(H_0/c) d_C``":
        ``c / (H_0 r_d)`` for ``D_M / r_d``, ``c / H_0`` in the input's unit for
        a distance, and ``10^((mu_offset - 25) / 5)`` for a reduced modulus,
        with ``mu_offset`` the zero point
        :class:`~CosmoRecon.validation.LambdaCDM` fits. Supplying it turns the
        test from "``Q`` is constant" into "``Q = 1``", and is required by
        :meth:`zunckel_clarkson`.

    Notes
    -----
    The paper's test is a flat one. In a curved universe it fails even when
    dark energy is a cosmological constant; :class:`CurvedLitmus` is the test
    that does not.
    """

    def __init__(self, *, hubble_distance: float | None = None) -> None:

        if hubble_distance is not None and hubble_distance <= 0.0:

            raise ValueError(
                f"hubble_distance must be positive, got {hubble_distance}."
            )

        self.hubble_distance = (
            None if hubble_distance is None else float(hubble_distance)
        )

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        return "litmus Q(z)" if self.hubble_distance is None else "litmus Q(z) vs 1"

    @property
    def null_value(self) -> float:

        return 1.0

    @property
    def null_is_free_constant(self) -> bool:
        """
        Without a calibration the null is "constant, value ``1 / alpha^2``",
        which is the flat test itself. With one it is the exact number 1.
        """

        return self.hubble_distance is None

    # ---------------------------------------------------------

    def _pieces(self, distance: Reconstruction):

        _check_grid(distance.z)

        D, first, second = _distance_and_derivatives(distance)

        draws = first.draws

        falling = np.any(draws <= 0.0, axis=1)

        if np.any(falling):

            where = distance.z[np.any(draws <= 0.0, axis=0)]

            raise DataError(
                f"{100.0 * falling.mean():.2g}% of the draws have a distance "
                f"that stops increasing with redshift, at z = {where.min():.3g} "
                f"to {where.max():.3g}. No expansion history does that, and the "
                "litmus test divides by D'^3, so those draws would dominate it "
                "with a value that means nothing. It happens where a flexible "
                "reconstruction returns something close to its prior -- "
                "usually beyond the last well-measured point. Confine the grid "
                "to where the data are dense."
            )

        return D, first, second

    def statistic(self, distance: Reconstruction) -> Reconstruction:
        """
        ``Q(z)`` from a distance or a reduced modulus.

        ``Q = 1 / alpha^2`` in flat Lambda-CDM, with ``alpha`` the calibration
        the input was measured in; multiplied by ``alpha^2`` when
        ``hubble_distance`` was given, so that it is then exactly 1.
        """

        _, first, second = self._pieces(distance)

        z = distance.z

        zeta = 2.0 * ((1.0 + z) ** 3 - 1.0)

        rise = 3.0 * (1.0 + z) ** 2

        curve = (zeta * second + rise * first) / (rise * first**3)

        if self.hubble_distance is not None:
            curve = curve * self.hubble_distance**2

        curve.label = self.name

        return curve

    def zunckel_clarkson(self, distance: Reconstruction) -> Reconstruction:
        """
        ``L(z) = zeta D'' + 3 (1+z)^2 D' (1 - D'^2)``, Eq. 6 of the paper,
        which is zero in every flat Lambda-CDM universe.

        Needs the distance normalised by ``c / H_0``, so it needs
        ``hubble_distance``. It carries the same information as
        :meth:`statistic` -- ``L = 3 (1+z)^2 D'^3 (Q - 1)`` -- and is here so
        that a result can be compared with the paper's figures directly.
        """

        if self.hubble_distance is None:

            raise ValueError(
                "L(z) compares D'^2 with 1, so it needs the distance normalised "
                "by c/H0, and no distance data supply that. Pass "
                "hubble_distance, or test the constancy of Q(z) with "
                "statistic(), which needs no calibration at all."
            )

        _, first, second = self._pieces(distance)

        z = distance.z

        first = first / self.hubble_distance

        second = second / self.hubble_distance

        curve = (
            2.0 * ((1.0 + z) ** 3 - 1.0) * second
            + 3.0 * (1.0 + z) ** 2 * first * (1.0 - first**2)
        )

        curve.label = "litmus L(z)"

        return curve


# ============================================================
# Any curvature
# ============================================================

class CurvedLitmus(NullTest):
    """
    A litmus test for Lambda in a universe of any curvature, from a BAO
    release's transverse and radial distances.

    ``O(z; z_r)`` equals ``Omega_m / A^2`` at every redshift in Lambda-CDM,
    whatever ``Omega_m`` and ``Omega_k``, with ``A = c / (H_0 r_d)``.

    >>> fit = Cosmography("y").fit(desi.select("DM_over_rs", "DH_over_rs"))
    ...                                                        # doctest: +SKIP
    >>> CurvedLitmus(z_reference=0.93).evaluate(fit["DM_over_rs"], fit["DH_over_rs"])
    ...                                                        # doctest: +SKIP

    The two inputs must come from one joint fit, for the reason given in
    :class:`~CosmoRecon.consistency.curvature.Curvature`: the statistic is built
    from both and their correlation sets its width.

    Parameters
    ----------
    z_reference
        The anchor. The grid must stay away from it -- the statistic is 0/0
        there -- and it is best placed where both distances are well measured.
    hubble_distance
        ``c / (H_0 r_d)``. Supplying it makes the statistic ``Omega_m``.
    omega_m
        Test against this matter density instead of against "some constant".
        Needs ``hubble_distance``.
    """

    def __init__(
        self,
        z_reference: float,
        *,
        hubble_distance: float | None = None,
        omega_m: float | None = None,
    ) -> None:

        if omega_m is not None and hubble_distance is None:

            raise ValueError(
                f"Testing against Omega_m = {omega_m:g} needs a calibration: the "
                "statistic is Omega_m / (c / H0 r_d)^2, and BAO measure neither "
                "H0 nor the sound horizon. Pass hubble_distance, or test "
                "constancy -- the litmus test itself -- which needs none."
            )

        if z_reference < _MIN_REDSHIFT:

            raise ValueError(
                f"z_reference = {z_reference:g} is below z = {_MIN_REDSHIFT:g}, "
                "where the transverse distance it is built from vanishes."
            )

        if hubble_distance is not None and hubble_distance <= 0.0:

            raise ValueError(
                f"hubble_distance must be positive, got {hubble_distance}."
            )

        self.z_reference = float(z_reference)

        self.hubble_distance = (
            None if hubble_distance is None else float(hubble_distance)
        )

        self.omega_m = None if omega_m is None else float(omega_m)

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        base = f"curved litmus O(z; {self.z_reference:g})"

        return base if self.omega_m is None else f"{base} vs {self.omega_m:g}"

    @property
    def null_value(self) -> float:

        return 0.0 if self.omega_m is None else self.omega_m

    @property
    def null_is_free_constant(self) -> bool:

        return self.omega_m is None

    # ---------------------------------------------------------

    def statistic(
        self,
        transverse: Reconstruction,
        radial: Reconstruction,
    ) -> Reconstruction:
        """
        ``O(z; z_r)`` from ``D_M / r_d`` and ``D_H / r_d`` of one joint fit.
        """

        z = transverse.z

        _check_grid(z)

        gap = np.abs(z - self.z_reference)

        if np.any(gap < _MIN_SEPARATION):

            raise DataError(
                f"The grid reaches z = {z[np.argmin(gap)]:.6g}, within "
                f"{_MIN_SEPARATION:g} of the reference redshift "
                f"{self.z_reference:g}, where this statistic is 0/0. Evaluate "
                "on a grid that stays away from its own anchor."
            )

        # Omega_k / A^2 from the Clarkson-Bassett-Lu relation, at each redshift.
        kappa = ((transverse.d(1) / radial) ** 2 - 1.0) / transverse**2

        anchor = radial.at(self.z_reference)

        zr = 1.0 + self.z_reference

        curve = (
            1.0 / radial**2
            - 1.0 / anchor**2
            - kappa * ((1.0 + z) ** 2 - zr**2)
        ) / ((1.0 + z) ** 3 - zr**3)

        if self.hubble_distance is not None:
            curve = curve * self.hubble_distance**2

        curve.label = self.name

        return curve
