"""
The Clarkson-Bassett-Lu curvature test.

Clarkson, Bassett & Lu (2008), Phys. Rev. Lett. 101, 011301, "A general test of
the Copernican principle". In **any** Friedmann-Lemaitre-Robertson-Walker
universe, whatever the dark energy does,

    ``Ok(z) = [ H^2(z) D'^2(z) - c^2 ] / [ H_0^2 D^2(z) ] = Omega_k``

with ``D`` the transverse comoving distance and ``D' = dD/dz``. The
dark-energy terms cancel identically, and what is left is a constant.

This is a stronger test than the Om family, and worth being precise about what
it tests. ``Om(z)`` departing from a constant says dark energy is not a
cosmological constant. ``Ok(z)`` departing from a constant says the universe is
**not FLRW** -- not homogeneous and isotropic about us on large scales. It is a
test of the Copernican principle, not of a dark-energy model, and a detection
would be a far larger claim than evolving dark energy.

What it costs, and what it does not
-----------------------------------

The test needs the transverse distance *and its derivative* and the expansion
rate, at the same redshifts, correlated. That is exactly what a BAO release
gives -- ``D_M/r_d`` and ``D_H/r_d`` together -- and exactly why they have to
be reconstructed jointly: the statistic is built from both, and an interval
computed as though they were independent is not the interval.

Written in the BAO observables, with ``T = D_M/r_d`` and ``R = D_H/r_d``,

    ``Ok(z) = (c / H_0 r_d)^2 * [ (T'(z) / R(z))^2 - 1 ] / T^2(z)``

so the whole calibration -- the Hubble constant and the sound horizon, neither
of which BAO measures -- collects into one multiplicative constant. Two of the
three questions therefore need no calibration at all:

- *Is the universe FLRW?* Test whether the bracket is **constant**. A
  multiplicative constant does not affect constancy.
- *Is it flat?* Test whether the bracket is **zero**. Zero times an unknown
  positive number is still zero.
- *Is ``Omega_k`` equal to some specific non-zero value?* This one needs
  ``c / H_0 r_d``, and the class asks for it rather than assuming one.

The same expressions work for a direct pair of distances -- pass ``D_M`` and
``c/H`` in the same length unit and the calibration constant becomes
``c / H_0`` in that unit.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DataError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import NullTest


__all__ = ["Curvature"]


#: Redshift below which the statistic is refused. ``T(z)`` vanishes at the
#: origin and sits in the denominator, so the diagnostic degrades long before
#: it is exactly singular. No BAO release comes near this anyway -- DESI's
#: lowest transverse measurement is at ``z = 0.51``.
_MIN_REDSHIFT = 0.05


class Curvature(NullTest):
    """
    ``Ok(z)``: constant in any FLRW universe, whatever the dark energy.

    >>> fit = Cosmography("y").fit(bao.select("DM_over_rs", "DH_over_rs"))
    ...                                                        # doctest: +SKIP
    >>> Curvature().evaluate(fit["DM_over_rs"], fit["DH_over_rs"])
    ...                                                        # doctest: +SKIP

    The two inputs must come from **one joint fit**. They enter the statistic
    together and are correlated -- ``r`` of about ``-0.4`` in the DESI DR2
    release -- so combining two separate fits would either be refused by
    :class:`~CosmoRecon.core.reconstruction.Reconstruction` or, if forced
    with ``assume_independent()``, give an interval that is simply wrong.

    Parameters
    ----------
    omega_k
        Test against a specific spatial curvature instead of against "some
        constant".

        ``omega_k=0`` is the flatness test and needs no calibration: the
        unknown constant multiplies zero. Any other value does need
        ``hubble_distance``, and the class says so rather than quietly
        assuming one.
    hubble_distance
        ``c / (H_0 r_d)`` when the inputs are BAO ratios, or ``c / H_0`` in the
        inputs' own length unit when they are distances. Supplying it turns
        the statistic into ``Omega_k`` itself; leaving it out keeps the
        statistic proportional to ``Omega_k``, which is all the first two
        questions need.

        For DESI DR2's published ``r_d h = 101.54 Mpc``, this is
        ``c / (100 * 101.54) = 29.52``.
    """

    def __init__(
        self,
        *,
        omega_k: float | None = None,
        hubble_distance: float | None = None,
    ) -> None:

        if omega_k is not None and omega_k != 0.0 and hubble_distance is None:

            raise ValueError(
                f"Testing against Omega_k = {omega_k:g} needs a calibration: "
                "the statistic is proportional to Omega_k with an unknown "
                "constant made of H0 and the sound horizon, neither of which "
                "BAO measures. Pass hubble_distance = c / (H0 r_d) -- about "
                "29.5 for DESI DR2's published r_d h = 101.54 Mpc.\n"
                "\n"
                "The two questions that need no calibration are whether the "
                "universe is FLRW (omega_k=None, testing constancy) and "
                "whether it is flat (omega_k=0, since the unknown constant "
                "multiplies zero)."
            )

        self.omega_k = None if omega_k is None else float(omega_k)

        self.hubble_distance = (
            None if hubble_distance is None else float(hubble_distance)
        )

        if self.hubble_distance is not None and self.hubble_distance <= 0.0:

            raise ValueError(
                f"hubble_distance must be positive, got {hubble_distance}."
            )

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        if self.omega_k is None:
            return "Ok(z)" if self.hubble_distance else "Ok(z) / calibration"

        if self.omega_k == 0.0:
            return "Ok(z) vs flat"

        return f"Ok(z) vs {self.omega_k:g}"

    @property
    def null_value(self) -> float:

        return 0.0 if self.omega_k is None else self.omega_k

    @property
    def null_is_free_constant(self) -> bool:
        """
        With no ``omega_k`` given the test is "constant, value unknown" --
        which is the test of FLRW itself, and the reason this diagnostic
        exists. Fixing ``omega_k`` narrows it to a test of that particular
        curvature.
        """

        return self.omega_k is None

    # ---------------------------------------------------------

    def statistic(
        self,
        transverse: Reconstruction,
        radial: Reconstruction,
    ) -> Reconstruction:
        """
        Build ``Ok(z)`` from a transverse and a radial distance.

        ``transverse`` is ``D_M/r_d`` (or ``D_M``); ``radial`` is ``D_H/r_d``
        (or ``c/H``) in the same units. Both must be on the same grid and from
        the same fit -- the second is not a formality here, since the
        statistic subtracts two quantities of similar size and their
        correlation sets how large the difference's uncertainty is.
        """

        z = transverse.z

        if np.any(z < _MIN_REDSHIFT):

            raise DataError(
                f"The grid reaches z = {z.min():.4g}. The transverse distance "
                "vanishes at the origin and sits in the denominator of this "
                f"statistic, so below z = {_MIN_REDSHIFT:g} it is dominated "
                "by that division rather than by the data. No BAO release "
                "reaches there in any case."
            )

        # T'(z) is asked of the reconstruction, which answers analytically or
        # refuses. Differencing it here would be a different estimator with a
        # tighter-looking and wronger interval -- see Reconstruction.d.
        slope = transverse.d(1)

        bracket = (slope / radial) ** 2 - 1.0

        curve = bracket / transverse**2

        if self.hubble_distance is not None:
            curve = curve * self.hubble_distance**2

        curve.label = self.name

        return curve
