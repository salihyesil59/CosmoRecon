"""
The Om family: null tests of the cosmological constant built from ``H(z)``
alone.

Sahni, Shafieloo & Starobinsky (2008), Phys. Rev. D 78, 103502 introduced the
two-point diagnostic

    ``Om(z2; z1) = [h^2(z2) - h^2(z1)] / [(1+z2)^3 - (1+z1)^3]``,  ``h = H/H0``

whose one-point form ``Om(z) = Om(z; 0)`` is the familiar one. For a spatially
flat universe with a cosmological constant this is ``Omega_m`` **at every pair
of redshifts** -- the dark-energy term cancels identically. So a measured
``Om`` that is not constant is a departure from LCDM, without any dark-energy
parametrisation ever being written down.

Shafieloo, Sahni & Starobinsky (2012), Phys. Rev. D 86, 103527 then noticed
that a *ratio* of two-point diagnostics sharing a reference redshift removes
even more:

    ``Om3(z1, z2, z3) = Om(z2; z1) / Om(z3; z1) = 1``  for LCDM.

The ``H0^2`` in ``h`` cancels between numerator and denominator, and so does
``Omega_m``. Om3 therefore needs no prior knowledge of the Hubble constant, of
the matter density, or of the distance to last scattering -- it is a statement
about the *shape* of ``H(z)`` and nothing else, and its null value is the
exact number 1 rather than an unknown constant.

That difference matters more in practice than it looks. ``Om(z)`` needs
``H(0)``, and the lowest cosmic chronometer sits at ``z = 0.07``, so every
``Om(z)`` reconstructed from chronometers alone is standing on a short
extrapolation whose uncertainty spreads into the statistic at every redshift.
Om3 stands on nothing of the kind.

Why this module is short
------------------------

Both diagnostics are one line of arithmetic on a
:class:`~CosmoRecon.core.reconstruction.Reconstruction`, and that is the point.
``Om3`` at different ``z3`` all share ``H(z1)`` and ``H(z2)``, so the curve is
strongly correlated along its length and strongly non-Gaussian -- it is a ratio
of two differences. Propagating pointwise error bars through it, which is what
a curve-with-error-bars representation forces, gives an interval that is simply
wrong. Here the correlation and the skew survive because the draws were never
summarised away, and the significance is computed against the number of
independent directions the curve actually has.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DataError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import NullTest


__all__ = ["Om", "Om3"]


#: How close a grid point may come to a reference redshift before the
#: statistic is refused. At ``z = z1`` both the numerator and the denominator
#: of a two-point Om vanish, so the ratio is 0/0 and its neighbourhood is
#: numerically worthless long before it is exactly singular.
_MIN_SEPARATION = 1e-3


def _two_point(H: Reconstruction, z_reference: float) -> Reconstruction:
    """
    ``[H^2(z) / H^2(z_ref) - 1] / [(1+z)^3 - (1+z_ref)^3]``.

    The two-point Om, normalised by the expansion rate at the reference
    redshift rather than by ``H0``. That choice does two things at once: it
    makes the statistic dimensionless without needing ``H0``, and at
    ``z_ref = 0`` it reduces to ``Om(z)`` exactly, since ``h(0) = 1``.

    For a cosmological constant the result is ``Omega_m / h^2(z_ref)`` --
    still a constant, still redshift-independent, still zero-slope. Only its
    *value* depends on the choice of reference, and only the ``z_ref = 0``
    case has ``Omega_m`` as that value.
    """

    z = H.z

    gap = np.abs(z - z_reference)

    if np.any(gap < _MIN_SEPARATION):

        raise DataError(
            f"The grid reaches z = {z[np.argmin(gap)]:.6g}, within "
            f"{_MIN_SEPARATION:g} of the reference redshift "
            f"{z_reference:g}. A two-point Om is 0/0 there and useless in a "
            "neighbourhood of it -- evaluate the reconstruction on a grid "
            "that stays away from its own reference point."
        )

    reference = H.at(z_reference)

    return ((H / reference) ** 2 - 1.0) / (
        (1.0 + z) ** 3 - (1.0 + z_reference) ** 3
    )


# ============================================================
# Om
# ============================================================

class Om(NullTest):
    """
    The ``Om`` diagnostic: constant if dark energy is a cosmological constant.

    >>> Om().evaluate(H)                                       # doctest: +SKIP
    >>> Om(omega_m=0.3).evaluate(H)                            # doctest: +SKIP
    >>> Om(z_reference=0.5).evaluate(H)                        # doctest: +SKIP

    Parameters
    ----------
    z_reference
        The redshift the diagnostic is anchored at. ``0`` gives the textbook
        ``Om(z)``, whose value is ``Omega_m`` -- and which needs ``H(0)``,
        an extrapolation for any dataset that does not reach it.

        A non-zero reference gives ``Om(z; z_ref)`` normalised by
        ``H^2(z_ref)``, which needs no extrapolation at all. Its value is no
        longer ``Omega_m``, but its *constancy* is the same null test, and on
        data that start at ``z = 0.07`` it is the better-founded one. Om3
        below removes the anchor entirely.
    omega_m
        Test against this specific value rather than against "some constant".
        A stronger test, and only meaningful at ``z_reference = 0``, where the
        diagnostic actually equals ``Omega_m``.

    Notes
    -----
    The difference diagnostic of the original paper,
    ``Om_diff(z1, z2) = Om(z1) - Om(z2)``, is this statistic minus itself at
    one point: ``curve - curve.at(z1)``. It needs no separate class, and
    written that way the correlation between the two terms is carried
    automatically.
    """

    def __init__(
        self,
        *,
        z_reference: float = 0.0,
        omega_m: float | None = None,
    ) -> None:

        if omega_m is not None and z_reference != 0.0:

            raise ValueError(
                f"Om is only equal to Omega_m when anchored at z = 0, and "
                f"this one is anchored at z = {z_reference:g}, where its "
                f"value is Omega_m / h^2({z_reference:g}) instead. Test "
                "against a free constant, or move the anchor to zero and "
                "accept the extrapolation."
            )

        self.z_reference = float(z_reference)

        self.omega_m = None if omega_m is None else float(omega_m)

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        if self.z_reference == 0.0:
            return "Om(z)" if self.omega_m is None else f"Om(z) vs {self.omega_m:g}"

        return f"Om(z; {self.z_reference:g})"

    @property
    def null_value(self) -> float:

        return 0.0 if self.omega_m is None else self.omega_m

    @property
    def null_is_free_constant(self) -> bool:
        """
        Whether the test is "constant, value unknown".

        ``True`` unless a specific ``omega_m`` was given -- and when it is
        true the ``null_value`` above is never used, because projecting out
        the uniform direction removes any constant offset along with it.
        """

        return self.omega_m is None

    def statistic(self, H: Reconstruction) -> Reconstruction:

        curve = _two_point(H, self.z_reference)

        curve.label = self.name

        return curve


# ============================================================
# Om3
# ============================================================

class Om3(NullTest):
    """
    The ``Om3`` diagnostic: exactly 1 if dark energy is a cosmological
    constant.

    ``Om3(z1, z2, z3) = Om(z2; z1) / Om(z3; z1)``, evaluated as a curve in
    ``z3`` over the reconstruction's grid.

    >>> Om3(z1=0.2, z2=0.5).evaluate(H)                        # doctest: +SKIP

    What makes it worth having alongside :class:`Om`:

    **No calibration enters.** ``H0`` cancels between numerator and
    denominator, and so does ``Omega_m``. There is no extrapolation to
    ``z = 0``, no sound horizon, no absolute magnitude -- only the shape of
    ``H(z)`` at three redshifts.

    **The null value is a number, not a parameter.** ``Om`` tests "is this
    curve flat?", which costs a degree of freedom to the unknown level it is
    flat at. Om3 tests "is this curve equal to one?", which costs nothing.

    **The deviation has a direction.** Quintessence (``w > -1``) drives Om3
    above 1 and phantom (``w < -1``) below it, and the departure grows with
    the separation ``z3 - z2`` -- so the diagnostic says not only that
    something is wrong but which way.

    Parameters
    ----------
    z1
        The shared reference. Both two-point diagnostics are anchored here, so
        the grid must stay away from it: the denominator is 0/0 at ``z3 = z1``.
    z2
        The fixed comparison redshift, giving the numerator. It must be far
        enough from ``z1`` that ``Om(z2; z1)`` is not itself a small
        difference of two similar numbers.
    """

    null_is_free_constant = False

    def __init__(self, z1: float, z2: float) -> None:

        z1, z2 = float(z1), float(z2)

        if abs(z2 - z1) < _MIN_SEPARATION:

            raise ValueError(
                f"z1 = {z1:g} and z2 = {z2:g} are the same redshift to within "
                f"{_MIN_SEPARATION:g}. The numerator of Om3 would be a "
                "difference of two nearly equal numbers divided by another, "
                "which carries no information and a great deal of noise."
            )

        self.z1 = z1

        self.z2 = z2

    # ---------------------------------------------------------

    @property
    def name(self) -> str:

        return f"Om3({self.z1:g}, {self.z2:g}, z)"

    @property
    def null_value(self) -> float:

        return 1.0

    def statistic(self, H: Reconstruction) -> Reconstruction:

        # Both terms come from the same fit, so they share a realisation
        # index: the H(z1) that appears in the numerator is the *same draw*
        # as the H(z1) in the denominator, and cancels the way it should.
        # Built from summaries instead, it would not.
        numerator = _two_point(H.at(self.z2), self.z1)

        denominator = _two_point(H, self.z1)

        curve = numerator / denominator

        curve.label = self.name

        return curve
