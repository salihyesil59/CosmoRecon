"""
Physical constants, and nothing that belongs to a cosmological model.

The line matters. ``c`` is a definition; ``Omega_m = 0.315`` is a result of an
analysis this library exists to avoid assuming. Anything with a subscript that
came out of a fit lives in the call that needs it, where a reader can see the
choice being made.
"""

from __future__ import annotations


__all__ = ["C_LIGHT_KM_S", "C_LIGHT_M_S", "MPC_IN_KM"]


#: Speed of light in km/s. Exact -- the metre is defined from it.
#:
#: This is the unit the library works in throughout, because it is the one in
#: which the observables arrive: ``H(z)`` from cosmic chronometers is quoted
#: in km/s/Mpc, ``D_M/r_d`` is dimensionless, and a comoving distance in Mpc
#: times ``H`` in km/s/Mpc lands in km/s without a conversion factor anywhere.
C_LIGHT_KM_S: float = 299_792.458

#: The same, in m/s, for the rare place that needs SI.
C_LIGHT_M_S: float = 299_792_458.0

#: One megaparsec in kilometres. IAU 2015 definition of the parsec via the
#: astronomical unit and the arcsecond.
MPC_IN_KM: float = 3.085_677_581_491_367e19
