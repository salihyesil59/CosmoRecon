"""
Expansion variables and polynomial families, and the calculus of composing
them.

A cosmographic reconstruction is a series in *something*. Which something is
not a detail of implementation -- it decides whether the series converges at
all over the redshifts being fitted, and it is the difference between an
honest reconstruction and one that diverges quietly.

The convergence problem
-----------------------

Cosmological distances, as functions of complex ``z``, carry a singularity at
``z = -1``. Every Taylor series in ``z`` around the origin therefore has radius
of convergence ``|z| = 1``, whatever the cosmology and however many terms are
kept -- and past that radius adding terms makes the truncation *worse*, not
better. Supernova compilations reach ``z ~ 2.3`` and DESI's Lyman-alpha BAO
reaches ``z = 2.33``, both comfortably outside it.

This is not a subtle point and it is not new: Cattoen and Visser introduced the
variable ``y = z / (1 + z)`` in 2007 precisely to fix it. The map sends
``z = -1`` to ``y = infinity`` and the whole of ``z > 0`` into ``y < 1``, so a
series in ``y`` converges over the entire observable range. It is nonetheless
routine to see a Taylor series in ``z`` fitted to high-redshift supernovae and
a jerk parameter read off the result.

So each variable here declares the redshift range over which a series in it
converges, and :mod:`CosmoRecon.reconstructors.cosmography` refuses to fit
outside it.

Families
--------

The second axis is which polynomials carry the series -- monomials, or
Chebyshev. It is worth being clear that this axis is *much* less consequential
than the first, in a way the literature tends to obscure by listing "Taylor,
Chebyshev and Pade" as three comparable choices:

**A monomial series and a Chebyshev series of the same degree in the same
variable span exactly the same functions.** They are two coordinate systems on
one space of polynomials. Fitted with a flat prior they give the identical
posterior; they differ here only because a prior on the coefficients is not
invariant under the change of basis, and because the Chebyshev design matrix
is vastly better conditioned at high degree.

The genuinely different choices are the *variable* (this module) and going
rational rather than polynomial (Pade, in the reconstructor).

Derivatives
-----------

Cosmography is about high derivatives -- the deceleration, jerk and snap
parameters are the second, third and fourth -- and the derivative wanted is
always with respect to ``z``, while the series is in ``x(z)``. Composing the
two is Faa di Bruno's formula, implemented here through the partial Bell
polynomials so that it works at any order rather than at the two or three
someone wrote out by hand.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import chebyshev as C

from CosmoRecon.typing import Array


__all__ = [
    "ExpansionVariable",
    "Redshift",
    "YRedshift",
    "LogRedshift",
    "VARIABLES",
    "get_variable",
    "design_matrix",
    "bell_polynomials",
    "compose_derivatives",
]


# ============================================================
# Expansion variables
# ============================================================

class ExpansionVariable(ABC):
    """
    A change of variable ``x(z)`` that a series is written in.

    Subclasses supply the map, its derivatives (needed for the chain rule at
    arbitrary order), and -- the part that matters -- the redshift range over
    which a series in it actually converges.
    """

    #: How this variable should be named in a caption.
    name: str = ""

    @abstractmethod
    def __call__(self, z: Array) -> Array:
        """The variable itself."""

    @abstractmethod
    def derivatives(self, z: Array, order: int) -> list[Array]:
        """
        ``[x'(z), x''(z), ..., x^(order)(z)]``.

        Closed form for every variable here, which is what keeps the chain
        rule exact at the orders cosmography cares about.
        """

    @abstractmethod
    def convergence_range(self) -> tuple[float, float]:
        """
        The redshifts over which a series in this variable converges.

        A property of the variable and of the singularity structure of a
        cosmological distance, not of the data or of how many terms are kept.
        """

    @abstractmethod
    def why(self) -> str:
        """
        One sentence on where :meth:`convergence_range` comes from, used in
        the error raised when a fit leaves it.
        """

    def __repr__(self) -> str:

        lo, hi = self.convergence_range()

        return f"<{type(self).__name__} valid for {lo:g} <= z <= {hi:g}>"


class Redshift(ExpansionVariable):
    """
    ``x = z``. The textbook cosmographic series, and the one that does not
    converge where the data are.
    """

    name = "z"

    def __call__(self, z):

        return np.asarray(z, dtype=float)

    def derivatives(self, z, order):

        z = np.asarray(z, dtype=float)

        return [np.ones_like(z)] + [np.zeros_like(z)] * (order - 1)

    def convergence_range(self):

        return (0.0, 1.0)

    def why(self):

        return (
            "a cosmological distance has a singularity at z = -1, so its "
            "Taylor series about the origin has radius of convergence "
            "|z| = 1 -- no choice of cosmology and no number of terms "
            "changes that (Cattoen & Visser 2007)"
        )


class YRedshift(ExpansionVariable):
    """
    ``x = z / (1 + z)``.

    The fix. The singularity at ``z = -1`` maps to ``y = infinity`` and the
    whole of ``z > 0`` maps into ``y < 1``, so the series converges over every
    redshift anyone has data at. Introduced by Cattoen and Visser for exactly
    this reason, and the right default.
    """

    name = "y = z/(1+z)"

    def __call__(self, z):

        z = np.asarray(z, dtype=float)

        return z / (1.0 + z)

    def derivatives(self, z, order):

        z = np.asarray(z, dtype=float)

        # d^n/dz^n [z/(1+z)] = (-1)^(n-1) n! / (1+z)^(n+1)
        return [
            (-1.0) ** (n - 1) * math.factorial(n) / (1.0 + z) ** (n + 1)
            for n in range(1, order + 1)
        ]

    def convergence_range(self):

        # |y| < 1 corresponds to z > -1/2, so every positive redshift is
        # inside it. The upper end is unbounded in principle; capped at
        # something no dataset reaches, so the check stays a real check.
        return (0.0, 1.0e6)

    def why(self):

        return (
            "y = z/(1+z) sends the z = -1 singularity to infinity and maps "
            "all of z > 0 into |y| < 1, which is why it exists"
        )


class LogRedshift(ExpansionVariable):
    """
    ``x = ln(1 + z)``.

    Also sends ``z = -1`` to infinity, so it is free of the obstruction that
    limits the ``z`` series, and it is the natural variable when the quantity
    varies over decades of scale factor.

    Its convergence range is declared conservatively as the same as ``y``'s.
    That is an honest statement of what is known rather than a derived radius:
    the ``z = -1`` singularity is gone, but no general bound is claimed here
    for whatever else sits in the complex plane for a given expansion history.
    """

    name = "ln(1+z)"

    def __call__(self, z):

        return np.log1p(np.asarray(z, dtype=float))

    def derivatives(self, z, order):

        z = np.asarray(z, dtype=float)

        # d^n/dz^n ln(1+z) = (-1)^(n-1) (n-1)! / (1+z)^n
        return [
            (-1.0) ** (n - 1) * math.factorial(n - 1) / (1.0 + z) ** n
            for n in range(1, order + 1)
        ]

    def convergence_range(self):

        return (0.0, 1.0e6)

    def why(self):

        return (
            "ln(1+z) sends the z = -1 singularity to infinity, so the "
            "obstruction that caps the z series at |z| = 1 is absent"
        )


#: Variables reachable by name.
VARIABLES: dict[str, type[ExpansionVariable]] = {
    "z": Redshift,
    "redshift": Redshift,
    "y": YRedshift,
    "y_redshift": YRedshift,
    "log": LogRedshift,
    "log1p": LogRedshift,
}


def get_variable(variable: ExpansionVariable | str) -> ExpansionVariable:
    """Resolve a variable given as an instance or a name."""

    if isinstance(variable, ExpansionVariable):
        return variable

    try:
        return VARIABLES[str(variable).lower()]()

    except KeyError:

        raise ValueError(
            f"Unknown expansion variable {variable!r}. Available: "
            f"{sorted(set(VARIABLES))}. The default 'y' is the one that "
            "converges over the redshifts data actually reach."
        ) from None


# ============================================================
# Polynomial families
# ============================================================

FAMILIES = ("chebyshev", "monomial")


def design_matrix(
    u: Array,
    degree: int,
    family: str,
    *,
    derivative: int = 0,
) -> Array:
    """
    ``(degree + 1, len(u))`` basis functions, or their ``derivative``-th
    derivative **with respect to** ``u``.

    The chain rule from ``u`` to ``z`` happens in :func:`compose_derivatives`;
    this is the easy half, done in closed form for both families.
    """

    u = np.atleast_1d(np.asarray(u, dtype=float))

    if family == "monomial":

        out = np.zeros((degree + 1, u.size))

        for j in range(degree + 1):

            if j < derivative:
                continue

            # d^n/du^n u^j = j! / (j - n)! * u^(j - n)
            factor = math.factorial(j) / math.factorial(j - derivative)

            out[j] = factor * u ** (j - derivative)

        return out

    if family == "chebyshev":

        out = np.empty((degree + 1, u.size))

        for j in range(degree + 1):

            coefficients = np.zeros(j + 1)
            coefficients[j] = 1.0

            if derivative:
                coefficients = C.chebder(coefficients, m=derivative)

            out[j] = C.chebval(u, coefficients) if coefficients.size else 0.0

        return out

    raise ValueError(f"Unknown polynomial family {family!r}; use one of {FAMILIES}.")


# ============================================================
# Faa di Bruno
# ============================================================

def bell_polynomials(
    inner: list[Array],
    order: int,
) -> list[list[Array]]:
    """
    Partial Bell polynomials ``B[n][k]`` of the inner derivatives.

    ``inner[i - 1]`` is ``x^(i)(z)``. Built by the standard recursion

        ``B[n][k] = sum_i C(n-1, i-1) x^(i) B[n-i][k-1]``

    which is short, is defined at every order, and beats writing the chain
    rule out by hand for the third and fourth derivatives -- the ones
    cosmography is actually about.

    Returns a lower-triangular list-of-lists, each entry an array over ``z``.
    """

    shape = inner[0].shape if inner else ()

    zero = np.zeros(shape)

    one = np.ones(shape)

    B: list[list[Array]] = [[zero] * (order + 1) for _ in range(order + 1)]

    B[0][0] = one

    for n in range(1, order + 1):

        for k in range(1, n + 1):

            total = np.zeros(shape)

            for i in range(1, n - k + 2):

                total = total + math.comb(n - 1, i - 1) * inner[i - 1] * B[n - i][k - 1]

            B[n][k] = total

    return B


def compose_derivatives(
    derivatives_in_x: list[Array],
    inner: list[Array],
    order: int,
) -> Array:
    """
    Faa di Bruno: turn derivatives with respect to the expansion variable into
    the derivative with respect to redshift.

        ``d^n/dz^n f(x(z)) = sum_k f^(k)(x) B[n][k](x', x'', ...)``

    ``derivatives_in_x[k - 1]`` is ``f^(k)(x)``, shaped ``(n_draws, n_z)``;
    ``inner[i - 1]`` is ``x^(i)(z)``, shaped ``(n_z,)``.
    """

    if order == 0:

        raise ValueError("compose_derivatives is for order >= 1.")

    B = bell_polynomials(inner, order)

    total = np.zeros_like(derivatives_in_x[0])

    for k in range(1, order + 1):

        total = total + derivatives_in_x[k - 1] * B[order][k][None, :]

    return total
