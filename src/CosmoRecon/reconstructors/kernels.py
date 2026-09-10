"""
Covariance kernels, and the spectral densities they are sampled from.

Each kernel here answers three questions, and the third is the one that is
usually skipped.

**What is the covariance?** ``covariance(r, ...)`` -- used for the exact
marginal likelihood that weights the hyperparameter posterior, and for
checking the sampled paths against the kernel they claim to come from.

**What is its spectral density?** ``log_spectral_density(...)``, from which
``spectral_weights(...)`` builds the quadrature that carries a sample path.
Bochner's theorem says a stationary kernel is the Fourier transform of a
positive measure, so the two statements are equivalent -- and they are written
here as independent pieces of code, which is what makes comparing them a real
check rather than a tautology. The test suite runs that comparison for every
kernel.

**How many times is it differentiable?** ``max_derivative(...)``, and this is
where the library differs from what is standard practice. A Matern process
with smoothness ``nu`` is ``m`` times mean-square differentiable only for
``m < nu``: a Matern-1/2 process is nowhere differentiable, and a Matern-3/2
process has a first derivative but no second. Reconstructions of ``w(z)``,
``q(z)`` and the whole cosmographic family are built from ``H'(z)`` and
``H''(z)``, so a Matern-3/2 kernel -- a common default -- cannot support the
quantity it is routinely used to produce. The numbers do come out, because a
finite sum of cosines is smooth whatever process it approximates; they are
just not a posterior on anything. This module reports the limit, and
:mod:`~CosmoRecon.reconstructors.gp` enforces it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy import special

from CosmoRecon.typing import Array


__all__ = [
    "Kernel",
    "SquaredExponential",
    "Matern",
    "RationalQuadratic",
    "Cauchy",
    "KERNELS",
    "get_kernel",
]


# ============================================================
# The contract
# ============================================================

class Kernel(ABC):
    """
    A stationary covariance kernel of unit amplitude.

    Amplitude and length scale are handled uniformly by the reconstructor --
    every kernel here has both, and duplicating them per class would only
    invite them to drift. What a subclass declares is its *shape*: the extra
    parameter, if any, that changes the character of the functions it
    generates rather than their size.
    """

    #: The shape parameters this kernel marginalises over. Empty for kernels
    #: whose only freedom is amplitude and length scale.
    shape_params: tuple[str, ...] = ()

    # ---------------------------------------------------------

    @abstractmethod
    def covariance(
        self,
        r: Array,
        length_scale: float,
        **shape: float,
    ) -> Array:
        """
        Unit-amplitude covariance at separations ``r >= 0``.
        """

    @abstractmethod
    def log_spectral_density(
        self,
        omega: Array,
        length_scale: float,
        **shape: float,
    ) -> Array:
        """
        Log of the spectral density at angular frequencies ``omega >= 0``,
        **up to an additive constant**.

        Bochner's theorem makes this an equivalent statement of the kernel,
        and it is the one a sample path is built from. The normalisation is
        dropped on purpose: :meth:`spectral_weights` renormalises so that the
        quadrature reproduces ``k(0) = 1`` exactly, which both removes every
        awkward constant from the expressions below and makes the variance
        right by construction rather than by cancellation.
        """

    def spectral_weights(
        self,
        n_nodes: int,
        length_scale: float,
        **shape: float,
    ) -> tuple[Array, Array]:
        """
        Quadrature nodes and weights with ``sum_m w_m cos(omega_m r) = k(r)``.

        This is the piece that makes accurate sample paths possible. The
        usual construction -- random Fourier features, ``omega`` drawn from
        the spectral density -- converges as ``1 / sqrt(M)``, and in practice
        that means several percent of error in the *width* of the
        reconstructed band at any affordable feature count. An error bar is
        the output of this library, so several percent of it being a
        representation artefact is not acceptable.

        In one dimension there is no need to sample. The spectral density is
        a one-dimensional integral, and the substitution ``omega = tan(t) /
        l`` maps it onto ``(0, pi/2)``, where the midpoint rule on a smooth
        periodic-in-effect integrand converges geometrically rather than as a
        square root. A few hundred nodes give what Monte Carlo would need
        billions of draws for.

        The substitution is also well matched to the kernels that need it
        most: for a Matern with ``nu = 1/2`` -- the heaviest spectral tail
        here -- it flattens the integrand exactly, and the quadrature is
        machine-precision at any node count.

        Weights are normalised to sum to one, which is ``k(0) = 1``. That
        absorbs both the density's normalisation and the truncation error at
        the tail, so a reconstruction never has the wrong variance even where
        the quadrature has to work hardest.
        """

        if n_nodes < 8:

            raise ValueError(f"{n_nodes} quadrature nodes is not enough.")

        t = (np.arange(n_nodes) + 0.5) * (0.5 * math.pi / n_nodes)

        omega = np.tan(t) / length_scale

        # d(omega)/dt, in logs -- the substitution's Jacobian.
        log_jacobian = -2.0 * np.log(np.cos(t)) - math.log(length_scale)

        log_w = self.log_spectral_density(omega, length_scale, **shape) + log_jacobian

        weights = np.exp(log_w - log_w.max())

        total = weights.sum()

        if not np.isfinite(total) or total <= 0.0:

            raise ValueError(
                f"The spectral quadrature for {type(self).__name__} "
                f"degenerated at length_scale={length_scale:g}, shape={shape}."
            )

        return omega, weights / total

    def spectral_quadrature(
        self,
        length_scale: float,
        r_max: float,
        *,
        tol: float = 3e-4,
        n_start: int = 256,
        n_max: int = 2048,
        **shape: float,
    ) -> tuple[Array, Array, float]:
        """
        Nodes and weights refined until they reproduce the kernel to ``tol``.

        Returns ``(omega, weights, achieved_error)``, the error being the
        largest absolute discrepancy between the quadrature and the closed-form
        covariance over separations ``[0, r_max]`` -- so it is measured on the
        range the data actually span, not on an abstract one.

        The default tolerance is set by what it has to be invisible against.
        An error ``eps`` in a covariance of unit amplitude moves the posterior
        standard deviation by around ``eps / 2`` in relative terms, so ``3e-4``
        puts the representation's contribution four orders of magnitude below
        any statistical uncertainty a reconstruction will ever quote. Chasing
        machine precision here would only buy quadrature nodes.

        This is a guarantee rather than a hope, and it is checked rather than
        assumed because the failure it prevents is silent: a sample path built
        on an under-resolved spectrum comes from a slightly different kernel
        than the marginal likelihood was computed under, and nothing about the
        output looks wrong.

        A kernel that cannot reach ``tol`` within ``n_max`` nodes raises. That
        happens for a genuinely rough Matern -- around ``nu <= 1``, where the
        spectral tail decays so slowly that a truncated representation of a
        sample path is not faithful at any affordable size. Such a process has
        no derivative either, so nothing downstream in this library can use it;
        refusing is more useful than a quiet approximation.
        """

        # Enough points to see the kernel's shape, weighted towards small
        # separations where it varies fastest.
        r = np.linspace(0.0, max(float(r_max), length_scale), 64)

        exact = self.covariance(r, length_scale, **shape)

        n = int(n_start)

        while True:

            omega, weights = self.spectral_weights(n, length_scale, **shape)

            approx = (weights[None, :] * np.cos(np.outer(r, omega))).sum(axis=1)

            error = float(np.max(np.abs(approx - exact)))

            if error <= tol:

                return omega, weights, error

            if n >= n_max:

                raise ValueError(
                    f"{self.describe(**shape)} at length_scale="
                    f"{length_scale:.4g} cannot be represented to {tol:g} "
                    f"with {n_max} quadrature nodes (best {error:.2g}). Its "
                    "spectral tail is too heavy for a truncated sample-path "
                    "basis. For a Matern this means a very small nu, which "
                    "also has no derivatives -- prefer nu >= 1, or raise "
                    "n_max and tol deliberately."
                )

            n *= 2

    def max_derivative(self, **shape: float) -> int | None:
        """
        The highest derivative that exists, in mean square.

        ``None`` means every order exists -- the analytic kernels. A finite
        value is a hard limit: past it, the process has no derivative, and any
        number produced for one is an artefact of the representation rather
        than a statement about the function.
        """

        return None

    def shape_grid(self) -> dict[str, Array]:
        """
        The values of each shape parameter to marginalise over.

        Empty for a kernel with no shape parameter. A single-element array
        pins the parameter instead of marginalising it -- which is what
        ``Matern(nu=1.5)`` does, and what this library argues against doing
        by default.
        """

        return {}

    @abstractmethod
    def describe(self, **shape: float) -> str:
        """This kernel, spelled for a figure caption."""

    def __repr__(self) -> str:

        return f"<{type(self).__name__}>"


# ============================================================
# Squared exponential
# ============================================================

class SquaredExponential(Kernel):
    """
    ``k(r) = exp(-r^2 / 2 l^2)``.

    Spectral density ``N(0, 1/l^2)``. Sample paths are analytic, so every
    derivative exists -- which is the kernel's attraction and its problem: it
    imposes infinite smoothness on an expansion history nobody has shown to
    be infinitely smooth, and its extrapolation is correspondingly
    overconfident. It is the default in most published GP reconstructions of
    ``H(z)``, almost always without that assumption being stated.
    """

    def covariance(self, r, length_scale, **shape):

        return np.exp(-0.5 * (np.asarray(r, dtype=float) / length_scale) ** 2)

    def log_spectral_density(self, omega, length_scale, **shape):

        return -0.5 * (np.asarray(omega, dtype=float) * length_scale) ** 2

    def describe(self, **shape):

        return "GP(squared exponential)"


# ============================================================
# Matern
# ============================================================

class Matern(Kernel):
    """
    ``k(r) = 2^(1-nu)/Gamma(nu) * (sqrt(2 nu) r / l)^nu * K_nu(sqrt(2 nu) r / l)``.

    Spectral density is Student-``t`` with ``2 nu`` degrees of freedom and
    scale ``1/l`` -- heavier-tailed than a Gaussian, which is exactly the
    roughness that ``nu`` controls.

    ``nu`` is free by default, and that is the point of this class. Fixing it
    (usually at 3/2 or 5/2, chosen by habit) picks the smoothness of the
    reconstructed expansion history by hand and then reports an interval that
    does not include that choice. Published analyses differ from one another
    by roughly as much as their kernels differ, which is the observation this
    whole library is built around. Here the data are allowed to say.

    Pass ``nu=1.5`` to pin it -- for reproducing someone else's analysis, or
    for an ensemble member that is deliberately one fixed choice.
    """

    shape_params = ("nu",)

    #: The grid ``nu`` is marginalised over when free. Runs from the roughest
    #: process whose sample paths can be represented faithfully, up to
    #: effectively analytic, and includes both conventional fixed choices
    #: (3/2 and 5/2) so that a marginalised result can be compared directly
    #: against the habit it replaces.
    #:
    #: It stops at ``nu = 1`` rather than at the Ornstein-Uhlenbeck case
    #: ``nu = 1/2`` for two reasons that point the same way: below ``nu = 1``
    #: the spectral tail is too heavy for a truncated sample-path basis to
    #: reproduce (see :meth:`Kernel.spectral_quadrature`), and such a process
    #: is nowhere differentiable, so no derived quantity in this library --
    #: ``w(z)``, ``q(z)``, anything cosmographic -- exists under it. Pass
    #: ``Matern(nu=0.5)`` to use it anyway.
    DEFAULT_NU_GRID = (1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5)

    def __init__(self, nu: float | Sequence[float] | None = None) -> None:

        #: ``None`` marginalises over :attr:`DEFAULT_NU_GRID`; a number pins;
        #: a sequence marginalises over exactly that set.
        #:
        #: The sequence form is how a derivative is bought. A free ``nu``
        #: posterior fitted to thirty-odd measurements reaches down to
        #: ``nu = 1``, where the process has no derivative at all, so ``H'(z)``
        #: is undefined on part of it and the library will say so. Passing
        #: ``nu=[1.5, 2.0, 2.5, 3.5, 5.0, 7.5]`` restricts the prior to
        #: processes that are once differentiable -- an assumption, stated
        #: where a reader can see it, rather than one smuggled in by fixing
        #: ``nu`` at a convention.
        self.nu = nu

    # ---------------------------------------------------------

    def covariance(self, r, length_scale, *, nu, **shape):

        r = np.abs(np.asarray(r, dtype=float))

        u = np.atleast_1d(math.sqrt(2.0 * nu) * r / length_scale)

        # K_nu diverges at the origin while u^nu vanishes; the product tends
        # to the normalisation, so the limit is set rather than computed.
        out = np.ones_like(u)

        far = u > 1e-10

        if np.any(far):

            uf = u[far]

            # Evaluated in logs, and through the exponentially scaled Bessel
            # function `kve = K_nu(u) exp(u)`. The direct product overflows
            # in `u**nu` and underflows in `2**(1-nu)` well before nu leaves
            # the range this kernel is used over -- the two cancel
            # analytically, and 0 * inf does not.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):

                log_k = (
                    (1.0 - nu) * math.log(2.0)
                    - special.gammaln(nu)
                    + nu * np.log(uf)
                    + np.log(special.kve(nu, uf))
                    - uf
                )

                value = np.exp(log_k)

            # `kve` itself overflows for small u at large nu -- K_nu(u) grows
            # like Gamma(nu)(2/u)^nu there, which leaves the range of a double
            # long before the *product* does. That is exactly the regime where
            # the small-argument expansion
            #
            #     k = 1 - u^2 / 4(nu - 1) + u^4 / 32(nu - 1)(nu - 2) + ...
            #
            # is accurate, so the two cover the line between them. As
            # nu -> infinity this expansion is the squared exponential's, which
            # is the limit it has to reproduce.
            broken = ~np.isfinite(value)

            if np.any(broken):

                ub = uf[broken]

                if nu <= 2.0 or np.any(ub**2 > 0.25 * 4.0 * (nu - 1.0)):

                    raise ValueError(
                        f"The Matern covariance cannot be evaluated at "
                        f"nu = {nu:g} for these separations in double "
                        "precision. Such a large nu is the squared "
                        "exponential in all but name -- use "
                        "SquaredExponential() instead."
                    )

                series = 1.0 - ub**2 / (4.0 * (nu - 1.0))

                series += ub**4 / (32.0 * (nu - 1.0) * (nu - 2.0))

                value[broken] = series

            out[far] = value

        return out.reshape(np.shape(r))

    def log_spectral_density(self, omega, length_scale, *, nu, **shape):

        # Student-t with 2 nu degrees of freedom and scale 1/l, up to its
        # normalisation. The heavier the tail, the rougher the paths -- which
        # is the entire content of nu, stated in the domain where it is a
        # single exponent.
        omega = np.asarray(omega, dtype=float)

        return -(nu + 0.5) * np.log1p((omega * length_scale) ** 2 / (2.0 * nu))

    def max_derivative(self, *, nu, **shape):
        """
        A Matern-``nu`` process is ``m`` times differentiable iff ``m < nu``.

        So Matern-1/2 supports no derivative at all, Matern-3/2 supports the
        first and not the second, and ``w(z)`` -- which needs ``H'`` -- is
        undefined under the first and defined under the second. Cosmographic
        reconstructions reaching ``H''`` need ``nu > 2``.
        """

        return int(math.ceil(nu)) - 1

    def shape_grid(self):

        return {"nu": _as_grid(self.nu, self.DEFAULT_NU_GRID)}

    def describe(self, *, nu=None, **shape):

        # `self.nu` is the configuration; `nu` is the value being used right
        # now, which is what an error message about one grid cell should
        # name. A free kernel asked to describe itself in general still says
        # so.
        if np.isscalar(self.nu):

            return f"GP(Matern, nu = {float(self.nu):g})"

        if self.nu is not None:

            grid = np.asarray(self.nu, dtype=float)

            return f"GP(Matern, nu in [{grid.min():g}, {grid.max():g}])"

        if nu is not None:

            return f"GP(Matern, nu = {float(nu):g})"

        return "GP(Matern, nu free)"

    def __repr__(self):

        return f"<Matern nu={'free' if self.nu is None else self.nu}>"


def _as_grid(value, default: tuple[float, ...]) -> Array:
    """
    Turn ``None`` / a scalar / a sequence into the grid to marginalise over.

    Shared by the two kernels with a shape parameter, so that pinning and
    restricting mean the same thing in both.
    """

    if value is None:
        return np.array(default, dtype=float)

    grid = np.atleast_1d(np.asarray(value, dtype=float))

    if grid.size == 0:
        raise ValueError("An empty shape grid has nothing to marginalise over.")

    return np.sort(grid)


# ============================================================
# Rational quadratic
# ============================================================

#: Used only for the rational quadratic's spectral density, which is a Matern
#: covariance in disguise. Module-level so the dual form does not rebuild it
#: on every quadrature.
_MATERN = Matern()

class RationalQuadratic(Kernel):
    """
    ``k(r) = (1 + r^2 / (2 alpha l^2))^(-alpha)``.

    A Gamma mixture of squared exponentials over length scale, so it describes
    a function varying on several scales at once -- which is what an expansion
    history reconstructed from data spanning ``z = 0`` to ``z = 2.3`` actually
    looks like. Recovers the squared exponential as ``alpha -> infinity``, and
    like it, sample paths are analytic.

    ``alpha`` is marginalised by default, for the same reason ``nu`` is.
    """

    shape_params = ("alpha",)

    DEFAULT_ALPHA_GRID = (0.75, 1.0, 2.0, 5.0, 20.0)

    def __init__(self, alpha: float | Sequence[float] | None = None) -> None:

        self.alpha = alpha

    def covariance(self, r, length_scale, *, alpha, **shape):

        r = np.asarray(r, dtype=float)

        return (1.0 + r**2 / (2.0 * alpha * length_scale**2)) ** (-alpha)

    def log_spectral_density(self, omega, length_scale, *, alpha, **shape):
        """
        The Fourier dual of a Matern.

        ``k(r) = (1 + r^2 / 2 alpha l^2)^-alpha`` has, as a function of ``r``,
        exactly the Student-t shape that a Matern kernel has as a function of
        ``omega``. Reading the correspondence the other way round gives the
        spectral density as a Matern *covariance* evaluated at ``omega``, with
        ``nu = alpha - 1/2`` and a dual length scale -- so the closed form is
        already in this module and needs only to be pointed at the right
        argument.

        The correspondence needs ``alpha > 1/2``; below it the dual order is
        not positive and the kernel's spectrum is not of this family.
        """

        if alpha <= 0.5:

            raise ValueError(
                f"The rational quadratic needs alpha > 1/2, got {alpha:g}. "
                "Below that its spectral density leaves the Matern family "
                "and this implementation does not cover it."
            )

        omega = np.abs(np.asarray(omega, dtype=float))

        nu_dual = alpha - 0.5

        length_dual = math.sqrt((2.0 * alpha - 1.0) / (2.0 * alpha)) / length_scale

        with np.errstate(divide="ignore"):

            return np.log(
                _MATERN.covariance(omega, length_dual, nu=nu_dual)
            )

    def shape_grid(self):

        return {"alpha": _as_grid(self.alpha, self.DEFAULT_ALPHA_GRID)}

    def describe(self, *, alpha=None, **shape):

        if np.isscalar(self.alpha):

            return f"GP(rational quadratic, alpha = {float(self.alpha):g})"

        value = alpha if self.alpha is None else None

        if value is not None:

            return f"GP(rational quadratic, alpha = {float(value):g})"

        return "GP(rational quadratic, alpha free)"


# ============================================================
# Cauchy
# ============================================================

class Cauchy(Kernel):
    """
    ``k(r) = 1 / (1 + (r / l)^2)``.

    Spectral density ``Laplace(0, 1/l)``. Long-range correlations that decay
    polynomially rather than exponentially, so a measurement at ``z = 0.1``
    still says something about ``z = 2`` -- appreciably more confident
    extrapolation than the other three, which makes it a useful ensemble
    member precisely because it disagrees with them where the data run out.
    """

    def covariance(self, r, length_scale, **shape):

        r = np.asarray(r, dtype=float)

        return 1.0 / (1.0 + (r / length_scale) ** 2)

    def log_spectral_density(self, omega, length_scale, **shape):

        return -length_scale * np.abs(np.asarray(omega, dtype=float))

    def describe(self, **shape):

        return "GP(Cauchy)"


# ============================================================
# Lookup
# ============================================================

#: Kernels reachable by name, so ``GaussianProcess(kernel="matern")`` works
#: and an ensemble can be written as a list of strings.
KERNELS: dict[str, type[Kernel]] = {
    "squared_exponential": SquaredExponential,
    "rbf": SquaredExponential,
    "matern": Matern,
    "rational_quadratic": RationalQuadratic,
    "cauchy": Cauchy,
}


def get_kernel(kernel: Kernel | str) -> Kernel:
    """
    Resolve a kernel given as an instance or a name.

    A bare name gives the kernel's *free* form -- ``"matern"`` marginalises
    ``nu`` rather than pinning it at a convention. Pinning is available and
    has to be asked for: ``Matern(nu=1.5)``.
    """

    if isinstance(kernel, Kernel):
        return kernel

    try:
        return KERNELS[str(kernel).lower()]()

    except KeyError:

        raise ValueError(
            f"Unknown kernel {kernel!r}. Available: "
            f"{sorted(set(KERNELS))}. Pass a Kernel instance to configure "
            "one -- e.g. Matern(nu=1.5) to pin the smoothness."
        ) from None
