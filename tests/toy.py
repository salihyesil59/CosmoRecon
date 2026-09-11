"""
Toy predictors, for testing the machinery before there is a reconstructor.

These stand in for a fitted method: a posterior over cosmological parameters,
turned into a posterior over ``H(z)`` that satisfies the
:class:`~CosmoRecon.core.reconstruction.Predictor` contract -- in particular
the determinism clause, since the parameter draws are fixed at construction
and every evaluation reuses them.

Their point is that the truth is known in closed form, so a test can check
not only that the machinery runs but that it gets the right answer.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DerivativeUnavailableError
from CosmoRecon.core.provenance import Provenance, new_origin
from CosmoRecon.core.reconstruction import Reconstruction


class FlatLCDM:
    """
    ``H(z) = H0 sqrt(Om (1+z)^3 + 1 - Om)``, with a Gaussian spread in the
    two parameters standing in for a posterior.

    The exact case: ``Om(z)`` built from this is constant at ``Om``, to
    machine precision, at every redshift.
    """

    def __init__(self, n_draws=4000, H0=70.0, sigma_H0=1.0, Om=0.30, sigma_Om=0.02, seed=0):

        rng = np.random.default_rng(seed)

        self.H0 = rng.normal(H0, sigma_H0, n_draws)
        self.Om = rng.normal(Om, sigma_Om, n_draws)

    def _E2(self, z):

        return self.Om[:, None] * (1 + z) ** 3 + (1 - self.Om)[:, None]

    def __call__(self, z, *, derivative=0):

        z = np.atleast_1d(np.asarray(z, dtype=float))

        E2 = self._E2(z)

        if derivative == 0:
            return self.H0[:, None] * np.sqrt(E2)

        if derivative == 1:
            dE2 = 3 * self.Om[:, None] * (1 + z) ** 2
            return self.H0[:, None] * dE2 / (2 * np.sqrt(E2))

        raise DerivativeUnavailableError("toy model provides orders 0 and 1")


class FlatCPL(FlatLCDM):
    """
    The same, with ``w(z) = w0 + wa z / (1+z)``.

    ``Om(z)`` built from this is *not* constant, which is what makes it the
    signal a null test has to find.

    ``w0`` and ``wa`` are **drawn, not fixed**, and that is not a detail. Hold
    them fixed and the deviation from a constant has an exact value with no
    uncertainty at all: the null test then divides a real residual by a
    covariance that is pure rounding error, and reports the same enormous
    chi-square for every choice of ``w0`` and ``wa``. A posterior with no
    width in the parameters that carry the signal is not a posterior, and a
    test built on one measures the floating-point unit.

    The default widths are roughly a DESI-DR2-scale constraint.
    """

    def __init__(self, w0=-0.9, wa=-0.6, sigma_w0=0.06, sigma_wa=0.25, **kwargs):

        super().__init__(**kwargs)

        rng = np.random.default_rng(kwargs.get("seed", 0) + 991)

        n = self.H0.size

        self.w0 = rng.normal(w0, sigma_w0, n)
        self.wa = rng.normal(wa, sigma_wa, n)

    def _E2(self, z):

        w0 = self.w0[:, None]
        wa = self.wa[:, None]

        de = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))

        return (
            self.Om[:, None] * (1 + z) ** 3
            + (1 - self.Om)[:, None] * de
        )

    def __call__(self, z, *, derivative=0):

        z = np.atleast_1d(np.asarray(z, dtype=float))

        if derivative == 0:
            return self.H0[:, None] * np.sqrt(self._E2(z))

        if derivative == 1:
            # Central difference on the closed form: the analytic route is
            # not the subject of these tests, and this is exact enough that
            # the derivative cross-check still bites.
            h = 1e-5
            up = np.sqrt(self._E2(z + h))
            dn = np.sqrt(self._E2(np.maximum(z - h, 0.0)))
            return self.H0[:, None] * (up - dn) / (z + h - np.maximum(z - h, 0.0))

        raise DerivativeUnavailableError("toy model provides orders 0 and 1")


def reconstruction(predictor, z, *, label="H", origin=None, method="toy"):
    """Wrap a toy predictor as a :class:`Reconstruction` on ``z``."""

    return Reconstruction.from_predictor(
        z,
        predictor,
        provenance=Provenance(
            method=method,
            data=("mock(30)",),
            seed=0,
            n_draws=predictor.H0.size,
        ),
        origin=new_origin() if origin is None else origin,
        label=label,
        unit="km/s/Mpc",
    )


# ============================================================
# Mock measurements
# ============================================================

class MockData:
    """
    A minimal dataset: redshifts, values, uncertainties, and a name.

    Duck-typed against what the reconstructors expect, so that the fitting
    machinery can be exercised before ``CosmoRecon.data`` exists.
    """

    def __init__(self, z, y, sigma, *, observable="H", unit="km/s/Mpc"):

        self.z = np.asarray(z, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.observable = observable
        self.unit = unit
        self.name = observable

    def __len__(self):
        return self.z.size


def chronometers(truth, *, n=32, z_max=2.0, sigma_frac=0.08, seed=11):
    """
    Cosmic-chronometer-like measurements of a known ``H(z)``.

    ``truth`` is a callable of ``z``. The redshift sampling and the ~8%
    fractional errors are roughly those of the real compilation, which is what
    makes a recovery test here mean something about a recovery test there.
    """

    rng = np.random.default_rng(seed)

    z = np.sort(rng.uniform(0.05, z_max, n))

    clean = np.asarray(truth(z), dtype=float)

    sigma = sigma_frac * clean

    return MockData(z, clean + rng.normal(0.0, sigma), sigma)


def lcdm_H(z, H0=70.0, Om=0.3):
    """The truth a recovery test is scored against."""

    z = np.asarray(z, dtype=float)

    return H0 * np.sqrt(Om * (1 + z) ** 3 + 1 - Om)


# ============================================================
# A curved FLRW universe, in BAO observables
# ============================================================

class CurvedFLRW:
    """
    ``D_M/r_d`` and ``D_H/r_d`` for a universe with spatial curvature.

    Built for the Clarkson-Bassett-Lu test, which needs both together. The
    transverse distance is integrated numerically and its derivative is taken
    numerically from that integral -- deliberately **not** from the FLRW
    relation ``D_M' = D_H sqrt(1 + Ok (H0 D_M/c)^2)``, since that relation is
    what the test is checking and using it here would make the check circular.

    ``distort`` multiplies the radial distance by ``1 + distort * z``, which
    breaks the FLRW relation without breaking either function's smoothness --
    an injected violation the test has to find.
    """

    C_KM_S = 299792.458

    def __init__(self, Om=0.30, Ok=0.0, H0=70.0, rd=147.0, n_draws=400,
                 sigma_Om=1e-9, sigma_H0=1e-9, sigma_Ok=1e-9,
                 distort=0.0, seed=0):

        rng = np.random.default_rng(seed)

        self.Om = rng.normal(Om, sigma_Om, n_draws)
        self.H0 = rng.normal(H0, sigma_H0, n_draws)

        # Drawn, not fixed. An exactly flat universe with no other freedom
        # gives Ok(z) = 0 with *zero* width, and a significance asked of a
        # posterior with no width divides rounding error by rounding error.
        # A real curvature posterior has a width, so the toy's does too.
        self.Ok = rng.normal(Ok, sigma_Ok, n_draws)

        self.rd = float(rd)
        self.distort = float(distort)

        # A fine grid the distances are built on, and differentiated on.
        self._z = np.linspace(0.0, 4.0, 40_000)

        E = np.sqrt(
            self.Om[:, None] * (1 + self._z) ** 3
            + self.Ok[:, None] * (1 + self._z) ** 2
            + (1 - self.Om - self.Ok)[:, None]
        )

        # Comoving distance in units of c/H0, then the curved transverse form.
        chi = np.concatenate(
            [np.zeros((n_draws, 1)),
             np.cumsum(0.5 * (1 / E[:, 1:] + 1 / E[:, :-1]) * np.diff(self._z), axis=1)],
            axis=1,
        )

        # sinh(sqrt(Ok) chi) / sqrt(Ok) analytically continues through zero
        # and covers both signs, so a per-draw curvature needs no branching.
        root = np.sqrt(self.Ok.astype(complex))[:, None]

        transverse = np.real(np.sinh(root * chi) / root)

        hubble = self.C_KM_S / self.H0[:, None]

        # In units of the sound horizon.
        self._T = hubble * transverse / self.rd
        self._R = hubble / E / self.rd * (1.0 + self.distort * self._z)

        self._dT = np.gradient(self._T, self._z, axis=1, edge_order=2)

    # ---------------------------------------------------------

    @property
    def curvature(self) -> float:
        """The mean spatial curvature the universe was built with."""

        return float(self.Ok.mean())

    @property
    def calibration(self) -> float:
        """``c / (H0 rd)``, the constant that turns the statistic into Ok."""

        return self.C_KM_S / (float(self.H0.mean()) * self.rd)

    def _interp(self, table, z):

        z = np.atleast_1d(np.asarray(z, dtype=float))

        return np.stack([np.interp(z, self._z, row) for row in table])

    def transverse(self):
        """A predictor for ``D_M/r_d``, with a numerical first derivative."""

        def predictor(z, *, derivative=0):

            if derivative == 0:
                return self._interp(self._T, z)

            if derivative == 1:
                return self._interp(self._dT, z)

            raise DerivativeUnavailableError("toy provides orders 0 and 1")

        return predictor

    def radial(self):
        """A predictor for ``D_H/r_d``."""

        def predictor(z, *, derivative=0):

            if derivative == 0:
                return self._interp(self._R, z)

            raise DerivativeUnavailableError("toy provides order 0")

        return predictor


class DistanceDuality:
    """
    The reduced distance modulus and ``D_M/r_d`` of a flat LCDM universe, with
    an optional cosmic opacity.

    ``d_L = (1 + z)^(1 + epsilon) D_M``, so ``epsilon = 0`` satisfies the
    Etherington relation exactly and anything else violates it in the
    power-law form the opacity slope measures. The supernova side carries a
    zero point ``offset`` in magnitudes and the BAO side is a ratio to ``rd``
    -- the two calibrations the statistic has to manage without.

    The reduced modulus ``mu - 5 log10 z`` is built from ``D_M / z``, whose
    limit at the origin is ``c / H0``, so there is no singularity for the toy
    to interpolate across -- the same property that makes the reduced modulus
    the right thing to reconstruct.
    """

    C_KM_S = 299792.458

    def __init__(self, Om=0.30, H0=70.0, rd=147.0, epsilon=0.0, offset=0.0,
                 n_draws=400, sigma_Om=1e-9, sigma_H0=1e-9, sigma_offset=0.0,
                 seed=0):

        rng = np.random.default_rng(seed)

        self.Om = rng.normal(Om, sigma_Om, n_draws)
        self.H0 = rng.normal(H0, sigma_H0, n_draws)

        # A supernova posterior's level is uncertain, and when sigma_offset is
        # set the toy's is too -- one zero point per realisation.
        self.offsets = offset + sigma_offset * rng.standard_normal(n_draws)

        self.rd = float(rd)
        self.offset = float(offset)
        self.epsilon = float(epsilon)

        self._z = np.linspace(0.0, 3.0, 6001)

        E = np.sqrt(
            self.Om[:, None] * (1 + self._z) ** 3 + (1 - self.Om)[:, None]
        )

        chi = np.concatenate(
            [np.zeros((n_draws, 1)),
             np.cumsum(0.5 * (1 / E[:, 1:] + 1 / E[:, :-1]) * np.diff(self._z), axis=1)],
            axis=1,
        )

        hubble = self.C_KM_S / self.H0[:, None]

        distance = hubble * chi

        self._T = distance / self.rd

        per_redshift = np.empty_like(distance)
        per_redshift[:, 1:] = distance[:, 1:] / self._z[1:]
        per_redshift[:, 0] = hubble[:, 0]

        self._m = (
            5.0 * np.log10((1 + self._z) ** (1 + self.epsilon) * per_redshift)
            + 25.0
            + self.offsets[:, None]
        )

    @property
    def calibration(self) -> float:
        """``rd 10^(offset / 5)``, the constant that turns the statistic into eta."""

        return self.rd * 10.0 ** (self.offset / 5.0)

    def _interp(self, table, z):

        z = np.atleast_1d(np.asarray(z, dtype=float))

        return np.stack([np.interp(z, self._z, row) for row in table])

    def _predictor(self, table):

        def predictor(z, *, derivative=0):

            if derivative == 0:
                return self._interp(table, z)

            raise DerivativeUnavailableError("toy provides order 0")

        return predictor

    def modulus(self):
        """A predictor for the reduced modulus ``mu - 5 log10 z``."""

        return self._predictor(self._m)

    def transverse(self):
        """A predictor for ``D_M / r_d``."""

        return self._predictor(self._T)


def duality_pair(universe, z, method="toy duality"):
    """
    The reduced modulus and ``D_M/r_d`` from one universe's draws, as one fit:
    same origin, so the statistic is exact draw by draw.
    """

    origin = new_origin()

    provenance = Provenance(
        method=method, data=("mock SN+BAO",), seed=0,
        n_draws=universe.H0.size,
    )

    def wrap(predictor, label):
        return Reconstruction.from_predictor(
            z, predictor, provenance=provenance, origin=origin, label=label
        )

    return (
        wrap(universe.modulus(), "mu_reduced"),
        wrap(universe.transverse(), "DM_over_rs"),
    )


def bao_pair(universe, z, method="toy FLRW"):
    """
    The two BAO observables as one aligned pair, as a joint fit would give
    them: same origin, so they combine without an independence claim.
    """

    origin = new_origin()

    provenance = Provenance(
        method=method, data=("mock BAO(12)",), seed=0,
        n_draws=universe.H0.size,
    )

    def wrap(predictor, label):
        return Reconstruction.from_predictor(
            z, predictor, provenance=provenance, origin=origin, label=label
        )

    return wrap(universe.transverse(), "DM_over_rs"), wrap(universe.radial(), "DH_over_rs")
