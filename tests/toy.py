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
