"""
The non-parametric methods.

Every member takes data and returns a
:class:`~CosmoRecon.reconstructors.base.ReconstructionSet`, so that
:mod:`CosmoRecon.consistency`, :mod:`CosmoRecon.inverse` and
:mod:`CosmoRecon.ensemble` can be written once and work with all of them.

Implemented so far: the contract, and the Gaussian process.

``gp.py`` (done)
    The incumbent -- GaPP (2012) -- optimises its kernel hyperparameters and
    fixes the Matern smoothness by hand. Both choices manufacture precision,
    and both are the subject of the kernel-dependence literature this library
    answers. Here the hyperparameters are marginalised, ``nu`` is inferred,
    sample paths come from a spectral quadrature accurate to a checked
    tolerance rather than from random features, and a derivative the fitted
    smoothness does not support is refused rather than returned.

The remaining methods land in this order, which is the order in which they are
load-bearing for the library's argument:

``cosmography.py``
    Taylor, Pade, Chebyshev, ``y``-redshift and log-polynomial expansions,
    each carrying the radius of convergence its order implies, because a
    Taylor series in ``z`` quoted at ``z > 1`` is not a reconstruction.

``nodal.py``
    Flexknot and nodal splines with the node count sampled rather than fixed.
    Needs the ``nested`` extra.

``pca.py``
    Principal components and binned ``w(z)`` with a correlation prior.

``ann.py``
    Neural reconstruction, REFANN-style. Needs the ``ann`` extra. No evidence.

``symbolic.py``
    Genetic-algorithm symbolic regression, GAME-style. No evidence.
"""

from __future__ import annotations

from CosmoRecon.reconstructors.base import (
    DEFAULT_N_DRAWS,
    ReconstructionSet,
    Reconstructor,
)

from CosmoRecon.reconstructors.kernels import (
    Cauchy,
    Kernel,
    Matern,
    RationalQuadratic,
    SquaredExponential,
    get_kernel,
)

from CosmoRecon.reconstructors.gp import GaussianProcess


__all__ = [
    # the contract
    "Reconstructor",
    "ReconstructionSet",
    "DEFAULT_N_DRAWS",
    # methods
    "GaussianProcess",
    # kernels
    "Kernel",
    "SquaredExponential",
    "Matern",
    "RationalQuadratic",
    "Cauchy",
    "get_kernel",
]
