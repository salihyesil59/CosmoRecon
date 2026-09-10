"""
CosmoRecon -- model-independent reconstruction and null tests of the
cosmological framework.

Two questions, either side of a model fit:

*What do the data say about* ``H(z)``, ``d_L(z)``, ``f sigma_8(z)`` *with no
model assumed at all?* -- and *how much of that answer is an artefact of the
method used to extract it?*

The second question is the one no existing tool answers. Kernel choice in a
Gaussian-process reconstruction is known to shift results by as much as
changing the cosmological model does; the response in the literature has been
papers about kernel dependence rather than results marginalised over it,
because nothing produces the spread. :mod:`CosmoRecon.ensemble` produces it.

Everything is built on one object. A
:class:`~CosmoRecon.core.reconstruction.Reconstruction` is a posterior over a
function, carried as draws sharing a realisation index, so that

>>> Ok = (H**2 * D.d(1)**2 - C_LIGHT_KM_S**2) / (H0**2 * D**2)   # doctest: +SKIP

propagates the correlation between ``H`` and ``D``, the non-Gaussianity of the
ratio and the covariance between redshifts exactly -- because none of them was
ever summarised away. See ``ARCHITECTURE.md`` for why that is the whole design.

Related, and deliberately out of scope: parametric model fitting and Bayesian
evidence (CosmoFit), Boltzmann solving (CAMB, CLASS), posterior-level tension
metrics (tensiometer, unimpeded), Fisher forecasting (cosmicfishpie).
"""

from __future__ import annotations

from CosmoRecon.core import (
    AlignmentError,
    ConvergenceError,
    CosmoReconError,
    C_LIGHT_KM_S,
    DerivativeUnavailableError,
    ExtrapolationWarning,
    GridMismatchError,
    NotResamplableError,
    Provenance,
    Reconstruction,
    common_support,
    linear_grid,
    log_grid,
    support_of,
)

from CosmoRecon.reconstructors import (
    Cauchy,
    Cosmography,
    GaussianProcess,
    Matern,
    RationalQuadratic,
    ReconstructionSet,
    Reconstructor,
    SquaredExponential,
)

from CosmoRecon.consistency import (
    Curvature,
    NullTest,
    Om,
    Om3,
    TestResult,
    significance,
)

from CosmoRecon.ensemble import MethodEnsemble, VarianceBudget


__version__ = "0.1.0.dev0"


__all__ = [
    "__version__",
    # the spine
    "Reconstruction",
    "Provenance",
    # reconstruction
    "GaussianProcess",
    "Cosmography",
    "SquaredExponential",
    "Matern",
    "RationalQuadratic",
    "Cauchy",
    # contracts
    "Reconstructor",
    "ReconstructionSet",
    "NullTest",
    "TestResult",
    "significance",
    "Om",
    "Om3",
    "Curvature",
    "MethodEnsemble",
    "VarianceBudget",
    # grids
    "linear_grid",
    "log_grid",
    "support_of",
    "common_support",
    # constants
    "C_LIGHT_KM_S",
    # the errors a user will actually meet
    "CosmoReconError",
    "AlignmentError",
    "ConvergenceError",
    "GridMismatchError",
    "DerivativeUnavailableError",
    "NotResamplableError",
    "ExtrapolationWarning",
]
