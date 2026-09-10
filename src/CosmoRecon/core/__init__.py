"""
The layer everything else is written against.

Nothing in :mod:`CosmoRecon.core` knows what a Gaussian process is, what a
null test is, or what dark energy is. It defines the object the rest of the
library passes around -- a posterior over a function of redshift -- together
with the bookkeeping that keeps combining two of them honest.
"""

from __future__ import annotations

from CosmoRecon.core.constants import (
    C_LIGHT_KM_S,
    C_LIGHT_M_S,
    MPC_IN_KM,
)

from CosmoRecon.core.errors import (
    AlignmentError,
    ConvergenceError,
    CosmoReconError,
    DataError,
    DerivativeUnavailableError,
    EvidenceUnavailableError,
    GridMismatchError,
    InsufficientDrawsError,
    NotResamplableError,
    OptionalDependencyError,
)

from CosmoRecon.core.grid import (
    ExtrapolationWarning,
    check_within_support,
    common_support,
    linear_grid,
    log_grid,
    suggested_n,
    support_of,
)

from CosmoRecon.core.provenance import (
    CONSTANT_ORIGIN,
    Origin,
    Provenance,
    new_origin,
)

from CosmoRecon.core.reconstruction import (
    Predictor,
    Reconstruction,
)


__all__ = [
    # the spine
    "Reconstruction",
    "Predictor",
    # provenance
    "Provenance",
    "Origin",
    "new_origin",
    "CONSTANT_ORIGIN",
    # grids
    "linear_grid",
    "log_grid",
    "support_of",
    "common_support",
    "check_within_support",
    "suggested_n",
    "ExtrapolationWarning",
    # constants
    "C_LIGHT_KM_S",
    "C_LIGHT_M_S",
    "MPC_IN_KM",
    # errors
    "CosmoReconError",
    "AlignmentError",
    "ConvergenceError",
    "GridMismatchError",
    "DerivativeUnavailableError",
    "NotResamplableError",
    "EvidenceUnavailableError",
    "InsufficientDrawsError",
    "DataError",
    "OptionalDependencyError",
]
