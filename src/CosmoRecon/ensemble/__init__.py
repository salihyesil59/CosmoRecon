"""
Running one question through several methods, and reporting the disagreement.

This is the layer the rest of the library was shaped to make possible. Because
:mod:`CosmoRecon.consistency` and :mod:`CosmoRecon.inverse` speak only
:class:`~CosmoRecon.core.reconstruction.Reconstruction`, the reconstructor
underneath can be swapped without touching them -- so the same null test, the
same inversion, the same figure can be produced four ways and differenced.

:mod:`.budget` holds the law-of-total-variance split; :mod:`.method` holds
``MethodEnsemble``, which fits every member, pools their draws into a
method-marginalised posterior, and reports a null test's significance before
and after that marginalisation -- the *"3.1 sigma becomes X sigma"*
statement.

Equal weighting is the default. Evidence weighting is available and is not the
default on purpose: a Bayesian evidence compares models of the *same* data
under the *same* likelihood, and two reconstruction methods with different
function-space priors are only loosely that. Some members (neural, symbolic)
have no evidence at all, and :class:`~CosmoRecon.reconstructors.base.Reconstructor`
raises rather than returning ``nan`` so that an evidence-weighted ensemble
cannot quietly shrink to the subset that happens to define one.
"""

from __future__ import annotations

from CosmoRecon.ensemble.budget import VarianceBudget, total_variance

from CosmoRecon.ensemble.method import (
    EnsembleFit,
    MethodEnsemble,
    SignificanceComparison,
)


__all__ = [
    "MethodEnsemble",
    "EnsembleFit",
    "SignificanceComparison",
    "VarianceBudget",
    "total_variance",
]
