"""
Optional adapters to other libraries.

Nothing here is imported by the rest of CosmoRecon, and nothing here is
required for anything to work. Each module guards its import and raises
:class:`~CosmoRecon.core.errors.OptionalDependencyError` naming the extra to
install.

``cosmofit.py``
    Two directions across the boundary between fitting a model and
    reconstructing a function.

    *In:* adapt a CosmoFit dataset -- its DESI DR2 BAO, Pantheon+, cosmic
    chronometer and growth loaders, with their covariances -- into the
    containers :mod:`CosmoRecon.data` expects, so neither library maintains a
    second copy of the same data release.

    *Out:* turn any CosmoFit ``Cosmology`` into a degenerate
    :class:`~CosmoRecon.core.reconstruction.Reconstruction` -- one draw, no
    uncertainty -- so that a fitted model can be overlaid on a reconstruction
    in the same units, on the same grid, through the same plotting code. The
    natural figure of this library is a non-parametric band with a best-fit
    LCDM curve through it, and that figure should not require the user to
    write the glue.

    Needs the ``cosmofit`` extra.
"""

from __future__ import annotations


__all__: list[str] = []
