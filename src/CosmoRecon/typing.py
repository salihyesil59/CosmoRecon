"""
The vocabulary this library's public surface speaks in.

CosmoRecon deals in *posteriors over functions*, and that shows up in the
annotations: almost nothing here returns a single number per redshift. The
distinction the three array aliases below draw is the one that matters most
when reading the source.

``Array``
    One number per redshift -- what a mean, a standard deviation or a
    single realisation looks like. Shape ``(n_z,)``.

``DrawArray``
    The representation. One number per redshift *per draw*, shape
    ``(n_draws, n_z)``, with the draw axis first so that ``arr[k]`` is
    realisation ``k`` and ``arr[:, i]`` is the marginal at ``z[i]``.

``Redshift``
    Anything numpy can make an array of floats from -- a scalar, a list,
    an array. Loose on purpose, at the boundary only.

The draw axis being *first* is not arbitrary. Every operation in
:mod:`CosmoRecon.core.reconstruction` is elementwise along it, and numpy
broadcasts a ``(n_z,)`` quantity against a ``(n_draws, n_z)`` one without a
reshape exactly when the draw axis leads. Reversing it would put an explicit
``[:, None]`` in every arithmetic method in the library.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "ArrayLike",
    "Array",
    "DrawArray",
    "PathLike",
    "Redshift",
    "Scalar",
]


#: A redshift, or a grid of them: anything numpy can make an array of floats
#: from. Used at the API boundary, where a user may reasonably pass a float.
Redshift = ArrayLike

#: One number per redshift. Shape ``(n_z,)``.
Array = NDArray[np.float64]

#: One number per redshift per draw. Shape ``(n_draws, n_z)``, draw axis
#: first. This is what a :class:`~CosmoRecon.core.reconstruction.Reconstruction`
#: actually holds.
DrawArray = NDArray[np.float64]

#: A single number that may itself be uncertain elsewhere in the library --
#: annotated separately from ``float`` so the places that deliberately collapse
#: a posterior to a point estimate are greppable.
Scalar = float | np.float64

#: Anywhere a file is read or written: a string, or anything implementing the
#: ``os.PathLike`` protocol.
PathLike = str | os.PathLike
