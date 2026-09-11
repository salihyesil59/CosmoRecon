"""
Observations, and the covariances that come with them.

A reconstruction is only as honest as the covariance it was given, so this
layer refuses a few things that are common elsewhere: a dataset with a
covariance of the wrong shape, two datasets covering the same galaxies
combined as if independent, and a compilation quoted without the correlations
its own paper published.

Bundled, in about 55 kB:

``chronometers()``
    32 differential-age measurements of ``H(z)``. The only probe that gives
    the expansion rate directly, with no calibration in the way.

``desi_dr2_bao()``
    DESI DR2: 13 BAO measurements at 7 redshifts, the dataset behind the
    evidence for evolving dark energy.

``union3()``
    22 binned supernova distance moduli with a dense systematic covariance.

``growth()``
    The Gold-2018 ``f sigma_8`` compilation -- the growth side of the
    argument, which is what separates modified gravity from dark energy.

Pantheon+ (33 MB of covariance) and DES-SN5YR (6 MB) are deliberately not
bundled; :mod:`CosmoRecon.bridges` reaches them through CosmoFit where it is
installed. There is no ``covariance.py``: a covariance here is a plain array,
factorised once per fit rather than once per likelihood call, and a class
around it would be indirection with nothing behind it.
"""

from __future__ import annotations

from CosmoRecon.data.dataset import (
    Dataset,
    MultiObservableDataset,
    check_combination,
    reduced_modulus,
)

from CosmoRecon.data.loader import (
    available,
    chronometers,
    desi_dr2_bao,
    growth,
    load,
    union3,
)


__all__ = [
    # containers
    "Dataset",
    "MultiObservableDataset",
    "check_combination",
    # transformations
    "reduced_modulus",
    # the bundled releases
    "chronometers",
    "desi_dr2_bao",
    "union3",
    "growth",
    "available",
    "load",
]
