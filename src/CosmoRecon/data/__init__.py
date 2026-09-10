"""
Observations, and the covariances that come with them.

A reconstruction is only as honest as the covariance it was given, so this
layer refuses a few things that are common elsewhere: a dataset with a
covariance of the wrong shape, two datasets covering the same galaxies
combined as if independent, and a compilation quoted without the correlations
its own paper published.

Planned contents:

``dataset.py``
    The containers: ``CCDataset`` (cosmic chronometers), ``BAODataset``
    (``D_M/r_d``, ``D_H/r_d``, ``D_V/r_d`` with their cross-covariance),
    ``SNDataset`` (distance moduli with the full systematic covariance),
    ``GrowthDataset`` (``f sigma_8``), ``TimeDelayDataset``.

``loader.py``
    Bundled data: cosmic chronometers, DESI DR2 BAO, Pantheon+ / Union3 /
    DES-SN5YR, and a growth compilation. Each with the reference it came
    from, and each declaring which other datasets it may not be combined
    with.

``covariance.py``
    Dense, diagonal and block covariances behind one interface, with the
    Cholesky cached -- reconstructors evaluate a likelihood thousands of
    times per fit.

Where CosmoFit is installed, :mod:`CosmoRecon.bridges` adapts its dataset
objects directly, so the two libraries do not maintain two copies of the same
DESI release.
"""

from __future__ import annotations


__all__: list[str] = []
