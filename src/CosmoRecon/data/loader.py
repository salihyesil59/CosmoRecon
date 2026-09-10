"""
The bundled observations.

Four releases, chosen to be small enough to ship, current enough to matter,
and between them to cover the two things a model-independent analysis needs:
an expansion history and a growth history.

    >>> from CosmoRecon.data import chronometers, desi_dr2_bao   # doctest: +SKIP
    >>> print(chronometers().describe())                         # doctest: +SKIP

Nothing here is re-derived. Each loader reads the numbers as the survey
published them and builds the covariance the way the survey's own likelihood
does -- the cosmic chronometer covariance from the released correlation
matrix, the growth covariance with the two internally correlated blocks the
reference likelihood overwrites, the BAO covariance as released. Where a
release's reduction assumed a fiducial cosmology, the loader says so in the
dataset's ``note`` rather than leaving the user to find out.

What is deliberately *not* bundled: Pantheon+ (a 33 MB covariance) and
DES-SN5YR (6 MB). A reconstruction library should not cost forty megabytes to
install for data most users will not touch. They are reachable through
:mod:`CosmoRecon.bridges` when CosmoFit is installed.
"""

from __future__ import annotations

import io
from importlib import resources

import numpy as np

from CosmoRecon.typing import Array

from CosmoRecon.core.errors import DataError

from CosmoRecon.data.dataset import (
    Dataset,
    MultiObservableDataset,
    check_combination,
)


__all__ = [
    "chronometers",
    "desi_dr2_bao",
    "union3",
    "growth",
    "available",
    "load",
]


# ============================================================
# File access
# ============================================================

def _read(*parts: str) -> str:
    """Text of a bundled file, working from a wheel as well as a checkout."""

    handle = resources.files("CosmoRecon.data")

    for part in parts:
        handle = handle / part

    try:
        return handle.read_text(encoding="utf-8")

    except (FileNotFoundError, OSError) as error:

        raise DataError(
            f"Bundled data file {'/'.join(parts)} is missing. If this is an "
            "editable install, the package data may not have been copied."
        ) from error


def _table(*parts: str) -> Array:
    """A whitespace-separated numeric table, comments stripped."""

    return np.loadtxt(io.StringIO(_read(*parts)), comments="#")


# ============================================================
# Cosmic chronometers
# ============================================================

def chronometers() -> Dataset:
    """
    32 differential-age measurements of ``H(z)``, ``0.07 <= z <= 1.965``.

    The only probe here that measures the expansion rate directly rather than
    a distance, and therefore the only one that constrains ``H(z)`` without a
    calibration -- no sound horizon, no absolute magnitude. That makes it the
    natural first dataset for a reconstruction, and the reason the examples
    use it.

    The covariance is **not** diagonal, and using it as though it were is a
    common shortcut with real consequences: the Moresco et al. systematic
    budget correlates the measurements through the stellar population
    modelling they share, and ignoring that overstates how much independent
    information the 32 points carry.
    """

    table = _table("cc", "favale2023", "CC_32_Favale2023_data.txt")

    z, values, sigma = table[:, 0], table[:, 1], table[:, 2]

    correlation = _table(
        "cc", "favale2023", "CC_32_Favale2023_Moresco2020_correlation.txt"
    )

    covariance = correlation * np.outer(sigma, sigma)

    return Dataset(
        z=z,
        y=values,
        cov=covariance,
        observable="H",
        unit="km/s/Mpc",
        name="CC(Favale2023)",
        reference=(
            "Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406, "
            "arXiv:2301.09591; systematic correlation matrix from Moresco "
            "et al. (2020), ApJ 898, 82, arXiv:2003.07362"
        ),
        note=(
            "Differential ages depend on a stellar population synthesis "
            "model; the released systematic covariance is where that "
            "dependence is accounted for, which is why it is used here in "
            "full rather than as error bars."
        ),
    )


# ============================================================
# BAO
# ============================================================

#: Quantity names as DESI writes them, and what they mean.
_BAO_UNITS = {
    "DV_over_rs": "",
    "DM_over_rs": "",
    "DH_over_rs": "",
}


def desi_dr2_bao() -> MultiObservableDataset:
    """
    DESI DR2 baryon acoustic oscillations: 13 measurements at 7 redshifts.

    The dataset behind the evidence for evolving dark energy, and the reason
    this library exists in its current form.

    Returned as a :class:`~CosmoRecon.data.dataset.MultiObservableDataset`
    rather than something directly fittable, because it is not one function of
    redshift: ``D_M/r_d`` and ``D_H/r_d`` are measured together and are
    correlated within each tracer. Use ``.select("DM_over_rs")`` for the part
    that can be reconstructed as a curve.

    Note what BAO does and does not give. Every entry is a **ratio to the
    sound horizon** ``r_d``, so a reconstruction of these is model-independent
    while a conversion to ``H(z)`` or ``D_M(z)`` is not -- that step needs
    ``r_d`` from somewhere else, which is a calibration and belongs to the
    analysis rather than to the data.
    """

    text = _read("bao", "desi_dr2", "desi_dr2_gaussian_bao_ALL_GCcomb_mean.txt")

    z_list: list[float] = []
    value_list: list[float] = []
    label_list: list[str] = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        first, second, third = stripped.split()

        z_list.append(float(first))
        value_list.append(float(second))
        label_list.append(third)

    covariance = _table("bao", "desi_dr2", "desi_dr2_gaussian_bao_ALL_GCcomb_cov.txt")

    # File order is covariance order. Sorting here would silently decouple
    # the two.
    return MultiObservableDataset(
        z=np.array(z_list),
        values=np.array(value_list),
        cov=covariance,
        quantity=np.array(label_list),
        units=dict(_BAO_UNITS),
        name="DESI-DR2-BAO",
        reference=(
            "DESI Collaboration (2025), DESI DR2 BAO measurements, "
            "arXiv:2503.14738"
        ),
        note=(
            "Every entry is a ratio to the sound horizon r_d. Converting to "
            "H(z) or a distance requires r_d from elsewhere, which is a "
            "calibration, not part of these data."
        ),
        excludes=("DESI-DR1-BAO", "SDSS-BAO"),
    )


# ============================================================
# Supernovae
# ============================================================

def union3() -> Dataset:
    """
    Union3: 22 binned Type Ia supernova distance moduli, ``0.05 <= z <= 2.26``.

    Binned rather than per-object, which is what makes it small enough to
    bundle -- the 22 points carry a dense 22x22 covariance built from the full
    systematic budget of 2087 supernovae.

    **The overall level is not measured.** A supernova distance modulus is
    ``mu = m_B - M_B``, and ``M_B`` is degenerate with ``H_0``: the shape of
    ``mu(z)`` is an observation, its zero point is a calibration. A
    reconstruction of these numbers therefore determines the shape and inherits
    whatever offset the release adopted, which matters the moment the result
    is used to say something about ``H_0``.
    """

    text = _read("sn", "union3", "union3_lcparam_full.txt")

    rows = [
        line.split()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    z = np.array([float(row[1]) for row in rows])       # zcmb

    mu = np.array([float(row[4]) for row in rows])      # mb, binned

    covariance = _covariance_from_flat(
        _read("sn", "union3", "union3_mag_covmat.txt"), z.size
    )

    return Dataset(
        z=z,
        y=mu,
        cov=covariance,
        observable="mu",
        unit="mag",
        name="Union3",
        reference=(
            "Rubin et al. (2023), Union3 / UNITY1.5 compilation, "
            "arXiv:2311.12098"
        ),
        note=(
            "Distance moduli carry an arbitrary zero point degenerate with "
            "M_B and H_0: the shape of mu(z) is measured, the level is not."
        ),
        excludes=("Pantheon+", "DES-SN5YR"),
    )


def _covariance_from_flat(text: str, n: int) -> Array:
    """
    Read a SALT2-style covariance: a count, then ``n * n`` entries in row-major
    order, one per line.
    """

    numbers = [
        float(line)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    declared = int(numbers[0])

    if declared != n:

        raise DataError(
            f"Covariance file declares {declared} entries but the light-curve "
            f"table has {n}."
        )

    body = np.array(numbers[1:])

    if body.size != n * n:

        raise DataError(
            f"Covariance file holds {body.size} numbers, expected {n * n}."
        )

    return body.reshape(n, n)


# ============================================================
# Growth
# ============================================================

#: Row ranges (half-open, zero-indexed) in the Gold-2018 table that are
#: internally correlated, and the released block for each. Taken from the
#: reference MontePython likelihood (snesseris/RSD-growth): WiggleZ's three
#: low-redshift points and eBOSS DR14 quasars' four tomographic bins. Every
#: other entry in the compilation comes from an independent survey.
_GROWTH_BLOCKS = (
    ((12, 15), "Cij_WiggleZ.txt"),
    ((18, 22), "Cij_SDSS.txt"),
)


def growth() -> Dataset:
    """
    The "Gold-2018" compilation: 22 measurements of ``f sigma_8(z)``.

    The growth side of the library's argument. Geometry alone cannot separate
    a modification of gravity from a dark-energy equation of state; the growth
    of structure can, which is what the growth-geometry consistency test in
    :mod:`CosmoRecon.consistency` will be built on.

    Two blocks of the compilation are internally correlated -- WiggleZ's three
    low-redshift points and eBOSS DR14 quasars' four tomographic bins -- and
    those blocks are laid over the diagonal here exactly as the reference
    likelihood does. Treating the whole compilation as 22 independent numbers
    is the usual shortcut and overstates its constraining power.

    **These numbers are not model-independent.** Each survey converted a raw
    redshift-space distortion measurement into ``f sigma_8`` assuming a
    fiducial cosmology, and the Alcock-Paczynski correction that undoes part
    of that assumption is itself model-dependent. A reconstruction of them is
    model-independent given the compilation, not from the raw data up.
    """

    table = _table("growth", "gold2018", "fsigma8_gold2018.txt")

    z, values, sigma = table[:, 0], table[:, 1], table[:, 2]

    covariance = np.diag(sigma**2)

    for (start, stop), filename in _GROWTH_BLOCKS:

        block = _table("growth", "gold2018", filename)

        if block.shape != (stop - start, stop - start):

            raise DataError(
                f"{filename} has shape {block.shape}, expected "
                f"{(stop - start, stop - start)}."
            )

        covariance[start:stop, start:stop] = block

    return Dataset(
        z=z,
        y=values,
        cov=covariance,
        observable="fsigma8",
        unit="",
        name="Growth(Gold2018)",
        reference=(
            "Sagredo, Nesseris & Sapone (2018), Phys. Rev. D 98, 083543, "
            "arXiv:1806.10822"
        ),
        note=(
            "Each entry was converted from a redshift-space distortion "
            "measurement under a fiducial cosmology, so the compilation is "
            "not model-independent from the raw data up."
        ),
    )


# ============================================================
# Registry
# ============================================================

_LOADERS = {
    "chronometers": chronometers,
    "cc": chronometers,
    "desi_dr2_bao": desi_dr2_bao,
    "desi": desi_dr2_bao,
    "union3": union3,
    "growth": growth,
    "fsigma8": growth,
}


def available() -> list[str]:
    """The bundled datasets, by canonical name."""

    return ["chronometers", "desi_dr2_bao", "union3", "growth"]


def load(*names: str):
    """
    Load one or several datasets by name, refusing combinations that overlap.

    >>> load("chronometers")                                   # doctest: +SKIP
    >>> load("chronometers", "union3")                         # doctest: +SKIP

    The check is the point of routing through here rather than calling the
    loaders directly: two compilations built from the same objects are not two
    measurements, and combining them narrows an interval without adding
    information.
    """

    if not names:

        raise DataError(f"Nothing asked for. Available: {available()}.")

    datasets = []

    for name in names:

        try:
            datasets.append(_LOADERS[str(name).lower()]())

        except KeyError:

            raise DataError(
                f"Unknown dataset {name!r}. Available: {available()}."
            ) from None

    check_combination(*datasets)

    return datasets[0] if len(datasets) == 1 else datasets
