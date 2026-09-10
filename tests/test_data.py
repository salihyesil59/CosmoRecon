"""
The bundled observations, checked against what the surveys published.

Loading data is the part of a library where a silent mistake is most
expensive and least visible: a column read in the wrong order, a correlation
matrix used as a covariance, a covariance transposed. None of those produce an
error, and all of them produce a reconstruction that looks entirely
reasonable.

So the test is not that the files parse. It is that fitting flat LCDM back to
each dataset reproduces the number the survey itself published, to the
precision the survey quoted. DESI DR2 published ``Omega_m = 0.2975 +/- 0.0086``
and ``r_d h = 101.54 +/- 0.73 Mpc`` from BAO alone; Union3 published
``Omega_m = 0.356 +/- 0.026``. If the loaders are right those numbers come
back out, and essentially nothing else does.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, linalg, optimize

from CosmoRecon.core.errors import DataError

from CosmoRecon.data import (
    Dataset,
    available,
    check_combination,
    chronometers,
    desi_dr2_bao,
    growth,
    load,
    union3,
)

from CosmoRecon.reconstructors import Cosmography, GaussianProcess


C_KM_S = 299792.458


# ============================================================
# A flat LCDM the fits below are scored against
# ============================================================

def E(z, Om):
    return np.sqrt(Om * (1 + z) ** 3 + 1 - Om)


def comoving(z, Om):
    """Comoving distance in units of c / H0."""

    fine = np.linspace(0.0, float(np.max(z)), 4000)

    integral = integrate.cumulative_trapezoid(1.0 / E(fine, Om), fine, initial=0.0)

    return np.interp(z, fine, integral)


def chi_square(residual, cov):

    factor = linalg.cho_factor(cov, lower=True)

    return float(residual @ linalg.cho_solve(factor, residual))


# ============================================================
# Shapes and integrity
# ============================================================

@pytest.mark.parametrize("loader, n, observable", [
    (chronometers, 32, "H"),
    (union3, 22, "mu"),
    (growth, 22, "fsigma8"),
])
def test_datasets_load_with_the_published_size(loader, n, observable):

    data = loader()

    assert len(data) == n
    assert data.observable == observable
    assert data.reference
    assert np.all(np.isfinite(data.y))


def test_desi_dr2_has_the_published_thirteen_measurements():

    bao = desi_dr2_bao()

    assert len(bao) == 13
    assert len(np.unique(bao.z)) == 7
    assert set(bao.quantities()) == {"DV_over_rs", "DM_over_rs", "DH_over_rs"}


@pytest.mark.parametrize("loader", [chronometers, union3, growth])
def test_covariances_are_usable_as_covariances(loader):
    """
    Symmetric and positive definite. Enforced at construction, so this is a
    check that the enforcement is reached rather than that the numbers happen
    to be fine.
    """

    cov = loader().cov

    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0.0


def test_a_bad_covariance_is_refused():

    with pytest.raises(DataError, match="positive definite"):
        Dataset(
            z=np.array([0.1, 0.2]),
            y=np.array([70.0, 75.0]),
            cov=np.array([[1.0, 2.0], [2.0, 1.0]]),   # indefinite
            observable="H",
            name="broken",
        )


# ============================================================
# Against what the surveys published
# ============================================================

def test_chronometers_refit_to_a_sensible_expansion_history():
    """
    Favale et al. (2023) fit H0 near 70 and Omega_m near 0.3 to these 32
    points with this covariance. The low chi-square per degree of freedom is
    expected and is itself informative: it is what the correlated systematic
    budget does, and a value near 1 here would mean the correlation matrix had
    been ignored.
    """

    data = chronometers()

    def objective(p):
        return chi_square(data.y - p[0] * E(data.z, p[1]), data.cov)

    result = optimize.minimize(objective, [70.0, 0.3], bounds=[(50, 90), (0.05, 0.9)])

    H0, Om = result.x

    assert 65.0 < H0 < 76.0
    assert 0.20 < Om < 0.40

    assert result.fun / (len(data) - 2) < 0.8


def test_desi_dr2_refits_to_the_published_omega_m_and_sound_horizon():
    """
    DESI DR2, BAO alone, flat LCDM: ``Omega_m = 0.2975 +/- 0.0086`` and
    ``r_d h = 101.54 +/- 0.73 Mpc`` (arXiv:2503.14738).

    This is the sharpest check in the suite. It exercises the redshifts, the
    three distinct quantities, their order in the file, and the full 13x13
    covariance simultaneously -- get any of them wrong and the recovered
    numbers move well outside the published uncertainties.
    """

    bao = desi_dr2_bao()

    def objective(p):

        Om, rd_over_hubble = p

        transverse = comoving(bao.z, Om) / rd_over_hubble
        radial = 1.0 / (E(bao.z, Om) * rd_over_hubble)
        volume = (bao.z * transverse**2 * radial) ** (1.0 / 3.0)

        model = np.where(
            bao.quantity == "DM_over_rs",
            transverse,
            np.where(bao.quantity == "DH_over_rs", radial, volume),
        )

        return chi_square(bao.values - model, bao.cov)

    result = optimize.minimize(
        objective, [0.3, 0.035], bounds=[(0.1, 0.6), (0.01, 0.08)]
    )

    Om, rd_over_hubble = result.x

    rd_h = rd_over_hubble * C_KM_S / 100.0

    assert Om == pytest.approx(0.2975, abs=3 * 0.0086)
    assert rd_h == pytest.approx(101.54, abs=3 * 0.73)

    assert result.fun / (len(bao) - 2) < 2.0


def test_union3_refits_to_the_published_omega_m():
    """
    Union3, flat LCDM: ``Omega_m = 0.356 +/- 0.026`` (arXiv:2311.12098).

    The absolute magnitude is free, as it must be -- these are distance
    moduli with an arbitrary zero point, which is exactly what the dataset's
    note says.
    """

    sn = union3()

    def objective(p):

        Om, offset = p

        mu = 5.0 * np.log10((1 + sn.z) * comoving(sn.z, Om) * C_KM_S / 70.0) + 25.0

        return chi_square(sn.y - mu - offset, sn.cov)

    result = optimize.minimize(
        objective, [0.3, 0.0], bounds=[(0.05, 0.9), (-2.0, 2.0)]
    )

    assert result.x[0] == pytest.approx(0.356, abs=3 * 0.026)

    assert result.fun / (len(sn) - 2) < 2.0


def test_the_chronometer_covariance_is_not_diagonal_and_it_matters():
    """
    The Moresco systematic budget correlates the 32 measurements through the
    stellar population modelling they share. Treating them as independent --
    the usual shortcut -- changes the answer, which is why the loader carries
    the full matrix.
    """

    data = chronometers()

    off_diagonal = data.cov - np.diag(np.diag(data.cov))

    assert np.abs(off_diagonal).max() > 0.0

    def best_fit(cov):

        return optimize.minimize(
            lambda p: chi_square(data.y - p[0] * E(data.z, p[1]), cov),
            [70.0, 0.3],
            bounds=[(50, 90), (0.05, 0.9)],
        ).x

    full = best_fit(data.cov)
    diagonal = best_fit(np.diag(np.diag(data.cov)))

    assert not np.allclose(full, diagonal, rtol=1e-3)


# ============================================================
# Several observables at once
# ============================================================

def test_a_multi_observable_dataset_cannot_be_fitted_as_one_curve():
    """
    Thirteen numbers that are alternately a transverse distance and a Hubble
    distance do not trace one function of redshift, and a reconstructor
    reaching for ``.y`` is about to pretend they do.
    """

    with pytest.raises(DataError, match="not one function of redshift"):
        _ = desi_dr2_bao().y


def test_selecting_an_observable_gives_the_right_block():

    bao = desi_dr2_bao()

    transverse = bao.select("DM_over_rs")

    index = np.flatnonzero(bao.quantity == "DM_over_rs")

    assert len(transverse) == index.size
    assert np.array_equal(transverse.z, bao.z[index])
    assert np.array_equal(transverse.cov, bao.cov[np.ix_(index, index)])

    # And the result says what was given up by selecting.
    assert "cross-covariance is not carried" in transverse.note


def test_selecting_something_absent_is_refused():

    with pytest.raises(DataError, match="has no"):
        desi_dr2_bao().select("DA_over_rs")


# ============================================================
# Combining
# ============================================================

def test_overlapping_datasets_are_refused():
    """
    Two supernova compilations built from the same objects are not two
    measurements. Combining them narrows the interval without adding
    information, which looks like a better result.
    """

    fake_pantheon = Dataset(
        z=np.array([0.1, 0.2, 0.3]),
        y=np.array([38.0, 40.0, 41.0]),
        cov=np.eye(3) * 0.01,
        observable="mu",
        name="Pantheon+",
    )

    with pytest.raises(DataError, match="overlapping observations"):
        check_combination(union3(), fake_pantheon)


def test_independent_datasets_combine_freely():

    check_combination(chronometers(), union3(), growth())


def test_load_by_name():

    assert isinstance(load("chronometers"), Dataset)

    both = load("chronometers", "union3")

    assert len(both) == 2

    assert set(available()) == {"chronometers", "desi_dr2_bao", "union3", "growth"}


def test_load_refuses_an_unknown_name():

    with pytest.raises(DataError, match="Unknown dataset"):
        load("planck")


# ============================================================
# They actually reconstruct
# ============================================================

def test_a_gaussian_process_fits_the_real_chronometers():

    data = chronometers()

    H = GaussianProcess(kernel="matern").fit(data, n_draws=800, seed=1)["H"]

    assert H.label == "H"
    assert H.provenance.data == ("CC(Favale2023)(32)",)

    at_one = H.at(1.0)

    # LCDM with H0 ~ 70, Omega_m ~ 0.3 gives H(1) ~ 123.
    assert abs(float(at_one.mean()[0]) - 123.0) < 4 * float(at_one.std()[0])


def test_cosmography_fits_the_real_desi_transverse_distances():

    transverse = desi_dr2_bao().select("DM_over_rs")

    fit = Cosmography().fit(transverse, n_draws=800, seed=1)

    curve = fit["DM_over_rs"]

    assert curve.label == "DM_over_rs"

    # D_M/r_d rises monotonically with redshift in any sane cosmology.
    assert np.all(np.diff(curve.mean()) > 0.0)


def test_a_z_series_is_refused_on_data_reaching_z_two():
    """
    The convergence guard, on real data rather than a mock: DESI's Lyman-alpha
    point sits at z = 2.33, well outside |z| < 1.
    """

    from CosmoRecon.core.errors import ConvergenceError

    transverse = desi_dr2_bao().select("DM_over_rs")

    with pytest.raises(ConvergenceError, match="outside the range"):
        Cosmography("z").fit(transverse, n_draws=100, seed=1)
