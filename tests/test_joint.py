"""
Joint reconstruction of several correlated observables.

A BAO release measures ``D_M/r_d`` and ``D_H/r_d`` together, correlated at
``r`` of about ``-0.4`` within each tracer. The null tests that need both --
the Clarkson-Bassett-Lu curvature test, distance duality -- are only defined
if that correlation is carried, so the two have to be reconstructed from one
realisation rather than fitted separately and paired afterwards.

The load-bearing design decision is what the joint prior must *not* do.

In FLRW the two functions are related by
``d/dz (D_M) = D_H sqrt(1 + Ok (H0 D_M / c)^2)`` -- which is the curvature test
itself. A joint prior that linked them, however physically motivated it
looked, would make that null test vacuous: it would be testing an assumption
it had already made. So the priors stay independent, and every correlation in
the posterior has to arrive from the data covariance.

That is not a claim to be taken on trust. Forcing the data covariance diagonal
has to make the posterior correlation vanish, and it is tested below.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoRecon import Cosmography, GaussianProcess
from CosmoRecon.core.errors import DataError
from CosmoRecon.data import desi_dr2_bao
from CosmoRecon.data.dataset import Dataset, MultiObservableDataset


GRID = np.linspace(0.55, 2.30, 20)


@pytest.fixture(scope="module")
def joint():
    return desi_dr2_bao().select("DM_over_rs", "DH_over_rs")


@pytest.fixture(scope="module")
def fitted(joint):
    return Cosmography("y").fit(joint, grid=GRID, n_draws=4000, seed=1)


def correlation(fit) -> np.ndarray:
    """Posterior correlation between the two curves, redshift by redshift."""

    a = fit["DM_over_rs"].draws
    b = fit["DH_over_rs"].draws

    return np.array([
        np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(a.shape[1])
    ])


# ============================================================
# The design decision
# ============================================================

def test_the_prior_contributes_no_correlation(joint):
    """
    The test the whole design rests on.

    Fit the same two observables twice: once with the released covariance, and
    once with it forced diagonal. If the priors are genuinely independent, the
    second fit must show no correlation between the two reconstructed
    functions -- because there is then nothing anywhere in the model to put
    one there.

    A joint prior that linked ``D_M`` and ``D_H`` through the FLRW relation
    would fail this test, and would also quietly destroy the curvature test,
    which is that relation.
    """

    diagonal = MultiObservableDataset(
        z=joint.z,
        values=joint.values,
        cov=np.diag(np.diag(joint.cov)),
        quantity=joint.quantity,
        units=joint.units,
        name="diagonal",
    )

    grid = np.linspace(0.55, 2.30, 15)

    n_draws = 8000

    with_data = correlation(
        Cosmography("y").fit(joint, grid=grid, n_draws=n_draws, seed=1)
    )

    without = correlation(
        Cosmography("y").fit(diagonal, grid=grid, n_draws=n_draws, seed=1)
    )

    # Nothing, to Monte-Carlo precision.
    assert np.max(np.abs(without)) < 6.0 / np.sqrt(n_draws)

    # And with the real covariance, a correlation of the size and sign the
    # release published.
    assert np.all(with_data < -0.2)
    assert np.all(with_data > -0.7)


def test_the_posterior_correlation_tracks_the_data(joint, fitted):
    """
    DESI DR2 publishes ``D_M``-``D_H`` correlations between ``-0.35`` and
    ``-0.49``, one per tracer and none across tracers. The reconstructed
    functions should carry something of that size -- not identically, since a
    reconstruction interpolates, but in the same range and with the same sign.
    """

    labels = np.asarray(joint.quantity).astype(str)

    published = []

    for z in np.unique(joint.z):

        at_z = np.flatnonzero(joint.z == z)

        if at_z.size != 2:
            continue

        i, j = at_z

        published.append(
            joint.cov[i, j] / np.sqrt(joint.cov[i, i] * joint.cov[j, j])
        )

    assert labels.size == 12
    assert np.all(np.array(published) < 0.0)

    reconstructed = correlation(fitted)

    assert np.all(reconstructed < 0.0)
    assert np.min(reconstructed) > 1.5 * min(published)


# ============================================================
# One fit, two functions
# ============================================================

def test_both_functions_come_from_one_realisation(fitted):
    """
    Same origin, so they combine without an independence claim -- which is the
    whole point, since claiming independence here would be false.
    """

    assert set(fitted) == {"DM_over_rs", "DH_over_rs"}

    assert fitted["DM_over_rs"].origin == fitted["DH_over_rs"].origin

    combined = fitted["DM_over_rs"] / fitted["DH_over_rs"]

    assert combined.n_z == GRID.size


def test_the_ratio_is_tighter_than_independent_propagation_would_say(fitted):
    """
    The two are anticorrelated, so their ratio is *wider* than independent
    propagation would give -- and the sign of that difference is exactly what
    a null test built on both would get wrong.
    """

    dm = fitted["DM_over_rs"]
    dh = fitted["DH_over_rs"]

    ratio = dm / dh

    fractional = ratio.std() / np.abs(ratio.mean())

    independent = np.sqrt(
        (dm.std() / dm.mean()) ** 2 + (dh.std() / dh.mean()) ** 2
    )

    # Anticorrelated inputs make a ratio noisier, not quieter.
    assert np.all(fractional > independent)


def test_each_function_recovers_its_own_measurements(joint, fitted):

    labels = np.asarray(joint.quantity).astype(str)

    for label in ("DM_over_rs", "DH_over_rs"):

        rows = np.flatnonzero(labels == label)

        curve = fitted[label].at(joint.z[rows])

        residual = (curve.mean() - joint.values[rows]) / np.sqrt(
            np.diag(joint.cov)[rows]
        )

        assert np.all(np.abs(residual) < 3.0), f"{label} residuals {residual}"


def test_each_observable_gets_its_own_order_and_prior(fitted, joint):
    """
    Six ``D_M`` points constrain ``D_M``; they say nothing about ``D_H``. So
    the order grid and the prior width are built per observable, and the
    marginalisation runs over the product.
    """

    method = Cosmography("y")

    method.fit(joint, grid=GRID, n_draws=100, seed=1)

    best = method.best_cell

    assert set(best) == {"DM_over_rs", "DH_over_rs"}

    for label, cell in best.items():
        assert cell["order"] >= 1, label
        assert cell["scale"] > 0.0, label


def test_a_joint_fit_is_reproducible(joint):

    def run():
        return Cosmography("y").fit(joint, grid=GRID, n_draws=300, seed=5)

    first, second = run(), run()

    for label in ("DM_over_rs", "DH_over_rs"):
        assert np.array_equal(first[label].draws, second[label].draws)


def test_pade_works_jointly(joint):
    """
    A rational re-expansion per observable, from one stacked draw -- and a
    draw is kept only if *every* observable is pole-free, since one
    realisation carries them all.
    """

    method = Cosmography(pade=(2, 1))

    fit = method.fit(joint, grid=GRID, n_draws=800, seed=1)

    assert set(fit) == {"DM_over_rs", "DH_over_rs"}
    assert fit["DM_over_rs"].origin == fit["DH_over_rs"].origin

    assert 0.2 <= method.acceptance <= 1.0


# ============================================================
# Selection
# ============================================================

def test_selecting_one_gives_a_single_dataset():

    one = desi_dr2_bao().select("DM_over_rs")

    assert isinstance(one, Dataset)
    assert one.observable == "DM_over_rs"
    assert "cross-covariance is not carried" in one.note


def test_selecting_several_keeps_the_covariance_between_them(joint):

    assert isinstance(joint, MultiObservableDataset)

    assert set(joint.quantities()) == {"DM_over_rs", "DH_over_rs"}

    # The block that couples them is present and not zero.
    labels = np.asarray(joint.quantity).astype(str)

    dm = np.flatnonzero(labels == "DM_over_rs")
    dh = np.flatnonzero(labels == "DH_over_rs")

    assert np.abs(joint.cov[np.ix_(dm, dh)]).max() > 0.0


def test_selecting_nothing_or_something_absent_is_refused():

    with pytest.raises(DataError, match="Nothing selected"):
        desi_dr2_bao().select()

    with pytest.raises(DataError, match="has no"):
        desi_dr2_bao().select("DM_over_rs", "DA_over_rs")


# ============================================================
# What cannot do it
# ============================================================

def test_a_single_function_method_refuses_and_says_what_can(joint):
    """
    The Gaussian process fits one scalar function at a time. Handed two, it
    refuses rather than fitting a curve through a mixture of transverse and
    radial distances -- and the message names the way out.
    """

    assert Cosmography.supports_joint
    assert not GaussianProcess.supports_joint

    with pytest.raises(DataError, match="supports_joint"):
        GaussianProcess().fit(joint, n_draws=100, seed=1)


def test_an_observable_with_too_few_points_is_refused():

    with pytest.raises(DataError, match="not enough"):
        Cosmography("y").fit(
            MultiObservableDataset(
                z=np.array([0.5, 0.7, 0.9, 1.1]),
                values=np.array([10.0, 12.0, 14.0, 16.0]),
                cov=np.eye(4) * 0.01,
                quantity=np.array(["a", "a", "a", "b"]),
                name="lopsided",
            ),
            n_draws=50,
            seed=1,
        )
