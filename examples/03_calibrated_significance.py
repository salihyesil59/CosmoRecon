"""
The significances of example 02, calibrated.

    python examples/03_calibrated_significance.py       # about twenty minutes

Example 02 prints nominal significances: a chi-square of each statistic's
posterior mean against its posterior covariance. On mock surveys in which the
null holds exactly, that number is wrong in both directions. A Gaussian process
almost never rejects, because the directions its prior dominates are counted as
degrees of freedom. A series with its order left free rejects most of the time,
because under the null it is biased by more than its posterior width, and the
chi-square reads the bias as a detection.

This example reruns the same analyses, on the same data, with the same seeds,
adds the two litmus tests for a cosmological constant, and calibrates each with
:func:`CosmoRecon.validation.calibrate`:

1. A null model is fitted to the same data -- flat Lambda-CDM for the Om
   diagnostics, Lambda-CDM with curvature for the curvature test, Lambda-CDM
   distances for distance duality.
2. Mock datasets are drawn from its parameter posterior with the released
   covariances, and every member of the ensemble is refitted to each.
3. Each realisation is ranked by the distance of its statistic from the null
   mocks' own mean, in their own covariance, so that the method's bias at the
   null is subtracted rather than reported.

The time goes into refitting the Gaussian processes. The p-values cannot go
below ``1 / (n_mocks + 1)``, and a result at that floor is printed as a bound.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from CosmoRecon import (                                          # noqa: E402
    C_LIGHT_KM_S,
    Cosmography,
    Curvature,
    CurvedLitmus,
    Duality,
    ExtrapolationWarning,
    GaussianProcess,
    LambdaCDM,
    Litmus,
    MethodEnsemble,
    Om,
    Om3,
    calibrate,
)
from CosmoRecon.data import (                                     # noqa: E402
    chronometers,
    desi_dr2_bao,
    reduced_modulus,
    union3,
)


START = time.time()


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def show(result) -> None:

    for line in result.summary().splitlines():
        print("   " + line.strip())

    print(f"   ({time.time() - START:.0f} s so far)")
    print()


# ============================================================
# 1. Om and Om3 on the chronometers
# ============================================================

rule("1. Om and Om3 on 32 cosmic chronometers, calibrated")

cc_fit = MethodEnsemble([
    GaussianProcess(kernel="matern"),
    GaussianProcess(kernel="squared_exponential"),
    Cosmography("y"),
    Cosmography("log"),
    Cosmography(pade=(2, 1)),
]).fit(chronometers(), grid=np.linspace(0.10, 1.90, 25), n_draws=2000, seed=4)

with warnings.catch_warnings():

    # Om is anchored at z = 0, below the lowest chronometer; example 01 is about
    # that extrapolation, and it is the same in the data and in every mock.
    warnings.simplefilter("ignore", ExtrapolationWarning)

    for test in (Om(), Om3(z1=0.15, z2=0.35)):

        show(calibrate(
            lambda s, t=test: t.statistic(s["H"]),
            cc_fit,
            test.null_value,
            null=LambdaCDM(),
            marginalise_constant=test.null_is_free_constant,
            n_mocks=200,
            seed=11,
            name=test.name,
        ))


# ============================================================
# 2. The curvature test on DESI DR2
# ============================================================

rule("2. Ok(z) from DESI DR2, calibrated")

ok_fit = MethodEnsemble([
    Cosmography("y"),
    Cosmography("log"),
    Cosmography("y", order=3),
    Cosmography("y", family="monomial"),
]).fit(
    desi_dr2_bao().select("DM_over_rs", "DH_over_rs"),
    grid=np.linspace(0.60, 2.25, 20),
    n_draws=4000,
    seed=4,
)

hubble_distance = C_LIGHT_KM_S / (100.0 * 101.54)

show(calibrate(
    lambda s: Curvature(hubble_distance=hubble_distance).statistic(
        s["DM_over_rs"], s["DH_over_rs"]
    ),
    ok_fit,
    0.0,
    null=LambdaCDM(curved=True),
    marginalise_constant=True,
    n_mocks=300,
    seed=12,
    name="Ok(z), tested for constancy",
))


# ============================================================
# 3. Distance duality from Union3 and DESI DR2
# ============================================================

rule("3. Distance duality from Union3 and DESI DR2, calibrated")

grid = np.linspace(0.51, 2.26, 20)

duality_ensemble = MethodEnsemble([
    GaussianProcess(kernel="matern"),
    Cosmography("y"),
    Cosmography("y", order=3),
    Cosmography("log"),
])

both = duality_ensemble.fit(
    reduced_modulus(union3()), grid=grid, n_draws=4000, seed=7
).with_independent(
    duality_ensemble.fit(desi_dr2_bao().select("DM_over_rs"), grid=grid, n_draws=4000, seed=7)
)

show(calibrate(
    lambda s: Duality().opacity(s["mu_reduced"], s["DM_over_rs"]),
    both,
    0.0,
    null=LambdaCDM(),
    n_mocks=300,
    seed=13,
    name="epsilon = 0",
))

show(calibrate(
    lambda s: Duality().statistic(s["mu_reduced"], s["DM_over_rs"]),
    both,
    1.0,
    null=LambdaCDM(),
    marginalise_constant=True,
    n_mocks=300,
    seed=14,
    name="eta(z), tested for constancy",
))


# ============================================================
# 4. The flat litmus test on the supernovae
# ============================================================

rule("4. Zunckel-Clarkson litmus test on Union3, calibrated")

# Q(z) is constant in flat Lambda-CDM whatever Omega_m and whatever the
# supernova zero point, and it needs only distances -- at the price of a second
# derivative. The grid stops at the last densely sampled bin, z = 1.39: beyond
# it Union3 has one point, at 2.26, and nothing constrains a second derivative.
#
# The squared-exponential Gaussian process is not a member. Its distance stops
# increasing at z = 1.39 in a tenth of a per cent of its draws, the test divides
# by D'^3, and the library refuses rather than letting those draws decide.

sn_litmus = MethodEnsemble([
    Cosmography("y"),
    Cosmography("y", order=3),
    Cosmography("log"),
]).fit(
    reduced_modulus(union3()), grid=np.linspace(0.10, 1.39, 15), n_draws=2000, seed=21
)

show(calibrate(
    lambda s: Litmus().statistic(s["mu_reduced"]),
    sn_litmus,
    1.0,
    null=LambdaCDM(),
    marginalise_constant=True,
    n_mocks=300,
    seed=22,
    name="litmus Q(z), Union3, z <= 1.39",
))


# ============================================================
# 5. The curved litmus test on DESI DR2
# ============================================================

rule("5. Curved litmus test on DESI DR2, calibrated")

# The flat test cannot tell curvature from dynamical dark energy. This one
# removes the curvature through the Clarkson-Bassett-Lu relation, using both
# BAO distances, and is constant in Lambda-CDM of any curvature -- with first
# derivatives only. With c / (H0 r_d) supplied, the constant is Omega_m.

litmus_grid = np.linspace(0.60, 2.25, 20)
litmus_grid = litmus_grid[np.abs(litmus_grid - 0.93) > 0.04]     # away from the anchor

bao_litmus = MethodEnsemble([
    Cosmography("y"),
    Cosmography("log"),
    Cosmography("y", order=3),
    Cosmography("y", family="monomial"),
]).fit(
    desi_dr2_bao().select("DM_over_rs", "DH_over_rs"),
    grid=litmus_grid,
    n_draws=2000,
    seed=23,
)

show(calibrate(
    lambda s: CurvedLitmus(0.93, hubble_distance=hubble_distance).statistic(
        s["DM_over_rs"], s["DH_over_rs"]
    ),
    bao_litmus,
    0.0,
    null=LambdaCDM(curved=True),
    marginalise_constant=True,
    n_mocks=300,
    seed=24,
    name="curved litmus O(z; 0.93), DESI DR2",
))
