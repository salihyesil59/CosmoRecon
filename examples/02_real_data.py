"""
The same machinery on real observations.

    python examples/02_real_data.py

Example 01 runs on mocks, because validating a method needs a known truth.
This one runs on the bundled releases -- 32 cosmic chronometers and DESI DR2
BAO -- where there is no truth to compare against and the output is a result
rather than a check.

Three things come out of it:

1. What the data are, including the parts a user has to know before trusting
   the numbers -- an arbitrary zero point, a fiducial cosmology in the
   reduction, a ratio to a sound horizon that is not itself measured.
2. Reconstructions of ``H(z)`` and ``D_M/r_d``, and the variance budget across
   five methods: how much of the error bar is a measurement and how much is a
   choice. Nothing else reports the second number.
3. The ``Om(z)`` null test on the real expansion history, with an effective
   number of degrees of freedom rather than a grid size.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from CosmoRecon import (                                          # noqa: E402
    Cosmography,
    ExtrapolationWarning,
    GaussianProcess,
    MethodEnsemble,
)
from CosmoRecon.data import chronometers, desi_dr2_bao            # noqa: E402


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def ensemble() -> MethodEnsemble:
    """
    Two Gaussian processes with different kernels, two series in different
    expansion variables, one rational. Chosen to disagree in different ways
    rather than to agree -- an ensemble of near-identical methods would report
    a small method variance and mean nothing by it.
    """

    return MethodEnsemble([
        GaussianProcess(kernel="matern"),
        GaussianProcess(kernel="squared_exponential"),
        Cosmography("y"),
        Cosmography("log"),
        Cosmography(pade=(2, 1)),
    ])


def show(budget, grid, indices):

    print(f"   {'z':>6s} {'statistical':>12s} {'method':>9s}"
          f" {'method share':>13s} {'inflation':>10s}")

    for i in indices:
        print(f"   {grid[i]:6.2f} {np.sqrt(budget.statistical[i]):12.3f}"
              f" {np.sqrt(budget.methodological[i]):9.3f}"
              f" {100 * budget.method_fraction()[i]:12.0f}%"
              f" {budget.inflation()[i]:10.2f}")

    print()
    print(f"   {budget.summary()}")


# ============================================================
# 1. What the data are
# ============================================================

rule("1. The bundled observations")

cc = chronometers()

bao = desi_dr2_bao()

for dataset in (cc, bao):
    print("   " + dataset.describe().replace("\n", "\n   "))
    print()


# ============================================================
# 2. H(z) from cosmic chronometers
# ============================================================

rule("2. H(z) from 32 cosmic chronometers, five methods")

cc_grid = np.linspace(0.10, 1.90, 25)

cc_fit = ensemble().fit(cc, grid=cc_grid, n_draws=2000, seed=4)

print("   at z = 1, every method agrees:")

for name, curve in cc_fit.members.items():
    at_one = curve.at(1.0)
    print(f"     {name:36s} {float(at_one.mean()[0]):7.2f}"
          f" +/- {float(at_one.std()[0]):4.2f}")

print()

show(cc_fit.budget(), cc_grid, (0, 8, 16, 24))


# ============================================================
# 3. D_M / r_d from DESI DR2
# ============================================================

rule("3. D_M/r_d from DESI DR2, five methods")

transverse = bao.select("DM_over_rs")

print(f"   {len(transverse)} measurements, z = {transverse.z.min():.3f}"
      f" to {transverse.z.max():.3f}")

bao_grid = np.linspace(0.55, 2.30, 25)

bao_fit = ensemble().fit(transverse, grid=bao_grid, n_draws=2000, seed=4)

print()

show(bao_fit.budget(), bao_grid, (0, 8, 16, 24))


# ============================================================
# 4. The Om(z) null test on the real expansion history
# ============================================================

rule("4. Om(z) null test on the chronometers")

# Om(z) = [E(z)^2 - 1] / [(1+z)^3 - 1] is constant at Omega_m in flat LCDM,
# whatever H0 is. Written as arithmetic on the reconstruction, so that the H0
# uncertainty cancels draw by draw instead of adding.
#
# The same expression goes to every member and to the mixture, so the only
# thing differing between the numbers below is which posterior it was
# evaluated on -- which is what makes the spread attributable to the method.
#
# H(0) is a short extrapolation: the lowest chronometer sits at z = 0.07, and
# the library flags it. Silenced here because that flag is the subject of
# example 01, not of this one.


def om_diagnostic(H):

    return ((H / H.at(0.0)) ** 2 - 1.0) / ((1.0 + cc_grid) ** 3 - 1.0)


with warnings.catch_warnings():

    warnings.simplefilter("ignore", ExtrapolationWarning)

    comparison = cc_fit.significance(
        om_diagnostic,
        null_value=0.30,
        marginalise_constant=True,
        name="Om(z) from 32 cosmic chronometers, tested against 'some constant'",
    )

    curve = om_diagnostic(cc_fit.marginalised)

    lo, hi = curve.interval(0.68)

print(f"   Om(z) runs {curve.mean().min():.3f} to {curve.mean().max():.3f}"
      f" over z = {cc_grid[0]:.2f} to {cc_grid[-1]:.2f},"
      f" width +/- {np.median(0.5 * (hi - lo)):.3f}")
print()
print("   " + comparison.summary().replace("\n", "\n   "))
print()

verdict = comparison.marginalised[2] >= 0.05

print("   " + ("consistent with a cosmological constant"
               if verdict else "DEVIATION from a cosmological constant"))
print("   32 differential ages cannot see a DESI-scale deviation, and a tool")
print("   that said otherwise would be measuring its own assumptions.")

print()
