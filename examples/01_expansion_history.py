"""
Reconstruct H(z) from cosmic-chronometer-like data, three ways.

Run it:

    python examples/01_expansion_history.py

It needs no install -- ``src/`` is put on the path below -- and no network. The
measurements are mocks generated from a known LCDM expansion history, so every
number printed can be scored against a truth.

What it shows, in order:

1. A reconstruction, and whether its band actually contains the truth.
2. What marginalising the kernel hyperparameters costs, against the standard
   recipe of optimising them. This is the library's central claim, as a number.
3. What happens when a derivative is asked of a posterior that does not have
   one -- the check no other GP reconstruction code makes.
4. The Om(z) null test, built by ordinary arithmetic on the reconstruction,
   with the correlations propagating exactly because they were never
   summarised away.
5. The variance budget across five reconstruction methods: how much of the
   error bar is a measurement and how much is a choice. This is the number the
   library exists to produce and that no other tool produces.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from CosmoRecon import (                                          # noqa: E402
    Cosmography,
    ExtrapolationWarning,
    GaussianProcess,
    Matern,
    significance,
)
from CosmoRecon.ensemble import total_variance                     # noqa: E402
from tests.toy import chronometers, lcdm_H                        # noqa: E402


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


data = chronometers(lcdm_H, n=32, seed=11)

grid = np.linspace(0.2, 1.9, 25)


# ============================================================
# 1. A reconstruction
# ============================================================

rule("1. Gaussian process, nu marginalised")

gp = GaussianProcess(kernel="matern")

fit = gp.fit(data, grid=grid, n_draws=3000, seed=6)

H = fit["H"]

print(f"   {H.provenance.describe()}")
print(f"   support        {fit.support[0]:.3f} to {fit.support[1]:.3f}")
print(f"   log evidence   {gp.log_evidence:.2f}")
print(f"   basis error    {H._predictor.quadrature_error:.1e}  (kernel reproduced to this)")

lo, hi = H.interval(0.68)

truth = lcdm_H(grid)

print(f"   68% band contains the truth at "
      f"{100 * ((truth >= lo) & (truth <= hi)).mean():.0f}% of grid points")

for z in (0.3, 1.0, 1.8):
    at = H.at(z)
    print(f"   H({z:.1f}) = {float(at.mean()[0]):6.2f} +/- {float(at.std()[0]):4.2f}"
          f"   (truth {float(lcdm_H(z)):6.2f})")


# ============================================================
# 2. What optimising the hyperparameters hides
# ============================================================

rule("2. Marginalised vs optimised hyperparameters")

pinned_kernel = Matern(nu=2.5)

marginalised = GaussianProcess(kernel=pinned_kernel).fit(
    data, grid=grid, n_draws=3000, seed=6
)["H"]

# The single cell the standard recipe would have stopped at.
reference = GaussianProcess(kernel=pinned_kernel)
reference.fit(data, grid=grid, n_draws=10, seed=6)
best = reference._cells[int(np.argmax(reference._log_weights))]

optimised = GaussianProcess(
    kernel=pinned_kernel,
    n_length=1,
    n_amplitude=1,
    length_range=(best["length_scale"],) * 2,
    amplitude_range=(best["amplitude"],) * 2,
).fit(data, grid=grid, n_draws=3000, seed=6)["H"]

ratio = marginalised.std() / optimised.std()

print(f"   maximum-likelihood cell:  length {best['length_scale']:.2f}, "
      f"amplitude {best['amplitude']:.0f}")
print(f"   interval widens by        {100 * (np.median(ratio) - 1):.0f}% (median), "
      f"{100 * (ratio.max() - 1):.0f}% (worst)")
print("   -- that width is real uncertainty the standard recipe discards")


# ============================================================
# 3. A derivative that does not exist
# ============================================================

rule("3. Asking for a derivative the posterior does not support")

try:
    _ = H.d(1).draws

except Exception as error:            # noqa: BLE001 -- printing it is the point
    print("   " + str(error)[:300].replace("\n", "\n   "))

smooth = GaussianProcess(kernel=Matern(nu=[1.5, 2.0, 2.5, 3.5, 5.0, 7.5])).fit(
    data, grid=grid, n_draws=2000, seed=6
)["H"]

dH = smooth.d(1)

print(f"\n   with nu restricted to >= 3/2:  H'(1.0) = "
      f"{float(dH.mean()[11]):.1f} +/- {float(dH.std()[11]):.1f}")


# ============================================================
# 4. A null test, by arithmetic
# ============================================================

rule("4. Om(z) null test")

# Om(z) = [E(z)^2 - 1] / [(1+z)^3 - 1] is constant at Omega_m in flat LCDM.
#
# H.at(0.0) is a scalar-valued posterior and broadcasts; because it is draw
# aligned with H, the H0 uncertainty cancels realisation by realisation.
#
# It is also an extrapolation, and the library says so -- the lowest
# chronometer here sits at z = 0.106, so H(0) is the prior reaching down
# rather than a measurement. That is the honest situation for any H0 taken
# from chronometers alone. The warning is silenced only because the point of
# this section is what comes after it.
print(f"   (H(0) is extrapolated: the data start at z = {data.z.min():.3f})")

with warnings.catch_warnings():

    warnings.simplefilter("ignore", ExtrapolationWarning)

    E = H / H.at(0.0)

    Om = (E**2 - 1.0) / ((1.0 + grid) ** 3 - 1.0)

    _ = Om.draws      # materialise inside the block, where the warning belongs

chi2, n_eff, pte, sigma = significance(
    Om, float(Om.mean().mean()), marginalise_constant=True
)

print(f"   Om(z) = {Om.mean().min():.3f} to {Om.mean().max():.3f}   "
      f"(truth 0.300, constant)")
print(f"   chi2 = {chi2:.2f} over {n_eff} effective dof "
      f"(from {Om.n_z} grid points)")
print(f"   p = {pte:.3f}  ->  {sigma:.2f} sigma from constant")
print(f"   expression: {Om.provenance.expression}")

print()


# ============================================================
# 5. How much of the error bar is a choice
# ============================================================

rule("5. Variance budget across five methods")

members = {
    "GP(Matern, nu free)": GaussianProcess(kernel="matern"),
    "GP(squared exp)": GaussianProcess(kernel="squared_exponential"),
    "Chebyshev(y)": Cosmography("y"),
    "Chebyshev(log)": Cosmography("log"),
    "Pade[2/1](y)": Cosmography(pade=(2, 1)),
}

means = {}
variances = {}

for name, method in members.items():

    curve = method.fit(data, grid=grid, n_draws=2000, seed=4)["H"]

    means[name] = curve.mean()
    variances[name] = curve.var()

budget = total_variance(means, variances, dict.fromkeys(members, 1.0), grid)

print("   at z = 1.0, every method agrees on the value:")

for name in members:
    print(f"     {name:22s} {np.interp(1.0, grid, means[name]):7.2f}"
          f" +/- {np.interp(1.0, grid, np.sqrt(variances[name])):.2f}")

print(f"     {'truth':22s} {lcdm_H(1.0):7.2f}")

print()
print("   but the disagreement between them is not uniform in redshift:")
print(f"   {'z':>6s} {'statistical':>12s} {'method':>8s} {'method share':>13s}"
      f" {'inflation':>10s}")

for i in (0, 6, 12, 18, 24):
    print(f"   {grid[i]:6.2f} {np.sqrt(budget.statistical[i]):12.2f}"
          f" {np.sqrt(budget.methodological[i]):8.2f}"
          f" {100 * budget.method_fraction()[i]:12.0f}%"
          f" {budget.inflation()[i]:10.2f}")

print()
print(f"   {budget.summary()}")
print("   -- the method share does not shrink with more data")

print()
