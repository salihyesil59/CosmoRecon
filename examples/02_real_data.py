"""
The same machinery on real observations.

    python examples/02_real_data.py

Example 01 runs on mocks, because validating a method needs a known truth.
This one runs on the bundled releases -- 32 cosmic chronometers, DESI DR2 BAO
and the Union3 supernovae -- where there is no truth to compare against and the
output is a result rather than a check.

Six things come out of it:

1. What the data are, including the parts a user has to know before trusting
   the numbers -- an arbitrary zero point, a fiducial cosmology in the
   reduction, a ratio to a sound horizon that is not itself measured.
2. Reconstructions of ``H(z)`` and ``D_M/r_d``, and the variance budget across
   five methods: how much of the error bar is a measurement and how much is a
   choice. Nothing else reports the second number.
3. Two null tests of the cosmological constant on the real expansion history,
   under every method and under the mixture -- and the difference between a
   diagnostic that needs a calibration and one that does not.
4. The Clarkson-Bassett-Lu curvature test on DESI DR2, which is where the
   library's argument stops being methodological and starts changing what a
   result is: the same twelve numbers support anything from no violation of
   the Copernican principle to a decisive one, depending only on the method.
5. Distance duality from Union3 and DESI DR2 -- two datasets, two fits, the
   independence between them declared -- where two nearly identical methods
   report a cosmic opacity of opposite sign.
6. Growth against geometry: whether the Gold-2018 growth rates follow from the
   DESI DR2 expansion history under general relativity, and the amplitude
   ``sigma_8`` they imply -- with no dark-energy model anywhere.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from CosmoRecon import (                                          # noqa: E402
    C_LIGHT_KM_S,
    Cosmography,
    Curvature,
    Duality,
    ExtrapolationWarning,
    Growth,
    GaussianProcess,
    MethodEnsemble,
    Om,
    Om3,
)
from CosmoRecon.data import (                                     # noqa: E402
    chronometers,
    desi_dr2_bao,
    growth,
    reduced_modulus,
    union3,
)


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

for name, curve in cc_fit.curves("H").items():
    at_one = curve.at(1.0)
    print(f"     {name:36s} {float(at_one.mean()[0]):7.2f}"
          f" +/- {float(at_one.std()[0]):4.2f}")

print()

show(cc_fit.budget("H"), cc_grid, (0, 8, 16, 24))


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

show(bao_fit.budget("DM_over_rs"), bao_grid, (0, 8, 16, 24))


# ============================================================
# 4. Null tests of the cosmological constant
# ============================================================

rule("4. Null tests on the chronometers")

# Om(z) = [E(z)^2 - 1] / [(1+z)^3 - 1] is Omega_m in flat LCDM, at every
# redshift. Om3(z1, z2, z3) = Om(z2;z1) / Om(z3;z1) is exactly 1 -- and, being
# a ratio, needs neither H0 nor Omega_m.
#
# That is not a cosmetic difference here. Om is anchored at z = 0 and the
# lowest chronometer sits at z = 0.07, so every Om(z) from this dataset is
# standing on an extrapolation; the library flags it, and it is silenced below
# only because the flag is the subject of example 01. Om3 is anchored at two
# redshifts inside the data and touches nothing that was not measured.
#
# Both statistics go to every member and to the mixture, so the only thing
# differing between the numbers is which posterior they were evaluated on.

for test, note in [
    (Om(), "anchored at z = 0, so it needs H(0) -- an extrapolation here"),
    (Om3(z1=0.15, z2=0.35), "anchored inside the data; no H0, no Omega_m"),
]:

    print()
    print(f"   {test.name}: {note}")

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ExtrapolationWarning)

        comparison = cc_fit.significance(
            lambda s, t=test: t.statistic(s["H"]),
            test.null_value,
            marginalise_constant=test.null_is_free_constant,
            name=test.name,
        )

        curve = test.statistic(cc_fit.marginalised["H"])

    print(f"     runs {curve.mean().min():.3f} to {curve.mean().max():.3f},"
          f" typical width +/- {np.median(curve.std()):.3f}")

    body = comparison.summary().splitlines()[1:]

    for line in body:
        print("     " + line.strip())

print()
print("   Both are consistent with a cosmological constant, under every method")
print("   and under the mixture. Thirty-two differential ages cannot see a")
print("   DESI-scale deviation, and a tool that said otherwise would be")
print("   measuring its own assumptions.")


# ============================================================
# 5. The curvature test, and a dataset that cannot support it
# ============================================================

rule("5. Ok(z) from DESI DR2, and what it is worth")

# Ok(z) = [H^2 D_M'^2 - c^2] / [H0^2 D_M^2] is Omega_k in *any* FLRW universe,
# whatever the dark energy does. A departure from constancy is therefore not
# evidence about dark energy; it is evidence against homogeneity and isotropy.
#
# It needs D_M/r_d and D_H/r_d together -- one of them differentiated -- so the
# two are reconstructed jointly and their correlation is carried into the
# statistic. Selecting them separately and pairing the results is refused.

joint = bao.select("DM_over_rs", "DH_over_rs")

print(f"   {len(joint)} measurements of {list(joint.quantities())},"
      " reconstructed together")

ok_grid = np.linspace(0.60, 2.25, 20)

ok_fit = MethodEnsemble([
    Cosmography("y"),
    Cosmography("log"),
    Cosmography("y", order=3),
    Cosmography("y", family="monomial"),
]).fit(joint, grid=ok_grid, n_draws=4000, seed=4)

# c / (H0 r_d) from DESI DR2's published r_d h = 101.54 Mpc. Needed only to
# turn the statistic into Omega_k itself; the question of whether it is
# constant needs no calibration at all.
calibration = C_LIGHT_KM_S / (100.0 * 101.54)

comparison = ok_fit.significance(
    lambda s: Curvature(hubble_distance=calibration).statistic(
        s["DM_over_rs"], s["DH_over_rs"]
    ),
    0.0,
    marginalise_constant=True,
    name="Ok(z), tested for constancy",
)

print()

for line in comparison.summary().splitlines()[1:]:
    print("   " + line.strip())

print()
print("   Read that column again. Four nearly identical polynomial fits to the")
print("   same twelve numbers report anything from a fraction of a sigma to a")
print("   formally infinite one. A violation of the Copernican principle at")
print("   several sigma is available to whoever picks the right expansion")
print("   variable -- and nothing inside a single-method analysis could tell.")
print()
print("   The honest reading is the marginalised one: six transverse and six")
print("   radial BAO measurements, one of which has to be differentiated, do")
print("   not constrain Ok(z). This dataset cannot answer the question, and")
print("   saying so is the result.")


# ============================================================
# 6. Distance duality, across two datasets
# ============================================================

rule("6. Distance duality from Union3 and DESI DR2")

# eta(z) = d_L / [(1+z) D_M] is exactly 1 if photons are conserved and travel
# on null geodesics. No expansion history, curvature or dark energy enters.
#
# Its two sides come from different probes, so this is two fits, and the
# independence between them is declared -- supernovae and galaxies share no
# data. with_independent pairs member m on one dataset with member m on the
# other, since a method is a choice an analyst makes once.
#
# The supernovae enter as the reduced modulus mu - 5 log10 z. A reconstruction
# of mu itself pays for its logarithmic singularity at the origin where the
# data are sparse, and invents a violation at 21 sigma in mock universes where
# duality holds exactly.
#
# Neither question asked below needs a calibration: the opacity slope epsilon
# in eta ~ (1+z)^epsilon absorbs the sound horizon and the supernova zero
# point into its free level.

supernovae = reduced_modulus(union3())

# Where both are measured: Union3 reaches z = 2.26, DESI's transverse
# distances start at z = 0.51.
duality_grid = np.linspace(0.51, 2.26, 20)

duality_ensemble = MethodEnsemble([
    GaussianProcess(kernel="matern"),
    Cosmography("y"),
    Cosmography("y", order=3),
    Cosmography("log"),
])

both = duality_ensemble.fit(
    supernovae, grid=duality_grid, n_draws=4000, seed=7
).with_independent(
    duality_ensemble.fit(transverse, grid=duality_grid, n_draws=4000, seed=7)
)

print(f"   {len(supernovae)} supernova bins and {len(transverse)} transverse BAO"
      f" distances, over z = {duality_grid[0]:.2f} to {duality_grid[-1]:.2f}")
print()
print("   opacity slope epsilon in eta ~ (1+z)^epsilon:")


def opacity(fit):
    slope = Duality().opacity(fit["mu_reduced"], fit["DM_over_rs"])
    return float(slope.mean()[0]), float(slope.std()[0])


for name, fit in both.members.items():
    mean, width = opacity(fit)
    print(f"     {name:36s} {mean:+.3f} +/- {width:.3f}"
          f"  ({abs(mean) / width:.1f} sigma)")

mean, width = opacity(both.marginalised)

print(f"     {'-- method-marginalised':36s} {mean:+.3f} +/- {width:.3f}"
      f"  ({abs(mean) / width:.1f} sigma)")

print()
print("   Two free-order Chebyshev fits to the same 28 numbers report a cosmic")
print("   opacity of opposite sign, each at more than two and a half sigma.")
print("   Either would have been a paper. Marginalised over the method the slope")
print("   is consistent with a transparent universe, and about four times wider")
print("   than either claimed.")
print()
print("   The signs are the methods', not the sky's. On mock surveys where the")
print("   slope is exactly zero, the series in y leans negative and the series")
print("   in ln(1+z) positive -- and the method-marginalised slope stayed within")
print("   2 sigma in every one of twelve.")

comparison = both.significance(
    lambda s: Duality().statistic(s["mu_reduced"], s["DM_over_rs"]),
    1.0,
    marginalise_constant=True,
    name="eta(z), tested for constancy",
)

print()

for line in comparison.summary().splitlines()[1:]:
    print("   " + line.strip())


# ============================================================
# 7. Growth against geometry
# ============================================================

rule("7. Growth against geometry: Gold-2018 and DESI DR2")

# Under GR, with matter the only component that clusters, the growth equation
# has a first integral that ties f sigma_8 to the expansion rate:
#
#     a H d/dln a [a^2 H f sigma_8] = (3/2) Omega_m H0^2 sigma_8(z),
#
# whatever the dark energy and the curvature. Anchoring at one redshift
# removes sigma_8, and G(z) is then constant at Omega_m H0^2.
#
# DESI's radial distance D_H/r_d is the expansion rate in units of c/r_d, so
# the constant is Omega_m h^2 (100 r_d / c)^2. Calibrating with the sound
# horizon and H0 = 100 makes it Omega_m h^2 -- a number the CMB fixes from
# early-universe physics alone, whatever dark energy did later. Planck 2018:
# Omega_m h^2 = 0.1430, r_d = 147.09 Mpc.
#
# Given that number, the same relation returns sigma_8 today at every
# redshift. The growth compilation and the BAO release are different surveys,
# and the independence is declared.

OMEGA_M_H2, SOUND_HORIZON = 0.1430, 147.09

growth_grid = np.linspace(0.55, 1.5, 16)
growth_grid = growth_grid[np.abs(growth_grid - 0.93) > 0.04]   # the anchor

radial = desi_dr2_bao().select("DH_over_rs")

growth_ensemble = MethodEnsemble([
    Cosmography("y"),
    Cosmography("log"),
    Cosmography("y", order=3),
    Cosmography("y", family="monomial"),
])

growth_fit = growth_ensemble.fit(
    growth(), grid=growth_grid, n_draws=4000, seed=31
).with_independent(
    growth_ensemble.fit(radial, grid=growth_grid, n_draws=4000, seed=32)
)

geometry = Growth(
    0.93,
    hubble_distance=C_LIGHT_KM_S / (100.0 * SOUND_HORIZON),
    omega_m=OMEGA_M_H2,
)

print(f"   {len(growth())} growth rates and {len(radial)} radial BAO distances,"
      f" over z = {growth_grid[0]:.2f} to {growth_grid[-1]:.2f}")
print()
print(f"   {'':36s} {'Omega_m h^2 at z = 0.55 / 1.5':>29s}  {'sigma_8(0) at z = 0.55 / 1.5':>29s}")

for name, fit in list(growth_fit.members.items()) + [("-- method-marginalised", growth_fit.marginalised)]:

    density = geometry.statistic(fit["fsigma8"], fit["DH_over_rs"])
    amplitude = geometry.sigma8(fit["fsigma8"], fit["DH_over_rs"])

    cells = [
        f"{c.mean()[i]:.2f}+/-{c.std()[i]:.2f}"
        for c in (density, amplitude) for i in (0, -1)
    ]

    print(f"   {name:36s} {cells[0]:>14s} {cells[1]:>14s}  {cells[2]:>14s} {cells[3]:>14s}")

comparison = growth_fit.significance(
    lambda s: Growth(0.93).statistic(s["fsigma8"], s["DH_over_rs"]),
    0.0,
    marginalise_constant=True,
    name="G(z), tested for constancy",
)

print()

for line in comparison.summary().splitlines()[1:]:
    print("   " + line.strip())

print()
print("   Growth and geometry agree under GR, under every method. The growth")
print("   reads Omega_m h^2 above Planck's 0.143 and sigma_8 below its 0.811, each")
print("   within about one sigma -- the direction of the S8 tension, nowhere near")
print("   its significance.")
print()
print("   Agreement is not confirmation. A coupling G_eff/G = 1 + Omega_DE(a),")
print("   which moves G(z) by 20% on this grid, is not detected at these errors:")
print("   calibrated on mock surveys, 0 of 6 realisations at the real errors and 3")
print("   of 6 at a tenth of them. Twenty-two growth rates cannot see an order-")
print("   unity change in the strength of gravity.")

print()
print("   Every significance in this example is nominal, and none should be")
print("   quoted as it stands. On mocks where the null holds exactly, the nominal")
print("   chi-square almost never rejects for a Gaussian process and rejects most")
print("   of the time for a free-order series, whose bias under the null it reads")
print("   as a detection. examples/03_calibrated_significance.py reruns these")
print("   same analyses against universes in which each null is true, and reports")
print("   the significances those universes imply.")

print()
