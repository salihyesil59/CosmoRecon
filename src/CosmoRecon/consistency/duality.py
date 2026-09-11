"""
Distance duality: the Etherington relation, tested without a model.

Etherington (1933), Phil. Mag. 15, 761. In any metric theory of gravity in
which light travels on null geodesics and photons are conserved,

    ``d_L(z) = (1 + z)^2 d_A(z) = (1 + z) D_M(z)``

so the ratio

    ``eta(z) = d_L(z) / [ (1 + z) D_M(z) ] = 1``

at every redshift -- whatever the expansion history, the curvature or the dark
energy. None of those appear. A departure means photons were lost or gained on
the way (grey dust, photon-axion mixing; Bassett & Kunz 2004, Phys. Rev. D 69,
101305), or that light does not travel on the null geodesics of a metric at
all. Like the curvature test, it is a statement about the framework rather than
about a model inside it.

The two sides come from different probes -- the luminosity distance from
standard candles, the transverse distance from a standard ruler -- and
therefore from **different datasets**. That makes this the first test in the
library whose inputs are two separate fits, and the independence between them
has to be declared: see
:func:`~CosmoRecon.reconstructors.base.combine_independent` for a pair of fits
and :meth:`~CosmoRecon.ensemble.method.EnsembleFit.with_independent` for a pair
of ensembles. For Union3 and DESI the claim is true -- supernovae and galaxies
-- and it belongs in the source, where a reader can check it.

What it needs, and what it does not
-----------------------------------

A supernova compilation measures a distance modulus whose zero point is not
measured -- it is degenerate with the absolute magnitude and with ``H_0`` -- and
a BAO release measures ``D_M / r_d`` with ``r_d`` not measured either. Written in
those observables,

    ``eta(z) = 10^{(mu(z) - 25) / 5} / [ (1 + z) r_d T(z) ]``,  ``T = D_M / r_d``

so every calibration collects into one multiplicative constant,
``K = r_d 10^{Delta / 5}``, with ``Delta`` the offset between the distance moduli
as supplied and absolute ones. The same structure as the curvature test, and
the same consequence:

- *Is* ``eta`` *constant?* Needs no calibration. This is the question every
  physical mechanism answers: absorption and mixing accumulate along the path,
  so they grow with redshift.
- *What is the opacity slope?* ``eta = K (1 + z)^epsilon`` (Avgoustidis et al.
  2010, JCAP 10, 024) has ``epsilon`` as the slope of ``ln eta`` against
  ``ln(1+z)``, with the level free -- so :meth:`Duality.opacity` needs no
  calibration either.
- *Is* ``eta = 1``? Needs ``K``, which means ``r_d`` **and** the supernova zero
  point, and the class asks for both. A violation that does not depend on
  redshift is exactly degenerate with the calibration, and nothing in these
  data can see it.

Why the supernovae are not reconstructed as ``mu``
--------------------------------------------------

Near the origin ``mu(z) ~ 5 log10(z)``: a logarithmic singularity at ``z = 0``
that no polynomial represents and no stationary kernel expects. A series fitted
to ``mu`` over the range of a real compilation pays for it where the data are
sparsest, which for Union3 is the gap between its last two bins at
``z = 1.39`` and ``z = 2.26`` -- and a distance ratio then inherits a
redshift-dependent bias that looks exactly like a violation.

This was measured, not supposed. On twelve mock Union3 + DESI DR2 realisations
of a universe in which duality holds **exactly**, with the real redshifts and
covariances:

======================  =================  ================
fitted to               Chebyshev order 3  Gaussian process
======================  =================  ================
``mu``                  21 sigma, 12 / 12  0 sigma, 0 / 12
``mu - 5 log10(z)``     0.6 sigma, 0 / 12  0 sigma, 0 / 12
======================  =================  ================

(median significance of the constancy test; realisations beyond 2 sigma.) The
series fails by inventing a violation in every universe; the Gaussian process
fails the other way, with an interval three magnitudes wide inside that gap.

The **reduced modulus** ``m(z) = mu(z) - 5 log10(z)`` is smooth at the origin
and removes both failures. The transformation is exact -- a known number
subtracted from each measurement -- so the covariance is unchanged and no
approximation enters. :func:`CosmoRecon.data.reduced_modulus` does it, and
:meth:`Duality.statistic` takes the reduced modulus and refuses a
reconstruction of ``mu`` itself.

How far to trust the significance
---------------------------------

The same mocks say something less comfortable, about the chi-square rather than
about this test. Under an exactly true null the constancy test should exceed
2 sigma about 5% of the time. A Gaussian process never does, because modes its
prior still dominates are counted as degrees of freedom and dilute the
chi-square; a series with its order left free does so in half the
realisations, because its posterior is narrower than its error. Only a pinned
low-order series came out close to nominal.

The opacity slope, a single number, shows the failures more plainly and the
remedy with them. On those mocks, where ``epsilon = 0``, a Chebyshev series in
``y`` leans negative (mean pull ``-1.5``) and one in ``ln(1+z)`` positive
(``+0.75``). Single methods exceed 2 sigma in 20 to 58 per cent of
realisations. The method-marginalised slope exceeded it in none of twelve. The
per-method spread :mod:`CosmoRecon.ensemble` reports is therefore not
decoration here -- it is the only protection against both failures at once.
"""

from __future__ import annotations

import numpy as np

from CosmoRecon.core.errors import DataError
from CosmoRecon.core.reconstruction import Reconstruction

from CosmoRecon.consistency.base import NullTest


__all__ = ["Duality", "REDUCED_MODULUS"]


#: The observable name :func:`CosmoRecon.data.reduced_modulus` gives its
#: output, and so the label a reconstruction of it carries.
REDUCED_MODULUS = "mu_reduced"

#: Redshift below which the statistic is refused. Both distances vanish at the
#: origin, so the ratio is 0/0 there and dominated by that division long
#: before it is exactly singular. No BAO release reaches below ``z = 0.29``.
_MIN_REDSHIFT = 0.05


class Duality(NullTest):
    """
    ``eta(z) = d_L / [(1 + z) D_M]``: exactly 1 if photons are conserved and
    travel on null geodesics.

    >>> sn = GaussianProcess().fit(reduced_modulus(union3()), grid=grid)
    ...                                                        # doctest: +SKIP
    >>> bao = GaussianProcess().fit(desi.select("DM_over_rs"), grid=grid)
    ...                                                        # doctest: +SKIP
    >>> both = combine_independent(sn, bao)                    # doctest: +SKIP
    >>> Duality().evaluate(both["mu_reduced"], both["DM_over_rs"])
    ...                                                        # doctest: +SKIP

    Parameters
    ----------
    eta
        Test against a specific value instead of against "some constant".
        Needs the calibration, since the statistic is otherwise known only up
        to a multiplicative constant -- and ``eta = 1``, the value that
        actually matters, is the case that needs it.
    sound_horizon
        The length, in Mpc, that one unit of the transverse input represents:
        ``r_d`` when it is ``D_M / r_d``, ``1`` when it is ``D_M`` in Mpc.
    magnitude_offset
        ``mu`` as supplied minus ``mu`` absolute, in magnitudes -- ``0`` if the
        compilation's zero point is to be taken at face value.

        Required together with ``sound_horizon``, never defaulted. A supernova
        zero point is a calibration choice with a real literature behind it,
        and leaving it at zero silently would put an absolute-magnitude
        assumption into a result that looks model-independent.
    """

    def __init__(
        self,
        *,
        eta: float | None = None,
        sound_horizon: float | None = None,
        magnitude_offset: float | None = None,
    ) -> None:

        if (sound_horizon is None) != (magnitude_offset is None):

            raise ValueError(
                "The calibration is one constant made of two things, the "
                "sound horizon and the supernova zero point, and supplying "
                "one of them without the other would fix the constant by "
                "assumption. Pass both sound_horizon and magnitude_offset -- "
                "magnitude_offset=0.0 if the distance moduli are to be taken "
                "as absolute -- or neither."
            )

        if eta is not None and sound_horizon is None:

            raise ValueError(
                f"Testing against eta = {eta:g} needs a calibration: without "
                "one the statistic is known only up to a constant made of the "
                "sound horizon and the supernova zero point, neither of which "
                "either dataset measures. Pass sound_horizon and "
                "magnitude_offset.\n"
                "\n"
                "The questions that need no calibration are whether eta is "
                "constant (eta=None) and the opacity slope (Duality.opacity) "
                "-- and they are the ones that any physical mechanism for a "
                "violation answers, since absorption and mixing accumulate "
                "with distance."
            )

        if sound_horizon is not None and sound_horizon <= 0.0:

            raise ValueError(
                f"sound_horizon must be positive, got {sound_horizon}."
            )

        self.eta = None if eta is None else float(eta)

        self.sound_horizon = (
            None if sound_horizon is None else float(sound_horizon)
        )

        self.magnitude_offset = (
            None if magnitude_offset is None else float(magnitude_offset)
        )

    # ---------------------------------------------------------

    @property
    def calibration(self) -> float | None:
        """``K = r_d 10^{Delta / 5}``, or ``None`` if none was given."""

        if self.sound_horizon is None:
            return None

        return self.sound_horizon * 10.0 ** (self.magnitude_offset / 5.0)

    @property
    def name(self) -> str:

        if self.eta is not None:
            return f"eta(z) vs {self.eta:g}"

        return "eta(z)" if self.calibration is not None else "eta(z) / calibration"

    @property
    def null_value(self) -> float:

        return 1.0 if self.eta is None else self.eta

    @property
    def null_is_free_constant(self) -> bool:
        """
        With no ``eta`` the test is "constant, value unknown", which is the
        test that needs no calibration. The ``null_value`` of 1 is then never
        used: projecting out the uniform direction removes any level with it.
        """

        return self.eta is None

    # ---------------------------------------------------------

    def statistic(
        self,
        modulus: Reconstruction,
        transverse: Reconstruction,
    ) -> Reconstruction:
        """
        Build ``eta(z)`` from a reduced distance modulus and a transverse
        distance.

        ``modulus`` is a reconstruction of ``mu - 5 log10(z)`` -- see
        :func:`CosmoRecon.data.reduced_modulus` and the module docstring for
        why not ``mu``. ``transverse`` is ``D_M / r_d``, or ``D_M`` in Mpc.

        The two will normally come from different datasets, and combining them
        needs an independence claim. Without one this raises
        :class:`~CosmoRecon.core.errors.AlignmentError` rather than pairing
        two unrelated realisation streams by index.
        """

        if modulus.label == "mu":

            raise DataError(
                "This is a reconstruction of the distance modulus itself. "
                "mu(z) ~ 5 log10(z) has a logarithmic singularity at the "
                "origin that no series represents and no stationary kernel "
                "expects, and a fit pays for it where the data are sparse: on "
                "mock Union3 + DESI data in which duality holds exactly, a "
                "third-order Chebyshev fit to mu reports a violation at 21 "
                "sigma in every realisation.\n"
                "\n"
                "Reconstruct the reduced modulus mu - 5 log10(z) instead -- "
                "CosmoRecon.data.reduced_modulus(dataset) -- which is smooth "
                "at the origin. The transformation is exact, so nothing about "
                "the data changes except that a reconstruction can now "
                "represent them."
            )

        z = transverse.z

        if np.any(z < _MIN_REDSHIFT) or np.any(modulus.z < _MIN_REDSHIFT):

            raise DataError(
                f"The grid reaches z = {min(z.min(), modulus.z.min()):.4g}. "
                "Both distances vanish at the origin, so their ratio is 0/0 "
                f"there and meaningless below z = {_MIN_REDSHIFT:g}. No BAO "
                "release reaches that low in any case."
            )

        # d_L in Mpc, up to the supernova zero point:
        # 10^((mu - 25) / 5) with mu = m + 5 log10(z) is z 10^((m - 25) / 5).
        luminosity = modulus.z * 10.0 ** ((modulus - 25.0) / 5.0)

        curve = luminosity / ((1.0 + z) * transverse)

        if self.calibration is not None:
            curve = curve / self.calibration

        curve.label = self.name

        return curve

    def opacity(
        self,
        modulus: Reconstruction,
        transverse: Reconstruction,
    ) -> Reconstruction:
        """
        The opacity slope ``epsilon`` in ``eta(z) = K (1 + z)^epsilon``, with
        the level ``K`` left free.

        Returned as a one-point reconstruction -- a posterior over a number,
        one value per draw -- so that it can be tested like anything else:

        >>> eps = Duality().opacity(both["mu_reduced"], both["DM_over_rs"])
        ...                                                    # doctest: +SKIP
        >>> significance(eps, 0.0)                             # doctest: +SKIP

        **What it is.** Each draw of ``ln eta(z)`` is regressed on
        ``ln(1 + z)`` with a free intercept, and the slope is that draw's
        ``epsilon``. Because the intercept absorbs ``ln K``, no calibration
        enters. The regression is weighted by the inverse pointwise variance
        of ``ln eta`` -- fixed weights, the same for every draw -- so redshifts
        where either input is poorly constrained contribute little.

        **What it is not.** A slope is a projection. A violation that is not a
        power law still has one, and a redshift-independent violation has
        none: it is absorbed into ``K`` along with the calibration, which is
        the price of needing none.

        The single redshift the result carries is the regression's pivot,
        where the fitted level and ``epsilon`` are uncorrelated.
        """

        curve = self.statistic(modulus, transverse)

        if curve.n_z < 2:

            raise DataError(
                "A slope needs at least two redshifts; this grid has one."
            )

        draws = curve.draws

        unphysical = np.any(draws <= 0.0, axis=1)

        if np.any(unphysical):

            where = curve.z[np.any(draws <= 0.0, axis=0)]

            raise DataError(
                f"{100.0 * unphysical.mean():.2g}% of the draws of eta(z) are "
                f"not positive, at z = {where.min():.3g} to {where.max():.3g}, "
                "and a ratio of two distances cannot be. The luminosity side "
                "is positive by construction, so it is the transverse "
                "distance whose posterior reaches zero there -- usually "
                "across a redshift gap its data do not constrain, where a "
                "flexible method returns something close to its prior.\n"
                "\n"
                "The constancy test is still defined on these draws; the "
                "slope, which needs ln eta, is not, and it is refused rather "
                "than computed on the draws that happen to be positive. Narrow "
                "the grid away from the gap, or use a method whose posterior "
                "there stays physical."
            )

        log_eta = np.log(draws)

        variance = log_eta.var(axis=0)

        if np.any(variance <= 0.0):

            raise DataError(
                "ln eta(z) has no variance at some redshift, so the weights "
                "of the regression are undefined. A posterior with no width "
                "has no slope uncertainty to report."
            )

        weight = 1.0 / variance

        x = np.log1p(curve.z)

        pivot = float(np.sum(weight * x) / np.sum(weight))

        lever = weight * (x - pivot)

        epsilon = log_eta @ lever / float(np.sum(lever * (x - pivot)))

        result = Reconstruction.from_draws(
            np.array([np.expm1(pivot)]),
            epsilon[:, None],
            origin=curve.origin,
            provenance=curve.provenance.derive(
                "epsilon: weighted slope of ln eta against ln(1+z)",
                n_draws=int(epsilon.size),
            ),
            label="epsilon",
        )

        return result.assume_independent() if curve._independent else result
