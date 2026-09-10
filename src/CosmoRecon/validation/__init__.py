"""
Proving that a method deserves to be in the library.

The rule this package enforces: **a reconstruction method that reports a 68%
interval must contain the truth 68% of the time.** A method that reports 68%
and covers 45% is manufacturing significance -- which is the failure mode
CosmoRecon exists to expose, and would be an embarrassing one to ship.

Coverage is therefore a *test*, run in CI, not a figure produced once for a
paper.

Planned contents:

``mocks.py``
    Injection. Generate cosmic chronometer, BAO, supernova and growth mocks
    at a real survey's error budget from a known ``E(z)`` -- LCDM, CPL, an
    oscillating ``w``, a sign-switching Lambda -- so that the truth is known
    exactly and the recovery can be scored.

``coverage.py``
    Recovery. Over many realisations, the empirical coverage of each stated
    interval, per method and per redshift, with the binomial uncertainty on
    the coverage itself. Reported as a table a referee can read.

The cross-checks that live in the test suite rather than here follow the same
principle -- anything checkable two ways is checked two ways: an analytic GP
derivative against finite differences of the same draw, a Chebyshev expansion
against a Taylor one inside the radius of convergence, and the whole
null-test chain against a ``Reconstruction`` built analytically from a known
cosmology, where every statistic has a value known in closed form.
"""

from __future__ import annotations


__all__: list[str] = []
