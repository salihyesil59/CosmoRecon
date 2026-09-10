"""
From a reconstructed function back to the theory that would produce it.

The companion direction to symbolic tools that go action -> field equations ->
``E(z)``. Here the expansion history is the input and the Lagrangian function
is the output, with the posterior carried through intact.

Each of these maps is a few lines of arithmetic once the input is a
differentiable posterior over functions -- which is precisely why the core
object is shaped the way it is. In the literature each is instead re-derived
per paper on top of a Gaussian-process smoother, with the derivative
uncertainties propagated by hand or not at all.

Planned members:

``dark_energy.py``
    ``H(z) -> w(z), rho_de(z), Omega_de(z)``. Needs ``H'``, and the sign of
    the answer at high redshift is dominated by how that derivative was
    obtained -- so it is taken analytically, from the method, never by
    differencing a posterior mean.

``quintessence.py``
    ``H(z) -> V(phi), phi(z)``. The kinetic term can go negative where the
    data allow phantom behaviour; the module reports that region rather than
    clipping it, because it is the observationally interesting part.

``designer.py``
    ``H(z) -> f(R), f(T), f(Q)`` by the designer construction: the free
    function that reproduces a given expansion history exactly. Note what
    this does and does not show -- a background-level match is necessary and
    nowhere near sufficient, and the module says so alongside its output.

``coupling.py``
    ``{H, f sigma_8} -> mu(z) = G_eff / G``. The model-independent
    modified-gravity signal, and the natural partner to the growth-geometry
    null test in :mod:`CosmoRecon.consistency`.
"""

from __future__ import annotations


__all__: list[str] = []
