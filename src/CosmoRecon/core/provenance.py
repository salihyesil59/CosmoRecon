"""
Where a reconstruction came from.

In a library whose central claim is *the answer depends on the method*, the
method is part of the answer. A curve that says "H(z), 68% interval" and
nothing else is not a result here -- the same data through a Matern kernel
with free smoothness and through a fifth-order Chebyshev expansion give
visibly different curves, and a figure that does not carry which one it is
cannot be checked by anyone.

So every :class:`~CosmoRecon.core.reconstruction.Reconstruction` carries a
:class:`Provenance`, and arithmetic *composes* them rather than dropping them:
the ``Ok(z)`` built out of ``H`` and ``D`` records both parents and the
expression that combined them. Plot labels, figure captions and the ensemble
layer's per-method breakdown all read from it, so it stays honest by being
used rather than by being audited.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace


__all__ = ["Provenance", "new_origin", "Origin"]


# ============================================================
# Fit identity
# ============================================================

#: The identity of a single fit. Every reconstruction produced by one call to
#: ``Reconstructor.fit`` shares one of these, and that shared value is what
#: makes draw ``k`` of ``H`` and draw ``k`` of ``D`` the same universe.
Origin = int

_origin_counter = itertools.count(1)


def new_origin() -> Origin:
    """
    Mint a fresh fit identity.

    A monotone integer rather than a UUID: these are compared constantly in
    arithmetic, they show up in error messages a human has to read, and they
    never leave the process.
    """

    return next(_origin_counter)


#: The identity used by anything that is not the output of a fit -- a constant
#: promoted into an expression, an analytic curve built from a known
#: cosmology. Combining with it is always allowed, because a constant is
#: correlated with everything and nothing.
CONSTANT_ORIGIN: Origin = 0


# ============================================================
# Provenance
# ============================================================

@dataclass(frozen=True, slots=True)
class Provenance:
    """
    The record of how a curve came to exist.

    Frozen: a provenance is evidence, and evidence that can be edited in
    place is not evidence.
    """

    #: The reconstructor, spelled the way it should appear in a caption --
    #: ``"GP(Matern, nu free)"``, ``"Chebyshev(order=5)"``. Empty for a
    #: derived quantity, which describes itself through :attr:`expression`
    #: and :attr:`parents` instead.
    method: str = ""

    #: The datasets that entered the fit, with their sizes:
    #: ``("CC(32)", "DESI-DR2-BAO(13)")``.
    data: tuple[str, ...] = ()

    #: Settings that would change the result if changed -- kernel, order,
    #: node count, prior ranges. Not performance knobs.
    hyperparameters: dict[str, object] = field(default_factory=dict)

    #: The seed of the draw stream, where one was used. ``None`` for an
    #: analytic result.
    seed: int | None = None

    #: How many draws stand behind the curve. Carried because a quantile is
    #: only as good as this number, and because the ensemble layer needs it
    #: to weight members.
    n_draws: int = 0

    #: The expression that produced a derived quantity, in terms of its
    #: parents: ``"(H**2 * D'**2 - c**2) / (H0**2 * D**2)"``.
    expression: str = ""

    #: The provenances this one was built from. Empty for a direct fit.
    parents: tuple["Provenance", ...] = ()

    # ---------------------------------------------------------

    @property
    def is_derived(self) -> bool:
        """Whether this came out of arithmetic rather than out of a fit."""

        return bool(self.parents)

    # ---------------------------------------------------------

    def methods(self) -> tuple[str, ...]:
        """
        Every distinct reconstructor anywhere in this curve's history, in
        first-seen order.

        A derived quantity built from two different fits has two, and the
        caller usually wants to know that before quoting a significance from
        it.
        """

        seen: dict[str, None] = {}

        def walk(p: "Provenance") -> None:

            if p.method:
                seen.setdefault(p.method, None)

            for parent in p.parents:
                walk(parent)

        walk(self)

        return tuple(seen)

    def datasets(self) -> tuple[str, ...]:
        """Every distinct dataset anywhere in this curve's history."""

        seen: dict[str, None] = {}

        def walk(p: "Provenance") -> None:

            for name in p.data:
                seen.setdefault(name, None)

            for parent in p.parents:
                walk(parent)

        walk(self)

        return tuple(seen)

    # ---------------------------------------------------------

    def derive(
        self,
        expression: str,
        *others: "Provenance",
        n_draws: int | None = None,
    ) -> "Provenance":
        """
        The provenance of a quantity built from this one and ``others``.

        Called by every arithmetic operator on ``Reconstruction``; not
        normally called by hand.
        """

        parents = (self, *others)

        return Provenance(
            method="",
            data=(),
            hyperparameters={},
            seed=self.seed,
            n_draws=self.n_draws if n_draws is None else n_draws,
            expression=expression,
            parents=parents,
        )

    def with_draws(self, n_draws: int) -> "Provenance":
        """A copy carrying a different draw count."""

        return replace(self, n_draws=n_draws)

    # ---------------------------------------------------------

    def describe(self) -> str:
        """
        One line, for a legend or a log.

        A direct fit describes itself by method and data; a derived quantity
        by its expression and the methods underneath it.
        """

        if not self.is_derived:

            data = ", ".join(self.data) if self.data else "no data"

            return f"{self.method or 'unknown method'} [{data}]"

        methods = " + ".join(self.methods()) or "unknown method"

        return f"{self.expression or 'derived'} [{methods}]"

    def __str__(self) -> str:

        return self.describe()
