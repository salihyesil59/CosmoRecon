"""
Figures.

Three defaults, chosen because they are the ones the library's argument needs
and the ones that are laborious to build by hand:

``curves.py``
    A reconstruction as a band, drawn from its quantiles rather than from
    ``mean +/- std`` -- the ratio statistics in the null tests are skewed, and
    a symmetric band misrepresents them. Shades the region outside the data's
    support differently, because a Gaussian process there is showing its prior.

``budget.py``
    The variance budget: total uncertainty split into its statistical and
    methodological halves, with each method's own mean curve overlaid. The
    figure that says how much of an error bar is a choice.

``tests.py``
    A null test against its null value, annotated with the effective degrees
    of freedom rather than the grid size -- so a reader can see immediately
    that a two-hundred-point curve carries perhaps five independent numbers.

Every figure reads its caption text from the
:class:`~CosmoRecon.core.provenance.Provenance` attached to what it is
plotting, so a plot cannot lose track of which method produced it.
"""

from __future__ import annotations


__all__: list[str] = []
