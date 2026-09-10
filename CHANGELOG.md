# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`core.Reconstruction`** — the object the rest of the library is written
  against: a posterior over a function of redshift, carried as draws sharing a
  realisation index. Arithmetic, numpy ufunc routing, quantiles from the draws
  rather than from a Gaussian assumption, and analytic derivatives asked of the
  fitting method rather than finite-differenced behind the user's back.

  - **Draw alignment.** Reconstructions from one fit share an `Origin` and
    combine freely; reconstructions from different fits raise `AlignmentError`
    until `assume_independent()` is called. Independent members have their
    draws permuted before combination, so no accident of ordering can leak in
    as a correlation.
  - **Scalar broadcasting.** A one-point reconstruction — `H.at(0.0)`, a
    curvature, a growth normalisation — broadcasts across a grid. This is what
    makes `E = H / H.at(0.0)` cancel the `H0` uncertainty exactly, draw by
    draw, instead of adding to it.
  - **Re-evaluability.** `at()` puts two reconstructions built from datasets
    with different redshifts onto one grid, which is the redshift-mismatch
    problem every distance-duality analysis has to solve. Derived quantities
    carry draws rather than a function and say so (`NotResamplableError`)
    rather than interpolating silently.

- **`core.Provenance`** — method, data, hyperparameters, seed and expression
  history, composed through arithmetic rather than dropped. `Om(z)` built from
  `H` records the kernel, the dataset and the expression that produced it.

- **`core.grid`** — grid construction, support tracking, and
  `ExtrapolationWarning` for a reconstruction evaluated past the data that
  constrain it. Promoted to an error inside the test suite.

- **`consistency.significance`** — chi-square in the eigenbasis of the
  statistic's own covariance, with **effective** degrees of freedom from the
  resolved eigenspectrum, a Hartlap correction for the Monte-Carlo covariance,
  and an explicit refusal when the draws cannot support the number of modes.
  The dof is the rank of the statistic's own covariance, measured rather than
  taken from the grid size — a posterior with three shape parameters resolves
  three or four modes however finely it is sampled, while a Gaussian-process
  posterior resolves many more. A naive chi-square over a fine grid assumes the
  grid and reports several sigma of deviation from data containing none.

- **`ensemble.VarianceBudget`** — the law of total variance applied across
  reconstruction methods, splitting an error bar into its statistical and
  methodological halves. `method_fraction()` is the fraction that is a choice
  rather than a measurement.

- **`reconstructors.Reconstructor`** and **`consistency.NullTest`** — the two
  abstract contracts, with the bookkeeping every implementation must get right
  (seeding, support, provenance) in the base class rather than in each member.

- **`reconstructors.GaussianProcess`** — the first method, and the one the
  field runs on, with the habits that make its published intervals too narrow
  removed.

  - **Hyperparameters are marginalised, not optimised.** The standard recipe
    maximises the marginal likelihood over amplitude and length scale and
    reconstructs at those values. On mock chronometer data, marginalising them
    instead widens the interval by 16% at the median and 28% at worst — real
    uncertainty the standard recipe discards.
  - **Matérn `nu` is inferred, not chosen.** Free by default; `Matern(nu=2.5)`
    pins it, and `Matern(nu=[...])` restricts the grid.
  - **A derivative the posterior cannot support is refused.** A Matérn-`nu`
    process is differentiable `ceil(nu) - 1` times, so a Matérn-3/2 kernel has
    no second derivative and nothing cosmographic exists under it. `d()` raises,
    reports what fraction of the posterior *does* support the order, and names
    the prior restriction that would buy it. No other GP reconstruction code
    makes this check.
  - **Sample paths come from a spectral quadrature, not random features.**
    Random Fourier features converge as `1/sqrt(M)`: measured against the exact
    posterior, 8192 of them still misstate the width of the band by ~5%. Since
    the width is this library's output, the basis is instead a quadrature on
    the spectral density, which converges geometrically in one dimension,
    reaches a part in 10⁴ with a few hundred nodes, and **measures its own
    error at fit time** rather than assuming it. Kernels whose spectral tail is
    too heavy to represent (Matérn `nu <= 1/2`) are refused rather than
    approximated quietly.
  - Four kernels — squared exponential, Matérn, rational quadratic, Cauchy —
    each declaring a covariance and a spectral density as independent code, so
    that comparing them is a real cross-check. The rational quadratic's
    spectral density is derived as its Matérn Fourier dual.

- Test suite (82 tests) covering the core contract, the kernels, and the GP:
  sample paths checked against the exact GP posterior to the Monte-Carlo floor,
  empirical coverage of the 68% interval over repeated realisations, the
  analytic derivative against finite differences, and the determinism clause
  that `at()` returns the same realisation on every grid.

- `ARCHITECTURE.md`, documenting the single design decision the rest of the
  library follows from.
