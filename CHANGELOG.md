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

- **`reconstructors.Cosmography`** — series expansions, with the failure mode
  the cosmographic literature is built on made impossible.

  - **A series must converge where it is fitted.** A cosmological distance has
    a singularity at `z = -1`, so its Taylor series in `z` has radius of
    convergence `|z| = 1` — and past it, more terms make the truncation worse.
    Supernova compilations reach `z ~ 2.3`. Fitting a `z`-series there and
    reading off a jerk parameter is not an approximation with a large error
    bar; it is not an approximation. `ConvergenceError` names the redshift at
    which the fit left the radius and points at `y = z/(1+z)`, the variable
    introduced for exactly this reason and the default here.
  - **The order is marginalised, not chosen.** The model is linear and
    Gaussian, so each order's evidence is closed-form and the order can carry
    a posterior like anything else.
  - **Derivatives to any order**, through Faà di Bruno and the partial Bell
    polynomials, so that the chain rule from `x(z)` to `z` is exact at the
    second, third and fourth derivatives cosmography is actually about.
  - **Padé** built per draw from the fitted series, so the non-linear map is
    propagated exactly. Draws whose denominator has a root in the fitted range
    are rejected and redrawn — an approximant with a pole among the data is
    not an expansion history — and the acceptance rate is part of the result.
  - Corrects a comparison the literature makes routinely: a Chebyshev series
    and a monomial series of the same degree in the same variable **span the
    same functions**. They differ here only through the coefficient prior and
    through conditioning. Tested: forced flat, the two posteriors coincide to
    the Monte-Carlo floor.

- **`CosmoRecon.data`** — four real releases, about 55 kB: 32 cosmic
  chronometers (Favale+ 2023 with the Moresco+ 2020 systematic correlation
  matrix), DESI DR2 BAO, Union3, and the Gold-2018 `f sigma_8` compilation.

  - **Every loader is validated against the survey's own published fit**, not
    against itself. Refitting flat ΛCDM gives `Omega_m = 0.297` and
    `r_d h = 101.5 Mpc` from DESI DR2 (published: `0.2975 ± 0.0086`,
    `101.54 ± 0.73`) and `Omega_m = 0.356` from Union3 (published:
    `0.356 ± 0.026`). A column read in the wrong order or a correlation matrix
    used as a covariance raises no error and produces a reasonable-looking
    curve, so the test has to be the thing that would move.
  - **A BAO release is not one function of redshift.** `D_M/r_d` and
    `D_H/r_d` are measured together and correlated, so
    `MultiObservableDataset` refuses to be fitted whole and `.select()` gives
    one observable with its own covariance block — stating, in the result's
    note, the cross-covariance that choice gives up.
  - **Overlapping compilations cannot be combined.** Union3 and Pantheon+ are
    built from the same supernovae; using both counts them twice and narrows
    the interval without adding information. `check_combination` refuses.
  - Covariances are used as published — the chronometer correlation matrix in
    full, the growth compilation's two internally correlated blocks laid over
    the diagonal exactly as the reference likelihood does. Nothing is invented
    and nothing is discarded.
  - Pantheon+ (33 MB) and DES-SN5YR (6 MB) are deliberately not bundled.

- **The variance budget is now a measurement on real data.** Five members —
  two Gaussian processes, two Chebyshev series in different variables, one
  Padé. On 32 cosmic chronometers, method variance is 11% of the total on
  average and 27% at `z = 1.7`; on DESI DR2 `D_M/r_d`, 12% and 25% at
  `z = 1.5`. Error bars inflate by a median factor of 1.04 and 1.06. That
  share does not shrink with more data.

- **First real null test.** `Om(z)` from the 32 chronometers is `0.00 σ` from
  constant, determined to about `± 0.09`. Chronometers alone cannot see a
  DESI-scale deviation from a cosmological constant, which is the correct
  answer and the one a tool measuring its own assumptions would not give.

- Test suite (144 tests) covering the core contract, the kernels, the GP,
  cosmography and the data layer: sample paths checked against the exact GP posterior to the
  Monte-Carlo floor, empirical coverage of the 68% interval over repeated
  realisations, each kernel's covariance against its spectral density, Faà di
  Bruno against finite differences at three orders and against the chain rule
  written out by hand at low order, and the determinism clause that `at()`
  returns the same realisation on every grid.

- `ARCHITECTURE.md`, documenting the single design decision the rest of the
  library follows from.
