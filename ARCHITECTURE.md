# CosmoRecon — architecture

> The design document. What the library is, the one decision the rest of it
> follows from, and the contract each layer owes the next.

---

## 1. What this library is for

CosmoFit answers *"which model fits the data?"* — it assumes a parametrisation
and returns a posterior over its parameters.

CosmoRecon answers the two questions that come before and after that one:

1. **Before.** *What do the data say about `H(z)`, `d_L(z)`, `f sigma_8(z)`
   with no model assumed at all?* — non-parametric reconstruction.
2. **After.** *Is the concordance model internally consistent with what the
   data say, and how much of the answer is an artefact of the reconstruction
   method?* — null tests, and the method-variance budget.

The second half of question 2 is the reason this library exists. The published
state of the art is that kernel choice in a Gaussian-process reconstruction
"introduces systematic variations comparable to those arising from different
cosmological models" (Gen GP, arXiv:2510.03742; EPJC 2025 on kernel
dependence). Every group re-implements one method, per paper, and reports the
number that method gives. Nobody reports the spread across methods, because no
tool produces it. That number is this library's headline output.

---

## 2. The one design decision

**A reconstruction is a posterior over functions, carried as a shared sample
axis — not a mean with error bars.**

Everything else is a consequence.

The naive representation (`z`, `y`, `sigma_y`) breaks the moment a null test
needs more than one quantity at once. The curvature test

```
Ok(z) = [ H(z)^2 D'(z)^2 - c^2 ] / [ H_0^2 D(z)^2 ]
```

needs `H`, `D` and `D'` *jointly*. They are strongly correlated — `D'` is a
functional of `H` in any reconstruction that used the same data — and
propagating diagonal errors through that expression gives a confidence
interval that is simply wrong. This is the most common defect in the
null-test literature.

So the core object holds **draws**, indexed on a common sample axis, and every
downstream operation is applied *per draw*:

```python
Ok = (H**2 * D.d(1)**2 - C_LIGHT**2) / (H0**2 * D**2)
```

Correlations, non-Gaussianity and the derivative covariance all propagate
exactly, for free, because they were never thrown away. Uncertainty
propagation stops being something the user has to get right and becomes a
property of the data structure.

Two corollaries follow immediately, and both are enforced in code:

- **Draws from one fit are aligned.** Draw `k` of `H` and draw `k` of `D` come
  from the same underlying realisation. Arithmetic between them is therefore
  correct by construction.
- **Draws from different fits are not.** Combining them silently would invent
  a correlation structure nobody chose. `Reconstruction` refuses, and makes
  you write `assume_independent()` — which is a physical claim, so it should
  look like one in the source.

---

## 3. Layer map

```
                    data/            bridges/
              (CC, BAO, SN,        (optional CosmoFit
               fs8, time delays)    adapter)
                        |
                        v
   reconstructors/  ---------->  core.Reconstruction  <---- the spine
   GP | cosmography |            posterior over f(z),
   nodal | PCA |                 differentiable, composable,
   ANN | symbolic                provenance-carrying
                        |
          +-------------+--------------+---------------+
          v             v              v               v
   consistency/     inverse/       ensemble/       validation/
   Om, Ok, DDR,   w(z), V(phi),   method-variance  injection-
   litmus,        f(R)/f(T)/f(Q), budget,          recovery,
   growth-        mu(z)=Geff/G    significance     coverage
   geometry                       deflation
                        |
                        v
                     plots/
```

Dependency runs strictly downward, to `core`. `consistency/`, `inverse/` and
`ensemble/` speak only `Reconstruction`; none of them knows which
reconstructor produced it. That is what makes a null test method-agnostic, and
what makes the ensemble layer possible at all.

---

## 4. `core.Reconstruction`

The spine. A `Reconstruction` is:

| field | meaning |
|---|---|
| `predictor` | callable `(z, derivative=n) -> (n_draws, n_z)` |
| `z` | the default evaluation grid |
| `origin` | identity of the fit these draws came from |
| `provenance` | method, data, hyperparameters, seed, expression history |
| `label` / `unit` | for plots, and for unit sanity checks |

Its public surface:

```python
r.at(z_new)            # re-evaluate on another grid; the fix for redshift mismatch
r.d(n)                 # n-th derivative, ANALYTIC where the method provides one
r.mean(), r.std()      # summary, never the representation
r.quantile(q)          # non-Gaussian intervals come out right
r.draws                # (n_draws, n_z) — the representation
r + - * / ** np.log …  # elementwise on the draw axis, origin-checked
```

Three points that are architecture, not convenience:

**Re-evaluability.** `at()` exists because the distance-duality test needs
`d_L` and `d_A` at the *same* redshifts and no two datasets share a redshift
grid. Every DDR paper solves this by hand; here it is a method call, and the
mismatch cannot be forgotten.

**Analytic derivatives.** A GP's derivative is another GP with kernel
`d^2 k / dz dz'`; a Chebyshev expansion differentiates term by term; a spline
differentiates exactly. So `d(n)` asks the *reconstructor* for the derivative,
and falls back to finite differencing only when the method genuinely has none
— and then it says so, loudly. Numerically differentiating a posterior mean
and calling the result a reconstruction of `H'(z)` is the second most common
defect in this literature; the API is shaped so the correct path is the
default one.

**Provenance survives arithmetic.** The `Ok` above carries a record of which
reconstructor, which kernel, which data, which seed, and the expression that
built it. In a library whose thesis is *the answer depends on the method*,
provenance is part of the result, not logging.

### Two kinds of reconstruction

Draws are the universal representation, but they are not always the whole of
what is held:

- **From a fit.** Carries a `predictor` — a function-space posterior evaluable
  at any redshift and any derivative order — so `at()` and analytic `d()`
  work. Draws are materialised from it on demand and cached. A method with
  closed-form moments may additionally attach `(mean, cov)`, used to answer
  `mean()` and `cov()` without Monte-Carlo noise.
- **From arithmetic.** Carries draws on a fixed grid and nothing else, because
  `H**2 * D.d(1)**2` was applied to samples, not to functions. `at()` and
  analytic `d()` raise, naming the fix: move the call to the operands.

The degradation is automatic and one-way. Users never choose.

---

## 5. `reconstructors/` — the contract

```python
class Reconstructor(ABC):
    provides_evidence: bool
    def fit(self, data) -> ReconstructionSet: ...
    @property
    def log_evidence(self) -> float: ...   # raises where undefined
```

A `ReconstructionSet` is the several functions one fit produces (`H`, `D`,
`mu`, …) sharing one `origin`, hence mutually aligned.

`log_evidence` raises rather than returning `None` or `nan`: an
evidence-weighted ensemble that silently dropped a member would report a
spread across fewer methods than the caller asked for, which is exactly the
number they came for. `provides_evidence` is what to check.

Members:

| module | method | derivative | evidence |
|---|---|---|---|
| `gp.py` ✅ | Gaussian process; SqExp, Matérn(ν), RQ, Cauchy, **ν free** | analytic | marginal likelihood |
| `cosmography.py` ✅ | Chebyshev/monomial in z, y or ln(1+z); Padé(m,n); **order free** | analytic | closed form |
| `nodal.py` | flexknot / nodal spline, node count sampled | analytic | nested sampling |
| `pca.py` | PCA and binned `w(z)` with a correlation prior | analytic | yes |
| `ann.py` | neural reconstruction (REFANN-style) | autodiff | no |
| `symbolic.py` | genetic-algorithm symbolic regression (GAME-style) | analytic | no |

Three commitments the incumbent (GaPP, 2012) does not make:

- **Hyperparameters are marginalised, not optimised.** Fixing the kernel
  amplitude and length scale at their maximum-likelihood values manufactures
  precision the data do not contain. The GP module carries a posterior over
  them and draws each path at hyperparameters drawn from it.
- **`nu` is a free parameter.** Matérn smoothness is a modelling choice
  currently made by hand and then not reported; here it is inferred.
- **A derivative that does not exist is refused.** A Matérn-ν process is
  differentiable `ceil(ν) - 1` times, so a Matérn-3/2 kernel — the commonest
  fixed choice — supports `H'(z)` and not `H''(z)`. Numbers can always be
  produced, because any finite representation of a path is smooth; they are
  not a posterior on anything. `d()` raises, reports what fraction of the
  posterior *does* support the order, and names the prior restriction that
  would buy it.

And two that `cosmography.py` adds:

- **A series is refused outside its radius of convergence.** A cosmological
  distance is singular at `z = -1`, so a Taylor series in `z` converges only
  for `|z| < 1`, and past that radius more terms make the truncation worse.
  Supernovae reach `z ≈ 2.3`. `ConvergenceError` is raised at fit time, not
  quietly absorbed into a wide interval — this is a different category from
  `ExtrapolationWarning`, which is about leaving the *data*, not about leaving
  the region where the model means anything.
- **The order is marginalised.** The model is linear-Gaussian, so each order's
  evidence is closed form and the order carries a posterior like anything else.

One correction the module makes to standard practice: a Chebyshev series and a
monomial series of the same degree in the same variable **span the same
functions**. Listing "Taylor, Chebyshev and Padé" as three comparable methods
double-counts one model. They differ here only through the coefficient prior
and through conditioning; forced flat, the two posteriors coincide, and the
test suite checks it.

The sample-path basis is a **quadrature** on the kernel's spectral density,
not a sample from it. Random Fourier features — the standard construction —
converge as `1/√M`; measured against the exact posterior, 8192 random features
still misstate the width of the band by about 5%. Since the width *is* this
library's output, that is not acceptable. In one dimension the spectral
density is a one-dimensional integral, and quadrature on it converges
geometrically: 256–2048 nodes reach a part in 10⁴, and the achieved error is
measured at fit time and refused if it misses.

### Joint fits: several correlated observables at once

A BAO release is not one function of redshift. DESI DR2 measures `D_M/r_d` and
`D_H/r_d` together at each tracer redshift, correlated at `r = −0.35` to
`−0.49` — measured, not remembered; the covariance is exactly six 2×2 blocks
with nothing across redshifts. The curvature test and distance duality are
built from *both*, so they are not defined unless that correlation survives
into the reconstruction. Hence `Reconstructor.supports_joint`, and a
`MultiObservableDataset` that refuses to be fitted as one curve.

**The design decision is about what the joint prior must not do.** In FLRW the
two functions are related by

```
d/dz (D_M) = D_H · sqrt(1 + Ω_k (H₀ D_M / c)²)
```

which *is* the Clarkson–Bassett–Lu curvature test. A joint prior that linked
them — however physically motivated it looked — would make that null test
vacuous: it would be testing an assumption it had already made. This is easy to
get wrong in a way that leaves no trace in the output.

So: **independent priors, joint likelihood.** Each observable gets its own
expansion variable map, its own column normalisation, its own order grid and
its own prior width, built from its own measurements and nothing else. The fit
stacks them into one design matrix against the full covariance, and one draw of
the stacked coefficient vector produces both curves — sharing an `origin`, so
everything downstream carries their correlation exactly.

Every correlation in the posterior therefore arrives from the data. That is
tested rather than asserted: forcing the data covariance diagonal drives the
posterior correlation between the two reconstructed functions to `−0.005` to
`+0.022`, zero to Monte-Carlo precision, while the real covariance gives
`−0.31` to `−0.54`.

`Cosmography` supports joint fits; the Gaussian process does not yet, because
its hyperparameter grid would become a product over two functions' kernels and
needs a different sampling strategy.

---

## 6. `consistency/` — the null tests

Each test consumes `Reconstruction`s and returns a `TestResult`: the
`z`-dependent statistic as a `Reconstruction`, the null value it is tested
against, and a global significance computed with the **full covariance across
`z`** and an effective number of degrees of freedom taken from the
eigenspectrum. Adjacent redshifts in a smooth reconstruction are not
independent, and a naive chi-square over a fine grid will report a detection
that is entirely correlation.

| module | test | null |
|---|---|---|
| `om.py` ✅ | `Om(z)` | constant (`= Omega_m`) |
| `om.py` ✅ | `Om3(z1,z2,z3)` — no `H0`, no `Omega_m`, no extrapolation | **exactly 1** |
| `curvature.py` ✅ | `Ok(z)` (Clarkson–Bassett–Lu) — needs the joint fit | constant, `= Omega_k` |
| `duality.py` ✅ | Etherington `eta(z) = d_L / [(1+z) D_M]`; opacity slope `epsilon` — two datasets, independence declared | constant (`= 1` with a calibration) |
| `litmus.py` | `L(z)` litmus test for `Lambda` | `0` |
| `growth.py` | growth–geometry consistency: does measured `f sigma_8` match the growth *implied by* the reconstructed geometry under GR? | `0` |
| `isotropy.py` | the cosmological principle, from BAO across the sky | `0` |

`growth.py` is the S8 tension restated without a model — the tension becomes a
statement about internal consistency rather than a disagreement between two
ΛCDM fits.

### Two datasets in one test

The curvature test takes two observables from **one** release, and a joint fit
keeps their correlation. Distance duality takes a luminosity distance from
supernovae and a transverse distance from BAO — two releases, two fits, two
realisation streams — and pairing draw `k` of one with draw `k` of the other is
only meaningful if the fits share nothing. So the claim is made in the source,
at one of two scopes:

- `Reconstruction.assume_independent()` for one curve. The claim survives
  arithmetic on its own side: `(1 + z) * D.assume_independent()` is still
  independent of whatever it meets, because it contains no data `D` did not.
- `combine_independent(fit_a, fit_b)` for every function two fits produced.
  The result is an ordinary `ReconstructionSet` on one realisation index —
  the first fit's draw order kept, the second's permuted by a fixed,
  seed-derived permutation — which a null test, an ensemble or a later `at()`
  uses without knowing it was ever two fits. It refuses a dataset that appears
  in both fits, an observable that appears in both, and supports that do not
  overlap.

The permutation is not a formality. Two fits drawn with the same seed reuse the
same random numbers, and index-pairing them correlates two posteriors that share
nothing — `r > 0.8` in the test that checks it, `|r| < 0.2` after combination.

**The observable matters as much as the pairing.** A distance modulus goes as
`5 log10 z` near the origin, a singularity no series represents and no
stationary kernel expects. Reconstructed directly, it biases the ratio at high
redshift enough that a third-order series reports a duality violation at
21 sigma in every mock universe where duality is exact. `data.reduced_modulus`
subtracts `5 log10 z` from each measurement — exact, covariance unchanged — and
the same series then reports 0.6 sigma. `Duality` takes the reduced modulus and
refuses `mu`.

---

## 7. `inverse/` — from a function back to a theory

This is where CosmoRecon meets `wljs-gr-toolkit` from the other side. The
toolkit goes action → field equations → `E(z)`. This layer goes back:

| module | map |
|---|---|
| `dark_energy.py` | `H(z) -> w(z), rho_de(z), Omega_de(z)` |
| `quintessence.py` | `H(z) -> V(phi), phi(z)` |
| `designer.py` | `H(z) -> f(R)`, `f(T)`, `f(Q)` designer reconstruction |
| `coupling.py` | `{H, f sigma_8} -> mu(z) = G_eff / G` |

Every one of these is currently done by hand, per paper, on top of GaPP. Each
is a few lines once the input is a differentiable posterior over functions —
which is the point of section 2.

---

## 8. `ensemble/` — the thesis, in code

```python
budget = MethodEnsemble(
    [GaussianProcess(kernel="matern-free"), Chebyshev(order=5),
     FlexKnot(), PCA(n=6)],
    weights="evidence",         # or "equal"
).fit(data).budget
```

The law of total variance splits the answer:

```
Var_total(z) = E_method[ Var_within(z) ]  +  Var_method[ E_within(z) ]
                    statistical                 methodological
```

`budget.method_fraction()` is the fraction of the error bar that is a choice
rather than a measurement. `fit.significance(statistic, null)` reports a null
test under each member and under the mixture — the *"3.1 sigma becomes X
sigma"* statement.

The mixture is a full `Reconstruction`, not a summary: each pooled draw still
knows which member's function it is, so it regrids and differentiates, and it
is only as differentiable as its roughest member.

**Two datasets, one choice of method.** `sn_fit.with_independent(bao_fit)`
pairs member `m` on one dataset with member `m` on the other, since an analyst
who reconstructs the supernovae with a Matérn process does the same to the BAO
distances. The pooled posterior is then a mixture of those *pairs*, sharing one
member index per draw across both observables — not the product of two
separate mixtures, which would mostly pair a Gaussian process on one side with
a polynomial on the other and describe analyses nobody runs. Weights multiply,
as evidences of independent data do.

**Why this is not bookkeeping.** A rigid method's posterior covers the
uncertainty in its own parameters, not the error it makes by being the wrong
shape — so it can be confidently wrong, and nothing inside a single-method
analysis can tell. Measured over 24 ΛCDM realisations, a Padé[2/1] fit's
nominal 68% interval covers 44.6% of the time. Pooling across methods supplies
exactly the missing term and brings coverage back to 70.8%.

Evidence weighting is offered; equal weighting is the default. Bayesian
evidence is comparable only across methods that are genuinely competing models
of the same function, and some members (ANN, symbolic regression) have no
evidence at all. The library will not quietly pretend otherwise.

---

## 9. `validation/` — the acceptance bar

**Anything that can be checked two ways is checked two ways**, and every
reconstructor must earn its place:

- **Injection–recovery.** Generate mock CC/BAO/SN at a survey's real error
  budget from a known `E(z)` (ΛCDM, CPL, an oscillating `w`), reconstruct, and
  require recovery of the truth within the stated interval.
- **Coverage.** Over many realisations, the 68% interval must contain the
  truth 68% of the time. A method that reports 68% and covers 45% is
  manufacturing significance, which is exactly what this library exists to
  catch. Coverage is a *test*, not a figure.
- **Cross-checks.** GP derivative against finite differences of the same draw;
  Chebyshev against Taylor inside the radius of convergence; the whole chain
  against a `Reconstruction` built analytically from a known cosmology.

### Calibrating a null test

The coverage rule has a counterpart for null tests: **a test that reports
`p = 0.05` must be wrong 5% of the time when the null is true.** The nominal
chi-square in `consistency.significance` fails it in both directions, and the
diagnosis matters because it decides the repair.

- A Gaussian process's posterior variance exceeds the variance of its posterior
  mean across data realisations by a median factor of 120 to 1400 in the
  directions its prior dominates. Those directions are counted as degrees of
  freedom, and the test almost never rejects.
- A free-order series is **biased** under the null, by two to four of its own
  standard deviations in some directions. Given the *true* sampling covariance,
  measured over 300 independent mocks, the bias alone takes it from 6–9% false
  positives to 64–100%. A bootstrap estimate of the sampling covariance around
  the fit fixes neither problem, because bias is not variance.

`validation.calibrate` therefore has two parts.

1. **The reference distribution.** A null model (`validation.nulls`) is fitted
   to the same data. Mocks are drawn from its parameter posterior with the
   released covariances, the whole analysis is rerun on each through `refit`,
   and the p-value is read off where the data fall among the mocks.
2. **The ordering.** Something has to rank realisations, and the nominal
   chi-square must not. Its largest terms sit where the posterior is narrowest,
   which for a biased method is where the bias lives. Used as the ordering, it
   gives the right size and almost no power: 0–15% detection of a `w = −0.6`
   universe whose oracle non-centrality against the best-fitting ΛCDM is 44.
   The ordering used instead is the Mahalanobis distance from **the null mocks'
   own mean, in their own covariance**, with each mock scored leave-one-out.
   The mean removes the method's bias at the null, and the covariance weighs
   each direction by how much the estimate actually moves. On the same
   universes it gives 90–100% detection and 0–5% false positives.

The model dependence is stated rather than hidden: a calibrated significance is
the significance at the fitted null model, and the result carries that model's
name and parameters. The floor is stated too: `n` mocks cannot report a p-value
below `1/(n+1)`, and a result at the floor is a bound.

---

## 10. What this library deliberately does not do

- **No sampler of its own for parametric cosmologies.** That is CosmoFit.
- **No Boltzmann code.** CAMB and CLASS exist and are excellent.
- **No tension metrics between posteriors.** `tensiometer` and `unimpeded`
  cover it. The growth–geometry null test is a different question, and stays.
- **No Fisher forecasting.** `cosmicfishpie` covers it.

Scope discipline is what keeps the library reviewable.

---

## 11. Dependencies

Core is `numpy`, `scipy`, `matplotlib` and nothing else. Everything with a
compiled or heavy dependency is an extra, guarded at import:

| extra | pulls | needed by |
|---|---|---|
| `nested` | `dynesty` | `nodal.py` node-count sampling, evidences |
| `mcmc` | `emcee` | GP hyperparameter marginalisation (alternative path) |
| `ann` | `torch` | `ann.py` |
| `cosmofit` | `cosmofit` | `bridges/cosmofit.py` |
| `docs`, `dev` | sphinx/furo/myst, pytest/ruff/mypy | — |

The test suite asserts that the optional paths really do skip when the extra
is absent, so a bare install is a tested configuration rather than a hope.

---

## 12. Directory map

```
src/CosmoRecon/
    typing.py            Array, Redshift, PathLike — the public vocabulary
    core/
        reconstruction.py  Reconstruction, backends, arithmetic
        provenance.py      Provenance, expression history
        grid.py            grid construction, alignment, regridding
        errors.py          the exception hierarchy
        constants.py       c, and nothing that belongs to a model
    reconstructors/
        base.py            Reconstructor ABC, ReconstructionSet
        kernels.py  gp.py  cosmography.py  nodal.py  pca.py  ann.py  symbolic.py
    consistency/
        base.py            NullTest ABC, TestResult, effective d.o.f.
        om.py  curvature.py  duality.py  litmus.py  growth.py  isotropy.py
    inverse/
        dark_energy.py  quintessence.py  designer.py  coupling.py
    ensemble/
        method.py          MethodEnsemble
        budget.py          VarianceBudget, law of total variance
    data/
        dataset.py  loader.py   + bundled CC / DESI DR2 / Union3 / growth
    bridges/
        cosmofit.py        optional adapter, guarded import
    validation/
        mocks.py  coverage.py
    plots/
        curves.py  budget.py  tests.py
```
