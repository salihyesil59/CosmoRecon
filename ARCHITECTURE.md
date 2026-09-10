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
| `om.py` | `Om(z)`, `Om3(z)` (Sahni–Shafieloo–Starobinsky) | constant |
| `curvature.py` | `Ok(z)` (Clarkson–Bassett–Lu) | constant, `= Omega_k` |
| `duality.py` | Etherington `eta(z) = d_L / [(1+z)^2 d_A]`; cosmic opacity | `1` |
| `litmus.py` | `L(z)` litmus test for `Lambda` | `0` |
| `growth.py` | growth–geometry consistency: does measured `f sigma_8` match the growth *implied by* the reconstructed geometry under GR? | `0` |
| `isotropy.py` | the cosmological principle, from BAO across the sky | `0` |

`growth.py` is the S8 tension restated without a model — the tension becomes a
statement about internal consistency rather than a disagreement between two
ΛCDM fits.

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
rather than a measurement. `ensemble.significance(test)` reports a null test
before and after method marginalisation — the *"3.1 sigma becomes X sigma"*
statement.

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
