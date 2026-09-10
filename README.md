# CosmoRecon

> **Model-independent reconstruction and null tests of the cosmological framework**

CosmoRecon asks what the data say about `H(z)`, `d_L(z)` and `f sigma_8(z)`
with no cosmological model assumed — and then asks the question the field
currently cannot answer: **how much of that result is the method rather than
the measurement?**

> **Status: pre-alpha.** The core — the object everything else is written
> against, its guarantees, and the significance machinery — is implemented and
> tested, and so are the first two reconstruction methods, which is enough for
> the variance budget below to be a real measurement rather than a plan. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the design, and the roadmap below for
> the order of work.

---

## The problem this exists to solve

Since DESI DR2, the central question in cosmology is whether dark energy
evolves. The evidence is model-independent reconstruction: fit `H(z)` without
assuming a parametrisation and look at what comes out.

Three things are true about how that is currently done.

1. **Nearly every paper depends on one package.** GaPP (Seikel, Clarkson &
   Smith 2012) is Python-2-era, with an unofficial Python 3 fork. Gaussian
   process reconstructions of `f(R,T)`, of quintessence potentials, of `w(z)` —
   all of them run on it.
2. **Everything else is re-implemented per paper.** Padé and Chebyshev
   cosmography, flexknots, PCA, redshift binning, neural reconstruction,
   symbolic regression. No shared interface, no shared conventions, no
   cross-method benchmark.
3. **The method is known to drive the answer, and nobody quotes the spread.**
   Kernel choice in a GP reconstruction produces "systematic variations
   comparable to those arising from different cosmological models"
   ([arXiv:2510.03742](https://arxiv.org/abs/2510.03742); see also the
   kernel-dependence analysis in EPJC 2025). The response has been papers
   *about* method dependence rather than results marginalised *over* it —
   because no tool produces the number.

CosmoRecon produces the number.

---

## The design, in one idea

**A reconstruction is a posterior over functions, carried as draws — not a
curve with error bars.**

That sounds like a detail. It is the whole library. Consider the
Clarkson–Bassett–Lu curvature test:

```
Ok(z) = [ H(z)^2 D'(z)^2 - c^2 ] / [ H_0^2 D(z)^2 ]
```

`H`, `D` and `D'` are strongly correlated — `D'` is a functional of `H` in any
reconstruction built from the same data. Propagate pointwise error bars through
that expression and the interval is simply wrong, which is the most common
defect in the null-test literature.

Here, every operation is applied per draw:

```python
Ok = (H**2 * D.d(1)**2 - C_LIGHT_KM_S**2) / (H0**2 * D**2)
```

The correlations, the non-Gaussianity of the ratio and the covariance between
redshifts all survive, because none of them was ever summarised away. Correct
uncertainty propagation stops being something the user has to get right and
becomes a property of the data structure.

### It is checkable in three lines

`Om(z) = [E(z)^2 - 1] / [(1+z)^3 - 1]` is exactly `Omega_m` in flat ΛCDM. Feed
the machinery a ΛCDM posterior with a 1.4% uncertainty on `H0` and 6.7% on
`Omega_m`:

```python
E  = H / H.at(0.0)
Om = (E**2 - 1.0) / ((1.0 + z)**3 - 1.0)

Om.mean()   # 0.30033 at every redshift, flat to 1e-12
Om.std()    # 0.02014 at every redshift — exactly the input Omega_m width
```

The `H0` uncertainty cancels draw by draw between `H(z)` and `H(0)`, leaving
only the `Omega_m` width. Nothing was assumed, nothing was linearised, and no
covariance matrix was passed by hand. This is a test in the suite.

### And the guardrails are real

```python
H_desi + H_pantheon          # AlignmentError: different fits, no independence claim
H + H.at(other_grid)         # GridMismatchError: put them on one grid first
(H / H0).d(1)                # DerivativeUnavailableError: pass numerical=True to opt in
```

Each of those is a mistake that appears in published analyses. None of them can
be made here without saying so out loud.

---

## The headline output

The significance machinery computes chi-square in the eigenbasis of the
statistic's own covariance and reports the **effective** degrees of freedom —
the rank the posterior actually has, measured, rather than the number of grid
points it was drawn on. Dividing by a pointwise error bar and counting the
grid instead will find a detection whatever the data say.

Injecting a known universe into a 60-point `Om(z)` test:

| injected truth | chi² | eff. dof | significance |
|---|---|---|---|
| flat ΛCDM (exact) | 0.00 | 0 | 0.00 σ |
| CPL `w0=-0.95, wa=-0.2` | 1.63 | 4 | 0.25 σ |
| CPL `w0=-0.9, wa=-0.6` | 28.66 | 4 | 4.44 σ |
| CPL `w0=-0.8, wa=-0.8` | 127.06 | 4 | 10.65 σ |

Read the dof column as a check on the machinery, not as a general claim about
smoothness. This injected posterior has exactly three parameters that change
the *shape* of `Om(z)` — `Omega_m`, `w0`, `wa`, with `H0` cancelling — and the
eigenvalue spectrum duly falls off a cliff after the third (1, 8×10⁻², 7×10⁻³,
then 10⁻⁵ and below). The routine recovers the dimensionality it was given.
A Gaussian-process posterior is genuinely high-dimensional and reports a much
larger dof on the same grid, which is equally correct and equally computed.

And the exact ΛCDM row is not a rounding: `Om(z)` there is constant to machine
precision, so after projecting out the constant there is no resolved direction
left and the routine says so instead of dividing by rounding error.

### And the method itself is measured

With more than one method implemented, `ensemble/` splits the answer by the law
of total variance:

```
Var_total(z) = E_method[ Var_within(z) ]  +  Var_method[ E_within(z) ]
                    statistical                 methodological
```

Five methods — two Gaussian processes with different kernels, two Chebyshev
series in different expansion variables, one Padé — on the **real bundled
data**:

| dataset | method share, mean | peak | error bars inflate by |
|---|---|---|---|
| 32 cosmic chronometers → `H(z)` | 11% | 27% at `z = 1.7` | 1.04 (median) |
| DESI DR2 → `D_M/r_d` | 12% | 25% at `z = 1.5` | 1.06 (median) |

Roughly one part in eight of the quoted uncertainty is which method was
picked — rising to a quarter where the data thin out — and that part does not
shrink when more data arrive. `budget.method_fraction()` is that number. It is
not large enough to overturn anything on its own; it is large enough that
nobody should be quoting an interval without it, and at present nobody can.

```python
fit = MethodEnsemble([
    GaussianProcess(kernel="matern"),
    GaussianProcess(kernel="squared_exponential"),
    Cosmography("y"), Cosmography("log"), Cosmography(pade=(2, 1)),
]).fit(chronometers())

fit.budget()                     # the split above
fit.marginalised                 # a full Reconstruction, not a summary
fit.significance(statistic, 0.0) # the same test under each method, and pooled
```

### Why it is not just bookkeeping

Run every method over 24 realisations of ΛCDM chronometers and count how often
its nominal 68% interval actually contains the truth:

| method | coverage of its own 68% interval |
|---|---|
| Padé[2/1] | **44.6%** |
| Chebyshev in `y` | 58.8% |
| Chebyshev in `ln(1+z)` | 62.5% |
| GP, Matérn with `nu` free | 70.4% |
| GP, squared exponential | 70.4% |
| **method-marginalised** | **70.8%** |

The Padé result is not a bug in the fit — the posterior is exactly right for
that model. The model is wrong: a three-parameter rational function cannot
contain a ΛCDM expansion history, and its posterior covers the uncertainty in
its coefficients, not the error it makes by being the wrong shape. Nothing
inside a single-method analysis can see that. It reports 68% and delivers 45%.

The between-method scatter *is* the missing term, so pooling restores the
calibration. That is the argument for the whole library, and it is measured
rather than asserted — `tests/test_ensemble.py` runs it.

Run `examples/02_real_data.py` to reproduce the table, and
`examples/01_expansion_history.py` for the same machinery on mocks, where a
known truth makes the recovery checkable.

The same example also runs the `Om(z)` null test on the real chronometers, and
the answer is worth stating plainly: **0.00 σ from constant**, with `Om(z)`
determined to about `± 0.09`. Thirty-two differential ages cannot see a
DESI-scale deviation from a cosmological constant, and a tool that said
otherwise would be measuring its own assumptions. Getting that answer is what
makes the machinery worth pointing at data that can.

---

## The first null tests

`consistency/om.py` implements the Sahni–Shafieloo–Starobinsky diagnostics.
Both are one line of arithmetic on a reconstruction, and both are exact
statements rather than approximations:

```
Om(z)            = [E²(z) − 1] / [(1+z)³ − 1]           = Ω_m   in flat ΛCDM
Om3(z₁, z₂, z₃)  = Om(z₂; z₁) / Om(z₃; z₁)             = 1     in flat ΛCDM
```

Fed an exact ΛCDM posterior, `Om3` comes back as **1.0000000 ± 8×10⁻¹⁶** —
not on average, but in every draw, because the cancellation is algebraic and
happens *inside* each realisation. A machinery that summarised before
combining would get the mean right and the width wrong.

**Om3 is the better test on real data, and for a reason worth knowing.** `Om`
is anchored at `z = 0`, so it needs `H(0)`; the lowest cosmic chronometer sits
at `z = 0.07`, so every `Om(z)` from that dataset is standing on an
extrapolation. `Om3` is a ratio in which `H₀` and `Ω_m` both cancel — no
Hubble constant, no matter density, no sound horizon, no absolute magnitude.
It touches nothing that was not measured, and its null value is the exact
number 1 rather than an unknown constant.

The implementation is checked against Figure 1 of the paper that defined it
(Shafieloo, Sahni & Starobinsky 2012): quintessence at `w = −0.9` drives Om3
to 1.11 by a separation of 2, phantom at `w = −1.1` to 0.90, ΛCDM stays pinned
at unity. Those are the published values, and they are a test in the suite.

### And this is where carrying draws earns its keep

`Om3` at different `z₃` all share `H(z₁)` and `H(z₂)`, and it is a ratio of two
differences — strongly correlated along its length, and skewed. Propagating
marginal error bars as though the three redshifts were independent, which is
the only thing a curve-with-error-bars representation can do, misstates the
width by a factor of **two in the middle of the range and seven at the ends**,
in *both* directions. There is no fudge factor that repairs that.

On the real chronometers, both diagnostics come back consistent with a
cosmological constant under every method and under the mixture. That is the
correct answer: 32 differential ages cannot see a DESI-scale deviation.

### The curvature test, where this stops being methodological

`Ok(z) = [H²(z) D'²(z) − c²] / [H₀² D²(z)]` is `Ω_k` in **any** FLRW universe,
whatever the dark energy does (Clarkson, Bassett & Lu 2008). So a departure
from constancy is not evidence about dark energy — it is evidence against
homogeneity and isotropy. It is the strongest claim in the library.

It needs `D_M/r_d` and `D_H/r_d` together, one of them differentiated, so it is
the first real consumer of the joint fit. Written in BAO observables the whole
calibration collapses into one constant, which means **two of the three
questions need no calibration at all**: whether the universe is FLRW (is the
statistic constant?) and whether it is flat (is it zero?). Only a specific
non-zero `Ω_k` needs `c / H₀ r_d`, and the class asks for it rather than
assuming one.

Run it on DESI DR2 with four nearly identical polynomial reconstructions of the
same twelve numbers:

| method | `Ok(z)` significance |
|---|---|
| Chebyshev in `ln(1+z)` | **∞** (p underflows) |
| Chebyshev in `y` | 4.42 σ |
| monomial in `y` | 2.09 σ |
| Chebyshev in `y`, order 3 | 0.28 σ |
| **method-marginalised** | **0.41 σ** |

A decisive violation of the Copernican principle is available to whoever picks
the right expansion variable, and nothing inside a single-method analysis could
tell. The honest reading is the marginalised one: six transverse and six radial
BAO measurements, one of which has to be *differentiated*, do not constrain
`Ok(z)`. **This dataset cannot answer the question, and saying so is the
result.**

That is what the library is for. It is a test in the suite, and
`examples/02_real_data.py` prints the table.

---

## The data

Four releases ship with the library, about 55 kB in total:

| | | |
|---|---|---|
| `chronometers()` | 32 differential-age `H(z)` points | Favale+ 2023, with the Moresco+ 2020 systematic correlation matrix |
| `desi_dr2_bao()` | 13 BAO measurements at 7 redshifts | DESI DR2, arXiv:2503.14738 |
| `union3()` | 22 binned SN distance moduli | Union3, arXiv:2311.12098 |
| `growth()` | 22 `f sigma_8` measurements | Gold-2018, arXiv:1806.10822 |

Each is tested by refitting flat ΛCDM and requiring the survey's **own
published number** back: DESI DR2 gives `Omega_m = 0.297` and
`r_d h = 101.5 Mpc` against the published `0.2975 ± 0.0086` and
`101.54 ± 0.73`; Union3 gives `Omega_m = 0.356` against `0.356 ± 0.026`.
A column read in the wrong order or a correlation matrix used as a covariance
produces no error and a perfectly reasonable-looking curve — so the loaders
are checked against the thing that would move if they were wrong.

Two more things the data layer refuses. A BAO release is `D_M/r_d` **and**
`D_H/r_d`, correlated, at shared redshifts — not one function of redshift — so
it cannot be handed to a reconstructor whole; `.select("DM_over_rs")` gives
the part that can be. And two compilations built from overlapping objects
(Union3 and Pantheon+) cannot be combined, because doing so counts the same
supernovae twice and narrows the interval without adding information.

Pantheon+ (33 MB of covariance) and DES-SN5YR (6 MB) are deliberately not
bundled; they are reachable through the optional CosmoFit bridge.

---

## Roadmap

- [x] **Core.** `Reconstruction`, draw alignment, provenance, grids, the
      significance machinery with effective degrees of freedom.
- [x] **`reconstructors/gp.py`.** Gaussian process with hyperparameters
      *marginalised* rather than optimised, Matérn `nu` inferred rather than
      fixed, a spectral-quadrature sample-path basis that measures its own
      error, and a refusal to differentiate past what the fitted smoothness
      supports.
- [x] **`reconstructors/cosmography.py`.** Chebyshev and monomial series in
      `z`, `y = z/(1+z)` or `ln(1+z)`, with Padé re-expansion; the order
      marginalised rather than chosen, derivatives to any order through Faà di
      Bruno, and a refusal to fit a series outside its radius of convergence.
- [x] **`data/`.** Cosmic chronometers, DESI DR2 BAO, Union3 and a growth
      compilation, with the covariances their papers published, each validated
      by refitting ΛCDM to the survey's own published parameters.
- [x] **`consistency/om.py`.** `Om` and `Om3`, validated against the figure in
      the paper that defined them.
- [x] **Joint fits.** `D_M/r_d` and `D_H/r_d` reconstructed together with the
      correlation between them kept, from independent priors — which is what
      keeps the curvature test from testing its own assumption. `Cosmography`
      supports it; the GP does not yet.
- [x] **`consistency/curvature.py`.** The Clarkson–Bassett–Lu test, which on
      DESI DR2 alone reports honestly that the data cannot support it.
- [ ] **`consistency/`, the rest.** Distance duality, litmus, growth–geometry,
      isotropy.
- [x] **`ensemble/method.py`.** `MethodEnsemble`: fits every member, pools
      their draws into a method-marginalised posterior that is itself a full
      reconstruction, and reports a null test under each method and under the
      mixture.
- [ ] **`inverse/`.** `w(z)`, `V(phi)`, designer `f(R)`/`f(T)`/`f(Q)`,
      `mu(z) = G_eff/G`.
- [ ] **`validation/`.** Injection–recovery and coverage, as CI tests.
- [ ] **`reconstructors/nodal.py`, `pca.py`, `ann.py`, `symbolic.py`.**
- [ ] **`bridges/cosmofit.py`.** Data in, fitted cosmologies out.

---

## Install

Not yet on PyPI. For now:

```bash
git clone https://github.com/salihyesil59/CosmoRecon
cd CosmoRecon
pip install -e ".[dev]"
```

The test suite needs no install at all — `pyproject.toml` puts `src/` on the
path itself:

```bash
python -m pytest
```

204 tests, all of which run in about two minutes.

Requires Python ≥ 3.11. The core depends on numpy, scipy and matplotlib and
nothing else; every heavier dependency is an optional extra, and the suite
asserts that those paths really do skip when the extra is absent.

---

## Scope

Deliberately **not** in this library, because good tools already exist:
parametric model fitting and Bayesian evidence
([CosmoFit](https://github.com/salihyesil59/CosmoFit)), Boltzmann solving
(CAMB, CLASS), posterior-level tension metrics (`tensiometer`, `unimpeded`),
Fisher forecasting (`cosmicfishpie`).

---

## License

MIT — see [LICENSE](LICENSE).
