# CosmoRecon

> **Model-independent reconstruction and null tests of the cosmological framework**

CosmoRecon asks what the data say about `H(z)`, `d_L(z)` and `f sigma_8(z)`
with no cosmological model assumed — and then asks the question the field
currently cannot answer: **how much of that result is the method rather than
the measurement?**

> **Status: pre-alpha.** The core — the object everything else is written
> against, its guarantees, and the significance machinery — is implemented and
> tested, and so is the first reconstruction method. See
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

On top of that, `ensemble/` splits the answer by the law of total variance:

```
Var_total(z) = E_method[ Var_within(z) ]  +  Var_method[ E_within(z) ]
                    statistical                 methodological
```

`budget.method_fraction()` is the fraction of a published error bar that is a
choice rather than a measurement — the part that will not shrink with more
data.

---

## Roadmap

- [x] **Core.** `Reconstruction`, draw alignment, provenance, grids, the
      significance machinery with effective degrees of freedom.
- [x] **`reconstructors/gp.py`.** Gaussian process with hyperparameters
      *marginalised* rather than optimised, Matérn `nu` inferred rather than
      fixed, a spectral-quadrature sample-path basis that measures its own
      error, and a refusal to differentiate past what the fitted smoothness
      supports. 82 tests.
- [ ] **`reconstructors/cosmography.py`.** Taylor, Padé, Chebyshev,
      `y`-redshift, log-polynomial, each carrying its radius of convergence.
- [ ] **`data/`.** Cosmic chronometers, DESI DR2 BAO, Pantheon+ / Union3 /
      DES-SN5YR, growth — with the covariances their papers published.
- [ ] **`consistency/`.** `Om`, `Om3`, `Ok`, distance duality, litmus,
      growth–geometry, isotropy.
- [ ] **`ensemble/method.py`.** `MethodEnsemble` and significance deflation.
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
