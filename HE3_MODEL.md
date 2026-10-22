# Full thin-film helium-3 model

This implements the full complex `3 x 2` order parameter of Chapter 9 of
`MEX.pdf`, with the existing model, update, measure, and analysis structure.
It includes amplitude, spin, orbital, and nonunitary fluctuations. There is
no imposed unit-vector constraint or separate Ising variable.

## Hamiltonian and units

At each site, `Psi[mu, a]` has three spin components and two orbital columns
`a = x, y`. The periodic square-lattice Hamiltonian is

\[
H=\sum_i\left[-2n_i+\sum_{m=1}^5\widetilde\beta_m I_m(\Psi_i)+K G_i\right],
\qquad n_i=\sum_{\mu,a}|\Psi_{\mu a,i}|^2.
\]

For `S = Psi @ Psi.T` and `B = Psi @ Psi.conj().T`, the invariants are

\[
I_1=|\operatorname{tr}S|^2,\quad
I_2=(\operatorname{tr}B)^2,\quad
I_3=\operatorname{tr}(SS^*),\quad
I_4=\operatorname{tr}(B^2),\quad
I_5=\operatorname{tr}(BB^*).
\]

Here the star denotes elementwise complex conjugation. In particular,
`I5` is different from `I4` for a general nonunitary field.

The weak-coupling coefficients are

\[
(\widetilde\beta_1,\ldots,\widetilde\beta_5)
=(-1/2,1,1,1,-1).
\]

These coefficients use the amplitude and energy normalization of Sections
8.7 and 9.2. The quadratic coefficient is fixed at `-2`; `K=1` corresponds
to the thesis choice of lattice spacing in these units. Temperature is
the dimensionless Monte Carlo temperature in `exp(-H/T)`.

The implemented gradient is the direct forward-difference expression in
Chapter 9, rather than a continuum integration-by-parts simplification.
For each spin component, define

\[
A=\Delta_x\Psi_x,\quad B=\Delta_x\Psi_y,\quad
C=\Delta_y\Psi_x,\quad D=\Delta_y\Psi_y,
\qquad\Delta_\nu\Psi_i=\Psi_{i+\hat\nu}-\Psi_i.
\]

Then

\[
G_i=\sum_\mu\left[
3|A|^2+|B|^2+|C|^2+3|D|^2
+2\operatorname{Re}(A^*D+B^*C)\right].
\]

The cross terms couple diagonal neighbors, so the existing four-color
checkerboard update is used. Periodic lattice sizes must be even, as
already required by the engine.

## Pressure and strong-coupling coefficients

Set pressure in bar directly on the model:

```python
model = He3Model(pressure=20.0, reduced_temperature=1.0)
```

`pressure=None` preserves the supplied `betas`, which default to weak
coupling. `pressure=0.0` includes the nonzero zero-bar correction. At
weak coupling, homogeneous planar and chiral states are degenerate.

The pressure dependence uses the fifth-degree polynomial fit to Wiman's
coefficients (thesis reference [46]), published in
[Regan, Wiman and Sauls, Appendix Table I](https://arxiv.org/pdf/1908.04190).
The source code contains the published coefficients, in ascending powers
of `x = pressure / 34`. The fit is intended for `0 <= pressure <= 34` bar;
the code does not clip inputs or switch to an extrapolation model.

The separate temperature factor follows
[Wiman and Sauls, Eq. (8)](https://arxiv.org/pdf/1605.01047), thesis
reference [45]. With the default weak-coupling baseline, the result is

\[
\delta_i(p)=\sum_{n=0}^5 a_n^{(i)}(p/34)^n,\qquad
\widetilde\beta_i=\tfrac12\left[b_i^{\rm wc}+t\delta_i(p)\right],
\qquad b^{\rm wc}=(-1,2,2,2,-2).
\]

Here `reduced_temperature = t` is the fixed physical coefficient-evaluation
temperature `T/Tc(p)`, not the Monte Carlo temperature. The factor `1/2`
is required by the Chapter 9 normalization. If explicit baseline `betas`
are supplied, the same `0.5 * t * delta_i(p)` correction is added to them.
`reduced_temperature=0` therefore recovers the supplied baseline.

`main_he3.py` sets `pressure=20.0` and `reduced_temperature=1.0`. The fit
gives, before conversion to float32,

```text
betas = (-0.51651232, 0.94202281, 0.98752904, 0.87906292, -1.15251495)
```

This replaces the previous four-decimal table entry with the polynomial.
Pressure, reduced temperature, and the effective GPU coefficients are
saved in `model_parameters`. Pressure affects the local quartic terms;
`K` and the dimensionless quadratic coefficient remain as specified by
the existing lattice model.

These bulk corrections are a phenomenological input to the strictly
two-dimensional model, as discussed in Section 5.6. All Hamiltonian
coefficients remain fixed across the Monte Carlo temperature ladder, so
the existing PT acceptance formula remains valid. This defines a
reproducible example Hamiltonian, rather than reconstructing the Chapter
13 coefficient evaluation point or its production simulations.

## Updates and engine integration

The GPU state is `complex64`, with shape `(replicas, L, L, 3, 2)`.
One local proposal adds independent uniform Cartesian increments to the
twelve real components, followed by one Metropolis decision for the
whole site. The proposal is symmetric. It samples the flat Cartesian
field measure without a polar-coordinate Jacobian or amplitude clipping.
`delta` controls the proposal half-width; the default is `0.15`.

The local energy change includes the site's potential and every affected
forward-difference cell. Existing `Runtime`, `Simulation`, and
`ParallelTempering` handle scheduling, recording, and exchanges. The
energy-only GPU kernel measures total energy in walker order for PT;
ordinary measurements support the engine's fixed temperature ordering.
Measurement sums are accumulated in float64.

## Measurements

The saved time series contain total energy, total density, the chiral
numerator, real and imaginary parts of the composite phase variable, and
the first and second twist derivatives in both lattice directions.

### Chirality

For every saved configuration, analysis forms the thesis's ratio of sums,

\[
m_\chi=
\frac{2\sum_{i,\mu}\operatorname{Im}
 (\Psi^*_{\mu x,i}\Psi_{\mu y,i})}
 {\sum_{i,\mu,a}|\Psi_{\mu a,i}|^2}.
\]

This differs from averaging individually normalized local chiralities.
It is `+1` and `-1` in the two pure chiral states. The scalar Binder
cumulant follows Section 10.7.1,

\[
U_\chi=1-\frac{\langle m_\chi^4\rangle}
 {3\langle m_\chi^2\rangle^2},
\]

whose ordered limit is `2/3`.

### Composite U(1) variable

The added variable is the proposed spin contraction

\[
\Phi_{2,i}=\sum_\mu
 (\Psi_{\mu x,i}^{\,2}-\Psi_{\mu y,i}^{\,2}),
\qquad p_2=\frac1{L^2}\sum_i\Phi_{2,i}.
\]

The squares here are complex squares, not squared magnitudes. Under
`Psi -> exp(i alpha) Psi`, this variable transforms as
`Phi2 -> exp(2 i alpha) Phi2`. It is invariant under real spin rotations.
For the chiral ansatz

\[
\Psi_{\mu a}=\rho e^{i\theta}d_\mu
 (\delta_{ax}+i\chi\delta_{ay})/\sqrt2,
\]

with real unit `d`, it gives `Phi2 = rho**2 * exp(2 i theta)`, independent
of the sign of chirality and of the spin direction. It therefore reduces
to Kawamura's doubled-phase variable when the amplitude is fixed.

Analysis uses the complex-vector Binder ratio

\[
g_2=2-\frac{\langle |p_2|^4\rangle}
 {\langle |p_2|^2\rangle^2},
\]

with ordered limit `1`. The unnormalized local composite retains amplitude
fluctuations. This is one orbital component of a charge-two composite:
it depends on the chosen orbital axes and can vanish in planar states.
It is suitable for studying the chiral sector, rather than a phase
coordinate for every possible complex matrix.

### Helicity modulus

A common phase twist replaces every forward difference in direction
`nu` by `exp(-i delta_nu) * Psi_neighbor - Psi_site`, exactly as in
Appendix A.9.2. Onsite terms do not contribute. The GPU records

\[
J_\nu=\left.\partial_{\delta_\nu}H\right|_0,
\qquad C_\nu=\left.\partial_{\delta_\nu}^2H\right|_0.
\]

For example, in the `x` direction let `u=Psi_x(i+x)` and
`v=Psi_y(i+x)`, retaining the differences `A,B,C,D` defined above. The
contribution summed over sites and spin components is

\[
J_x=2K\sum\operatorname{Im}
 (3A^*u+B^*v-u^*D-v^*C),
\]

\[
C_x=2K\sum\left[3|u|^2+|v|^2-
 \operatorname{Re}(3A^*u+B^*v+u^*D+v^*C)\right].
\]

In the `y` direction, use `u=Psi_x(i+y)`, `v=Psi_y(i+y)`:

\[
J_y=2K\sum\operatorname{Im}
 (C^*u+3D^*v+A^*v+B^*u),
\]

\[
C_y=2K\sum\left[|u|^2+3|v|^2-
 \operatorname{Re}(C^*u+3D^*v+A^*v+B^*u)\right].
\]

Analysis evaluates the connected estimator

\[
\Upsilon=\frac1{2L^2}\sum_{\nu=x,y}
 \left[\langle C_\nu\rangle-
 \frac{\langle J_\nu^2\rangle-\langle J_\nu\rangle^2}{T}\right].
\]

Each direction has its own current variance. The current is not divided
by two before squaring. The twist is a gradient per lattice link, rather
than the total angle across the sample. Differentiating with respect to
the phase of `Phi2` instead would change the stiffness normalization by
a factor of four.

The analysis uses the existing blocked jackknife and recomputes nonlinear
observables in each jackknife sample. Plots use error bars. It does not
automatically fit a transition temperature or a Weber–Minnhagen form.

## Derivation audit

The checks below cover the equations needed for this implementation,
including Chapters 2–10 and Appendices A.1–A.9. They do not certify
every derivation or the numerical conclusions of the 99-page thesis.
The PDF itself has not been edited.

- **Microscopic basis assumptions:** the reduced-state calculation in
  Appendix A.1 needs an explicit independent-pair basis and compatible
  canonical coherence factors to justify switching between spin rows and
  columns. Eq. (2.9) also uses special coherence-factor conventions: for
  the general stated transformation the canonical inverse is
  `a_k = u_k.dagger() b_k - v_minus_k.T b_minus_k.dagger()`.
  Real diagonal `u` and the appropriate triplet conventions recover the
  form written in the thesis. These are missing qualifications, rather
  than a demonstrated failure of the zero-field GL coefficients.
- **Spectrum with a magnetic field:** Eq. (2.10), printed page 9 / PDF
  page 16, needs a zero-field or appropriate equal-spin-pairing assumption;
  diagonalizing `Delta Delta.dagger()` alone is insufficient in a field.
  For example, opposite-spin triplet pairing `Delta = d sigma_x` with
  Zeeman splitting `h` has branches `sqrt(xi**2 + abs(d)**2) +/- h`,
  rather than `sqrt((xi +/- h)**2 + abs(d)**2)`. The implemented model
  has no magnetic field.
- **Quartic invariants and angular averages:** the five invariants,
  the `1/15` three-dimensional and `1/8` two-dimensional fourth moments,
  weak-coupling coefficient ratios, and planar/chiral degeneracy follow
  correctly. The two-dimensional quartic trace was also checked against
  a direct angular average for a general complex `3 x 2` matrix.
- **Gradient coefficient:** the equality of the three weak-coupling
  coefficients is correct. The strict-2D coefficient in Section 6.2 is
  consistent with the bulk coefficient multiplied by the angular-factor
  ratio `15/8`, with the stated density-of-states conventions.
- **Chiral reduction:** the local-energy and fixed-chirality gradient
  contractions in Chapter 7 and Appendix A.5 are correct. The prefactor
  `K` multiplies both terms inside the displayed square brackets; this
  was checked against the rendered PDF, not just extracted text.
- **Matrix identity correction:** immediately before Eq. (4.3), printed
  page 18 / PDF page 25, the product should read
  `Delta† Delta = sigma_y (sigma.d*) (sigma.d) sigma_y`. The missing
  similarity transformation does not change the traces used later.
- **Pauli identity correction:** Appendix A.2, printed page 79 / PDF
  page 86, incorrectly states `sigma_y sigma_k sigma_y = -sigma_k`
  for all three components. The sign is positive for `k=y` and negative
  for `k=x,z`; the compact correct identity is
  `sigma_y sigma_k sigma_y = -sigma_k.conj()`. Its final trace
  `tr(A_i† A_j)=2 delta_ij` is nevertheless correct.
- **Continuum versus lattice:** the integration-by-parts identities in
  Appendices A.7 and A.8 are correct for smooth periodic continuum
  fields. They are not exact identities for the forward differences
  of Chapter 9. For instance, the periodic `4 x 4` field
  `psi(x,y)=exp(i*(pi*x+pi*y/2))` has mean
  `abs(Dx psi + i Dy psi)**2 = 10`, whereas mean
  `abs(Dx psi)**2 + abs(Dy psi)**2 = 6`. Thus discretizing the
  continuum-simplified spin-frozen Eq. (8.9) and restricting the directly
  discretized full Chapter 9 model to one spin row give different finite
  lattice models. This implementation retains Chapter 9's expression.
- **Chirality and helicity:** the ratio of sums in Section 10.7.1 and
  the connected twist estimator in Appendix A.9.2 are correct, including
  the `1/L^2` normalization for a twist per link.
- **Strong coupling:** applying bulk corrections to a strict-2D Fermi
  surface remains a phenomenological assumption, already acknowledged
  in Section 5.6. A common positive rescaling of quartic coefficients
  preserves homogeneous phase comparisons, but changes amplitude and
  energy scales unless the remaining normalization is also transformed.
- **Pressure-fit transcription:** the published Table I gives `-0.1774`
  for the quadratic term of `delta_4`. With the full published table,
  Section 5.6's quadratic coefficient in `g(x)` is `0.55901`, rather than
  `0.562`. The corrected polynomial still satisfies the same negativity
  argument and favors the chiral state in that comparison. The code uses
  the published table directly.
- **Jump interpretation:** Eq. (10.1), `2T/(pi*q**2)`, is conditional on
  a BKT transition driven by freely proliferating vortices of minimal
  phase winding `2 pi q`. The chiral identification
  `(d,theta) ~ (-d,theta+pi)` permits half-quantum vortices; it does not
  establish which defects drive a transition or prove a separate BKT
  transition. Coupled spin textures and the long-distance spin stiffness
  also matter. A Binder curve or a better fit to `8T/pi` alone does not
  settle that question, consistent with the thesis's Chapter 13 caveat.

The invariant definitions, bulk weak-coupling coefficients, and
temperature-scaled phenomenological corrections agree with
[Wiman and Sauls, Eqs. (3)–(8)](https://arxiv.org/pdf/1605.01047).

## Files and use

New files:

- `models/he3/model.py`: state, parameters, and the existing model interface.
- `models/he3/update.py`: local potential, forward-cell energies, and GPU updates.
- `models/he3/measure.py`: observables, twist derivatives, and GPU-only PT energy.
- `models/he3/analysis.py`: model-specific moments, jackknife estimates, and plots.
- `main_he3.py`: a short run with an editable temperature ladder and lattice sizes.
- `tests/test_he3.py`: independent Hamiltonian, symmetry, derivative, and engine checks.
- `HE3_MODEL.md`: implementation details and this audit.

Run:

```powershell
python main_he3.py
```

Analyze the resulting data with the model-specific analysis class:

```python
from models.he3.analysis import He3Analysis

analysis = He3Analysis.from_file("data/he3_run.npz")
analysis.plot()
```

The example uses `L = 16, 24, 32`, 24 temperatures from `0.15` to `0.45`,
1,000 thermalization sweeps, 4,000 measurement sweeps, a sample every 10
sweeps, and a PT exchange interval of 10 sweeps. These are short runs to
exercise the implementation, not converged production estimates.

No new project dependencies, general analysis framework, or model-specific
engine changes are needed.

## Validation performed

`python -B -m unittest discover -s tests -p test_he3.py -v` passed all
nine tests: two CPU coefficient tests and seven GPU tests. The pressure
coefficients agree with the published rounded Table II at eight pressures
within `6e-5` in the lattice beta normalization. The tests also cover the
zero-bar correction, fixed temperature scaling, custom baseline betas,
and persistence of the pressure settings and effective coefficients.
The GPU tests compare the orbital-contraction GPU
potential with independent full-spin matrix contractions, check local
energy changes including periodic boundaries and diagonal couplings,
and compare both current and curvature estimators with finite differences
of the twisted Hamiltonian. They also check phase/spin symmetries,
ordered Binder normalizations, updates, temperature ordering, PT, saving,
and analysis integration.

The pressure-based `main_he3.main()` completed all three sizes in a
temporary output directory and saved and reloaded `data/he3_run.npz`
with the correct pressure and coefficient metadata. Each of the nine
measurement channels has shape
`(400, 24)` per size; every computed observable and error was finite,
and the four-panel error-bar plot rendered successfully. This verifies
execution, not equilibration or transition temperatures. Project datasets
were unchanged. The pressure addition changes only `models/he3/model.py`,
`main_he3.py`, `tests/test_he3.py`, and this documentation; the GPU kernels,
engine, and analysis implementation remain unchanged.
