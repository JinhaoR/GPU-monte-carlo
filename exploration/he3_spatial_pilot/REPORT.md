# Spatial He3 observables and a bounded GPU pilot

**Outcome:** the chirality, charge-two composite and spin-nematic measurements validate against independent NumPy calculations and work in the unchanged GPU Gaussian/PT simulation. Two L=64,gD=0 histories completed 100,000 thermalization plus 100,000 production sweeps each. The spatial measurements add about 1% runtime at the tested cadence.

The most useful physical observation is that **spin-nematic correlations remain comparable to the box scale near the ordering region**. The composite spatial ratio also differs substantially from a specified simple harmonic reference constructed from the measured common-phase helicity. These observations strengthen the case for spatial finite-size studies and caution against a mass-only HQV interpretation at the existing sizes. **This single-size pilot does not resolve the U(1)/Z2 transition splitting.**

The principal deliverable is the step-by-step [MAIN_BRANCH_IMPLEMENTATION.md](MAIN_BRANCH_IMPLEMENTATION.md). It explains the physics, tensor normalization, Fourier signs, GPU reductions, data layout, tests and a minimal manual production implementation. No automatic port or merge was made. All new code and data are under this exploratory directory; production code, the Hamiltonian, sampler, exchange logic, earlier exploratory work and original datasets are unchanged.

## 1. Definitions and mathematical checks

The field is the full complex matrix A_mu,a on a periodic L-by-L lattice, with spin index mu and orbital columns x,y. Define N=L² and

\[
\kappa(r)=2\operatorname{Im}\sum_\mu A_{\mu x}^*(r)A_{\mu y}(r),\qquad
\mathcal P(r)=\sum_\mu[A_{\mu x}^2(r)-A_{\mu y}^2(r)],
\]
\[
Q_{\mu\nu}(r)=\operatorname{Re}\sum_a A_{\mu a}(r)A_{\nu a}^*(r)-\frac{n(r)}3\delta_{\mu\nu},
\qquad n=\sum_{\mu,a}|A_{\mu a}|^2.
\]

These expressions require no chiral projection or director extraction. Kappa is a real chirality field, P a spin-invariant complex field of gauge charge two, and Q a density-weighted real symmetric traceless spin tensor. The existing normalized global chirality C/D_A is retained for comparison but is not substituted for the raw local kappa in its structure factor.

Fourier amplitudes use sum_r O(r) exp(-2pi i k.r/L), without normalization. The saved momentum order is 0,+kx,-kx,+ky,-ky with integer indices (0,0),(1,0),(-1,0),(0,1),(0,-1). Thus kappa_0 equals the existing extensive C, and P_0 equals `phase_real+i phase_imag`.

The spin tensor is represented by the orthonormal components

\[
q_1=(Q_{xx}-Q_{yy})/\sqrt2,\quad q_2=(Q_{xx}+Q_{yy}-2Q_{zz})/\sqrt6,
\]
\[
q_3=\sqrt2 Q_{xy},\quad q_4=\sqrt2 Q_{xz},\quad q_5=\sqrt2 Q_{yz}.
\]

At every momentum sum_a |q_a|² equals the full nine-entry Frobenius norm sum_mu,nu |Q_mu,nu|², including twice the off-diagonal powers. Under a global real spin rotation Q_k transforms as R Q_k R^T, so this scalar is invariant even though the components change.

The spectra are

\[
S_\kappa(k)=\langle|\kappa_k|^2\rangle/N,\quad
S_P(k)=\langle|P_k|^2\rangle/N,\quad
S_Q(k)=\langle\sum_a|q_{a,k}|^2\rangle/N.
\]

For complex P both signs are retained. The GPU computes Fourier transforms of Re(P) and Im(P) separately. If these are a_k,b_k, then P_+k=a_k+i b_k and P_-k=a_k*+i b_k*. Imposing P_-k=P_+k* would be wrong. Kappa and the real q_a do satisfy the usual conjugacy.

Let Sx and Sy be the averages over the two signs on each axis, and Sk=(Sx+Sy)/2. The finite-size diagnostic is

\[
R=S(0)/S_k,\qquad \xi_2/L=\frac{\sqrt{R-1}}{2L\sin(\pi/L)}.
\]

Directional estimates use Sx or Sy in the denominator. **Ensemble powers are averaged first**, then ratios and square roots are formed. The finite-lowest-momentum expression is a second-moment diagnostic, not an exact exponential decay length in an arbitrary full-field state. In a finite chirality-ordered box its unconnected zero mode includes the ordering contribution; xi/L can exceed one. This does not imply a measurable exponential decay length larger than a periodic box.

The default moments are unconnected finite-volume moments. Connected zero-mode estimates and sample-mean fractions are retained as diagnostics, with no automatic mean subtraction. A slowly sampled symmetry sector can make such subtraction misleading. Nonzero SOC would also require a deliberate distinction between mean spin alignment and connected spin fluctuations.

If R<1, its value and uncertainty are retained and xi/L is NaN with a negative-radicand flag. A zero spectrum is undefined, and an absent nonzero mode with a finite zero mode is singular. A declared double-precision cancellation threshold recognizes numerically zero denominators. No negative radicand is clamped to zero. The guide derives these conventions and gives the component reconstruction formulas.

## 2. Validation and implementation

The initial small-field and real-PT checks passed before the pilot. The final suite additionally checks partial/multiple reduction tiles through L=256. [validate.py](validate.py) passes **17 check groups**, covering thirteen synthetic cases/sizes and the following independent tests:

- Uniform chiral and nonchiral fields, a planar case with vanishing P, and the zero field.
- Known real density sinusoids and a periodic complex modulation whose P positive and negative modes differ analytically.
- Spatial spin rotations/textures, global SO(3) rotations, global U(1) charge-two transformation, spatial x/y exchange and translation phases.
- Random unrestricted matrices, independent NumPy FFTs and a contraction of all nine tensor entries rather than the GPU's five-component representation.
- Nontrivial walker-to-temperature permutations, actual Gaussian/PT updates and existing global zero-mode/Qzz checks.
- Nonnegative powers, singular/undefined ratios and measurement read-only behavior.

The maximum random/analytic CPU-GPU amplitude discrepancy is 1.15e-13 in this suite. Relative discrepancies measured against the largest amplitude are below 7e-16. SO(3)/U(1) transformed complex64 fields use looser tolerances to include rounding of the transformed input.

Six sparse pilot snapshots, each containing four selected temperatures, were also independently transformed on the CPU. Maximum amplitude differences are 2.28e-13 and 4.55e-13 in the ordered and hot runs. Across **all** saved events, new zero modes differ from production globals by at most 4.88e-6 for chirality and 4.97e-6 for P. This is consistent with promoting components before local multiplication in the new kernel while the existing kernel uses complex64 local products. It is not a physical normalization difference.

[Saved-data verification](results/saved_data_checks.json) checks exact synchronization of spatial/global records, valid maps, source hashes and snapshot FFT agreement. The kernel source hashes match the validated implementation and production source hashes still match the run manifests.

The independent [GPU module](gpu_measure.py) uses three modes of eight real local fields, tiled into 1,024-site blocks with 128 threads and float64 reductions. A second kernel sums tile outputs and scatters to the current temperature index. It precomputes sine/cosine tables and reuses small buffers. It never copies full fields at normal measurement events. The [NumPy reference and ensemble estimators](observables.py) have no CUDA dependency.

## 3. Bounded run settings and temperature ladder

The model/runtime, measurements, Gaussian updater and PT wrapper were read directly. The existing [transition study](../he3_transition_study/REPORT.md), [SOC study](../he3_soc/REPORT.md), relevant thesis helicity derivations and previous pilot were reviewed for normalization and sampling context. The coefficient temperature remains fixed at `reduced_temperature=1`; the MC temperature changes exp(-H/T). The full extensive H, including all implemented onsite and gradient terms, is saved at each spatial event.

| Setting | Actual pilot |
|---|---|
| GPU | Local NVIDIA GeForce RTX 4070 SUPER |
| Model | Full complex 3-by-2 GL; P=20 bar, K=1, gD=0 |
| Quartic coefficients | (-0.5165123343,0.9420228004,0.9875290394,0.8790628910,-1.1525149345) |
| L and replicas | 64; 47 temperatures |
| Temperature interval | 0.37–0.415 |
| Gaussian updater | Existing full-Hamiltonian MH; rho=-0.9, proposal mass 0 |
| PT exchange | Every 5 complete sweeps |
| Thermalization / production | 100,000 / 100,000 sweeps in each history |
| Global cadence | 50 sweeps; 2,000 production events per history |
| Spatial cadence | 250 sweeps; 400 production events per history |
| Walker-history cadence | 100 sweeps |
| Sparse fields | At sweeps 100k,150k,200k; T=0.37,0.3815,0.385,0.415 |
| Seeds | Ordered 2026102951; hot 2026102953; PT seeds one larger |
| Hard loop cap | 120 seconds per history; neither cap reached |

Ordered initialization is the existing homogeneous positive-chirality d=z field. Hot initialization independently draws every full matrix entry from a zero-mean complex Gaussian with expected total local norm equal to the homogeneous n0. It has no imposed local chiral or spin form. It is copied before PT construction and the runtime energy cache is explicitly invalidated. Both then use the unchanged updater.

The [ladder generator](ladder.py) uses the existing L64 energy-variance/Cv records and the finer reweighted curve around the feature. It targets Delta(beta) sigma_H approximately 0.65, with sigma_H=T L sqrt(Cv), then inserts exact controls. The largest predicted edge value is 0.667. This replaces the old pilot's broad coarse gaps rather than merely narrowing the interval for high acceptance. The hot endpoint is about 0.033 above the thermodynamic feature and is disordered in the measured channels.

Both requested histories completed: ordered loop **41.92 s**, hot loop **40.19 s**, about **86.94 s total including setup and saving**. Each records file is about 29.8 MB, including thermal histories. No production-size scan, SOC scan, cluster job, negative-start expansion or thermal extension was run. A separate performance smoke used only 1,900 L128 sweeps; L160/L256 tests were synthetic measurements only.

## 4. Spatial results

Main errors use joint delete-one blocks of 20 spatial events = 5,000 sweeps, leaving twenty blocks per production history. Each deletion is simultaneous across momenta, components and all PT temperatures. Scalar errors use matching 5,000-sweep blocks. These are conditional history errors, not proof of equilibration.

| Temperature | Start | xi_kappa/L | xi_P/L | xi_Q/L |
|---:|---|---:|---:|---:|
| 0.3700 | ordered | 4.302 ± 0.084 | 1.048 ± 0.025 | 0.840 ± 0.009 |
| 0.3700 | hot | 4.313 ± 0.064 | 1.080 ± 0.015 | 0.861 ± 0.013 |
| 0.3815 | ordered | 1.264 ± 0.045 | 0.658 ± 0.013 | 0.619 ± 0.006 |
| 0.3815 | hot | 1.269 ± 0.034 | 0.684 ± 0.016 | 0.611 ± 0.007 |
| 0.3850 | ordered | 0.232 ± 0.016 | 0.201 ± 0.012 | 0.363 ± 0.008 |
| 0.3850 | hot | 0.244 ± 0.013 | 0.198 ± 0.012 | 0.380 ± 0.006 |

### Chirality

The large cold S0 and unconnected ratio are consistent with an ordered finite box. Both decrease rapidly through the known ordering region. At T=0.3815 the directional difference is only 1.01 and 0.27 paired block errors in the two histories. Across all reported sector/temperature directional checks the largest finite value is 2.97 errors; there is no demonstrated coherent x/y discrepancy.

At the hot end, S0 and Sk become almost equal. Eleven central sector/temperature estimates across the two histories have R<1, all in hot chirality/composite channels. For example the hot-start chirality estimate at 0.415 is R=0.9970 ± 0.0733, so xi is undefined at that sample mean. This is a useful example of why a noisy near-zero radicand must remain flagged. Some positive central estimates also have invalid jackknife deletes, and their Gaussian xi error is not reported. The raw R error remains usable.

### Composite U(1) channel

The low-temperature spatial ratio is of order the box size and falls to a short-range value above the feature. At 0.415, mass helicity is -0.008 ± 0.043 and +0.004 ± 0.042, consistent with zero, while the composite spectrum is nearly flat. The low-temperature behavior is compatible with a phase-coherent finite system and possible quasi-long-range order. Establishing algebraic decay requires more sizes and better resolved real-space correlations; an L64 ratio alone cannot distinguish a long finite correlation length from an algebraic phase.

The P zero mode reproduces the old global charge-two measurement. Its S0=N<p²> allows exactly the previous two-size exponent analysis when multiple sizes are eventually measured. The new Sk adds independent shape information that the global exponent lacked. Both momentum signs and separate directions are retained for current/anisotropy checks.

### Spin-nematic channel

The scalar spin tensor has substantial long-wavelength weight: at 0.3815 the second-moment diagnostic is about **39–40 sites**. At 0.385 it remains about 23–24 sites, while the chirality and P diagnostics are about 13–16 sites. These are observable-dependent finite-box scales, not evidence for a separate spin ordering transition or a proven intermediate phase.

They do show that this box is not demonstrably in an L much larger than the measured spin-nematic scale regime. Inferring that an HQV's spin contribution has already screened out from mass helicity alone would be premature. The nematic scale is a physical gauge-invariant proxy; it is not a direct measurement of spin stiffness, director decay length or HQV string tension.

Component information is also useful for sampling. At 0.3815, zero-mode power fractions range 0.192–0.205 in the ordered history and 0.179–0.232 in the hot history; exact spin isotropy would give 0.2 in a fully sampled ensemble. Mean spin alignment is -0.0058(78) and -0.0211(66), respectively. These residual orientation differences warrant longer independent checks before precision inference, even though the invariant spatial ratios agree well.

![Temperature-dependent spatial spectra and finite-size ratios](results/spatial_vs_temperature.png)

### An independent helicity/composite consistency check

For a deliberately restricted reference, take a Gaussian common phase with action rho/2 sum_<ij>(theta_i-theta_j)² on the same periodic square, zero winding, and P proportional to exp(2i theta). Define

\[
g(r)=\frac1N\sum_{k\ne0}\frac{e^{ik\cdot r}}{4\sin^2(k_x/2)+4\sin^2(k_y/2)}.
\]

The finite-torus prediction is

\[
G_P(r)/G_P(0)=\exp\{-4T[g(0)-g(r)]/\rho\}.
\]

Fourier transforming this deterministic covariance gives a reference xi_P/L without fitting a spatial exponent. This is not a second Monte Carlo simulation. Substituting measured common-phase helicity for rho is an **additional effective-theory assumption**, not an identity for the full GL field. The corresponding infinite-plane spin-wave exponent would be eta_P=2T/(pi rho).

At 0.3815 the reference gives 0.856 and 0.846 for the two histories, compared with measured 0.658 and 0.684. Jointly deleting the same scalar/spatial 5,000-sweep intervals gives differences **-0.198 ± 0.016** and **-0.162 ± 0.017**. Sampling error within the histories does not account for the gap to this particular reference. At 0.37 the differences are smaller, -0.066(24) and -0.029(15).

This corroborates the earlier composite/helicity consistency concern through a different spatial statistic. It does not disprove BKT physics, identify a vortex charge or show that the helicity kernel is wrong. The reference omits vortices, winding-sector contributions, nonconstant/internal tensor amplitudes and coupled spin/chirality effects, and uses a response-derived stiffness at a finite scale. Conventional XY finite-size matching itself needs logarithmic corrections ([Hasenbusch](https://arxiv.org/abs/cond-mat/0502556)); the full GL model requires additional checks. The spin-disordered paired-superfluid theory motivating an anomalous jump remains a useful analogy rather than an established reduction here ([Mukerjee, Xu and Moore](https://arxiv.org/abs/cond-mat/0605102)).

Sparse fields provide qualitative real-space correlations. The figure pools six fields per selected temperature, including each run's burn-boundary snapshot. Chirality has a cold plateau; composite and nematic correlations remain appreciable across the cold box and decay much more rapidly at the hot control. There are too few snapshots for exponent or exponential-length fits, so none is reported.

![Illustrative spatial correlations and conditional harmonic comparison](results/spatial_interpretation.png)

## 5. PT, initialization and reliability

| Diagnostic | Ordered | Hot |
|---|---:|---:|
| Minimum production swap acceptance | 0.6315 | 0.6345 |
| Mean production swap acceptance | 0.7068 | 0.7063 |
| Minimum adjacent energy-histogram overlap | 0.714 | 0.708 |
| Completed round trips, entire run | 96 | 88 |
| Completions occurring during production | 54 | 55 |
| Walkers with a production completion / 47 | 38 | 37 |
| Sampled walkers reaching both endpoints during production | 45 | 39 |
| Mean completed-trip time, entire run | 67,870 sweeps | 71,011 sweeps |

The acceptance and actual histogram overlap support the ladder design, and the earlier severe pilot bottleneck is absent. The different scopes of exact completion counters and sampled endpoint visits are retained. A completion during production can include a trip that started in burn-in; the mean times above include the entire run. Short endpoint visits can be missed by 100-sweep history sampling.

Energy and chirality magnitude settle toward similar temperature-dependent values. Cold signed chirality and spin orientation lose their initial uniform bias and fluctuate in both histories. At 0.3815, first/second production-half xi values are 1.211/1.322 for chirality, 0.649/0.667 for P and 0.617/0.621 for Q in the ordered run; the hot values are 1.308/1.231,0.702/0.667,0.610/0.611. Their differing directions of fluctuation do not establish a common systematic drift, but do not certify equilibrium either.

There are residual discrepancies. The cold Cv estimates are 10.29(33) and 12.12(36), a 3.72 combined-error difference. The largest other start differences include hot, nearly flat chirality ratios. Many correlated temperatures and observables were examined, so isolated deviations are not independent significance tests. Conversely, they should not be erased by pooling histories. All per-history outputs are retained.

Against the earlier long L64 reweighted history, energy, chirality, helicity and P² at four selected temperatures agree within 2.24 combined errors; this is a useful external consistency check with different seeds, not a proof that every sector is sampled.

The short-history ACF estimates at 0.3815 give about 1,035–1,109 effective energy records and 933–1,084 absolute-chirality records out of 2,000, but only 337–399 for the spatial S0 observables out of 400. These fixed-temperature estimates can benefit from frequent exchanges while missing slower hidden modes. They are finite-history diagnostics, not certified independent equilibrium counts.

Blocking is not a precision plateau demonstration. At 0.3815, the ordered chirality xi error rises from 0.039 at 1,250-sweep blocks to 0.047 at 10,000; the hot composite error rises from 0.0134 to 0.0178. At 20,000 sweeps only five blocks remain and some errors fall sharply. The primary 5,000-sweep errors describe this pilot conditionally; they should not be extrapolated to precision production targets without more history and seeds.

The initial 100,000-sweep burn-in is adequate for this feasibility comparison and shows no obvious persistent ordered-versus-hot branch separation in the principal spatial curves. It is not demonstrated sufficient for precision spin-sector or transition inference. No thermal extension was needed to establish measurement correctness and practical feasibility; unresolved statistical issues are documented rather than turning this pilot into production.

![Histories, acceptance, blocking and signed hot radicands](results/sampling_checks.png)

## 6. GPU cost and practical scaling

Three warm repetitions use CUDA-event batches plus synchronized host timing. The incremental measurement itself is slower than the old scalar event, but is called five times less frequently than the 50-sweep global measurement.

| L, 47 replicas | Cached global GPU event | New GPU event | New event with host read | New workspace |
|---:|---:|---:|---:|---:|
| 64 | 0.168 ms | 0.343 ms | 0.439 ms | 91 KB |
| 128 | 0.603 ms | 1.106 ms | 1.210 ms | 309 KB |
| 160 | 0.947 ms | 1.910 ms | 2.143 ms | 472 KB |
| 256 | 2.372 ms | 4.756 ms | 4.953 ms | 1.18 MB |

At L64 the repeated update/measurement loop changes from 188.63 to 190.48 microseconds per sweep, **0.98%** overhead. The isolated-event cadence estimate is 0.93%. The short L128 loop gives 0.23% observed overhead versus 0.71% from isolated events; short wall-clock comparisons and desktop/clock variation limit that precision. These are feasibility numbers, not guaranteed production timings.

The L128 smoke totals 1,900 sweeps, including warm-up and repeated timing segments. L160/L256 measurements use static fields with no updates. No second substantial physics campaign was performed. With a production ladder containing more temperatures, cost and workspace increase approximately with replica count; state/RNG/update costs also grow. The new meter is practical at L128–256 at the proposed cadence without further optimization. FP64 reductions and three local-field passes are the main arithmetic cost, and there is no evidence that a more complicated implementation is justified now.

## 7. Recommended next steps

1. **Work through the main-branch guide manually.** Begin with one common 250-sweep cadence so named-channel storage and the existing engine/PT mapping can be reused. Add dual cadence after correctness and save/load tests are understood.
2. **Retain independent starts and all three sectors.** A future statistical check could use four L64 histories with longer production, e.g. 200k burn plus 300k production as an initial allocation, extended only if measured blocking/sector diagnostics require it. These are proposed settings, not runs performed here.
3. **Add a second physical size before fitting critical behavior.** At L128, redesign the ladder from the existing L128 Cv data, targeting Delta(beta) sigma_H about 0.65 and retaining the 0.415 hot control. Roughly 95–110 replicas and critical spacings around 0.0002 are a starting design, subject to overlap and flow checks. The 47-replica L128 timing smoke was not a validated physics ladder. Use at least ordered/hot preparations and longer blocks/histories justified by observed autocorrelation.
4. **Measure real-space shapes more often when doing physics.** Sparse full matrices are sufficient for independent measurement checks. For algebraic/exponential discrimination, occasionally compute FFT correlations and save compact directional/radial averages instead of large field dumps. Preserve enough raw synchronized low modes for reweighting and joint errors.
5. **Follow scaling flow, not one-size feature locations.** Compare xi_kappa/L, xi_P/L and the spin scale over useful size ratios. A stable interval with different large-L ordering in the two sectors would support an intermediate phase. One box with three different finite-size ratios does not. The earlier larger-campaign proposal remains conditional on implementing and reviewing these validated observables.

The full spin-singlet orbital tensor could be added later if P continues to disagree with other phase diagnostics, since the nonchiral validation case shows how one composite projection can vanish in a nonzero field. It was not needed to validate the immediate observables. Spin-twist stiffness remains a future derivation; no unvalidated estimator was added. No HQV pair-pinning work was resumed.

## 8. Reproduction and files

Run read-only analyses on the delivered records:

```powershell
python -B exploration/he3_spatial_pilot/check_saved.py
python -B exploration/he3_spatial_pilot/analyze.py
python -B exploration/he3_spatial_pilot/figures.py
```

To reproduce measurements in a fresh result location, the original execution sequence was:

```powershell
python -B exploration/he3_spatial_pilot/validate.py
python -B exploration/he3_spatial_pilot/ladder.py
python -B exploration/he3_spatial_pilot/benchmark.py
python -B exploration/he3_spatial_pilot/pilot.py
```

The pilot refuses to overwrite existing ordered/hot directories. Review the explicit output location before a future repetition. It requires a passed current-source validation and a bounded preflight runtime; it has no cluster launcher or CPU MC substitute. `validate.py` itself runs only a 100-sweep L8 GPU smoke plus synthetic read-only tests.

| Artifact | Contents |
|---|---|
| [Implementation guide](MAIN_BRANCH_IMPLEMENTATION.md) | Derivations, GPU design, minimal manual changes, tests and statistics |
| [validation.json](results/validation.json), [saved checks](results/saved_data_checks.json) | Independent numerical agreement and synchronization/provenance checks |
| [benchmark.csv](results/benchmark.csv), [benchmark.json](results/benchmark.json) | Incremental timing, raw repeats and workspace |
| [ladder.json](results/ladder.json) | Actual 47 temperatures and overlap design |
| `results/ordered/`, `results/hot/` | Full manifests, compact per-temperature histories and sparse fields |
| [spatial summary](results/spatial_summary.csv), [global summary](results/global_summary.csv) | Per-history values, errors, statuses and components of ratios |
| [PT summary](results/pt_summary.csv), [edges](results/pt_edges.csv) | Acceptance, transport and energy overlap |
| [blocking](results/blocking.csv), [ACF/ESS](results/autocorrelation.csv), [halves](results/half_history.csv) | Conditional sampling diagnostics |
| [Start comparison](results/initialization_comparison.csv), [long-history check](results/long_history_comparison.csv) | Independent-preparation and external consistency checks |
| [Directional checks](results/directional_checks.csv), [momentum asymmetry](results/composite_momentum_asymmetry.csv), [spin components](results/spin_components.csv) | Information hidden by scalar averages |
| [Gaussian reference](results/gaussian_reference.csv), [paired gaps](results/gaussian_comparison_gap.csv) | Explicit conditional spatial/helicity comparison |
| [Sparse correlations](results/sparse_realspace_correlations.csv), [time blocks](results/time_history.csv) | Figure inputs; no fitted real-space critical exponents |

The runs record source hashes and revision `84fe77bc5c90822a1c74a613fb75a1ce04ac58fe`. The tested environment is Python 3.12.10, NumPy 2.4.6, Numba 0.68.0, with the installed CUDA backend on the local GPU; exact recorded versions are in each manifest. No tracked source modifications or existing-data writes were required.
