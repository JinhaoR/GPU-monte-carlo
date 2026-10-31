# Stage 2: thermal snapshots, HQV detection and an unrelaxed pair-energy baseline

**Status:** one short zero-SOC pilot and the synthetic energy baseline completed. No weak-SOC pilot or constrained minimization was run. All new files are under `exploration/he3_hqv_stage2/`; production code, Stage 1 and original datasets are unchanged.

## Answers to the scientific questions

| Question | What this stage establishes |
|---|---|
| Did convincing HQVs appear in actual MC configurations? | **None could be confirmed by the tested detector.** All proposed raw charged clusters have invalid surrounding contours. This does not establish that HQVs are absent. |
| Were the surrounding fields sufficiently chiral? | Generally not at the microscopic site scale under the Stage-1 tolerance. Even the lower-temperature control has about 79% locally invalid sites; near 0.3815 the fraction is 93%. |
| What prevented classification? | Substantial opposite-orbital and nonreal-spin weight, mixed selected chirality, director ambiguity and unresolved phase links. No raw site falls below the density floor. Sampling and the simple contour/cluster search also remain limitations. |
| Does the synthetic pair energy reproduce the continuum benchmark? | **Yes, as an unrelaxed kinematic consistency check in a suitable regime.** At L=512, core width 3 and R=24–64, the open-plane trial slope is 18.7353 versus the predicted 18.7959, a −0.32% difference. |
| What is the next credible interaction calculation? | Full twelve-component relaxation of a neutral pair with small independent annular constraints, a common outer reference, and multiple surrounding topology checks. See §5 for a concrete proposed problem. |
| Which statements remain hypotheses? | Equilibrium HQV populations, their relaxed interaction, a weak-SOC crossover and genuine confinement are all unresolved. No separation free energy or real-time dynamics has been measured. |

The most useful new lesson is that **synthetic success does not by itself calibrate microscopic thermal defect counting**. The full-matrix manifold diagnostic exposes a limitation that would be hidden by counting every composite plaquette winding. The energy calculation independently verifies the phase-plus-spin logarithmic reference while showing that an imposed finite-box trial texture can produce large nonlogarithmic contributions even at zero SOC.

## 1. A minimal, verified spatial snapshot workflow

### Existing interfaces and the implementation

`Runtime.snapshot()` already returns the full state in physical walker order. `ParallelTempering.snapshot()` copies all walkers and reorders them by temperature. Both are correct, but the latter transfers more fields than this pilot needs.

[pilot.py](pilot.py) uses the existing Gaussian runtime and PT wrapper without a production hook. At an observation event it:

1. Completes the local sweep and its scheduled exchange and synchronizes the default CUDA stream.
2. Copies both exact inverse temperature↔walker maps and checks their inverse relationship.
3. Copies only the five corresponding contiguous physical walker slices from the existing GPU state.
4. Takes the matching ordered scalar measurement, verifies field/scalar density agreement, and checks that assignments and sweep count did not change during capture.
5. Writes one independently readable small NPZ chunk.

Every saved field has shape `(64,64,3,2)` and dtype complex64. There is no spatial downsampling. A chunk contains five fields, exact maps, selected temperature indices/values, physical walker IDs, sweep and exchange round, axis/component order, periodic boundaries, seeds, sampler settings, model coefficients, pressure, observation event, source revision and SHA256 hashes. `run.json` supplies the full 35-temperature ladder and cumulative diagnostics. Together they specify a reproducible run, not a complete RNG restart checkpoint or a promise of bitwise reproducibility on other hardware.

The copy is selective: five walker fields are transferred, rather than all 35 simulated replicas or the old campaign's 160 replicas. Thirteen chunks occupy about 11.9 MB compressed. No existing simulation file is overwritten; the runner refuses to reuse an existing output directory.

[pilot_checks.py](pilot_checks.py) independently reopens all thirteen chunks and verifies maps, indices, temperatures, shape, dtype, density and signed chirality against the saved scalars. Source hashes match the files used for the run. [snapshot_validation.json](results/snapshot_validation.json) records per-chunk field hashes and arithmetic differences. The small L=8 GPU smoke test also compares the saved-field CPU energy to the production GPU energy, with maximum absolute difference 1.23×10^−6. A later comparison on five actual L=64 fields has maximum absolute energy difference 1.55×10^−5 for total energies of order 3600–6600.

### Pilot actually executed

| Setting | Actual value |
|---|---|
| Pressure / K / coefficient reduced_temperature | 20 bar / 1 / 1 |
| SOC / lattice size | g_D=0 / L=64 |
| Sampler | Existing conditional-Gaussian full-Hamiltonian MH, rho=−0.9, proposal mass 0 |
| PT ladder | 35 temperatures, 0.33–0.46; denser from 0.372–0.396 |
| Saved temperatures | 0.35, 0.375, 0.3815, 0.389, 0.41 |
| Thermalization / measurement | 30,000 / 60,000 sweeps |
| Exchange interval | 5 sweeps |
| Scalar measurements | Every 250 sweeps, including thermalization for diagnostics |
| Spatial observations | Thermalization endpoint and every 5,000 subsequent sweeps: 13 per temperature, 65 fields total |
| Local / PT seeds | 20261031 / 20261032 |
| Recorded runtime | 18.36 s total, including setup; sampling/capture loop about 15.11 s |

The runner has a 300-second sampling-loop cap and never changes L or adds a coupling automatically. An initial L=8 smoke attempt exposed nearly duplicate floating-point temperature labels that collided after conversion to float32 inverse temperatures. Rounding the intended ladder to seven decimal places removed those duplicates before the L=64 pilot started. Its small preliminary smoke file is retained under `results/failed_initial_ladder_smoke/`.

Only this zero-SOC pilot was run. Successful pair-energy and thermal-analysis scripts take tens of seconds each; development reruns and checks also remained well within the requested roughly ten-minute computation allowance. No cluster job or large parameter campaign was launched.

### Sampling quality: exploratory, not demonstrated equilibrium

The pilot starts from the production homogeneous positive-chirality z-spin configuration. There is one seed and thirteen correlated observations per saved temperature. Mean cumulative swap acceptance is 0.3218, but the weakest edge, 0.40↔0.41, has acceptance only 0.002889. There are zero completed PT round trips across the full ladder. A future short pilot should use more closely spaced temperatures through such edges rather than assume the average acceptance is adequate.

Energy and absolute chirality have reasonably similar production-half averages, but this is limited evidence. At T=0.375, Qzz changes from −0.06655 to +0.02198 between halves; its contrast is about 3.11 times the descriptive combined error using 1250-sweep blocks. This is neither a calibrated significance test nor proof of a specific slow mode: the chosen blocks and one ordered-start history do not establish independence or absence of longer memory. No equilibrium error bars are assigned to the snapshot defect counts. [pilot_half_checks.csv](results/pilot_half_checks.csv) contains all five temperatures and four observables.

![Short pilot traces and PT acceptance](results/pilot_sampling_diagnostics.png)

## 2. Applying the unchanged Stage-1 detector to real fields

### Definition and primary results

For the complete matrix, Stage 1 computes the nearest unitary rank-one chiral fit, with relative **squared** residual

\[
\epsilon=\min_{\sigma=\pm1}\left[1-
\frac{\lambda_{\max}(\operatorname{Re}z_\sigma z_\sigma^\dagger)}n\right],
\quad z_\sigma=(A_x-i\sigma A_y)/\sqrt2.
\]

An accepted loop also requires coherent fixed chirality, a director eigengap, nonzero charge-two fields, resolved links, agreement of composite and lifted phase charge, and the correct director parity. The automatic finder proposes elementary seeds, groups adjacent seeds, and requires a consistent nonzero charge on at least two surrounding contours. The original code is reused unchanged. Its finite-radius square contours and connected-seed grouping are a proof of concept, not an exhaustive search over all possible enclosing paths.

The primary analysis uses the original strict tolerance epsilon≤0.12 and the original remaining gates. Counts below are **per snapshot**, shown as mean and range over thirteen fields. Accepted positive HQVs, negative HQVs and integer-net-charge clusters are all zero at every saved temperature and snapshot.

| T | Accepted +HQV / −HQV / integer clusters | Unresolved clusters, mean [range] | Locally invalid area | Invalid attempted contours | Mean snapshot median epsilon |
|---:|---:|---:|---:|---:|---:|
| 0.3500 | 0 / 0 / 0 | 40.54 [32–52] | 79.44% | 100% | 0.2056 |
| 0.3750 | 0 / 0 / 0 | 149.38 [123–174] | 89.89% | 100% | 0.2901 |
| 0.3815 | 0 / 0 / 0 | 190.38 [154–211] | 92.90% | 100% | 0.3360 |
| 0.3890 | 0 / 0 / 0 | 261.46 [243–294] | 98.01% | 100% | 0.4284 |
| 0.4100 | 0 / 0 / 0 | 239.00 [217–267] | 98.78% | 100% | 0.4389 |

There are 11,450 raw seed clusters and 34,350 attempted enclosing contours across the fields. Every strict attempted contour fails the manifold residual criterion; most also fail other gates. These are overlapping rejection reasons, not disjoint defect species. Many unresolved clusters can arise from microscopic phase noise and composite cancellation. They must not be presented as a measured vortex density.

Detailed per-snapshot, candidate and reason tables are [thermal_snapshots.csv](results/thermal_snapshots.csv), [thermal_candidates.csv](results/thermal_candidates.csv), [thermal_rejections.csv](results/thermal_rejections.csv) and [thermal_summary.csv](results/thermal_summary.csv).

![Actual raw thermal fields, including invalid areas and unresolved seeds](results/thermal_raw_fields.png)

### What the local-quality failure means

The raw field has no site below the density floor n=0.01. The dominant issue is therefore not a vanishing full matrix amplitude on the contour. The exact residual decomposition exposes two contributions:

\[
\epsilon=\frac{\|z_{-\sigma}\|^2}{n}
+\frac{\|z_\sigma\|^2-|z_\sigma\cdot z_\sigma|}{2n}.
\]

The first is opposite-orbital weight in the fit-selected sector; the second penalizes a complex spin vector that is not a real director times a phase. Averaged over raw sites and snapshots:

| T | Mean epsilon | Opposite-orbital contribution | Nonreal-spin contribution |
|---:|---:|---:|---:|
| 0.3500 | 0.2305 | 0.1857 | 0.0448 |
| 0.3750 | 0.3042 | 0.2445 | 0.0597 |
| 0.3815 | 0.3379 | 0.2716 | 0.0664 |
| 0.3890 | 0.4111 | 0.3287 | 0.0824 |
| 0.4100 | 0.4217 | 0.3361 | 0.0855 |

This is a direct observation that these microscopic fields differ substantially from the synthetic chiral reference. It does not prove that their long-wavelength ordered sector lacks a chiral or HQV description. The local criterion requires every vertex of a loop to be suitable; it becomes restrictive in a noisy twelve-component field. Nor are local sign changes automatically macroscopic chirality domains: ultraviolet fluctuations and poorly defined local fits contribute.

### Threshold sensitivity and an explicitly separate smoothing diagnostic

Two raw-field tests increase the residual ceiling to 0.20 and 0.30. Their chirality, eigengap and composite-coherence floors are adjusted to the corresponding exact bound 1−2 epsilon; link-resolution safeguards remain in place. Neither produces accepted HQV or integer clusters. At epsilon≤0.30, locally valid fractions rise to approximately 73.6%, 52.3%, 42.5%, 21.6% and 18.5% for increasing T, but nearly all proposed loops still cross a bad region or fail another consistency check. Relaxing a single site-quality threshold does not establish a defect.

To distinguish microscopic noise from broader structure, an offline full-grid linear heat-kernel diagnostic uses

\[
A'_{ij}=\tfrac12A_{ij}+\tfrac18(A_{i+1,j}+A_{i-1,j}+A_{i,j+1}+A_{i,j-1})
\]

with periodic rolls, for four or eight passes. Its per-axis variance is passes/4, giving smoothing scales 1 and sqrt(2) lattice spacings. **Saved fields are untouched and no coordinates are decimated.** The resulting fields are smoothed proxies, not independent Monte Carlo samples or energy minima. Smoothing can merge defects, change their cores, or remove a close pair.

The same processing recovers the known well-separated Stage-1 synthetic charges and rejects its nonchiral controls; [smoothing_calibration.csv](results/smoothing_calibration.csv) records this limited calibration. It does not establish faithful topology preservation in a thermal field. After eight passes and the original strict gates:

| T | Locally invalid fraction | Mean unresolved clusters | Accepted HQV / integer clusters |
|---:|---:|---:|---:|
| 0.3500 | 4.73% | 0 | 0 / 0 |
| 0.3750 | 32.94% | 1.38 | 0 / 0 |
| 0.3815 | 53.11% | 14.77 | 0 / 0 |
| 0.3890 | 94.11% | 118.46 | 0 / 0 |
| 0.4100 | 97.72% | 183.00 | 0 / 0 |

The low-T proxy is mostly chiral and has no elementary seed in these snapshots. That is compatible with few well-separated defects in the observed, ordered-start state, but does not determine equilibrium rarity. Near and above the ordering region, even smoothing does not supply reliable surrounding loops. A missing proxy count can also reflect pair cancellation, merging and the candidate algorithm's incompleteness.

![Director line structure on raw and separately labelled smoothed fields](results/thermal_director_proxies.png)

Glyphs are unoriented line segments using the spin components (dx,dz), not velocity arrows in physical space. They are drawn only where the local geometry passes. Gray crosses are unresolved seeds; any accepted half or integer charges would receive distinct signed markers, but none passes here.

![Temperature and detector-quality sensitivity](results/thermal_quality_summary.png)

### Individual candidate checks

The representative last T=0.3815 eight-pass proxy has a seed anchored at (8.5,1.5). On successive periodic enclosing squares:

| Radius parameter | Raw composite winding | Maximum epsilon | Director parity | Accepted? |
|---:|---:|---:|---|---|
| 1 | +1 | 0.6366 | Undefined by quality gate | No |
| 2 | +1 | 0.6141 | Undefined by quality gate | No |
| 4 | 0 | 0.5099 | Undefined by quality gate | No |

The smaller loops also have phase-link/coherence problems. Calling the initial +1 an HQV would discard precisely the evidence that its surrounding matrix cannot support that interpretation. Wider loops with radius parameters 8 and 12 were checked for all sixteen representative proxy candidates at T=0.375 and 0.3815. Across 80 candidate/loop-size checks, none passes. This does not exhaust adaptive non-square loops, but shows that simply extending the fixed square search does not solve the problem. See [representative_candidates.json](results/representative_candidates.json) and [wider_candidate_loops.csv](results/wider_candidate_loops.csv).

### Decision about weak SOC

**No g_D=0.001 pilot was executed.** It would be cheap, but the zero-SOC workflow does not yet yield a validated thermal HQV population to compare. Repeating unresolved counts at another coupling would not answer the confinement question. The runner retains an explicit optional weak-SOC switch for a later deliberate run; it does not execute it automatically.

The next thermal-method test should calibrate a scale-dependent coarse field/contour definition on realistic fluctuations, quantify topology changes caused by coarse graining, and test adaptive loops or ordered-region boundaries while preserving invalid-region reporting. Existing saved fields suffice for that methodological work. An independent preparation or improved small ladder would then address sampling bias. More snapshots alone would not repair the current definition-to-thermal-field mismatch.

## 3. Exact lattice-energy validation and trial pair construction

### Full energy and boundary conventions

[energy.py](energy.py) evaluates the full six-complex-component polynomial and actual three-contraction forward gradient in float64. It uses the saved pressure-20 coefficients, K=1, and the production float32 literal for 2/3 in the dipole term. It does not replace the gradient by an XY interaction or project an arbitrary input field onto the chiral manifold.

Independent tests evaluate the actual production scalar helper definitions on small random matrices, removing only their CUDA decorators in memory. The zero-SOC source-loop/CPU differences are below 5×10^−13. Tests also cover g_D=0.001 and both explicit boundary conventions. The actual thermal GPU/CPU comparisons above exercise normalization and the periodic stencil. [energy_validation.json](results/energy_validation.json) records the checks.

Production boundaries are periodic. For the open-plane trial family, the chosen **open complete-cell convention** includes every site potential and the same gradient density only on cells with both forward neighbors inside the box. It is a declared boundary truncation, not an assertion that all possible free-boundary discretizations coincide. Every open trial and its reference use this identical convention. Finite-L comparisons expose its boundary dependence.

For an ideal chiral field, beta_A=beta2+beta4+beta5=0.6685707569. The homogeneous zero-SOC branch has

\[
n_0=1/\beta_A=1.4957279984,\qquad
H_0=-L^2/\beta_A.
\]

The evaluator reproduces that energy and zero gradient exactly within numerical precision. Trial fields are scaled to this stationary bulk amplitude; using density 1 with the same Hamiltonian would introduce a different amplitude baseline. The physical coefficient parameter remains `reduced_temperature=1`; the later phrase “zero-temperature relaxation” means minimizing this fixed classical Hamiltonian, not recalibrating helium coefficients to zero Kelvin.

### Neutral full-matrix trial fields

The Stage-1 branch-independent pair is reused, with opposite half charges and the same planar phase/spin half-angle path. Core amplitude is changed consistently to

\[
f=\sqrt{n_0}\prod_{a=1,2}\tanh[(r_a/\xi)^2],
\]

with the default width xi=3 lattice spacings. Quadratic vanishing gives a C1 matrix at the idealized core, unlike a bare half-angle sign seam; the sampled centers lie between lattice sites. These resolved, deliberately prescribed cores are not optimized GL cores. Widths 4 and 6 test this choice.

Two texture families are compared:

- **Open plane:** the single-valued Stage-1 pair texture evaluated with open complete-cell boundaries. Its outer trace changes with R; finite-box tails are part of the measured trial energy.
- **Compact periodic:** the same pair angle is unchanged within radius 0.30L and smoothly tapered to zero by 0.46L with a quintic cutoff. The only angle branch seam lies between the cores, entirely where the cutoff is one, so the physical matrix stays continuous. Outside the cutoff the field is the homogeneous reference to machine precision. Opposite boundary values match to 10^−13. This is a genuinely periodic but intentionally unrelaxed texture, not the periodic Green-function minimum.

Multiplying an arbitrary discontinuous half-angle field by a cutoff would be unsafe; the construction preserves exact branch cancellation in the only cut-containing region. No connecting wall is inserted into the energy functional.

Fifty-two small CPU trial evaluations cover L=128,256,512 and R=8,12,16,24,32,48,64, with two additional core widths at the largest box. These larger L are **synthetic CPU fields**, not additional Monte Carlo systems. All 52 configurations have two accepted separate core contours per defect and an accepted enclosing neutral contour: 260 checks with the original detector. [pair_topology_checks.csv](results/pair_topology_checks.csv) records them. No global field minimization occurs.

![One branch-safe periodic trial and its excess energy](results/pair_trial_texture.png)

## 4. What the unrelaxed energy-versus-separation calculation shows

The restricted long-wavelength gradient is

\[
F=\frac12\int[\rho_c(\nabla\theta)^2+\rho_d|\nabla d|^2],d^2r,
\quad \rho_c=\rho_d=4Kn_0.
\]

For a neutral pair with phase charges ±1/2 and opposite planar spin paths ±1/2,

\[
E(R)=C+\frac\pi2(\rho_c+\rho_d)\ln(R/\xi)
=C+4\pi Kn_0\ln(R/\xi),
\]

so the bare reference coefficient is **18.79587237**. Omitting the spin contribution would halve this benchmark. This relation is a reference within the prescribed chiral texture, not a theorem about a relaxed full-field pair or a finite-temperature free energy.

The full lattice excess energies and descriptive least-squares log slopes are in [pair_energies.csv](results/pair_energies.csv) and [pair_log_fits.csv](results/pair_log_fits.csv). No random errors, physical confidence intervals or universality claims are attached to deterministic fits.

| Trial family / L / core width | Fit range R | Log slope | Relative difference from 18.7959 |
|---|---|---:|---:|
| Open plane / 512 / 3 | 24–64 | 18.7353 | −0.32% |
| Open plane / 512 / 3 | 16–64 | 18.8831 | +0.46% |
| Open plane / 256 / 3 | 24–64 | 18.1076 | −3.66% |
| Open plane / 128 / 3 | 24–64 | 15.4805 | −17.64% |
| Compact periodic / 512 / 3 | 24–64 | 20.4340 | +8.72% |
| Compact periodic / 256 / 3 | 24–64 | 24.9129 | +32.54% |
| Compact periodic / 128 / 3 | 24–64 | 43.3508 | +130.64% |
| Open plane / 512 / 4 | 16–64 | 19.0555 | +1.38% |
| Open plane / 512 / 6 | 16–64 | 19.4005 | +3.22% |

For the most favorable 24–64 open-plane fit, the maximum absolute departure from its log line is 0.035 energy units. This small numerical deviation is a consistency result, not an uncertainty on a measured interaction potential.

The onsite excess for the width-three open pair is essentially constant, 56.38650776, for R≥16. Its change at R=8 is visible core-overlap contamination. At L=512 the gradient contribution rises from 54.3591 at R=16 to 80.5462 at R=64; it accounts for nearly all the separation dependence there. Enlarging the box reduces the missing far-tail effect of the open construction. Increasing core width changes near-core and overlap corrections, as expected for a prescribed profile.

The compact periodic family is branch safe and seam free, yet its angle-tapering annulus adds R-dependent energy. That contamination decreases with L but can overwhelm the logarithmic slope in a small box. It must not be mistaken for a confining term: **all these trials have g_D=0**. Fitting a log-plus-linear curve to a short unrelaxed finite-box family would not establish a string tension.

![Full lattice energies, logarithmic comparison and local slopes](results/pair_energy_baseline.png)

This establishes a useful baseline: the evaluator, normalization, half charges and combined spin/phase coefficient agree in an appropriate open-plane regime. It does not determine the minimum energy at fixed separation. Trial energies are variational values for their particular boundary fields; the two families even have different boundary problems. Their comparison quantifies sensitivity and cannot be used to rank thermodynamic phases or to identify a measured relaxed potential.

## 5. Concrete proposal for the subsequent constrained relaxation

The simplest credible next calculation is a small **zero-SOC neutral-pair constrained relaxation**, with the following declared boundary-value problem. No solver or campaign has been implemented here.

1. **Common outer data:** evaluate the existing periodic lattice Hamiltonian but keep a one- or two-cell outer frame at the same homogeneous chiral A0 for every R. The neutral compact initializer is compatible with this frame. The periodic wrap bonds then connect identical boundary values; the remaining outer-frame image forces still require an L check. Start with a few R well beyond core overlap, for example 16,24,32,48 in a method-validation box, and a second L only to test boundaries.
2. **Independent small annuli around the two centers:** impose or softly pin admissible +1/2 and −1/2 loop templates on narrow rings of radius roughly 2–3 sites. Vary that radius and pin strength. Unlike fixing a core matrix value, a loop carries the intended winding. A small annulus leaves the interior core fields free, permitting density suppression, nonchiral filling and nonunitary structure. A hole-based variant is simpler but measures only an exterior energy and cannot claim an optimized core.
3. **Full GL variation elsewhere:** relax all twelve real components inside and outside the annuli, with the exact stencil. Do not constrain a line joining the cores, force d to one spin plane throughout the bulk, or project the relaxed field back onto the chiral ansatz. Permit phase offsets and allowed template orientation parameters to relax, especially at zero SOC, so unnecessarily rigid pin data do not define the interaction.
4. **Preserve identification, not merely the nominal centers:** check several larger loops around each core, the neutral outer loop, extra defects and local energy/manifold maps throughout relaxation. If a nonchiral screening shell or compensating defects make larger loops trivial, report topology loss/screening as a physical alternative, and do not label the final energy an HQV pair potential. Conversely, forbidding every such escape would change the question. Retain constrained metastable branches and released/unrestricted results separately.
5. **Energy and optimizer controls:** independently validate full-field derivatives against finite differences before optimizing. Record gradient norms, constraint residuals and energy changes. Use several initial spin textures and continuation directions. Compare E(R)−E(R0) with unchanged pins to reduce core constants, then test pin-radius dependence. A homogeneous reference cannot also obey winding annular maps, so bulk subtraction alone does not remove pin/self-energy contributions.

For zero SOC, test the long-distance log coefficient and finite-size force against the combined-sector benchmark, while allowing the full field to choose a lower-energy route. Only after that problem is controlled should the same operational defect sector be compared at g_D=0.001.

At weak SOC, the restricted bare length sqrt(2K/g_D) is about 44.7 lattice spacings for 0.001. A convincing confinement test needs retained HQVs, an identified physical connecting texture, and a regime beyond the actual anisotropy crossover while remaining far from boundaries. A fit of C+A ln R+tau R requires stable positive excess force after the A/R and boundary terms are accounted for; it should survive changes of L, pin prescription, initialization and spin path. A smooth director sign convention is not a physical string, and a short-range curve in a small box cannot settle asymptotic confinement.

There is an additional important comparison: two same-sign mass HQVs splitting an integer vortex have different far-field data from the neutral pair. Their restricted phase/spin logarithmic terms can cancel at equal bare stiffness. They should not replace the neutral HQV–anti-HQV baseline without redefining the problem.

Finally, keep the physical quantities distinct:

| Calculation | Quantity actually obtained |
|---|---|
| This pilot | Configurations generated by a short canonical-MC/PT sampling procedure, without established equilibrium |
| This synthetic baseline | Energies of prescribed, unrelaxed static matrix fields |
| Proposed constrained minimization | Minimum classical energy of a stated defect/pin/boundary sector |
| Future equilibrated constrained sampling | Separation free energy or potential of mean force, including entropy and a specified measure |
| Real-time dynamics | Requires a justified dynamical equation and time calibration; neither sweeps nor optimizer steps supply it |

## 6. Reproduction, scope and evidence files

All commands run from the project root. The pilot is deliberately opt-in as a separate command; analysis commands never launch it.

```powershell
# Runs the one bounded GPU pilot; refuses existing output directories.
python -B exploration/he3_hqv_stage2/pilot.py

# Reuse the saved fields; no more Monte Carlo.
python -B exploration/he3_hqv_stage2/pilot_checks.py
python -B exploration/he3_hqv_stage2/thermal_analysis.py
python -B exploration/he3_hqv_stage2/geometry_diagnostics.py
python -B exploration/he3_hqv_stage2/pair_baseline.py
```

The existing pilot is already saved. To inspect or reproduce the analysis now, run the last four commands. A deliberate future rerun of the pilot requires a fresh output location or explicit archival of the previous exploratory directory; the runner does not silently overwrite it. Its optional `--weak-soc` flag selects 0.001, but was not used in this investigation.

No new dependency was installed. The environment uses Python 3.12.10, NumPy 2.4.6, SciPy 1.18.1, Matplotlib 3.11.2 and Numba 0.68.0 with the available local NVIDIA GPU. The `-B` flag prevents incidental bytecode edits to tracked production caches. Trial energies and detection run on the CPU. The source hashes, arithmetic checks, all 260 trial topology checks, saved fields, six selected PNG figures and numerical tables remain in [results](results/).

The strongest justified conclusions are the verified snapshot pipeline, the observed failure of microscopic thermal loop quality under the tested detector, and the successful unrelaxed logarithmic normalization check. They provide a concrete route toward the supervisor's question without assuming HQV abundance, relaxed logarithmic interactions, or SOC confinement.
