# Proposed measurements and pair-interaction calculation

**Status:** design only. No snapshot hooks, production observables, minimizer or new Monte Carlo runs are implemented here. Stage 1 should be reviewed before authorizing either proposal below. The first targets are zero SOC and weak nonzero SOC in an identified chiral regime.

## 1. Stage 2: minimal spatial measurements

### Existing capabilities and the smallest useful change

The full field already exists as a complex64 array with shape `(walkers,L,L,3,2)`. [Runtime.snapshot()](../../core/runtime.py) returns it in **physical walker order**. [ParallelTempering.snapshot()](../../core/parallel_tempering.py) already copies the full array and permutes it into **fixed temperature order**, using `temperature_to_walker`. Scalar measurements are scattered into temperature order separately. Configurations and site random-number streams remain in their original physical walker slots; only temperature assignments move.

Thus an initial snapshot study needs an occasional saver in the host orchestration, rather than a new update algorithm or a GPU defect-detection kernel. At a declared observation point, after a complete local sweep and its scheduled exchange, copy the selected full-resolution fields and exact assignment maps before another sweep starts. The existing same-default-stream ordering provides the necessary event order; a future asynchronous saver must explicitly preserve completion and buffer ownership. Never infer a snapshot's temperature from a coarse walker history or a later final mapping.

The simplest first version can use the existing temperature-ordered snapshot method, select a few temperatures on the CPU and discard the rest. This transfers all walkers, which may be costly. If that cost is material, copy the exact mapping once and copy only the corresponding physical walker slices, with no intervening update/exchange. That is a proposed optimization, not a requirement for a first scientific test. Snapshotting is read-only and must not change the update schedule, field state, RNG stream or energy cache.

At L=160 one complex64 3×2 field costs `160²×6×8=1,228,800` bytes, about 1.17 MiB. A complete 160-temperature snapshot costs 187.5 MiB before compression. This argues for a sparse schedule and a small selected temperature set, rather than saving all fields at the scalar-measurement cadence. Do not downsample the spatial lattice to save space: that can remove cores, merge nearby HQVs or alias their winding.

### Snapshot contents

Each independently readable snapshot or small chunk should carry:

- Full complex64 matrix field, the array axes and spin/orbital component order, lattice spacing convention and periodic boundaries.
- Sweep count since run construction, production-relative sweep, exchange round and matching/event order, snapshot phase such as `after_exchange`, temperature index/value and physical walker ID for each saved field.
- The exact two assignment maps at that observation point, or an unambiguous equivalent; selected temperatures should not be silently renumbered.
- L, K, all five numerical beta coefficients, g_D, pressure, physical coefficient parameter `reduced_temperature`, Monte Carlo temperature, initialization protocol, run identifier, independent seeds and source revision.
- Snapshot schedule, dtype, schema version and file integrity identifier. Detector version and thresholds belong in derived outputs, while the raw field is retained so classifications can be revisited.
- Scalar energy, density and existing order parameters at the same event when practical. They help verify field ordering and detect save/restart mistakes; they are not substitutes for the field.

Use a predetermined sweep schedule, or randomized observation times independent of the current state. Defect-triggered debug snapshots can be valuable, but must be flagged and excluded from naive population averages. Write each small chunk once and close it, rather than repeatedly rewriting a growing production NPZ archive. A snapshot is not a full restart checkpoint: RNG and all runtime state would be additional requirements for exact restart.

### What to compute offline first

Run the validated Stage-1 detector on full-resolution fields. Preserve local density, chirality, chiral-manifold residual, composite coherence, director eigenvalue gap, link resolution diagnostics and loop classification. Report at least:

1. Resolved positive/negative HQV candidates, resolved integer vortices, unresolved charged clusters, chirality/domain features and invalid regions, with coordinates and loop support.
2. Defect counts per lattice area and per snapshot, plus the area/loop fraction that fails quality criteria. A low count obtained by rejecting most of the field is not evidence of a low defect density.
3. Candidate survival under nearby loop sizes, mesh neighborhoods and a range of quality thresholds. Closely spaced opposite defects can disappear inside one loop; a charge-two loop may be one integer core or two HQVs. Record that ambiguity instead of forcing a one-core count.
4. Density/manifold-residual profiles near each candidate, local chirality maps, and the geometry of any SOC-related spin texture or nonchiral strip. A branch cut of a chosen director eigenvector is not a measured string.
5. Fixed-temperature time traces of these quantities and physical-walker traces separately. PT can switch the configuration seen at a temperature without creating or annihilating a defect inside a physical walker.

Start with CPU offline detection on occasional snapshots. Only after the definitions are stable should a frequently measured GPU count be considered. A count-only kernel would otherwise discard the evidence needed to distinguish false positives and unresolved loops.

Snapshot spacing must be assessed using the autocorrelation and slow-sector behavior of the **defect observables themselves**, not just energy or PT acceptance. Block uncertainties in sweep time and compare independently prepared runs if population inference is attempted. Snapshots at different temperatures share PT walkers and are not independent replications. A walker path through a tempering ladder is also not a fixed-temperature physical trajectory.

Periodic boundaries require consistent noncontractible-cycle bookkeeping as well as contractible core loops. For a well-defined, periodically sampled scalar composite, the total oriented plaquette winding sums to zero; net apparent charge signals invalid links, an inconsistent seam or a detection issue. This charge-balance check does not guarantee that all individual candidates are correctly resolved.

Pair-separation histograms can be descriptive but do not directly give an interaction energy. They include radial phase-space factors, finite-area geometry, many-body screening, arbitrary pairing in a dense gas and detector selection. Even an equilibrium dilute-pair inversion would require the measure/Jacobian and the constrained ensemble to be specified. Do not infer confinement merely because a nearest-neighbor distance histogram has a short tail.

## 2. Stage 3: define the interaction before minimizing

### A neutral pair is the clean first problem

The first calculation should be a **neutral HQV–anti-HQV pair**, with composite windings +1 and −1 on separate reliable loops, each accompanied by odd director holonomy. Enclosing both gives zero total composite winding and even director holonomy. Specify the spin texture used for initialization; an integer signed spin winding is not an invariant of a general three-dimensional director.

In a planar initialization one can choose

\[
\theta=\tfrac12(\varphi_+-\varphi_-),\qquad
\alpha=\tfrac12(\varphi_+-\varphi_-),\qquad
\hat d=(\sin\alpha,0,\cos\alpha),
\]

and `A=sqrt(n/2) exp(i theta) d (1,i sigma)`, with smooth density cores. Each half-angle branch flip changes both exp(i theta) and d by a sign and leaves A single valued. The Stage-1 construction is an initializer, not a relaxed solution. On a torus the initializer must be truly periodic; simply wrapping a plane atan2 field can add an energetic seam. A periodic phase/Green-function construction or compatible periodic fixture is needed before energy comparisons. A neutral pair also permits a large box with common homogeneous outer data, provided boundary effects are tested.

Do not confuse this problem with splitting an integer vortex into **two same-sign mass HQVs**. Their far-field charge and interaction differ, even if their spin paths oppose. A single unbalanced vortex in an ordinary periodic scalar-composite sector is not the same boundary-value problem as a neutral pair.

### Restricted continuum reference: include both stiffnesses

For an ideal constant-density fixed-chirality field, substitution into this repository's long-wavelength gradient energy gives

\[
F_{\rm grad}=2Kn\int[(\nabla\theta)^2+(\nabla\hat d)^2]d^2r
=\frac12\int[\rho_c(\nabla\theta)^2+\rho_d(\nabla\hat d)^2]d^2r,
\]

with bare rho_c=rho_d=4Kn. This is a derivation within the ansatz, not a statement about finite-temperature renormalized stiffnesses. A neutral planar pair with phase charges ±q and opposite spin-angle charges ±s has

\[
E_{+-}(R)\simeq E_{\rm cores}+2\pi(\rho_c q^2+\rho_d s^2)\ln(R/\xi),
\qquad \xi\ll R\ll L.
\]

For q=s=1/2, the coefficient is pi(rho_c+rho_d)/2. For an integer phase vortex–antivortex pair with no spin texture it is 2 pi rho_c. By contrast, two same-sign q=1/2 vortices with opposite planar spin paths have a separation-dependent coefficient pi(rho_d−rho_c)/2 at fixed far-field integer winding; this cancels at bare equal stiffnesses. The distinction is crucial for a proposed logarithmic-interaction test. General full-spin relaxation can change the planar texture, so the chosen spin-path calculation is a reference to challenge.

The combined phase/spin origin of HQVs and possible dipole-related connecting solitons appears in the primary helium work of [Volovik and Salomaa, Sov. Phys. JETP 61, 986 (1985)](https://jetp.ras.ru/cgi-bin/dn/e_061_05_0986.pdf). Its particular field/geometry assumptions are not imported into this zero-field 3×2 model. The distinction between spin-active zero-temperature energies and spin-disordered finite-temperature theories is also explicit in the related-manifold calculation of [Mukerjee, Xu and Moore, Phys. Rev. Lett. 97, 120406 (2006)](https://arxiv.org/pdf/cond-mat/0605102). Neither source substitutes for testing the full GL relaxation here.

## 3. Operational constraints and full-field relaxation

An unconstrained neutral pair can translate, annihilate, leave the box, split into other features, or leave the chiral manifold. Minimizing such a field and recording its final energy at the **initial** separation does not measure a pair potential.

A useful first controlled boundary-value problem would use small holes/annuli around two fixed centers, or localized pins, with imposed admissible HQV boundary maps and no constraint on a connecting line. Outside those small regions, relax **all twelve real field components** using the exact discrete Hamiltonian. Keep chirality, density and topology diagnostics on several contours around each center, not just on a pinned core site. A core value alone cannot impose a winding. If holes exclude the core, call the result an exterior pinned-defect energy and do not claim to have optimized its internal core energy.

The operational constrained energy should be stated explicitly:

\[
E_{\rm pair}^{\cal C}(R;L,g_D)
=\min_{A\in\mathcal C(R)}[H[A]-H[A_{\rm vac}]],
\]

where C specifies pin radii/strengths, inner boundary maps, external data, admissibility checks and any restrictions on extra defects. This is conditional on those constraints. It need not be the unrestricted ground-state energy, which may contain no HQVs.

Important controls are:

- Use identical hole/pin geometry and reference volume at every R; vary pin radius/strength and inner-loop texture. Allow phase offsets and spin paths to relax where compatible with the intended loop charges. Compare several initial spin planes and pair orientations relative to the square lattice.
- Do not hold A inside the chiral ansatz throughout the bulk, fix the analytic atan2 field outside the cores, or prescribe a strip of suppressed amplitude. Such restrictions can forbid genuine full-GL escape or manufacture a connecting object.
- Detect the actual defect coordinates, loop charges and extra defects after relaxation. Pinning a nominal center does not prove a unique defect remains there.
- Compare continuation in both increasing and decreasing R and several initial textures. Distinct relaxed minima must be retained as branches, rather than selecting whichever the optimizer encounters first. Report full gradient norm, constraint violations and convergence with tolerance, not only energy monotonicity.
- After obtaining a constrained branch, propose release controls: remove or weaken pins and check whether the pair survives, moves together, or loses its identity. This is a stability diagnostic of an energy landscape, not real-time dynamics.

**Full-field escape is a result, not an optimization nuisance.** Because the unconstrained nonzero matrix field has a much larger target space than the chiral vacuum manifold, inner-loop templates alone do not guarantee long-range HQVs. A nonchiral screening annulus, additional defects or a strong distortion can make larger enclosing loops trivial. A flat energy curve in that situation measures loss or screening of the proposed defects, rather than an unconfined HQV pair. Conversely, enforcing manifold proximity throughout the bulk to prevent that loss would change the variational problem. Report the constrained metastable excitation and the best unrestricted/released state separately. If no robust surrounding annulus supports the HQV labels, the pair-energy interpretation has failed for that branch.

Stage 1 includes [check_algebra.py](check_algebra.py), a small read-only CPU check that extracts the production polynomial helpers without their CUDA decorators. It evaluates a small periodic random field to test the Hamiltonian symmetries and the exact restricted-chirality gradient identity; it neither relaxes a field nor measures an energy-versus-separation curve. A future pair solver still needs its full energy accumulation, boundary/pin terms and derivatives independently verified, including directional finite-difference checks of the minimization gradient. No pair optimizer or Stage-3 interaction calculation has been added in Stage 1.

## 4. Weak SOC and a confinement test that can fail

Within the ideal chiral manifold, the excess dipole density is `g_D n(1−d_z²)`. The SOC-locked physical vacuum for fixed chirality is still a connected U(1) circle: `(d,theta)=(+z,phi)` and `(-z,phi+pi)` represent the same field. The two signs of a director lift are not disconnected bulk vacua and do not by themselves establish a topologically stable free wall. For a planar spin-tilt channel that retains a prescribed alpha:0→pi lift, the constant-density continuum spin functional is

\[
F_{\rm wall}/\text{length}=\int dx[2Kn(\partial_x\alpha)^2+g_Dn\sin^2\alpha].
\]

The first integral in this constrained spin channel gives the reference width and spin-wall contribution per unit length

\[
\ell_D=\sqrt{2K/g_D},\qquad
\tau_{\rm chiral}=4n\sqrt{2K g_D}.
\]

This is a **restricted long-wavelength channel estimate**, relevant only when the joint phase/director defect constraints retain that nontrivial lift. The common-phase texture and its energy must be included separately in an actual pair configuration. For a standalone interface with only opposite-phase physical end fields, a broad U(1) phase rotation with locked director is an alternative to the spin-tilt path; imposing the spin path would select a channel rather than prove a stable wall. Thus the formula alone does not demonstrate confinement. It is not a measured full-GL tension or a strict bound on the discrete lattice problem. Density depletion, nonchiral intermediate fields and different spin/orbital paths can lower its energy. If a complete trial field is later discretized with exactly the same joint defect and boundary constraints and evaluated using the actual lattice Hamiltonian, its energy supplies a variational upper bound on that constrained discrete minimum. A lower-energy escaped configuration must be reported, not projected back onto the assumed wall.

For a well-defined metastable HQV pair, the expected qualitative regimes are: both mass and spin logarithms below an effective anisotropy length; then a spin contribution cut off at that scale, a remaining mass logarithm, and possibly an added line-energy term. Schematically, a large-distance fit may use

\[
E(R)=C+A\ln(R/\xi)+\tau R+\text{finite-size/crossover corrections},
\]

but not across all distances indiscriminately. The identifying evidence for confinement is a reproducible positive long-distance slope after accounting for A/R and image forces, accompanied by a localized physical excess-energy/spin texture joining still-identifiable HQVs. A director eigenvector's arbitrary sign seam is not that texture. A smooth nonlinear curve over a short interval can fit both log and log-plus-linear forms without deciding the physics.

Useful diagnostics would compare finite-difference forces, `(E(R+delta)−E(R−delta))/(2 delta)`, under changes of L, pair orientation, pin prescription and initialization. A logarithm gives a force proportional to 1/R; a genuine tension adds a roughly constant term over a controlled interval. Energy differences between separations reduce the unknown core constant, while force differences at multiple L help identify boundary contamination. A nonzero fitted tau is insufficient if it drifts toward zero with increasing L, is confined to R comparable to L, or disappears when a pin-induced strip is released.

The desired hierarchy is `xi_core << ell_eff << R << L`, not merely `g_D>0`. At K=1, bare ell_D is about 44.7 at g_D=0.001 and 14.1 at 0.01. The previous L≤160 data sizes offer little clean R≫44.7 while keeping R≪L. Thus a failure to see a linear regime in a small weak-SOC box would not refute asymptotic confinement. Conversely, the bare length does not prove that a proposed box reaches the renormalized crossover. First establish cores, spin profiles and constraint independence in inexpensive authorized tests before specifying larger sizes.

Finite-size and baseline requirements include:

1. Subtract a homogeneous bulk reference of the **same** g_D, beta coefficients, outer boundary and geometric domain. Such a field cannot simultaneously obey winding inner-boundary maps; pin/hole self energies and their separation dependence must be accounted for separately. Alternatively compare E(R)−E(R0) with an identical pin prescription and test its residual dependence. SOC changes the vacuum amplitude and bulk energy; subtracting a zero-SOC reference can produce a spurious extensive contribution.
2. In a periodic neutral problem, compare with the torus/discrete-Laplacian Green-function form rather than an infinite-plane logarithm at R approaching L. Wrap distances consistently and test the pair's orientation and noncontractible sectors. In an open box, vary boundary distance and outer data to quantify image forces.
3. Separate core overlap at R of order xi, the dipole crossover, and image-dominated distances. Changing the nominal lattice spacing without consistently rescaling the discrete model is not automatically a continuum-limit check; mesh refinement of analytic detector fixtures is a different test from changing the physical lattice Hamiltonian.
4. Resolve excess energy into onsite, gradient and dipole contributions and map it spatially. This tests whether an apparent linear term is a connecting texture, a broad misaligned bulk region in an unrelaxed initializer, a boundary seam, or a changing constrained core.
5. Do not average distinct metastable branches into one smooth energy curve. Their ordering, escape routes and topology validity may themselves answer why a simple HQV interaction picture fails.

## 5. Three distinct physical questions

| Procedure | Quantity addressed | What it does not establish |
|---|---|---|
| Zero-temperature full-GL constrained minimization | Classical minimum excess energy of a specified pinned/metastable defect sector | Entropic interaction, equilibrium populations, a finite-T transition, or physical motion |
| Finite-temperature constrained/umbrella sampling | Potential of mean force or free-energy difference for a declared collective variable and measure, after equilibration checks | A bare zero-T energy; a raw separation histogram requires geometric/Jacobian and many-body corrections |
| Physical dynamical evolution | Time-dependent motion only after choosing a justified dynamical model and time calibration | Metropolis sweeps, stochastic overrelaxation, PT exchanges and generic optimization steps are not such a real-time model |

A later finite-temperature calculation would need reversible sampling of constrained sectors, overlap between windows and an explicit definition of separation and charge identity. It would measure a free energy including entropy and screening, not the deterministic curve above. None of these further calculations is authorized or executed by the Stage-1 proof of concept.
