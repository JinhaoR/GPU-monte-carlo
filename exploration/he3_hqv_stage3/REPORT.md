# Stage 3: full-GL relaxation of a constrained HQV–anti-HQV pair

**Outcome:** a validated constrained interaction-energy prototype, with surviving HQV loop charges and explicit pin/boundary limitations. All calculations are deterministic classical energy minimizations. No Monte Carlo, GPU campaign or production-code change was made.

## Direct answers

1. **Can an identifiable pair survive full twelve-component relaxation? Yes, under the declared annular constraints.** All 21 pinned calculations converge and retain the individual +1/2 and −1/2 charges beyond the pins, odd director parity and a neutral outer loop. All 245 final extended-loop checks agree with their expected charges. The full-field candidate search finds exactly the two accepted HQVs and no additional accepted or unresolved seed clusters. This supports the particular constrained excitations, not unconstrained stability.
2. **How does its zero-SOC energy depend on separation? It increases in every tested box, with substantial finite-size corrections.** At L=192, the relaxed excess energies at R=16,24,32,48 are 62.036144, 69.611060, 75.317568 and 84.015311. Their differences are compatible with a logarithmic contribution near the bare benchmark, but the interval-dependent slopes are not constant. An asymptotic interaction law or unique intrinsic coefficient is not established.
3. **Does the full field find a different low-energy structure? It reconstructs the cores and substantially lowers the trial energies, but does not screen away the pinned HQVs in the tested minima.** Core density stays finite at sampled sites, and complex-spin/opposite-chirality components develop. A distinct nonplanar initialization reaches the same minimum to about 2.5×10^−9 energy units. No competing lower constrained branch was found in this limited test. Releasing the annuli removes the pair and reaches the homogeneous reference.
4. **Is the method ready for weak SOC? Numerically, yes for a controlled preliminary comparison; physically, not yet for a confinement claim.** Two g_D=0.001 calculations converge with surviving pairs. Relative to zero SOC, their extra excess energies are 0.662569 at R=16 and 1.717362 at R=32. Both separations are below the bare dipole length 44.7. Fixed pin textures and outer boundaries still affect the curve, so these results do not measure an asymptotic string tension.

The central advance over Stage 2 is an actual stationary full-matrix field with verified exterior charges and measured energy differences. The central remaining limitation is that **the energy is conditional on fixed annular texture data**. Rotating one pin's spin plane changes the small-box separation-energy difference by about 14%, making that qualification quantitatively important.

## 1. The constrained problem, precisely defined

### Hamiltonian and reference

The calculation uses the production pressure-20 coefficients, K=1, and the full complex 3×2 matrix A at each site. The energy density is reused directly from [Stage 2's evaluator](../he3_hqv_stage2/energy.py), which was checked against the production polynomial and GPU energies. All five quartic invariants, the SOC invariant and the actual three-contraction forward-difference gradient remain present. There is no XY replacement or chiral-only minimization.

The periodic gradient for one spin row, with a=Delta_x A_x, b=Delta_x A_y, c=Delta_y A_x and d=Delta_y A_y, is

\[
G=3(|a|^2+|d|^2)+|b|^2+|c|^2+2\operatorname{Re}(a^*d+b^*c).
\]

The coefficients include beta_A=beta2+beta4+beta5=0.6685707569. At zero SOC the homogeneous chiral reference has n0=1/beta_A=1.4957279984 and energy density −1/beta_A. For nonzero coupling the code uses the production float32 literal c0=float32(2/3):

\[
n_0(g_D)=\frac{1+g_Dc_0/2}{\beta_A},\qquad
e_0(g_D)=-\frac{(2+g_Dc_0)^2}{4\beta_A}.
\]

The objective is the excess Hamiltonian

\[
\mathcal E[A]=\sum_r[h_r(A)-e_0(g_D)].
\]

Subtracting the constant density before summation reduces cancellation and prevents an extensive negative bulk offset from making the optimizer's relative function tolerance misleading. It does not change the gradient. This is a double-precision evaluation of the same lattice polynomial and coefficients, rather than a claim of bitwise equivalence to float32 GPU arithmetic.

The model's physical coefficient parameter `reduced_temperature=1` remains fixed. “Zero temperature” here means minimizing this fixed classical functional, not changing the phenomenological helium coefficients to zero Kelvin.

### Common outer boundary and pin geometry

The evaluator retains all periodic bonds. A two-site outer frame is fixed at the same homogeneous positive-chirality field

\[
A_0=\sqrt{n_0/2}\,\hat z(1,i).
\]

Wrap bonds consequently connect identical fixed values. The surrounding frame is common to every R at fixed coupling; it still exerts finite-box image forces. Periodic bonds plus a frozen outer frame are a **fixed-boundary constrained problem**, not a freely fluctuating periodic torus.

For even L and R the centers are

\[
(c_{x1},c_y)=(L/2-1/2-R/2,L/2-1/2),\quad
(c_{x2},c_y)=(L/2-1/2+R/2,L/2-1/2).
\]

At each center a square annulus is the set of lattice sites at Chebyshev distance p+1/2. The primary p=2 pins have radius 2.5 and 20 sites each. Their core interiors remain free. A p=3 control uses radius 3.5 and 28 sites per annulus. No site on a line connecting the annuli is specially constrained.

The isolated annular templates are

\[
A_i=\sqrt{n_0}\,e^{i\theta_i}\hat d(\alpha_i)(1,i)/\sqrt2,
\quad \hat d(\alpha)=(\sin\alpha,0,\cos\alpha),
\]

with left theta=alpha=(phi_left−pi)/2 and right theta=alpha=−phi_right/2. These are single-valued physical matrices because a half-angle branch change changes both factors' signs. The annular maps do **not** contain the other core's R-dependent angular field. Their site count, amplitude and local angle pattern are identical as R varies; only the centers translate.

All twelve real components at every site outside the outer frame and two annuli are optimization variables, including the core interiors. For example, L=64,p=2 has 42,720 free real variables; L=192 has 423,648. No residual-to-manifold penalty, spin-plane restriction, density floor or topology projection enters the objective.

The initial field reuses the branch-safe compact Stage-2 trial, then overwrites the frame and annuli with these declared data. A small complex perturbation in **all** free components prevents exact initialization symmetries from silently restricting the search. A separate smooth nonplanar initialization tests another route to a minimum.

### Why fixed annuli first, and what they overconstrain

Exact fixed variables give unambiguous constraints, zero constraint error and a physical Hamiltonian without an extra penalty energy. A soft term lambda sum_pin ||A−A_pin||² would require separating penalty from physical energy and checking whether finite lambda permits charge loss. Its nominal strength would not guarantee a particular topological sector. Fixed annuli are therefore the clearer first prototype.

They nevertheless fix more than charge: amplitude, phase offsets, local spin path, spin-plane orientation, chirality and detailed lattice texture. An unpinned HQV pair can optimize those choices. Radius and relative spin-plane controls below measure part of this dependence; independent phase offsets have not yet been optimized. Neither a homogeneous subtraction nor E(R)−E(R0) automatically removes all pin/bulk coupling energy. The present result is a conditional local minimum for an explicit boundary/pin prescription, not the minimum over all ways of holding HQVs apart.

## 2. Analytical gradient and numerical validation

### Derivative convention and stencil

Let G_A=partial H/partial A*, so dH=2 Re sum(conj(G_A) delta A). A complex free variable is packed into its real and imaginary parts; the real derivative is therefore 2(Re G_A, Im G_A). Pinned variables are excluded from that vector and reconstructed exactly for each objective evaluation.

For the onsite derivative write x=A_x, y=A_y, nx=x†x, ny=y†y, n=nx+ny, X=x·x, Y=y·y, Z=x·y and O=x†y. Before SOC,

\[
\begin{aligned}
G_x={}&(-2+2\beta_2n)x+2\beta_1(X+Y)x^*\\
&+2\beta_3(n_xx+Oy)+2\beta_4(n_xx+O^*y)
+2\beta_5(Xx^*+Zy^*),\\
G_y={}&(-2+2\beta_2n)y+2\beta_1(X+Y)y^*\\
&+2\beta_3(n_yy+O^*x)+2\beta_4(n_yy+Ox)
+2\beta_5(Yy^*+Zx^*).
\end{aligned}
\]

Here nx x and ny y denote scalar norm times vector. The dipole derivative is the derivative of the actual Hermitian quadratic form: in the planar block it adds g_D[(2−c0)Axx+Ayy] and its exchanged counterpart, g_D[Ayx−c0 Axy] and its exchanged counterpart, and −g_D c0 A in the z row.

For forward D_nu, the periodic adjoint is D_nu†f(r)=f(r−nu)−f(r). The gradient part adds

\[
\begin{aligned}
G_x^{\rm grad}&=K[D_x^\dagger(3D_xx+D_yy)
+D_y^\dagger(D_yx+D_xy)],\\
G_y^{\rm grad}&=K[D_x^\dagger(D_xy+D_yx)
+D_y^\dagger(3D_yy+D_xx)].
\end{aligned}
\]

Expressions such as D_x x mean a difference acting on the column vector; all spin components remain included. [relax.py](relax.py) implements these vectorized expressions and reuses the Stage-2 energy density for the objective.

### Validation before pair calculations

[validate.py](validate.py) checks 120 individual real/imaginary component derivatives, spanning all six complex channels, four wrap-bond corners and an interior site, at g_D=0 and 0.001. Two full-field directional checks bring the total to 122. The largest absolute finite-difference error is **7.64×10^−8**. This tests the periodic couplings and SOC derivative as well as local quartics.

A perturbed L=16 homogeneous field with only the outer frame fixed relaxes from excess energy 17.62337 to 4.55×10^−14 in 71 iterations. The maximum difference from the known homogeneous field is 4.03×10^−8, and its free real-gradient infinity norm is 6.68×10^−8. Fixed variables remain exact.

The pair solver uses SciPy L-BFGS-B on the free real vector, with gtol=3×10^−6, ftol=10^−15, maximum 3500 iterations, bounded line search and a 60-second callback time limit. All reported pair/control cases reach the gradient criterion before their caps. A zero-time-limit test independently verifies the stopping path returns an explicitly nonconverged result rather than a falsely successful energy measurement.

### Convergence evidence and its scope

All 22 distinct calculation cases satisfy the free-gradient criterion and have zero pin/frame error. The maximum final gradient infinity norm is 2.994×10^−6. Recorded accepted objective values never increase. Histories retain objective, gradient infinity/RMS norms, iteration and elapsed time; topology is checked initially, every 100 iterations and finally.

A tighter **same-pin** restart of L=64,R=24 reaches a free gradient of 6.80×10^−8 and changes energy by only −2.34×10^−9. Six selected curvature probes—smooth out-of-plane spin, amplitude, opposite-chirality core and three random full-component directions—are positive. These are useful stability checks, not a complete Hessian spectrum or proof of a global constrained minimum.

The summed recorded solver times, including a duplicate release check retained but excluded from scientific comparisons, are 250.1 seconds. Validation and postprocessing add only several seconds. No optimizer hit a time/iteration cap. The full investigation stays below the requested aggregate 10–15 minute compute allowance.

![Convergence and core reconstruction](results/convergence_and_core.png)

Numerical details are in [validation.json](results/validation.json), [additional_validation.json](results/additional_validation.json), each case's `history.csv`, and [case_summary.csv](results/case_summary.csv). Small optimization residuals do not supply statistical error bars or bound pin, boundary and initialization systematics.

## 3. The pair survives beyond its pins

The unchanged Stage-1 detector checks composite winding, director Z2 holonomy, lifted phase charge, chiral-manifold quality and link resolution. Analysis adds progressively larger, separate surrounding squares, up to radius 21.5 for R=48, plus an enclosing neutral contour. This reaches well outside the 2.5-site annuli.

All 245 final extended-loop checks for the 21 pinned cases pass with the expected charge. Individual contours have wP=+1/−1 and odd director parity; outer contours have wP=0 and even parity. No compensating defect or screening annulus is found by those checks or by the full-field candidate search. This is positive evidence for the identifiable constrained pair, not an exhaustive classification of every conceivable full-matrix texture.

For representative L=128,R=32 zero-SOC contours:

| Location | Result |
|---|---|
| Pinned annuli | Exact imposed +1/2 and −1/2 maps |
| First free surrounding loops | Correct half charges; residual of order 0.001 or less |
| Larger exterior loops | Same charges, with still smaller residuals |
| Outer pair loop | Accepted zero net charge |
| Whole-field automatic search | One accepted HQV, one accepted anti-HQV, no other accepted or unresolved clusters |

Both initial and final extended checks are retained in [topology_checks.csv](results/topology_checks.csv); [detected_clusters.csv](results/detected_clusters.csv) records the independent candidate search. Acceptance does not imply the field is restricted to the chiral manifold everywhere. The core is allowed to depart from it, and the sampled lattice cannot exclude structure between sites.

![Representative fully relaxed matrix field](results/relaxed_field_maps.png)

### Core reconstruction and alternative outcomes

At L=64,R=24, the relaxed minimum density is about 0.82177, rather than the very small sampled density in the prescribed depleted-core trial. The maximum squared manifold residual is about 0.08652. The maximum opposite-chirality weight is 0.01432 and the maximum nonreal-spin penalty is 0.08511; their maxima need not occur at the same site. The dominant core departure is therefore complex spin structure, with a smaller opposite-chirality component. Chirality remains at least 0.97136 at sampled sites.

This demonstrates that permitting full GL components changes the core and lowers the energy; it is not a frozen-amplitude or real-director-only solve. Finite sampled core density does not prove a nonsingular continuum texture: the centers lie between lattice sites, where director/composite structure can be unresolved. The trustworthy HQV identification comes from resolved exterior loops.

The standard initialization and a smooth nonplanar plus complex-noise initialization at L=64,R=24 converge to energies differing by about 2.45×10^−9. Their final spin-y norm fractions are approximately 10^−11, so the tested branch returns close to the initial spin plane even though that restriction was never imposed. A pin whose relative spin plane is rotated instead yields a three-dimensional texture, but that is a **different constrained problem**, not a competing minimum under identical constraints.

Releasing the annuli while holding the outer frame fixed reduces E=74.208760 to approximately 4.03×10^−9. Every former core loop then has zero charge and the automatic search finds no charged clusters. An unpinned neutral pair is therefore not retained by this optimization. This released result is an energy-landscape control; optimizer iterations do not describe how physical vortices move or annihilate in real time.

## 4. Zero-SOC separation energy

### Data and comparison with Stage 2

The primary prescription is p=2 with aligned annular spin planes. All energies below are excess energies relative to the same-coupling homogeneous branch. Unknown pin/core constants are reduced by comparing E(R)−E(16).

| L | E(16) | E(24) | E(32) | E(48) |
|---:|---:|---:|---:|---:|
| 64 | 64.223091 | 74.208760 | 83.359111 | — |
| 96 | 62.832480 | — | 78.224733 | — |
| 128 | 62.363508 | 70.301995 | 76.516288 | 86.656145 |
| 192 | 62.036144 | 69.611060 | 75.317568 | 84.015311 |

The sequence began with the three L=64 cases. Their large apparent slopes justified a second box; persistent changes justified the inexpensive L=128 and L=192 checks and the fourth separation. This was an explicit bounded extension, not an automated size/parameter search.

For a concrete comparison at L=128,R=32:

| Field | Excess energy | Same annular constraints? |
|---|---:|---|
| Original Stage-2 compact trial | 130.997553 | No |
| Quiet trial with the declared annuli overwritten | 144.182459 | Yes |
| Actual all-component perturbed initial field | 202.124254 | Yes |
| Converged full-field configuration | 76.516288 | Yes |

The strongest valid variational comparison is against the initial field with identical constraints. The unpatched Stage-2 trial is a useful reference but is not admissible for the same pin data. Added random perturbations have an extensive energy cost, so a large noisy-initial energy drop should not all be called core reconstruction. Both quiet constrained and original unpatched trial energies are retained separately in [case_summary.csv](results/case_summary.csv).

The relaxed separation differences are also smaller than the unrelaxed trial differences. At L=192, E(48)−E(16) is 21.97917, compared with 27.14986 for the original compact trial and 25.41134 for the quiet pin-compatible trial. Full relaxation thus changes more than an additive constant in that finite box.

### Logarithmic benchmark and finite-size effects

The [Stage-1 derivation](../he3_hqv_stage1/THEORY.md) gives bare rho_c=rho_d=4Kn0 for the restricted long-wavelength chiral field. A neutral pair with both phase and planar spin half paths has

\[
E(R)=C+\frac\pi2(\rho_c+\rho_d)\ln(R/\xi),
\quad A_{\rm bare}=4\pi Kn_0=18.79587237.
\]

This is a benchmark, not a condition imposed on the optimizer. It includes the spin term, unlike a mass-only half-vortex estimate. The full field can alter cores, amplitudes and spin paths.

| L | E(24)−E(16) | E(32)−E(16) | E(48)−E(16) |
|---:|---:|---:|---:|
| 64 | 9.985669 | 19.136020 | — |
| 96 | — | 15.392254 | — |
| 128 | 7.938487 | 14.152780 | 24.292637 |
| 192 | 7.574916 | 13.281424 | 21.979167 |
| Bare log reference | 7.621070 | 13.028306 | 20.649376 |

The trend is consistent with a substantial logarithmic contribution plus boundary/finite-R corrections. It does not establish a pure logarithm. In the largest box the interval slopes Delta E/Delta ln R are 18.6820 for 16→24, 19.8362 for 24→32 and 21.4513 for 32→48. They do not form a constant plateau. The fixed outer frame becomes increasingly important as R/L grows; L=64 gives much larger values, 24.6277 and 31.8072.

![Separation differences and interval slopes](results/interaction_energies.png)

A deliberately exploratory fit of fixed-R differences to a leading 1/L² correction illustrates why a near match at one box is insufficient. For R=32 relative to 16, using all four sizes gives an extrapolated difference 12.5084; excluding L=64 gives 12.5744. These correspond to effective log coefficients 18.0458 and 18.1410, below the bare 18.7959. This is not a controlled infinite-box limit, and finite-R/core/pin effects remain. It shows that agreement at L=192 can include cancellation between different corrections, rather than uniquely measuring the asymptotic coefficient.

No logarithm-plus-linear fit is used to infer an interaction law. Four separations, fixed pin textures and visible finite-size curvature cannot distinguish all plausible asymptotic forms. The established observation is a rising conditional energy with quantitatively important boundary corrections.

## 5. Pinning and initialization controls

Changing p from 2 to 3 changes E(32)−E(16) in L=64 from 19.136020 to 18.774626, a reduction of 0.361394, about 1.89%. The annuli therefore influence more than an exactly removable core constant, although their effect in this comparison is modest.

Rotating only the right annulus's spin plane by pi/2 about the outer-frame z axis gives:

| Relative pin texture | E(16) | E(24) | E(32) | E(32)−E(16) |
|---|---:|---:|---:|---:|
| Aligned spin planes | 64.223091 | 74.208760 | 83.359111 | 19.136020 |
| Right spin plane rotated | 71.126942 | 79.324656 | 87.555948 | 16.429006 |

The difference of separation energies is about −14.15%. A global spin rotation at g_D=0 would be a symmetry; this is a **relative** rotation of one imposed boundary map while the other is held fixed. It is a meaningful test of unnecessarily constrained texture data. It neither changes the Hamiltonian nor demonstrates two equilibrium phases.

The nonplanar-initialization test under identical aligned pins reaches the same observed minimum, rather than this rotated-pin solution. Hence initialization dependence was small in that tested basin, but pin-prescription dependence was material. No global minimization proof or exhaustive branch search is claimed.

![Pin dependence and preliminary weak-SOC energetic cost](results/pinning_and_weak_soc.png)

The next improvement should allow independent annular phase offsets and appropriate spin-frame parameters to vary while retaining their charges, followed by radius/outer-frame checks. Replacing exact pins with a very stiff arbitrary penalty would not by itself remove these restrictions. A soft-pin version should only be used with separately reported physical energy, penalty energy, constraint errors and exterior topology, especially if its weakening allows screening.

## 6. Two preliminary weak-SOC calculations

The zero-SOC gradient, convergence, topology and release controls passed before proceeding. Two otherwise matching L=128,p=2 cases were then run at g_D=0.001, with the corresponding homogeneous amplitude on the outer frame and annuli. This changes their amplitude by the declared same vacuum prescription, rather than keeping a zero-SOC bulk reference.

| R | Zero-SOC excess | Weak-SOC excess | Additional cost |
|---:|---:|---:|---:|
| 16 | 62.363508 | 63.026077 | 0.662569 |
| 32 | 76.516288 | 78.233649 | 1.717362 |

The separation difference grows from 14.152780 to 15.207572, an additional 1.054793. The fields converge and retain both exterior HQV signatures. Re-evaluating a uniformly amplitude-scaled zero-SOC minimum in the SOC functional gives valid same-pin trial energies 63.061673 and 78.334410; further SOC relaxation lowers those by 0.035596 and 0.100761. Thus the reported SOC fields are relaxed responses, not just frozen zero-SOC textures evaluated with an extra term.

The actual dipole excess energy is approximately 0.62525 at R=16 and 1.62114 at R=32; onsite and gradient changes also contribute. Full parts are in [energy_parts.csv](results/energy_parts.csv). The dipole map uses the implemented invariant and the correct homogeneous subtraction, not a director-sign seam.

![Weak-SOC energy and spin structure](results/weak_soc_field.png)

The bare length sqrt(2K/g_D) is 44.7, larger than both separations. The observed positive cost is a short-range, finite-box, pin-dependent SOC response. It is compatible with an anisotropy perturbation, but does not identify an asymptotic linear term or its tension. Two points cannot distinguish a logarithm, crossover, finite-size curvature or a linear addition. A physical connecting spin texture is permitted, but no sharp long-distance soliton regime is demonstrated by this broad texture.

The method is numerically ready to extend this **declared constrained problem**. A credible confinement calculation additionally needs reduced/variational pin restrictions, larger-separation exterior topology checks, independent boundary control and a hierarchy beyond the actual anisotropy length while remaining far from the outer frame. Existing R=16–32 SOC results cannot settle whether the full field avoids confinement by another route.

## 7. Prioritized next calculation

1. **Reduce fixed texture information before expanding a campaign.** Add a small set of variational annular phase/spin-frame parameters or carefully monitored softer constraints. First verify that the zero-SOC separation difference becomes stable under these choices and pin radii. If it does not, retain the result as a family of constrained excitation energies rather than a unique HQV potential.
2. **Separate finite R from finite box.** Use the prepared solver for a few larger boxes at fixed R, and then a wider R interval at controlled R/L. Compare forces or interval slopes with the neutral-pair Green-function/boundary reference; do not fit all curvature to a new interaction term. This is scientifically necessary because the measured corrections are already several energy units.
3. **Repeat the matching zero/weak-SOC problem across the crossover only after the first two checks.** Keep surviving HQVs verified outside the cores; inspect full excess-energy and spin/tensor maps. A reproducible additional long-distance cost needs stability against initialization, pin prescriptions and boundaries before being interpreted as confinement.
4. **Challenge branch survival.** Preserve any lower-energy screened or nonchiral branch and compare it to the retained-HQV branch. A flat curve after topology loss would not be an unconfined HQV potential. Released-constraint optimization is useful for energy-landscape stability but supplies no real-time vortex dynamics.

No new MC data or finite-temperature interaction free energy is obtained here. All reported minima are local numerical solutions under constraints, not certified global constrained minima or thermodynamic phase-transition evidence.

## 8. Reproduction and evidence map

Run from the project root with the existing NumPy/SciPy environment:

```powershell
python -B exploration/he3_hqv_stage3/validate.py
python -B exploration/he3_hqv_stage3/run_cases.py
python -B exploration/he3_hqv_stage3/run_cases.py --controls
python -B exploration/he3_hqv_stage3/run_extensions.py
python -B exploration/he3_hqv_stage3/additional_checks.py
python -B exploration/he3_hqv_stage3/analyze.py
python -B exploration/he3_hqv_stage3/verify.py
```

`run_extensions.py` contains the explicit, bounded size, relative-pin, release and two-SOC tests justified above. Completed cases are reused rather than overwritten; inspect or archive an incomplete case before rerunning it. `analyze.py` never launches an optimizer. NumPy/SciPy BLAS threads are limited for these modest CPU problems. No CUDA runtime is imported for execution, and no cluster job is submitted.

| Artifact | Evidence |
|---|---|
| [relax.py](relax.py) | Exact derivative, fixed-variable packing, constraints, bounded solver and topology checkpoints |
| [validate.py](validate.py) | Finite-difference, wrap-bond and homogeneous-minimum tests |
| [additional_checks.py](additional_checks.py) | Tighter same-pin restart, selected curvatures and tested stopping path |
| [verify.py](verify.py) and [verification.json](results/verification.json) | Reopened field energies/gradients, exact declared pins and matching source hashes; no optimizer launched |
| [case_summary.csv](results/case_summary.csv) | All 22 distinct initial/relaxed cases, convergence, field diagnostics and counts |
| [energy_differences.csv](results/energy_differences.csv) | Same-prescription E(R)−E(16), trial comparisons and bare benchmark |
| [effective_slopes.csv](results/effective_slopes.csv) | Interval slopes without a claimed universal fit |
| [finite_size_diagnostic.csv](results/finite_size_diagnostic.csv) | Explicitly exploratory size extrapolation and subset sensitivity |
| [topology_checks.csv](results/topology_checks.csv) | Initial/final nested loops, winding, parity, quality and rejection reasons |
| [detected_clusters.csv](results/detected_clusters.csv) | Full-field candidate search, independent of specified core loops |
| [weak_soc_comparison.csv](results/weak_soc_comparison.csv) | Same-coupling excesses and frozen-field SOC control |
| [energy_parts.csv](results/energy_parts.csv) | Onsite, gradient and dipole excess components |
| Each `results/<case>/` directory | Initial/final complex128 fields, pinned mask, parameters, source hashes, optimizer history and topology history |

Five selected figures are embedded above. The repeated release calculation is flagged `exclude_from_comparison` and contributes only to the compute accounting. All original datasets and production files remain unchanged. No stage in this report treats optimizer iteration as physical time.
