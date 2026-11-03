# From weak spin locking to the strong-SOC sampling anomaly

Investigation of the implemented full thin-film He3 lattice model, 1 November 2026. All new calculations and local pilot data are confined to this directory. Production code, prior data, the running campaign and the Git branch were left unchanged. No cluster access or Slurm submission was performed.

## A. Main discoveries and confidence

**The weak-SOC coherent channel and the strong-SOC anomaly are related through the failure of P as a complete phase diagnostic, but the measured coherent structures are different.** The new small-box strong-SOC population has substantial antisymmetric-planar character and charge-one coherence in that spin–orbital projection. It is not simply the weak-SOC z-spin A+ channel continued to larger coupling.

The strongest results are:

1. **The old apparent low-temperature front is quantitatively explained by sorting an almost conserved inventory of configurations.** A constrained-population PT model, using measured inventory and conditional energy differences, predicts the high-population occupancy to within .015–.021 over the entire old ladder. The inventory largely fixes the location; the measured energy difference accounts for its width, including a held-out-half check. This explains how a sharp, size-dependent front can arise without measuring an equilibrium transition. Confidence: high for the recorded histories and this conditional explanation.
2. **No new onsite minimum was found between g_D=.01 and .1, and the tracked chiral/planar branches show no stability change.** Chiral-z and antisymmetric-planar minima are both stable throughout. Their energy gap decreases smoothly, .206454→.097376 per site. Verified full-space connecting paths also change smoothly. The strong sampling problem cannot be attributed to the planar minimum first appearing near .1. Confidence: high for these stationary structures; the finite search is not a global completeness proof.
3. **Preparation trapping already appears at g_D=.06 on L=16.** At the common cold endpoint, the three starts agree at .04 but split sharply at .06,.08,.1. At .06 the near-feature mixture also drifts while helicity remains nearly constant. This brackets a change in accessibility under this particular short-run protocol between .04 and .06, not a thermodynamic coupling threshold.
4. **A stiff planar-like population is directly identified in new spatial records.** At g_D=.08,T=.376522,L=16, all three preparations give Y≈2.17, low chirality, negative spin alignment, a local antisymmetric-planar weight ≈.60, and strong correlations in the corresponding charge-one field. At .1 the same pattern is stronger. Confidence: high as a finite-box structural observation; equilibrium weights and an asymptotic ordered phase remain unresolved.
5. **The mixed stiffness is symmetry-allowed in the full equilibrium ensemble.** Independent finite twists validate the mixed-curvature formula. The actual forward stencil lacks the axial mirrors that would force the mixed response to vanish. Averaging opposite chiralities does not cancel it. Small pilots measure a nonzero mixed component, with meaningful conditional errors.
6. **Temperature round trips and physical conversion are demonstrably different.** Some new planar histories make 9–14 complete PT loops while retaining a different cold population from another initialization. On L=8, reverse structural visits are possible with the unchanged sampler, but their weights still drift. Thus the sampler does not forbid conversion; ordinary local evolution becomes inadequate on the larger boxes and sampled time scales.

The old g_D=.1 equilibrium phase diagram, a first-order transition, HQV confinement and the vortex mechanism remain undetermined. No precise new transition temperature is claimed.

## B. Why the old g_D=.1 histories failed

### Direct reconstruction, with physical identities preserved

The four original archives have L=64,96,128,160, 160 temperatures .33–.47, 2M thermalization and 3M production sweeps, Gaussian rho≈−.9, and exchanges every five sweeps. Physical configurations remain in fixed walker slots; PT exchanges their temperature assignments. Raw measurements were matched to the saved physical-walker permutations at identical sweeps, rather than interpreting temperature-slot changes as conversions.

The empirical high population has normalized spin alignment s_z>.3. Thresholds .2 and .4 give the same inventory losses; |normalized global chirality|>.1 almost exactly identifies the same old population. These are empirical structural labels, not assumed phases.

| L | High inventory, production start→end | Physical high→low / low→high conversions | Fixed-temperature label changes at 500-sweep cadence | Exact complete PT loops over 5M sweeps |
|---:|---:|---:|---:|---:|
| 64 | 33→31 | 2 / 0 | 14,938 | 0 |
| 96 | 44→43 | 1 / 0 | 12,087 | 0 |
| 128 | 50→49 | 1 / 0 | 8,497 | 0 |
| 160 | 55→53 | 2 / 0 | 6,180 | 0 |

All 28 sampled full-run endpoint passages are cold→hot; none is hot→cold. At L=160, unchanged high walkers never exceed .377156 and unchanged low walkers never go below .366703 in the sampled histories. They diffuse over substantial, overlapping temperature ranges without exploring the whole ensemble. The final inventory of 53 high configurations persists for 2.916M sweeps. Agreement between production halves can therefore coexist with persistent trapping.

The low population has substantial common-phase response: around T=.388056 its Y is about 2.19–2.20 across the four sizes; at .442331 it is about 1.18–1.19. This confirms the earlier finding. Recomputed scalar means and helicities agree with the previous tables within 1e−8. Conditional population responses describe the visited histories; they are not certified equilibrium phase stiffnesses.

### A predictive sorting test

If the high-population count M changes extremely slowly, the allowed label distribution at the PT temperatures is approximately

\[
\mathcal P(\{b_i\}\mid M)\propto
\delta_{\sum_i b_i,M}\prod_i w_i^{b_i},\qquad
w_i=Z_H(\beta_i)/Z_L(\beta_i).
\]

Across a narrow front, a roughly constant measured conditional energy difference gives

\[
\ln w_i\simeq c-\Delta E(\beta_i-\beta_*),\qquad
\Delta E=E_H-E_L.
\]

The temperature-independent additive constant **in the log partition-function ratio** cancels at fixed M. An arbitrary energy offset would not cancel, because beta varies. Exact finite-inventory marginals were computed by a short dynamic-programming calculation and checked against explicit enumeration on a small example.

| L | Measured total ΔE at front | Maximum occupancy error | Observed / predicted 10–90% width |
|---:|---:|---:|---:|
| 64 | −113.668 | .0152 | .004768 / .004745 |
| 96 | −178.569 | .0174 | .002970 / .002938 |
| 128 | −244.963 | .0160 | .002163 / .002165 |
| 160 | −302.725 | .0205 | .001596 / .001649 |

The same-M sharp-step control has maximum errors .23–.29. Using only the first half to measure ΔE predicts the second-half occupancy with maximum errors .012–.021, conditional on its measured inventory. This tests more than the supplied front location. At L=160, as M falls 55→54→53, the observed half-occupancy temperature moves .371889→.371268→.370675; the model predicts .371907→.371295→.370684.

![Restricted-population temperature occupation and the fixed-inventory sorting prediction.](transport/sorting_and_occupation.png)

This does not reconstruct the missing equilibrium branch weights: conditioning on M removes their normalization information. Nor does it exclude a real transition somewhere nearby. It demonstrates that the observed front and its drift have a strong nonequilibrium inventory explanation.

### Why more swaps or a hotter endpoint are insufficient answers

The old worst acceptance edge is near .465949, reaching .1215 at L=160. Better placement there could improve temperature diffusion. Yet the structural-front edges accepted roughly .46–.71, and high/low walkers already exchanged their positions near that front. The problem is not an entirely uncrossed temperature edge.

The old hot endpoint is close to the upper heat feature, but Y(.47)=.01178±.00765 at L=128 and .00521±.00778 at L=160. It is already consistent with a phase-disordered large box. Raising the endpoint may improve structural forgetting, or simply destroy chiral configurations faster without regenerating them on cooling. The new .65-endpoint pilots below demonstrate that a broad hot range alone does not guarantee structural exploration; they are not a controlled estimate of changing only the old endpoint.

The Gaussian update proposes all twelve real components at one site, with the correct proposal-density ratio and full energy change. Its quadratic precision eigenvalues are 10K+mass and 6K+mass. Mean acceptance is about .59 in unchanged high walkers and .65 in low walkers. The source review found no sampler or PT correctness problem requiring a production change. Accepted local fluctuations can coexist with extremely slow collective structural reorganization.

Detailed raw checks, conversion brackets, temperature occupation, sampler review and model limitations are in [TRANSPORT.md](transport/TRANSPORT.md).

## C. Evolution of the local landscape and spin locking

### The same full Hamiltonian and conventions

The model remains

\[
H=\sum_r[-2n+\sum_{i=1}^5\beta_i I_i+g_D D+K G],
\quad n=\mathrm{Tr}(AA^\dagger),\quad K=1,
\]

with unrestricted complex 3×2 A and the actual P=20 coefficients

\[
(-.5165123343,.9420228004,.9875290394,.8790628910,-1.1525149345).
\]

The physical coefficient parameter reduced_temperature=1 stays fixed. The MC temperature controls exp(−H/T); it is not an experimental T/Tc. The dipole invariant is

\[
D=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\operatorname{Re}(A_{xy}A_{yx}^*)-c_0 n,
\quad c_0=\mathrm{float32}(2/3).
\]

Its lowest quadratic sector is antisymmetric planar, with eigenvalue approximately −5/3 per norm, versus −2/3 for the z-spin row. Within the restricted chiral manifold it favors a z-directed spin director. These are different statements: full-field SOC allows structural competition beyond spin locking.

### Stable competitors, smooth energy changes

Let βA=β2+β4+β5=.6685707569 and βP=β1+β2+(β3+β4+β5)/2=.7825489640. The optimized chiral-z and antisymmetric-planar representatives have

\[
A_C=\sqrt{n_C/2}\,\hat z(1,i),\quad
n_C=(1+g_Dc_0/2)/\beta_A,
\]
\[
(A_P)_{xy}=-(A_P)_{yx}=\sqrt{n_P/2},\quad
n_P=[1+g_D(1+c_0)/2]/\beta_P.
\]

| g_D | Chiral energy/site | Planar energy/site | Gap | Smallest positive Hessian mode, chiral / planar |
|---:|---:|---:|---:|---:|
| .01 | −1.505716 | −1.299262 | .206454 | .019894 / .020000 |
| .02 | −1.515738 | −1.320826 | .194911 | .039579 / .040000 |
| .04 | −1.535880 | −1.364487 | .171393 | .078326 / .080000 |
| .06 | −1.556155 | −1.408858 | .147298 | .116258 / .120000 |
| .08 | −1.576564 | −1.453938 | .122626 | .153392 / .160000 |
| .10 | −1.597105 | −1.499729 | .097376 | .189744 / .200000 |

Both have one gauge zero mode and no negative Hessian eigenvalues at nonzero coupling. The planar minimum already exists at zero SOC. Fifteen targeted full-space minimization starts per coupling found no lower state than the chiral representative, without proving global optimality. The analytic energies would cross near .1732, outside this interval; that is not a thermal phase boundary.

The pure rank-one polar field with A_x=A_y proportional to z is an onsite stationary **saddle with three negative modes**, not the planar competitor. The weak-SOC coherent A+,z channel therefore does not imply a new homogeneous onsite minimum.

Full twelve-dimensional index-one saddles were found from three path preparations and connected downhill to both minima. Verified continuous paths give accessible barrier upper bounds above chiral of .414810→.395902 and above planar of .208356→.298526 as g_D rises .01→.1. The initial strings reached their iteration cap; they are only saddle seeds. The final connecting paths and their segment maxima were checked independently. These are not proven globally minimal barriers, spatial nucleation barriers or finite-temperature free energies.

![Deterministic energies, stability and verified full-space connecting paths.](landscape/landscape.png)

The spatial environment changes the conversion problem substantially. Replacing one site A inside a uniform field C costs the exact gradient increment

\[
\Delta H_g=K[8\|A-C\|^2+
4\operatorname{Re}((A_x-C_x)^\dagger(A_y-C_y))].
\]

An abrupt chiral→planar replacement costs 22.52–23.54 total energy units across the range, much larger than the free-onsite barrier or MC T≈.4. This is an illustrative fixed-neighbor cost, not the true activation barrier or a Gaussian acceptance estimate. It explains why a small isolated onsite barrier does not imply fast local-MC conversion.

### Entropy is a plausible contributor, but a simple harmonic prediction fails

A bounded full-lattice Gaussian calculation around each minimum uses the exact onsite Hessian plus the actual Fourier gradient kernel. The planar structure has a Gaussian entropy advantage at every tested coupling. In Δf_harm=ΔV+T a(g_D), a changes from −.5892 at .01 to −.4407 at .1. Thus the entropy advantage decreases with coupling; the falling bare energy gap drives the increasing competitiveness in this approximation.

Its formal crossings, for example .3504 at .01 and .2599 at .08, are **not** inferred transition temperatures. The .01 prediction already overpredicts planar preference relative to the observed weak-SOC channel pattern. Nonphase fluctuations are 18–31% of the reference density at T=.38, and phase variance grows with box size. This supplies a qualitative entropy mechanism and a useful negative result: the bare harmonic approximation cannot locate the thermal crossover.

All Hessians, derivative checks, paths, exact-energy comparisons and the harmonic control are in [LANDSCAPE.md](landscape/LANDSCAPE.md).

### Bare dipole lengths versus measured tilt scales

The restricted benchmark sqrt(2K/g_D) decreases through 14.14,10.0,7.07,5.77,5.00,4.47 for the six requested couplings. In the chiral-start L=16 cold histories at T=.28, connected Qxz/Qyz second-moment lengths are 7.07±.32,5.76±.28,4.92±.24,4.40±.22 for .04,.06,.08,.1. This close agreement is a useful cold-branch check. At .06 and above it characterizes a sampled chiral population whose equilibrium weight is unresolved. A planar-like population has different soft variables; interpreting all its Q correlations as the same director-locking mode would be unjustified. None of these lengths measures a confinement string tension.

## D. Controlled local experiments and their limits

### Design and cost

The existing He3GaussianRuntime, ParallelTempering and spatial measurement code were reused unchanged. A short exploration loop saves their measurements and computes only the missing mixed curvature and an antisymmetric-planar projection from occasional CPU copies. No production measurement module or data format was edited.

| Cases | Preparations per case | Total / analyzed sweeps per history | Temperatures | Purpose |
|---|---|---|---|---|
| L=8,g_D=.1 | chiral, unrestricted random, planar | 16k / 12k | 24 uniform, .28–.65 | Test whether reverse structural visits are possible at all |
| L=16,g_D=.04,.08 | same three | 32k / 24k | same | Compare an intermediate coupling with a stronger competitor |
| L=16,g_D=.06,.1 | same three | 32k / 24k | same | Refine the accessibility bracket and connect to the old anomaly |

All preparations use independent site, PT and host-initialization random seeds. Chiral and planar amplitudes use their actual SOC-adjusted onsite densities; random fields are unrestricted complex Gaussians with expected density equal to n_C. The random preparation is not claimed to be an already equilibrated high-temperature ensemble. Initialization energies were independently checked.

Scalars are saved every 20 sweeps, Fourier/structure/Kxy data every 100, and physical identities every 20. Exchanges remain every five sweeps. The benchmark preceded sampling; each run printed a cost estimate. Conservative data/workspace estimates were 16.0 MiB at L8 and 24.7 MiB at L16, beyond the CUDA context. The 15 histories used 432,000 summed sweeps, 10.368M replica sweeps, and 2.433B site updates. Their timed update-and-measurement loops totaled **99.9 seconds**; this excludes process startup, compilation and saving. Two 400-sweep preflight benchmarks are additional correctness/timing work. No L=24 or32 expansion was warranted.

Full-history errors use 12 joint time blocks, 1k sweeps at L8 and 2k at L16. Halves, up to 6k-sweep blocks, preparation comparisons and empirical CDF differences were checked independently. Different preparations are never pooled into an equilibrium mixture. All errors remain conditional on visited histories.

### Cold agreement at .04; clear trapping by .06

At T=.28:

| g_D, L16 | Spin alignment: chiral / random / planar start | Y: same order |
|---:|---|---|
| .04 | .737 / .733 / .732 | 3.669 / 3.657 / 3.641 |
| .06 | .751 / .754 / −.320 | 3.715 / 3.715 / 3.155 |
| .08 | .763 / .767 / −.338 | 3.786 / 3.773 / 3.341 |
| .10 | .772 / −.346 / −.346 | 3.855 / 3.456 / 3.445 |

For .06–.1 the trapped populations have disjoint sampled spin/chirality distributions. At .08, halves within each trapped history agree very well, while preparations disagree completely. This is stationary-looking sampling within different populations, not equilibrium coexistence.

At .04,T=.376522 the spin means .304±.063,.352±.043,.438±.023 differ by at most two conditional errors. That is not by itself decisive disagreement, but broad uncertainties and only 2–4 completed loops do not certify mixing. At .06 the chiral-start spin mean changes −.1979±.0040→+.0417±.0797 between production halves, accompanied by energy, chirality and projection changes. Its full-history spin error grows .0595→.1096 when blocks grow2k→6k. Y stays near 1.98 while the structural mixture changes.

![Joint energy, residual and planar-projection measurements at .08. Cold preparations occupy different populations; the higher-temperature probe agrees much better.](pilot_results/joint_structure.png)

### Round trips are not conversions

New persistent visits use s_z>.25 and |m_chi|>.25 for a chiral-like region, versus s_z<−.1 and |m_chi|<.15 for a low-Q/low-chirality region. Neither threshold alone identifies the local planar structure. The reported strict production counts require at least400 sweeps between the first and last sample in each qualifying visit, with both visits entirely after burn. 100/200-sweep controls are retained.

| L,g_D | Exact full-run loops, chiral / random / planar | Production persistent visits, forward/reverse in the same order |
|---|---|---|
| 8,.1 | 34 / 31 / 43 | 0/2; 2/2; 2/3 |
| 16,.04 | 2 / 4 / 3 | 0/1; 0/0; 0/0 |
| 16,.06 | 0 / 0 / 14 | 0/2; 0/1; 0/0 |
| 16,.08 | 0 / 0 / 9 | 1/0; 0/0; 0/0 |
| 16,.1 | 0 / 14 / 9 | 1/0; 0/0; 0/0 |

Reverse visits at L8 show that the algorithm does not prohibit structural regeneration. Its cold planar-start history nevertheless has strong half drift, so equilibrium weights are not obtained even there. At L16, planar-only histories can traverse the whole ladder repeatedly without regenerating the other population. The .06 chiral inventory actually grows during production, while .08/.1 mixed starts retain a nearly fixed small inventory. These reproduce the large-box failure mechanism on a controlled small scale.

A missing reverse event alone would not prove that both populations have appreciable equilibrium probability. All independent starts could ultimately settle into one equilibrium population. What invalidates the present global averages is their persistent preparation-dependent inventory and, in some cases, continuing drift.

The independent [pilot review](transport/PILOT_REVIEW.md) contains block sensitivity, distribution comparisons, production-only walker counts and inventory plots.

## E. What coherence survives, and how anisotropic is it?

### Distinguishing the coherent operators

Define

\[
A_\pm=(A_x\pm A_y)/\sqrt2,\quad
B_{\pm\pm}=A_\pm\cdot A_\pm,\quad
P=2A_+\cdot A_-.
\]

For planar structure also use the directly measured charge-one projection

\[
\psi_P=(A_{xy}-A_{yx})/\sqrt2,\qquad
w_P=\langle\sum_r|\psi_P|^2\rangle/\langle\sum_r n\rangle.
\]

This local weight is not a configuration-phase probability. For the ideal planar matrix wP=1, s_z=−.5, P=0 and B++=B−−=n/2. For a pure z-polar plus field wP=0, s_z=1, P=0, B++=n and B−−=0. Both can carry common-phase stiffness while P vanishes, for different structural reasons.

The old g_D=.01,L160,T=.3869501 result had dominant A+,z and B++ coherence, with P and A− finite ranged. At that point roughly97.6% of total original-field zero-mode power was in A+. This was verified in the [previous spatial report](../he3_weak_soc_spatial_01/REPORT.md); its field-polynomial checks and saved spectral identities were also inspected here.

By contrast, the new strong/intermediate-SOC planar-like regime at the common L16,T=.376522 probe has:

| g_D | s_z, range across starts | Y, range | wP | S_psiP(0) | S_A+,z(0) | A+ fraction of total A power |
|---:|---:|---:|---:|---:|---:|---:|
| .08 | −.229 to−.236 | 2.166–2.178 | .596–.603 | 128–131 | 2.5–3.2 | .578–.588 |
| .10 | −.249 to−.253 | 2.358–2.410 | .627–.630 | 142–143 | 2.3–2.5 | .568–.572 |

At .08, xi_psiP/L≈2.01–2.07 while xi_A+,z/L≈.27–.31. Both B++ and B−− are correlated over the box: their second-moment ratios are approximately .89–.94 and .76–.80. The local residual is about .49 rather than the ideal planar .75, so the thermal fields contain substantial admixture. These observations support a planar-like coherent sector, not an exact local ansatz. Finite sampled charged means account for a few percent of the planar zero-mode power; the power is not predominantly an unremoved constant mean.

![Original-field and charge-two orbital channels across the four new L16 couplings. All preparations are shown separately.](pilot_results/orbital_channels.png)

At .04,T=.408696 the plus preference is stronger, but both z-plus and antisymmetric-planar projections contribute; this is not a clean replica of the large-L .01 pattern. The tiny boxes, different temperatures and trapping prevent assigning a continuous thermodynamic path between them.

The relationship to the old g_D=.1 population is therefore **supported but not proven microscopically**: its negative spin alignment, low chirality and large Y match the new planar-like population. The old large runs saved no fields or B tensor, so their exact orbital structure cannot be reconstructed. The new coherent structures differ from the weak-SOC plus-z sector; their common lesson is that P alone misses relevant U(1) correlations.

### Exact mixed response and symmetry

For the same uniform phase-twist convention as the production currents,

\[
\mathcal K_{xy}=2K\sum_{r,\mu}\operatorname{Re}[
A_{\mu x}(r+\hat x)^*A_{\mu y}(r+\hat y)
+A_{\mu y}(r+\hat x)^*A_{\mu x}(r+\hat y)],
\]
\[
\Upsilon_{xy}=\langle\mathcal K_{xy}\rangle/N
-\operatorname{Cov}(I_x,I_y)/(NT).
\]

Independent finite twists of the actual production polynomial agree within 3.56e−9 per site. SOC is onsite and has no direct common-twist derivative.

The stencil's Fourier off-diagonal kernel contains

\[
2[(1-\cos k_x)(1-\cos k_y)+\sin k_x\sin k_y].
\]

An axial reflection changes its magnitude at generic k, changing the eigenvalues of the kinetic matrix. Thus no unitary internal change can simply repair the missing reflection. The surviving coordinate swap and simultaneous inversion preserve qx qy. Time reversal flips both currents and chirality but leaves Kxy and their covariance unchanged. Consequently nonzero Upsilon_xy is allowed even without choosing one chirality or spontaneously breaking an additional reflection. A+ and A− uniform onsite energies coincide, but their fluctuating lattice theories are not related by an exact axial-mirror symmetry.

This permits an orbital preference without proving its microscopic cause. Indeed, a restricted frozen-polar phase-wave entropy calculation favors A− rather than the observed A+ by .193965 T/site. That controlled counterexample rules out claiming that the observed sign follows automatically from broken mirrors. Full internal fluctuations and incomplete equilibration still matter. The independent derivation and tests are in [SYMMETRY.md](symmetry/SYMMETRY.md).

### Bare versus measured tensors

For a homogeneous field at density n, the bare tensor is

\[
\rho^{\rm bare}=2K[nI+2\operatorname{Re}(A^\dagger A)].
\]

The plus-z polar field has rho=[[4,2],[2,4]]Kn, whereas the chiral-z and ideal antisymmetric-planar fields have4Kn I. All have the same averaged diagonal4Kn; internal rearrangement at fixed density alone does not reduce that average.

The pilots estimate the fluctuation tensor using all 20-sweep currents and diagonal curvatures, plus100-sweep mixed curvature, with joint blocks spanning the same physical intervals. Sparse-only tensor estimates are retained as a cadence check. Representative conditional results are:

| L16 case | Yxx | Yyy | Yxy | sqrt(det Y) |
|---|---:|---:|---:|---:|
| .04,.408696, chiral start | .801±.062 | .778±.025 | .248±.048 | .750±.039 |
| .04,.408696, random start | .794±.065 | .810±.048 | .255±.051 | .761±.052 |
| .08,.376522, random start | 2.182±.043 | 2.151±.030 | .144±.021 | 2.162±.026 |
| .1,.376522, random start | 2.384±.019 | 2.401±.021 | .162±.024 | 2.387±.012 |

The third .04 preparation has Yxy=.133±.048, so a precise universal anisotropy magnitude is not established. At .08/.1 in this planar-like regime, the determinant correction to the diagonal average is small although the mixed component is resolved. At .04 it is more noticeable. Noisy non-positive tensors at hot temperatures are left undefined for sqrt(det), not repaired by clipping.

![Conditional tensor estimates at .04; all three preparations remain separate.](pilot_results/stiffness_tensor.png)

Only after deriving an effective single phase with a positive long-distance stiffness tensor can a coordinate transformation make sqrt(det rho) the stiffness governing logarithmic vortex energy and Gaussian exponents. The measured finite-box response need not already be that effective coupling. Applying an ordinary or half-charge jump criterion here would require the allowed defect and internal-field content as additional input.

The prior .01 q=1/2 and higher-temperature q=1 WM candidates remain conditional fits, not vortex identifications. These new one-size coupling pilots do not support a new WM fit. For the planar-like branch, the observed charge-one psiP channel is a more direct diagnostic than P, but does not by itself assign the elementary vortex charge.

## F. Competing interpretations and falsifiable tests

| Interpretation | Evidence here | What would discriminate or falsify it |
|---|---|---|
| Growing competition with a planar thermal structure | Smooth onsite gap reduction; new planar projections and coherence; persistent preparation dependence | Larger, independently equilibrated spatial records that show a different low-Q orbital structure |
| The old low front is mainly fixed-inventory sorting | Quantitative width/location/held-out checks; almost no physical conversion | Reversible independent-start sampling with a stable front independent of initial inventory |
| Weak plus-z and stronger planar coherence are one simple unchanged channel | Disfavored by opposite spin signatures and different A/B power partitions | Demonstrate a common dominant eigenchannel with controlled size and temperature matching |
| A new bare minimum appears near .06 or .1 | Ruled out for the examined competitors | A newly found lower or unstable full-field stationary branch outside the targeted search |
| An entropic structural crossover explains thermal preference | Gaussian entropy has the right qualitative sign, but overpredicts planar preference | Controlled free-energy or equilibrated distribution comparisons; bare harmonic crossings are inadequate |
| A first-order equilibrium transition produces the populations | Still possible, but trapping and bimodality do not establish it | Reversible equilibrium coexistence, size-dependent interface barriers and stable weights |
| An axial-symmetry-breaking transition is required for mixed stiffness | Not required by this stencil | A demonstrated additional exact symmetry of the implemented full lattice model |
| HQV unbinding or confinement determines the observed features | Not measured by these observables | Qualified thermal topology or constrained free-energy measurements with finite-size controls |

The strongest current account is that SOC changes both spin locking and the relative structural competition, while local dynamics and temperature sorting can preserve a non-equilibrium inventory for very long times. The available calculations do not separate all entropic, interface and topological contributions to the collective conversion barrier.

## G. Smallest useful next numerical step

1. **Prioritize g_D=.02 and .04 at L=8 and 16**, retaining chiral, random and planar preparations. The present .04 cold agreement is encouraging; demonstrate near-feature distribution agreement and block stability before enlarging the box. A single .05 comparison would refine the observed .04–.06 accessibility bracket. This is an algorithmic diagnostic, not a proposed phase boundary.
2. **Improve the small-pilot ladder, not the PT engine.** The deliberately simple24-point broad ladder gives some L16 acceptance minima near .019. Use the saved energy widths and physical-walker flow to redistribute a bounded set, perhaps32–40 temperatures, around the observed bottlenecks while retaining a hot region. Measure both transport and structural regeneration. Compare original and broadened endpoints only in a controlled matched-size experiment before attributing benefit to the endpoint alone. Exchanges are already every 5 sweeps; faster attempts are not the first justified intervention. The distinction between swap acceptance and round-trip optimization is established in [Hukushima and Nemoto](https://arxiv.org/abs/cond-mat/9512035) and [Katzgraber et al.](https://arxiv.org/abs/cond-mat/0602085).
3. **Gate expansion to L=24 on evidence.** Require preparation-independent energy/structure distributions, stable blocks/halves, and repeated physical conversions when both populations have appreciable weight. If all starts instead settle into the same single population with the other weight negligible, that is a different acceptable outcome; do not demand artificial coexistence. Passing a round-trip count alone is insufficient. Do not extend the current .06–.1 histories into a production campaign on the strength of their conditional averages.
4. **If ordinary sampling fails again, evaluate SOC tempering or a structural umbrella separately.** For H_g=H0+gD_bare, exchanging replicas i,j requires the log acceptance ratio `(beta_i-beta_j)(H0_i-H0_j)+(beta_i*g_i-beta_j*g_j)(D_i-D_j)`. A biasing umbrella must enter every acceptance ratio and be removed by controlled reweighting; windows need overlap. A collective proposal requires its reverse density and any Jacobian. Global spin rotations alone do not change a chiral rank-one field into a planar rank-two field. These methods cost extra replicas/windows and are justified only if they produce demonstrably reversible physical exploration. None is implemented here.

No parameter changes or follow-up simulations are enabled automatically. MC sweeps are sampling steps, not real-time dynamics; onsite minimization, harmonic approximation and finite-temperature sampling have been kept separate throughout.

## Validation, reproduction and output map

Independent checks cover the actual production potential and gradient polynomial, full-space derivatives, mixed twists, final thermal fields, scalar/FFT normalization and physical mappings. Across all 15 pilots, the largest final-field CPU/GPU energy discrepancy is 5.48e−8/site, Fourier discrepancy5.69e−14, and local-residual discrepancy2.23e−16/site. Every saved physical mapping agrees exactly with the independent PT history. No nonfinite rejections occurred. [validation.json](pilot_results/validation.json) retains individual results.

The final preservation audit matches all 122 pre-existing Python-source hashes and the sizes/mtimes of all 27 original input archives. Git branch and HEAD are unchanged, with no tracked or staged diff. All nine new scripts pass syntax checks and the report links resolve. The input check is a size/mtime preservation check, not a new whole-file cryptographic hash. Approximately189 MiB of new pilot and analysis artifacts are retained in this directory.

All ordinary analysis commands run from the repository root:

```text
python -B exploration/he3_intermediate_soc/transport/reconstruct.py
python -B exploration/he3_intermediate_soc/landscape/onsite.py
python -B exploration/he3_intermediate_soc/landscape/harmonic.py
python -B exploration/he3_intermediate_soc/symmetry/check_symmetry.py
python -B exploration/he3_intermediate_soc/analyze_pilots.py
python -B exploration/he3_intermediate_soc/verify_pilots.py
python -B exploration/he3_intermediate_soc/figures.py
python -B exploration/he3_intermediate_soc/transport/review_pilots.py
```

These commands analyze existing records or perform deterministic calculations; they launch no MC. `pilot.py --benchmark` and `pilot.py label L gd preparation seed sweeps hot_endpoint` are the explicitly separate sampling entry points. The latter refuses to overwrite a saved label and checks a prior benchmark and runtime bounds. Actual labels, temperatures, seeds, initial energies, timings and settings are retained in each [pilot JSON/NPZ pair](pilots/); they are the authoritative experiment specifications.

| Output | Contents |
|---|---|
| [transport report](transport/TRANSPORT.md) | Old raw-data reconstruction, population sorting, failure diagnosis and controls |
| [landscape report](landscape/LANDSCAPE.md) | Onsite states, Hessians, connected paths, fixed-neighbor cost and harmonic limitation |
| [symmetry report](symmetry/SYMMETRY.md), [critical review](symmetry/REVIEW.md) | Independent mixed-response derivation, exact symmetries and inference checks |
| [pilot review](transport/PILOT_REVIEW.md) | Independent preparation, stationarity, block and physical-conversion checks |
| [curves.csv](pilot_results/curves.csv) | Every preparation and half: energies, structure, all selected orbital/spin spectra and tensor estimates |
| [joint_samples.csv](pilot_results/joint_samples.csv) | Event-level joint energy, spin, chirality, residual, planar weight and B powers |
| [transport.csv](pilot_results/transport.csv), [conversions.csv](pilot_results/conversions.csv) | Pilot transport and full-run threshold sensitivity; strict production counts are in the independent review |
| [preservation audit](preservation_after.json) | Source/data/branch preservation and final file checks |

The original [SOC investigation](../he3_soc/REPORT.md), [helicity-mechanism report](../he3_helicity_mechanism/REPORT.md), and [weak-SOC spatial report](../he3_weak_soc_spatial_01/REPORT.md) remain intact. Their scalar results are preserved; the new contributions identify the population-sorting mechanism, resolve the competing small-box orbital channel, and establish where ordinary short sampling already becomes unreliable.
