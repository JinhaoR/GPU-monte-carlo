# Independent mechanism review

The defensible conclusion is narrower than either "HQV BKT established" or "BKT ruled out." The data favor the half-charge **leading helicity fit** over the integer-charge leading fit on the measured sizes, while failing independent consistency checks for an already asymptotic, single-phase description. Local chiral-manifold disorder does not settle the infrared theory. The next data should test an entire scaling pattern, including which charge-one and charge-two tensor channels remain coherent, instead of electing a defect species from one fitted jump.

This review reads existing results and production definitions only. It does not run Monte Carlo, access Kraken, alter old reports, or fit new simulation data. The deterministic [audit script](audit_scaling.py) reproduces the algebraic checks in [results](results/audit_checks.json). The separate [evidence ledger](evidence_ledger.csv) and [falsification matrix](mechanism_falsification_matrix.csv) distinguish measurements from conditional interpretations. Parent and parallel-agent calculations are identified explicitly below.

## 1. What the helicity and exponent comparison really tests

The source `models/he3/measure.py` differentiates a twist applied to **all six complex components**. Its ensemble response is

\[
Y_\nu=[\langle K_\nu\rangle-\operatorname{Var}(I_\nu)/T]/N.
\]

This response is meaningful even when no local phase/director decomposition is meaningful. In contrast, writing

\[
\mathcal P(r)=Z(r)e^{2i\theta(r)},\qquad
H_{\rm IR}=\tfrac12\int\rho(\nabla\theta)^2,
\qquad \eta_P=2T/(\pi\rho)
\]

adds assumptions: a unique gapless compact phase, a nonzero infrared overlap of P with its charge-two vertex, sufficiently short-ranged neutral amplitude/tensor factors, and scale matching between the correlation and stiffness measurements. If another critical field multiplies the phase vertex, its scaling dimension contributes. A constant multiplicative amplitude renormalization cancels from a structure-factor ratio and cannot repair the observed ratio deficit; an additive/contact or correlated internal contribution can change that ratio.

The finite-size exponent from two global moments is not automatically the asymptotic spatial exponent. Writing \(S_P(0)=N\langle p^2\rangle\), its size ratio contains amplitudes, analytic backgrounds, irrelevant corrections, and proximity to a transition. It does not directly sample a single power law in distance. Near the existing largest-pair chirality crossing, the zero-SOC P effective exponents range approximately 0.66--0.90, changing strongly over a temperature shift of 0.0003. That makes identifying a transition by setting this exponent to 1/4 unjustified without joint finite-size consistency.

Two more qualifications matter:

* A generic stiffness tensor gives \(\eta_P=2T/(\pi\sqrt{\det\rho})\), after a coordinate transformation. Substituting the arithmetic mean of two diagonal responses requires isotropy and negligible off-diagonal response. Directional data should check the approximation; the old saved scalars do not contain the full mixed curvature. The geometric-mean eigenvalue entering the anisotropic defect criterion is also stated explicitly in [Korshunov, footnote 31](https://home.itp.ac.ru/~serkor/pdf/12-FFXY-honeycomb.pdf). Only that tensor statement is used here.
* The Gaussian coupling and the fully sector-averaged square-torus helicity are slightly different even in the ordinary XY reference. Hasenbusch explicitly sums winding sectors and obtains the helicity and second-moment ratio from the same finite-volume Gaussian measure. Its critical ratio is approximately 0.7506912, with logarithmic corrections; the winding correction to the critical helicity is small. These facts motivate consistent matching, not a direct transfer of that ratio to arbitrary P in this full GL model. [Primary calculation, Eqs. 29--36](https://arxiv.org/pdf/cond-mat/0502556).

The root investigation tests compact ordinary and paired scalar references with a self-consistent inversion of the torus response. It also finds a nonzero mixed response in descriptive sparse-snapshot checks; comparing only x and y diagonal components would miss that anisotropy. The determinant correction is several percent, too small by itself to explain the old ratio gap. Consult the [finite-volume table](../correlations/finite_volume_references.csv) for measured numbers and retain its sparse-snapshot limitation. Even a failed improved reference diagnoses an inadequate reduction, not a wrong production helicity estimator or a uniquely identified alternative mechanism.

An independent [reference audit](reference_audit.py) verifies the factors: paired rho=8T/pi and theta period pi map exactly to a charge-one phi=2theta scalar with rho_phi=rho/4; Y_paired/(4T)=0.6365081781905169. At L=64 the compact charge-two ratio is 0.75032992232, approaching the cited infinite-size Gaussian value. Independent covariance construction agrees to 2.8e-16. A synthetic contact background recovers its imposed amplitudes and local moment in the root code's Fourier normalization. [All controls pass](results/reference_audit.json); they validate algebra, not the He3 reduction.

## 2. Positive spin stiffness does not fix every discrepancy

On a smooth fixed-amplitude chiral branch,

\[
A=\sqrt n e^{i\theta}d(1,i\sigma)/\sqrt2,
\quad H_{\rm IR}=\tfrac12\int[\rho_c(\nabla\theta)^2+\rho_s|\nabla d|^2].
\]

A half-phase vortex with the shortest director half-path has single-defect logarithmic energy \(\pi(\rho_c+\rho_s)\ln L/4\). In this restricted decoupled model, comparing positional entropy gives marginality \((\rho_c+\rho_s)/T=8/\pi\), rather than \(\rho_c/T=8/\pi\). The spin coefficient is a **defect texture cost**; an independently measured nematic second-moment length is not an estimator of it. On nonabelian spin scales it runs and need not be represented by one constant over the whole box.

Thus a measured P exponent greater than 1/4 is not by itself incompatible with HQVs when unscreened spin cost matters. Combining the restricted formulas gives \(\rho_s/\rho_c=4\eta_P-1\) at marginality. For the old zero-SOC effective exponents, this conditional expression is 0.33--0.74. Applying exactly the same expression to the helicity-inferred exponents instead gives **negative** values, -0.18 to -0.15. These are not physical stiffness estimates: the calculation demonstrates that adding positive spin cost cannot simultaneously reconcile both sides of the P/helicity mismatch while leaving the single-Gaussian mapping intact. Finite-scale RG corrections mean neither expression should be enforced at the finite-size WM temperature.

The polar-spinor result motivating an 8T/pi jump concerns an isotropic, **spin-disordered paired phase** with algebraic charge-two order; it also finds ordinary KT behavior for easy-axis anisotropy in its own model. This supplies a conditional infrared example, not a mapping of the unrestricted He3 GL Hamiltonian, its chirality walls, or its finite-size data. [Mukerjee, Xu and Moore](https://arxiv.org/pdf/cond-mat/0605102).

## 3. Why the leading WM model comparison is not mechanism selection

The existing four-size q=1/2 zero-SOC fit has chi-square/dof 3.45 with only two nominal degrees of freedom. Excluding L=64 changes the fitted temperature by approximately 0.000063, exceeding its conditional block error. The q=1 fit is much worse (272.7). This strongly disfavors that particular q=1 leading ansatz **on L=64--160**. It does not test every integer-vortex crossover trajectory.

The [correction-size audit](results/wm_correction_sizes.csv) makes the limitation quantitative. With the fitted zero-SOC C=-4.3927, \(1/(2\ln L+C)\) is 0.255 at L=64 and 0.174 at L=160 for q=1/2. For q=1, C=-7.7462 produces corrections 1.75 and 0.416. A leading expansion with an order-one correction has little right to discriminate an asymptotic universality class without size-flow evidence. The formal pole length exp(-C/2) is an ansatz parameterization, not a measured core size. Holding C fixed at L=256 is only an illustration, not an uncertainty-qualified prediction.

The existing almost identical chirality and P Binder roots likewise do not independently locate two transitions: their underlying magnitudes have correlation approximately 0.964. They can follow the same finite-box amplitude/ordering fluctuation. The adjacent-size chirality roots are nonmonotonic. A genuinely discriminating observation would be stable ordering of size-flow curves and an interval of physically distinct spatial correlations, or a common limiting flow across several independent observables. Tiny paired errors on the difference of two roots do not bound the thermodynamic gap.

Relevant counterexamples exist. Large simulations of frustrated XY and coupled Ising-XY models find long nonmonotonic crossover regimes before separate nearby Ising and BKT transitions become visible. Their conclusion that the coincident transitions they studied are first order is specific to those model families; it cannot prohibit a continuous coupled transition here. The transferable lesson is to test the crossover scale and both sectors, not copy their phase diagram. [Hasenbusch, Pelissetto and Vicari](https://arxiv.org/pdf/cond-mat/0509511).

## 4. Local disorder and structure: what is and is not excluded

At T=0.3815, the earlier Stage-2 detector finds approximately 93% of microscopic sites invalid under its strict near-chiral tolerance and no accepted charged contours. The raw mean squared residual is approximately 0.338, with opposite-orbital weight 0.272 and complex-spin loss 0.066. This is a substantive failure of a microscopic London-field approximation. It is **not** a measurement of zero HQV density. Its finite search and contour acceptance have essentially no sensitivity in that regime.

The full nonzero field space is simply connected; the ordered chiral far-field manifold can still support meaningful defects whose cores escape it. Equally, a coarse-grained paired order can exist although averaging A itself cancels because the spin sector is disordered. Smoothing A until its manifold residual falls is not sufficient validation: filtering can remove defect pairs and project onto the chosen structure. A scale study must report P, B, A and Q correlations alongside the filtered residual, and preserve success and failure controls. No filtering parameter should be chosen because it returns the desired charges.

The density-weighted residual \(\langle n\epsilon\rangle/\langle n\rangle\), unweighted residual distribution, and separate orbital/spin terms answer different questions. A large residual need not imply vanishing P overlap, and a small residual need not imply macroscopic chirality order. In particular, local selected-chirality sign changes can be ultraviolet fluctuations rather than domain walls.

There is a useful new null from the root agent's simultaneous pilot checks: an additive **white/contact** background chosen to match the old harmonic prediction to both measured S0 and Sk would require a contact value far above the measured local \(\langle|P|^2\rangle\). At T=.3815 the required C0 is 7.80/6.23 for ordered/hot histories, versus the production-snapshot 0.394/0.420. The [background table](../correlations/white_background_test.csv) also checks the tensor reference. This excludes that particular repair, subject to the sparse-snapshot estimate of the local moment. It does not exclude a correlated short-range contribution extending over several sites. Its illustrative background correlation area is not a measured exponential length or a strict bound. The [two-size algebra audit](results/conditional_consistency_algebra.csv) shows why fitting only exponents is insufficient: 8--16% independent background fractions can reproduce the zero-SOC exponent pairs algebraically, yet those fractions alone enforce neither a common background across sizes nor the local-moment constraint.

## 5. Use the charge-two tensor before declaring phase disorder

The saved P is only one projection of \(B=A^T A\). Introduce orthonormal coordinates

\[
b_0=(B_{xx}+B_{yy})/\sqrt2,\quad
b_2=(B_{xx}-B_{yy})/\sqrt2=P/\sqrt2,\quad
b_{xy}=\sqrt2 B_{xy}.
\]

Then \(\|B\|_F^2=|b_0|^2+|b_2|^2+|b_{xy}|^2\). On the ideal chiral branch, \(b_0=0\) and \(b_{xy}=i\sigma b_2\). Therefore the two nonzero projections can have different infrared correlations when chirality fluctuates: P need not be suppressed simply because sigma disorders, while the off-diagonal channel carries sigma P. In a factorized ideal reference, \(C_B=(C_P+C_{\sigma P})/2\). This is a useful model identity, not a factorization assumption for thermal data.

Kraken's complex amplitudes support spectra of all three projections and their full 3-by-3 Hermitian cross-spectral matrix at each saved momentum. Predeclared projections are easier to interpret than noisy data-selected eigenvectors. Persistent P/B proportionality would support one charge-two channel; coherent b0 with weak P would falsify P as a sufficient U(1) diagnostic. An algebraic B/P sector accompanied by short-ranged full A is a strong paired-order signature, while coherent charge-one A would contradict the simplest spin-disordered paired description. The latter comparison needs multiple sizes because xi_A much larger than one box can mimic algebraic order.

Spectra alone still do not assign HQV topology or prove an equilibrium vortex-unbinding mechanism. Full-field qualified loops or controlled free energies would be additional evidence, currently absent from the saved sparse moment channels.

## 6. Full versus connected Q and sector averaging

At gD=0, exact global spin symmetry gives \(\langle Q\rangle=0\) in the finite-volume Gibbs ensemble. A finite sampled mean may reflect slow orientation coverage. Subtracting it can artificially reduce the long-wavelength nematic power; it must be reported as a sampling sensitivity, not automatically as the correct thermodynamic connected correlator. At gD>0, explicit alignment can produce a genuine nonzero mean. Then

\[
S_Q^{\rm full}(0)=S_Q^{\rm conn}(0)+N\|\langle Q\rangle\|^2.
\]

If connected correlations remain short ranged, this disconnected term alone can make the full xi_Q/L grow with L. Report the mean, full and connected spectra, and the m=0,1,2 tensor channels. A Q ratio elevated by alignment is not a spin correlation-length or string-tension measurement. Recompute mean subtraction inside each joint resample; do not subtract each configuration's spatial mean, which erases its zero mode by definition.

Even connected Q is density weighted. On a perfectly locked branch with d=z, \(Q=n(zz^T-I/3)\), so the connected axial m=0 channel inherits density correlations without any director tilt. A large total connected Q length therefore does not by itself show failed spin locking or a long director screening length under SOC. Compare the axial channel with density/amplitude diagnostics and the orientation-sensitive xz/yz channels (linear in small tilts), retaining xx-yy/xy as complementary quadratic-tilt channels. A disappearing full-to-connected excess identifies a disconnected mean contribution; labeling its cause as induced alignment additionally uses the model and symmetry context.

Spin Mermin-Wagner reasoning also needs care: exact SO(3) forbids ordinary spontaneous spin order at finite T under the usual assumptions, but absence of such order is not a direct proof that the measured Q length is already microscopic, nor by itself a proof of exponential rather than algebraic correlations. The finite spin correlation length of the appropriate nonabelian low-energy theory is a further dynamical/RG statement whose applicability must be checked.

There is an exact finite-sample/ensemble issue for helicity as well. For any fixed partition into sectors s,

\[
Y=\sum_s p_sY_s-\operatorname{Var}_s[\langle I\rangle_s]/(NT).
\]

The between-sector term follows from total variance. Suppressing visits to winding sectors can overestimate stiffness when their conditional currents differ. Mixing equilibrium sectors is physical and should not be "corrected away"; mixing nonstationary branches does not certify equilibrium either. Symmetry-related sectors with identical conditional current means need not change Y. PT swap acceptance and round trips do not test every structural/winding sector. Compare independent starts, invariant spectra, signed currents, blocking and history segments before interpreting universal flow.

## 7. New exact curvature decomposition: a useful, limited discriminator

The theory investigation derives, and the root investigation independently checks, an exact lattice identity:

\[
K_x+K_y=8K N_A-E_{\rm grad}+2K W,\qquad
W=\sum_r\operatorname{Re}[(A_x(r+x)-A_x(r+y))^\dagger
(A_y(r+x)-A_y(r+y))].
\]

Consequently the trace-averaged diamagnetic response splits into
\(D=4K\langle n\rangle-\langle E_{\rm grad}\rangle/(2N)+K\langle W\rangle/N\).
The incoming saved density, curvature and gradient-energy moments can reconstruct the last term; no measurement change is required. This separates a mechanical roughness contribution from an additional signed orbital-mixing term before the current variance is subtracted.

To see the limitation, set \(z_\pm=(A_x\mp iA_y)/\sqrt2\) and \(\delta f=f(r+x)-f(r+y)\). Direct substitution gives
\(W=\sum_r\operatorname{Im}[(\delta z_+)^\dagger\delta z_-]\).
It vanishes for a globally fixed orbital chirality even when the spin field is arbitrarily complex and rough. Nonzero W therefore falsifies that restricted kinematics, and its contribution to D can be quantified exactly. But W=0 is not evidence for pure chirality: every uniform nonchiral state also has W=0, and derivative overlaps can cancel. Neither Egrad nor W individually measures spin stiffness, a spin disorder fraction, or a vortex population. Combine them with the nonnegative local orbital/spin losses and B projections.

## 8. Current statistics constrain accounting, not a defect species

The parallel [data investigation](../data/EVIDENCE.md) finds that, across its stated L=160 windows, increased current screening contributes approximately 58--67% of the loss of trace-averaged helicity and decreased curvature contributes 33--42%. An exact symmetric product decomposition assigns approximately 59% of the curvature loss to the mean-density factor. These are useful non-negligible contributions to one response identity. They are not two causally independent microscopic mechanisms, and changing the window changes their shares. The weak-SOC upper endpoint also retains more helicity, so comparing fractions without the hotter shoulder would conflate mechanism with unequal progress through the crossover.

Near-Gaussian current histograms and non-dominant extreme records exclude a sampled-variance story driven by one or two outliers. They do not contradict vortex unbinding: currents are extensive sums, defect currents can cancel spatially, and a response can change through many ordinary-sized fluctuations. A vortex mechanism does not require a visibly heavy-tailed total-current histogram.

The law-of-total-variance check finds negligible contributions from different mean currents in the tested chirality, density and spin-alignment classes. It leaves unobserved winding labels unconstrained. The current-power/absolute-chirality correlation is weak and largely disappears after a linear density adjustment. This weakens attributing the scalar association uniquely to chirality walls. It does not exclude walls acting through density, and a conditioned variable can be a mediator or collider; partial correlation is not an intervention. No causal allocation between walls, spin textures and vortices follows from these scalar histories. The data agent recomputed current centering within each jackknife deletion after review; numerical changes to the reported partial-correlation errors were negligible.

## 9. Prioritized Kraken decisions

1. Preserve per-history estimates and joint blocks across all channels and temperatures. Confirm field/scalar zero-mode identities, finite values, histogram normalization, and total energy as the sum of bulk, gradient and SOC pieces. Inspect raw full and connected Q before pooling.
2. At matched temperatures and at matched dimensionless P/Y states, chart Y/T, xi_P/L, xi_A/L, projected B spectra, xi_kappa/L, connected xi_Q/L, residual distributions and energy components against L. Matching a reduced state is necessary because a small transition shift can otherwise make a weak-SOC difference look enormous near a steep feature.
3. At zero SOC, demand simultaneous evidence for a paired regime: scale-stable P/B coherence, A and connected Q becoming short relative to L, and a consistent mass-only compact-boson correlation/response relation. A persistent Q scale comparable to L leaves the asymptotic mass-only test inconclusive; a persistent P/Y discrepancy after its internal scales separate falsifies that simple reduction.
4. At gD=0.01, test whether explicit alignment removes the charge-one suppression and changes the large-L flow toward a spin-locked ordinary phase, or whether structural channel changes intervene. Use orientation-sensitive Q channels to assess tilts; long connected axial Q correlations can instead come from density fluctuations on an already locked branch. The bare chiral spin-kink length \(\sqrt{2K/g_D}\approx14.1\) is a zero-temperature restricted benchmark, not proof that L=64 already exceeds the thermal confinement scale. Existing pinned zero-temperature pair energies are not pair free energies or string tensions.
5. A claimed Ising/BKT split needs an interval that persists with L, with distinct spatial behavior in kappa and the appropriate charged channel. A coincident critical claim needs common scaling compatible with corrections. A first-order claim needs reproducible equilibrium switching, nonzero limiting energy-density separation and a growing interface barrier; present unimodal energy data provide no such positive evidence.

These tests can reject simplified mechanisms with the saved observables. They cannot guarantee that L=256 is asymptotic, recover a spin twist modulus from Q, measure a confinement free energy, or assign vortex populations from five Fourier modes. A scientifically useful outcome is an explicit unresolved crossover scale with several mechanisms falsified, rather than an unsupported named universality class.
