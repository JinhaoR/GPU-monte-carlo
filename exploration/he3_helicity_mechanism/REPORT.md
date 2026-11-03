# What causes the helicity-modulus drop in the full thin-film He3 model?

Research synthesis, 1 November 2026. This investigation uses existing weak-SOC histories, earlier exploratory results, deterministic CPU calculations, and three independent research/review workstreams. It ran **no Monte Carlo, relaxation campaign, GPU workload, or cluster job**. All new work is confined to this directory.

## A. Main findings

**The measured drop is a loss of phase rigidity involving both local energetic curvature and collective current screening. Its microscopic defect mechanism and thermodynamic classification remain unresolved.** The evidence is stronger than a favorable half-charge fit, but does not establish HQV unbinding. The main advances are:

1. **Both terms in the exact response change substantially.** In stated L=160 transition windows, decreasing curvature accounts for 32.6%, 35.1%, and 42.1% of the helicity loss at g_D=0, .001, and .01; increasing current screening accounts for the rest. Screening becomes relatively more important with increasing L. These are finite-window response contributions, not percentages of independent microscopic causes.
2. **Density matters, but density alone fails.** In an explicitly defined product decomposition, density accounts for about 59% of the curvature decrease. Curvature per unit density also falls. The apparent current-power/chirality association largely disappears after linear adjustment for density. That weakens an inference of a unique wall mechanism from those scalar correlations; it does not exclude walls acting through density or nonlinear effects.
3. **A new exact curvature identity provides a useful Kraken test without additional measurements:**
   \[
   D=4K\langle n\rangle-\frac{\langle H_g\rangle}{2N}
       +\frac{K\langle W_{\rm diag}\rangle}{N}.
   \]
   Here D is mean directional curvature per site, H_g is gradient energy, and W_diag is a signed derivative overlap between opposite orbital chiralities. The saved density, gradient energy, and curvatures suffice to reconstruct the last term. It separates gradient roughness from this particular orbital interference, but does not count defects.
4. **Several tempting microscopic explanations fail exact checks.** Uniform internal rearrangement at fixed density cannot lower the averaged bare curvature. The bare chiral minimum has no additional soft mode that appears as MC temperature crosses the observed feature. Chirality walls retain phase coupling in the actual gradient stencil; they are not identically broken phase bonds. Local wall currents also need not yield a total current.
5. **The composite/helicity discrepancy survives better finite-volume references.** Compact winding sectors and the tested anisotropic Gaussian corrections, including a sparse tensor estimate, are far too small to explain the L=64 pilot gap. A white/contact operator background sufficient to repair it requires 15–20 times the local composite second moment present in the saved fields. Correlated internal backgrounds, scale-dependent non-Gaussian phase fluctuations, and incomplete separation of spin/chirality scales remain viable.
6. **Neither local disorder nor near-Gaussian total currents settle HQV physics.** Bare Gaussian massive modes predict a mean local chiral residual of .105 at T=.3815, versus the earlier measured .338. This rules out that quantitative bare approximation, not an infrared paired phase. Nearly Gaussian extensive currents do not imply vortex-free configurations.
7. **The most discriminating next test is a joint scaling pattern.** At zero SOC, test whether charge-two P/B correlations remain coherent while charge-one A and spin correlations become short relative to L, and whether the composite correlations become consistent with the same phase stiffness. At finite SOC, test whether spin locking changes that pattern. A helicity fit alone cannot answer either question.

Confidence is high in the exact identities, normalization checks, and measured finite-history accounting. Confidence is moderate in the descriptive finite-size trends. A coupled amplitude/texture/phase crossover is a useful working interpretation of the available boxes; identifying a distinct coupled transition, BKT transition, or defect species would exceed the evidence.

The [evidence ledger](review/evidence_ledger.csv) labels measurements, exact results, conditional theories, and hypotheses. Detailed supporting notes are [numerical evidence](data/EVIDENCE.md), [theory](theory/THEORY.md), and [independent critique](review/REVIEW.md).

## B. Verified numerical evidence

### B1. What was already known, and what was added

The old weak-SOC campaign contains twelve datasets at g_D=0,.001,.01 and L=64,96,128,160. Each has 2M thermalization sweeps followed by 3M production sweeps, with 60,000 scalar records. The present work reuses the validated block caches, independently checks selected raw histories, and adds response decomposition, covariance derivatives, current-distribution and conditional-association analysis. Raw/cache means agree to 1.46e-11.

The [transition study](../he3_transition_study/REPORT.md) established a preference for the tested q=1/2 leading Weber–Minnhagen ansatz over the tested q=1 ansatz. At zero SOC the four-size fits have chi-square/dof 3.45 versus 272.7. This is a model-comparison result, not an observed vortex charge. For the half-charge fit, the relative logarithmic correction is still 25.5% at L=64 and 17.4% at L=160. For the unsuccessful integer fit it is 175% and 41.6%; that leading expansion is plainly not controlled at its smaller sizes. Dropping L=64 shifts the half-charge fitted temperature by approximately -0.000063, larger than its conditional statistical error. The [correction audit](review/results/wm_correction_sizes.csv) documents these numbers.

The nearly coincident chirality and P Binder roots are also not independent transition determinations: their magnitudes have correlation about .964 in the old analysis. Nonmonotonic crossing drift and the absence of resolved energy coexistence leave close separate transitions, a coupled transition, and a large crossover open.

The [spatial pilot](../he3_spatial_pilot/REPORT.md) supplies two independent L=64,g_D=0 starts, each with 100k production sweeps, 2,000 global records, and 400 spatial records. This work independently reproduces its P/helicity comparison and extends the references. Only four production snapshots per selected temperature are available across both histories. Snapshot operator comparisons are therefore descriptive, not precision equilibrium estimates.

The [HQV studies](../he3_hqv_stage1/REPORT.md), [thermal detector study](../he3_hqv_stage2/REPORT.md), and [constrained relaxation study](../he3_hqv_stage3/REPORT.md) establish different things: qualified synthetic HQVs can be detected; critical thermal fields often fail the detector's microscopic quality criteria; constrained full-field HQV minima can survive core reconstruction. None measures an equilibrium thermal HQV population or a pair free energy. The short weak-SOC pinned-pair baseline does not establish asymptotic confinement.

### B2. Curvature and current screening

Write N=L² and distinguish the Hamiltonian coupling K=1 from twist curvatures \(\mathcal K_\mu\). Define

\[
D=\frac{\langle\mathcal K_x+\mathcal K_y\rangle}{2N},\quad
S=\frac{\operatorname{Var}I_x+\operatorname{Var}I_y}{2TN},\quad
Y=\frac{\Upsilon_x+\Upsilon_y}{2}=D-S.
\]

| g_D, L=160 | Temperature interval | D, low → high | S, low → high | Y, low → high | Fraction of Y loss from falling D |
|---|---|---|---|---|---|
| 0 | .38030047–.38413969 | 2.55476 → 2.18328 | 1.19559 → 1.96497 | 1.35917 → .21832 | .3256 ± .0024 |
| .001 | .38030047–.38413969 | 2.57084 → 2.19667 | 1.17493 → 1.86811 | 1.39591 → .32856 | .3506 ± .0036 |
| .01 | .38349459–.38739762 | 2.55700 → 2.21471 | 1.17661 → 1.64707 | 1.38039 → .56764 | .4211 ± .0047 |

Errors are joint delete-one estimates from 30 blocks of 100k sweeps. They condition on the sampled history and do not include unvisited sectors or window selection. Local canonical derivatives, computed independently from covariances with energy, give curvature shares .329±.010, .372±.013, .424±.020 at the central temperatures. Thus the conclusion does not depend only on two endpoints. Changing the windows changes the fractions; see [data/EVIDENCE.md](data/EVIDENCE.md).

![Response components for all twelve old weak-SOC datasets. Shading denotes the primary accounting windows.](data/decomposition.png)

For g_D=0 the curvature share decreases from .415 at L=64 to .326 at L=160. The analogous changes are .427→.351 at .001 and .457→.421 at .01. This is evidence that screening sharpens with size, not a measured critical exponent. At .01 the primary upper endpoint retains substantially more Y; its larger curvature fraction should not be read as a different universality class. From T=.38740 to .39340 its remaining Y falls .568→.022, with 72.5% of that further fall coming from screening.

To examine density, set R=D/(4Kn). The exact endpoint identity

\[
\Delta D=4K R_{\rm mid}\Delta n+4K n_{\rm mid}\Delta R
\]

assigns about 59% of the primary curvature loss to density and 41% to R for all three L=160 cases. In that accounting, density supplies about 19–25% of the total Y loss. R is a normalized curvature diagnostic, not a spin stiffness or a probability; it is not bounded by one for arbitrary lattice fields. This allocation is not unique causal attribution. The microscopic identity in section C gives a complementary decomposition tied directly to the implemented gradient.

### B3. What current fluctuations do and do not reveal

Across 240 selected directional distributions, standardized excess kurtosis ranges from -.038 to .091. The top 1% of absolute centered currents supply 8.22–8.83% of variance, close to the Gaussian value 8.45%. No single 100k-sweep block dominates those tails. The sampled screening therefore comes from a broad distribution, not one or two numerical bursts. An extensive sum can be nearly Gaussian even when defects contribute importantly.

Partitioning records by chirality sign, |chirality| quartiles, density quartiles, or spin-alignment quartiles gives at most .0261% of current variance from differences of conditional mean currents. The variance changes within those groups. This rejects a large between-mean contribution for the **tested scalar labels**; winding sectors were not measured and are not constrained by this result.

At the central L=160 temperatures, correlations between current power and |chirality| are -.063, -.065, -.049. After linear density adjustment they become -.003±.004, -.002±.003, -.009±.005. Density and |chirality| themselves correlate .86–.90. Scalar association therefore does not isolate wall-induced screening. Partial correlation cannot exclude nonlinear coupling, selection effects, or a causal path through density.

Diagonal responses agree within the sampled errors. However, current cross correlations near the feature are approximately -.06 to -.09. The actual forward stencil permits a mixed response even when x and y diagonal responses agree. Without the missing mixed curvature \(\mathcal K_{xy}\), the old scalar records do not determine the stiffness tensor.

### B4. Statistical limitations

The mean-current centering correction is at most .000231 across the weak-SOC grid, much smaller than the helicity errors. It cannot explain the drop. Cancellation does amplify statistical errors as Y approaches zero: for zero SOC,L=160, D and S are about 2.18 and 1.96 at T=.38414. Their difference is .218±.007; the ratio (|D|+|S|)/|Y| is 19 and rises to 64 at T=.38543632. This is uncertainty in ensemble estimates, not inadequate floating-point subtraction precision.

Transition-region Y is fairly stable between time halves; only one of 36 selected comparisons exceeds two diagnostic errors. Low-temperature signed chirality shows appreciable drift in several histories. Changing block sizes does not produce a clear universal error plateau. Completed L=160 round trips in the old data average roughly 2.77–3.06M sweeps, conditioned on completion, comparable to the entire production interval. These observations support useful finite-history even-moment estimates while leaving equilibration of slow internal/topological modes uncertified. The temperature-indexed PT histories are Monte Carlo sampling records, not real-time defect dynamics.

## C. Theoretical analysis of the actual model

### C1. Hamiltonian, twist, and soft modes

The implemented field is an unrestricted complex 3×2 matrix. Its onsite energy is \(-2n+\sum_j\beta_j I_j+g_D D_{\rm SOC}\), with n=tr AA†. The source definitions are [update.py](../../models/he3/update.py), [measure.py](../../models/he3/measure.py), and [model.py](../../models/he3/model.py). Pressure is 20 and the reduced-temperature parameter is fixed at one; changing MC temperature does **not** change the coefficients. The resulting beta vector is approximately (-.516512,.942023,.987529,.879063,-1.152515). The onsite SOC invariant is

\[
D_{\rm SOC}=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\operatorname{Re}(A_{xy}A_{yx}^*)-\tfrac23n.
\]

For forward differences, the exact gradient per spin row is

\[
G=3(|D_xA_x|^2+|D_yA_y|^2)+|D_xA_y|^2+|D_yA_x|^2
+2\operatorname{Re}[(D_xA_x)^*D_yA_y+(D_xA_y)^*D_yA_x].
\]

The common twist replaces each forward neighbor by \(e^{-iq_\mu}A(r+\hat\mu)\). Equivalently, using a unitary Fourier transform, \(H_g(q)=K\sum_k A_k^\dagger M(k-q)A_k\), where

\[
M(k)=\begin{pmatrix}3|p_x|^2+|p_y|^2&2\operatorname{Re}p_x^*p_y\\
2\operatorname{Re}p_x^*p_y&|p_x|^2+3|p_y|^2\end{pmatrix},\qquad p_a=e^{ik_a}-1.
\]

Thus \(I_a=-K\sum A^\dagger\partial_aM A\) and \(\mathcal K_{ab}=K\sum A^\dagger\partial_a\partial_bM A\). No onsite term has a direct common-twist derivative. SOC changes helicity through the sampled configurations and correlations. Independent finite-twist calculations of the full Hamiltonian reproduce both currents, both diagonal curvatures, and the mixed curvature to 4.3e-9 on six static full-field controls at g_D=0 and .01.

At g_D=0 there is exact global U(1) and spin SO(3), together with the discrete chirality-related symmetry. On either ideal chiral component,

\[
A=\sqrt{n/2}\,e^{i\theta}d(1,i\sigma),\quad
(d,\theta)\sim(-d,\theta+\pi),\quad d\in S^2.
\]

The fixed-chirality manifold is (S²×U(1))/Z2 and permits HQV far fields. The full nonzero matrix field space is simply connected; core escape from the manifold does not invalidate ordered-manifold defects, but makes their identification conditional on a meaningful far field.

Putting u=(A_x-iA_y)/√2 and v=(A_x+iA_y)/√2 makes the mode content explicit. Around u=√n₀ z,v=0, n₀=1.495728, the twelve real onsite Hessian modes are:

| Mode | Multiplicity | Hessian eigenvalue |
|---|---:|---:|
| Common phase | 1 | 0 |
| Real director rotations | 2 | 0 |
| Radial amplitude | 1 | 8 |
| Nonunitary transverse spin | 2 | 13.7908 |
| Opposite orbital chirality, longitudinal spin | 2 | 5.6361 |
| Opposite orbital chirality, transverse spin | 4 | 7.5443 |

These are static masses, not mode frequencies. Positive SOC gaps the director modes (leading eigenvalue 2g_D), leaving one exact phase mode. Neither MC temperature nor the sharp thermal feature creates a new zero in this *bare* fixed Hamiltonian Hessian.

The opposite-chirality gradient mixing starts at k². Eliminating its massive modes changes the quadratic phase kernel first at k⁴:

\[
H^{\rm eff}_{pp}(k)=4Ks-\frac{4K^2|c|^2}{\lambda_v+4Ks},\quad
s=|p_x|^2+|p_y|^2,\quad c=|p_x|^2-|p_y|^2-2i\operatorname{Re}(p_x^*p_y).
\]

Here p=√n₀ θ is the real phase fluctuation coordinate. Stable massive modes soften shorter wavelengths; they do not erase the leading bare phase stiffness or supply an additional harmonic relative phase. Thermal renormalization and strongly nonuniform configurations remain outside this bare argument.

### C2. Spin coupling, uniform rigidity, and the exact curvature identity

On a smooth fixed-chirality real-director branch,

\[
H_g=2K\int[(\nabla\sqrt n)^2+n(\nabla\theta)^2+n|\nabla d|^2]\,d^2r.
\]

Bare phase and director stiffnesses are both 4Kn, without bilinear drag. This does not decouple their full thermal physics. An actual fixed-density lattice bond has energy

\[
E_{ij}=4Kn[1-(d_i\cdot d_j)\cos(\theta_j-\theta_i)],
\]

which couples spin and phase beyond quadratic order. Complex spin deviations also produce a phase connection; near a real director it is proportional to the product of nonunitary displacement and director gradient. Integrating those massive deviations generates nonlinear interactions. Spin fluctuations can consequently affect mass rigidity without an additional harmonic gapless mode.

For **any uniform** matrix at fixed n, the bare stiffness tensor is

\[
\rho^{(0)}_{ab}=2K[n\delta_{ab}+2\operatorname{Re}(A_a^\dagger A_b)],\quad
\rho^{(0)}_{xx}+\rho^{(0)}_{yy}=8Kn.
\]

Its eigenvalues lie between 2Kn and 6Kn. Uniform chirality changes, spin nonunitarity, or orbital redistribution cannot by themselves collapse its average at fixed density. Spatial roughness, relaxation, density loss, or fluctuation screening are essential to that proposed route.

For arbitrary periodic full fields, differentiating M gives

\[
(\partial_x^2+\partial_y^2)M=8\mathbf1-M+|e^{ik_x}-e^{ik_y}|^2\sigma_x,
\]

and hence

\[
\mathcal K_x+\mathcal K_y=8KN_A-H_g+2KW_{\rm diag},\qquad
W_{\rm diag}=\sum_r\operatorname{Re}[(\delta A_x)^\dagger\delta A_y],
\quad\delta A=A(r+\hat x)-A(r+\hat y).
\]

Equivalently \(W_{\rm diag}=\operatorname{Im}\sum(\delta u)^\dagger\delta v\). It vanishes in a globally fixed orbital chirality even for arbitrary complex spin textures. Nonzero W therefore detects departure from that restricted kinematics. Zero W does not establish it: uniform nonchiral fields and cancellations are counterexamples.

The identity passes independent checks on all sixteen production pilot snapshots to 1.7e-9. At T=.3815 their pooled descriptive values are

\[
4\langle n\rangle=3.56948,\quad-\langle H_g\rangle/(2N)=-1.00632,
\quad\langle W\rangle/N=-.01143,\quad D=2.55173.
\]

The small direct interference correction coexists with a large density-weighted local residual, .324. This is an explicit warning that residual size and signed interference are different diagnostics. Four fields cannot establish the thermal derivative of these contributions; Kraken's synchronized measurements can.

### C3. Chirality walls: a possible coupling mechanism, not a demonstrated cause

Substituting a sharp straight wall into the actual gradient, with fixed n and common director, gives an x-normal opposite-chirality link

\[
E^{x}_{ij}=Kn[4-2\cos\Delta\theta],
\]

and a y-normal link \(Kn[4+2\cos\Delta\theta]\). At their preferred offsets the phase curvature is 2Kn, half the same-chirality value, rather than zero. The thesis's reduced model with absent phase coupling across opposite chirality is therefore not an exact restriction of the implemented full stencil.

For a smooth x-normal wall, A_x=f(x)e^{iθ}d,A_y=ig(x)e^{iθ}d, the energy contains a tangential current term

\[
2K(fg'-gf')\,\partial_y\theta.
\]

A chirality zero is not necessarily a density or rigidity zero. At constant n, writing f=√n cosη,g=√n sinη makes the current coefficient 2Knη′. An ordinary periodic pair of opposite walls cancels its integrated current. Local wall current is therefore insufficient to explain the measured variance of total I.

Walls can still soften a response by relaxing internal coordinates. On a stable constrained branch,
\(\partial_q^2H_{\min}=H_{qq}-H_{qa}H_{aa}^{-1}H_{aq}\).
Soft wall coordinates can make the second term substantial. This static mechanism motivates thermal current/structure tests; it is not a calculation of their thermal contribution. Wall geometry, amplitudes, spin textures, and vortices can interact, so these are not mutually exclusive causal categories.

### C4. What is required for an HQV jump or confinement?

In the restricted isotropic smooth regime an HQV carries a π phase winding and a director half-turn. Its scale-dependent energy is approximately

\[
E_h(R)=E_{\rm core}+\frac\pi4\int_a^R[\rho_c(r)+\rho_d(r)]\,d\ln r.
\]

Before the director contribution is screened, a dilute-defect marginality estimate involves \((\rho_c+\rho_d)/T=8/\pi\), not a mass-only jump. The nonabelian S² sector is not a second independent XY vortex gas, and a Q correlation length does not measure its running stiffness.

The familiar mass jump 8T/π additionally requires unconfined half vortices, a spin-screened paired regime, and an appropriate asymptotic compact-phase description. The polar-spinor model of [Mukerjee, Xu and Moore](https://arxiv.org/abs/cond-mat/0605102) realizes that scenario; its shared quotient manifold is a useful analogy, not a reduction of this full GL model with chirality and amplitude fluctuations.

Within the chiral restriction, positive SOC contributes h_D(1-d_z²), h_D=g_D n. The bare locking length is √(2K/g_D), about 44.7 at .001 and 14.1 at .01. If a locked chiral vacuum and nonzero thermal string tension are established, isolated HQVs become confined at long distances and ordinary vortices are the unconfined asymptotic defects; this alone does not establish BKT criticality. The bare length alone does not establish that regime: spin screening, anisotropy renormalization, full-tensor escape, and entropy can change it. Existing zero-temperature pinned-pair energies are neither equilibrium pair free energies nor real-time dynamics.

### C5. Composite correlations and the phase-only relation

On the ideal branch, P=Σ(A_x²-A_y²)=n e^{2iθ}; the director drops out. If its neutral prefactor has nonzero infrared overlap and short-ranged connected fluctuations, and a single Gaussian phase describes long distances,

\[
\eta_P=\frac{2T}{\pi\rho_c},
\quad\text{or}\quad \frac{2T}{\pi\sqrt{\det\rho_c}}
\text{ for a stiffness tensor}.
\]

An extra spin-wave exponent cannot simply be added to this spin singlet. Residual spin stiffness changes HQV texture cost, but does not repair an inconsistent P/mass-stiffness relation while retaining the same Gaussian assumptions. Additional critical neutral factors, vanishing operator overlap, scale-dependent non-Gaussian physics, or insufficient scale separation can invalidate that mapping.

The finite-volume test uses the exact periodic lattice Gaussian covariance, then includes winding sectors consistently. For a scalar phase period a=2π or π, their weights are \(w_m\propto e^{-\rho a^2m^2/(2T)}\). The torus response is \(Y=\rho-\rho^2a^2\langle m^2\rangle/T\). We solve this for ρ using the measured Y and use the **same** weights in the P correlation. The π-period reference assumes no additional spin cost. It is a conditional paired scalar theory, not an assumption that the He3 samples contain HQVs. The treatment follows the finite-volume distinction between coupling and response in [Hasenbusch's XY calculation](https://arxiv.org/abs/cond-mat/0502556).

| L=64,g_D=0,T=.3815 | Ordered start | Hot start |
|---|---:|---:|
| Measured Y | 1.23043 | 1.20378 |
| Measured ξ_P/L | .65816 ± .01260 | .68429 ± .01576 |
| Zero-winding Gaussian reference | .85612 | .84584 |
| Self-consistent paired-winding reference | .85612 | .84585 |
| Tensor reference using sparse mixed curvature | .85439 | .84415 |
| Measured minus paired reference, joint error | -.19796 ± .01563 | -.16156 ± .01719 |

The nonzero half-winding probability per axis is only 2.45e-7 or 3.46e-7 in this reference. Including spin winding cost would suppress it further. Diagonal anisotropy changes ξ/L by less than .0003. A mixed-response plug-in estimate gives Y_xy≈.274/.279 and changes the reference by about .002; it combines only two mixed-curvature snapshots per history with full-history covariance and cannot support a precise tensor inference. Two reasonable lattice realizations of the same tensor give essentially the same conclusion. None of these tested corrections approaches the .16–.20 gap.

![Measured composite correlation ratios and finite-volume references. Error bars describe the sampled histories. The tensor check uses sparse mixed-curvature snapshots.](correlations/finite_volume_references.png)

A constant multiplicative operator overlap cancels from ξ/L. For a more specific proposed repair, take C_P(r)=A C_G(r)+Bδ_r0 with C_G(0)=1. Matching the measured S(0) and S(k_min) determines A and B, and predicts C_P(0)=A+B. At .3815 it requires 7.80/6.23, compared with actual snapshot local moments .394/.420. The simple white-background explanation fails badly. A correlated short-range background is still possible; under a roughly flat low-k background its required integrated-to-local weight would correspond to an illustrative area of about 38/27 sites. This is not a measured length or a bound on all possible backgrounds.

Bare Gaussian normal-mode fluctuations are also quantitatively insufficient: at .3815 they predict mean chiral residual .1054, versus the old thermal .3379. Their leading orbital/spin contributions are .0858/.0196 versus .2716/.0664. This calculation is controlled at small fluctuations and deliberately does not fit thermal masses. Its failure cannot identify vortices, but demonstrates that the bare small-fluctuation explanation of local disorder is inadequate.

Four-snapshot comparisons of P, P/n, P/|P| and full B show that removing local amplitude weighting does not erase all changes in correlation shape. They are too sparse to settle an exponent: at .385 those four P fields even give S(0)<S(k_min), while the 400-event estimates yield a positive ratio. The [descriptive profiles](correlations/snapshot_operator_profiles.png) are retained to expose this sampling limitation, not to fit a new phase diagram. Correlated internal structure and non-Gaussian phase fluctuations remain open explanations of the robust full-history discrepancy.

## D. Competing mechanisms

| Explanation | Evidence and present status | Discriminating outcome |
|---|---|---|
| Asymptotic HQV BKT in a paired regime | Quotient topology and favorable leading half-charge fit support plausibility. No qualified thermal HQVs or established spin-screened scale; the simple finite-box P/Y relation fails. | P/B coherent across sizes while A and connected spin correlations become short relative to L, followed by consistent compact-phase scaling. Topological evidence still required to identify HQVs directly. |
| Ordinary integer-vortex BKT | Tested leading q=1 fit is very poor. Integer crossover trajectories remain possible, particularly with eventual SOC locking. | Spin-locked charge-one coherence and convergent ordinary-phase stiffness/correlation flow at larger L; failure of this pattern weakens that reduction. |
| Coupled chirality/phase transition or close transitions | Strong coincident finite-box features, changing density/curvature, and plausible wall/phase coupling. Scalar correlations do not establish wall causality. | A stable intervening interval with different charged/chiral correlations supports a split; persistent common limiting flow supports coincidence, without by itself distinguishing coupled from accidentally coincident decoupled critical sectors. |
| Structural/amplitude fluctuations renormalize stiffness | Directly present in curvature accounting; bare massive-mode approximation fails quantitatively. Uniform fixed-density redistribution alone is ruled out. | Correlated changes in density, gradient roughness, orbital/spin residuals and B projections; a new coherent charge-two projection with weak P would expose structural reorientation. |
| Large finite crossover | Large WM corrections, nonmonotonic crossings and box-scale Q correlations make this a serious possibility. | Correlation ratios and apparent transition estimates keep drifting with L without a stable independent scaling pattern. |
| Weak first order | No positive coexistence evidence in current zero-SOC energy histograms. Existing sizes do not exclude it. | Reproducible equilibrium switching, nonzero limiting energy-density separation, and a growing interface barrier, rather than a single noisy bimodal histogram. |
| Estimator or sampling artifact | Twist normalization, centering, and arithmetic pass checks. Slow-sector sampling remains uncertain. | Persistent independent-start or production-segment differences would limit all thermodynamic claims; verified stationarity across sectors would weaken this alternative. |

Frustrated XY and coupled Ising–XY examples demonstrate that long nonmonotonic crossovers can precede close separate transitions. Their particular phase diagrams, including first-order coincident transitions, are not theorems about He3. The useful lesson is to demand independent sector scaling and assess internal crossover lengths. [Hasenbusch, Pelissetto and Vicari](https://arxiv.org/abs/cond-mat/0509511).

## E. Independent validation and null findings

Three independent workstreams handled data, derivations, and adversarial review. The review checked normalization and causal interpretation, including the limitations of density adjustment and scalar-sector variance partitions.

| Check | Result | What it validates |
|---|---|---|
| Selected raw histories versus old block caches | Maximum mean difference 1.46e-11 | Reuse of old measurements |
| Response/product/variance algebra | Residual ≤8.9e-16 | Numerical accounting, not causal attribution |
| Source onsite potential versus independent implementation | Difference ≤9.1e-13 | Actual GL coefficients and contractions |
| Independent finite-q full-H derivatives | Error ≤4.3e-9 | Twist signs, normalization, mixed derivative |
| Exact curvature identity, synthetic fields | Error ≤4.1e-13 | Full-field lattice identity |
| Same identity, saved thermal fields | Error ≤1.7e-9 | Applicability to the stored fields and scalar channels |
| Homogeneous Hessian and phase Schur complement | Errors ≤7.2e-9 and 2.3e-16 | Mode masses and k⁴ correction |
| Original spatial-pilot reference | Reproduced within 2e-13 | Fourier and second-moment normalization |
| Independent compact reference | Paired phase maps exactly to φ=2θ ordinary scalar; covariance error 2.8e-16 | Charge and winding factors |
| Synthetic contact background | Recovers imposed A=.2,B=.1,C(0)=.3 | Local-moment rejection test |

Important nulls are explicit: no normalization error found; no dominant sampled rare-current tail; negligible current-mean correction; no new bare soft mode; no loss of averaged uniform curvature from fixed-density structural rotation; no identically broken phase bond at a chirality wall; no repair of the pilot gap by the tested winding, anisotropy, or contact-background corrections. None alone excludes vortex physics.

Reproduction from the repository root, using NumPy/SciPy/Matplotlib and the existing saved data:

```text
python -B exploration/he3_helicity_mechanism/data/decompose.py
python -B exploration/he3_helicity_mechanism/data/current_histories.py
python -B exploration/he3_helicity_mechanism/data/conditional_associations.py
python -B exploration/he3_helicity_mechanism/data/summarize_evidence.py
python -B exploration/he3_helicity_mechanism/theory/harmonic_checks.py
python -B exploration/he3_helicity_mechanism/correlation_checks.py
python -B exploration/he3_helicity_mechanism/review/audit_scaling.py
python -B exploration/he3_helicity_mechanism/review/reference_audit.py
```

These scripts read old results and write only this investigation directory. The raw-history stage is I/O-heavy; the deterministic theory and correlation calculations take seconds. Small CSVs preserve the quantitative results; figure-producing scripts recreate the figures. The [final audit](source_audit_after.json) confirms unchanged hashes for 107 pre-existing Python files, unchanged size/mtime for 19 archived datasets, and unchanged Git HEAD and branch. The only added Git status entry is this investigation directory; earlier exploratory work is preserved.

## F. Prioritized predictions for the incoming Kraken data

These are analysis proposals, not implemented campaign changes. Preserve the saved temperature indices and synchronized sample indices; spatial records have a different cadence from scalar records. Compute powers before ratios, and recompute nonlinear estimators and ensemble means inside each joint block deletion. Do not infer physical time evolution from PT-swapped temperature histories.

### Priority 1: Determine what changes in the response

At each L and g_D, plot D,S,Y together with n and gradient energy. On the synchronized spatial subset reconstruct

\[
\frac{W_{\rm diag}}N=\frac{D-4Kn+H_g/(2N)}K.
\]

Use the matching scalar curvatures at those exact indices; retain the full-cadence Y estimate for precision. The channel conversions are n=`density`/N, D=(`curvature_x`+`curvature_y`)/(2N), and H_g=`energy_gradient`, which is extensive and already includes K. Compare temperature derivatives and finite intervals, with joint errors, rather than assigning meaning to a single threshold.

**Outcomes:** density plus gradient roughness accounting for D with small W would disfavor a large *direct interference* contribution, while leaving indirect orbital effects possible. A sharp W feature would reject globally fixed orbital chirality as adequate kinematics for that part of the response. Large current screening with nearly unchanged D would isolate a different response pattern from the old narrow-window mixture. No outcome identifies vortices without topology.

Compare energy_bulk, energy_gradient and energy_soc at the same points. Because SOC has no direct common-twist derivative, a strong spin/alignment change with a modest Y change is physically possible; it does not require a faulty response estimator.

### Priority 2: Test paired order as a pattern, not a jump fit

For g_D=0, compare ξ_P/L, S_P(0), full B and A low-momentum spectra, and ξ_Q/L over L=64,96,128,256 at common temperatures. Also compare at matched dimensionless charged-order states to avoid confusing a small temperature shift with a large SOC effect. Keep individual histories separate before pooling.

**Paired-regime prediction:** P/B remain algebraic while A and spin correlations become short on the scale of the box. Once those internal scales separate, P's exponent and correlation shape should approach a consistent phase-stiffness description. Persistent charge-one A coherence over accessible sizes or Q scales comparable to L would leave the asymptotic spin-disordered interpretation unestablished. If internal scales do separate but the P/Y mismatch persists, the simple one-phase reduction fails more decisively.

Use the compact κ,P,Q real-space profiles where temperatures were recorded. In a window well beyond lattice effects and well below L/2, compare how the local log-slope changes with distance: algebraic decay tends toward a nearly constant slope, exponential decay adds a contribution growing with r/ξ. Periodic images, logarithmic corrections, and plateaus must be included or excluded by a justified distance window. Vary both the window and L. A finite ξ≫L can mimic algebraic decay; five Fourier modes or a visually straight log plot are insufficient. Do not fit noisy negative correlations after a logarithm.

At those profile temperatures, r=0 supplies repeated local moments ⟨|P|²⟩. Repeat the simultaneous S(0), S(k_min), C_P(0) background test with these measurements, replacing the pilot's four-snapshot estimate. This directly checks whether any proposed short-distance operator correction has enough local weight; a failed contact model still does not exclude a correlated background.

### Priority 3: Check whether P is the relevant charge-two projection

From the recorded complex amplitudes of B=AᵀA, construct the predeclared combinations

\[
b_0=(B_{xx}+B_{yy})/\sqrt2,\quad b_2=P/\sqrt2,\quad
b_{xy}=\sqrt2 B_{xy}.
\]

On the ideal branch b₀=0 and b_xy=iσb₂. Their spectra and cross spectra can be computed from the saved amplitudes without new measurement code.

**Outcomes:** algebraic P with short-ranged b_xy and disordered chirality is compatible with a chirality-disordered paired sector. Coherent b₀ while P loses coherence would falsify P as a sufficient U(1) diagnostic and support a change of orbital structure. Proportional low-k P and B channels, with charged coherence surviving size scaling while local residuals remain large, would support short-ranged internal contamination of a common charged sector. Avoid selecting a noisy covariance eigenvector separately at every temperature and then calling it an order parameter.

### Priority 4: Distinguish SOC alignment from spin fluctuations

Report full and ensemble-connected Q, its mean, and the five component spectra. Under SOC, group components into the axial m=0 channel, xz/yz channels, and xx−yy/xy channels. Explicit alignment supplies

\[
S_Q^{\rm full}(0)=S_Q^{\rm conn}(0)+N\|\langle Q\rangle\|^2.
\]

**Outcomes:** growth of full ξ_Q/L that disappears after a well-sampled ensemble subtraction comes from its disconnected mean, not a divergent connected length. Spin locking accompanied by growing charge-one A coherence would support eventual ordinary-phase behavior at g_D=.01. To assess director locking, inspect orientation-sensitive channels, especially xz/yz, rather than interpreting total connected Q as a director correlation length. Even a perfectly locked d=z gives Q=n(zz−I/3): its connected axial channel can inherit long density/amplitude correlations. Compare the axial channel with density fluctuations and the tilt channels before claiming that the thermal locking scale exceeds the bare benchmark 14.1.

At zero SOC the finite-volume Gibbs mean Q vanishes by spin symmetry. A persistent sampled mean can reflect poor orientation coverage; subtracting it mechanically can remove genuine long-wavelength power. At nonzero SOC the mean can be physical. Recompute subtraction inside each block resample, never separately subtract each configuration's spatial mean, which would erase its zero mode by construction.

### Priority 5: Separate local tensor disorder from an infrared change

Track density-weighted ε, its orbital and nonunitary-spin pieces, and the residual histograms alongside B projections, gradient energy, and the response. Compare the distribution as well as its mean: dilution by low-density sites and genuine rearrangement need not look alike.

**Outcomes:** large but smoothly varying local residuals with stable low-k charged-channel ratios support ultraviolet contamination if charged coherence also survives size scaling. Stable ratios between short-ranged disordered channels would not establish this. A new structural feature synchronized with a change of coherent B projection supports a more substantial reconstruction. Residuals alone distinguish neither domain walls nor HQVs. Kraken's saved local moments do not provide the spatial spectrum of the residual or a coarse-graining test; those require additional configurations in a future authorized study.

### Priority 6: Resolve transition splitting and sampling limitations

Compare κ and the appropriate charged channel over increasing sizes. A thermodynamically distinct phase needs an interval with reproducibly different spatial behavior that survives size drift, not two nearby roots with tiny correlated errors. For a first-order interpretation demand equilibrium switching and consistent size scaling of energy-density separation and barriers. For a crossover interpretation quantify the continuing drift instead of forcing a universal jump.

Before precision claims compare time segments, signed chirality, Q component means, invariant spectra, and production-only walker transport; compare independent starts if supplied in the run manifests. The current local runner has one ordered initialization per case, so independent-start agreement may remain an evidence gap rather than an immediately available test. Stable even moments can coexist with poor sector coverage. Neither high swap acceptance nor a completed round trip establishes topological equilibration.

Two limitations should be explicit in the handoff. First, \(\mathcal K_{xy}\) is absent from the current scalar measurements, so the complete response tensor and \(\sqrt{\det\rho}\) cannot be reconstructed from those records alone. A future small mixed-curvature measurement would address this; **none was added here**. Second, sparse Fourier amplitudes and compact profiles cannot recover qualified contour topology, vortex populations, or thermal string tension. Those require separately authorized configuration/topology or free-energy work.

## G. Unresolved questions and stopping point

The existing data do not determine which spatial objects carry the increasing screening, whether an asymptotic spin-screened paired phase is reached, whether the chirality and phase transitions split, or whether weak SOC produces a measurable thermal confinement tension. The bare Hessian does not determine renormalized masses near the feature. The P/helicity gap remains a meaningful diagnostic, not a unique explanation.

The most economical next step is to apply the tests in section F to the incoming data before designing another simulation. If they reveal separated spin and phase scales, a compact-phase/HQV test becomes better founded. If they reveal a different coherent B projection or a strong orbital-interference feature, the effective theory should change accordingly. If histories remain inconsistent, uncertainty in the thermodynamic mechanism must remain explicit.

Further manipulation of the present scalar records cannot recover missing topology or spatial structure; fitting an extra arbitrary correlation scale would not resolve that absence. The completed calculations therefore stop after ruling out specific reductions and identifying tests that the new measurements can actually perform.
