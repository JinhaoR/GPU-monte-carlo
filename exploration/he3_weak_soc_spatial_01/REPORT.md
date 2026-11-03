# Weak-SOC spatial campaign: results through L=160

Analysis of `data/weak_soc_spatial_simple_01`, using the eight completed runs at g_D=0,.01 and L=64,96,128,160. No new g_D=.001 files were present; L=256 is not included. No simulation, cluster access, production-code change, or Git commit was performed.

## 1. Main findings

**The new observables materially change the interpretation of the chirality/P feature. P is not a sufficient diagnostic of the surviving U(1) coherence.** The strongest evidence is at g_D=.01: above the chirality feature, P and one orbital combination become short ranged, while a different charge-one field and its charge-two composite remain coherent across the available sizes.

The useful orbital basis is

\[
A_+=(A_x+A_y)/\sqrt2,\qquad A_-=(A_x-A_y)/\sqrt2,
\]

with exact identities

\[
P=2A_+\cdot A_-,\qquad B_{++}=A_+\cdot A_+
=\frac{B_{xx}+B_{yy}+2B_{xy}}2.
\]

This basis is a fixed x/y combination, not a fitted covariance eigenvector.

At g_D=.01,T=.3869501, as L increases from 64 to 160:

- ξ_P remains approximately 20–21 lattice spacings, so ξ_P/L decreases .327→.126.
- ξ_A− is approximately 20–21 spacings, while ξ_B++/L stays near .52.
- ξ_A+,z/L increases 1.027→1.090. This is the z-spin component of the plus orbital field, not the sum over all spin components.
- The largest-size effective exponents are η_A+,z=.1295±.0197 and η_B++=.5202±.0363. Their correlated difference η_B++−4η_A+,z=.002±.052 is consistent with a common phase field.
- Helicity remains Y=.6317±.0070. The relevant orbital correlations agree between production halves and are not produced by a trapped nonzero ensemble mean.

**Interpretation:** there is strong finite-size evidence for a chirality-disordered, phase-coherent regime at weak nonzero SOC. Calling it an asymptotic phase, locating its upper transition, or assigning the proliferating vortex species still requires size-flow and sampling checks. The evidence is substantially more informative than the earlier favorable half-charge helicity fit.

Other principal results:

1. **The old scalar results are reproduced.** At matched critical-region temperatures, 168 comparisons covering energy, helicity, chirality, both Binders, heat capacity and spin alignment all lie within 2.82 combined conditional errors. The new result comes from resolving previously unmeasured channels, rather than overturning the old scalar measurements.
2. **The helicity decomposition is confirmed.** At L=160, falling curvature supplies 32.60±.25% of the specified zero-SOC helicity loss and 42.29±.34% at .01; increasing current screening supplies the rest.
3. **The new energy channels resolve that curvature loss further.** Density and increasing gradient energy dominate. The signed orbital-interference term is nonzero but changes in the direction that slightly offsets the curvature decrease; it is not its dominant direct contribution.
4. **SOC-induced Q alignment is decisively separated from spin fluctuations.** At .01 the full Q ratio can grow with L while connected tilt correlations approach finite lengths of approximately 15–25 sites. At zero SOC, spin correlations remain comparable to the box near the ordering feature.
5. **The P/helicity discrepancy survives, but much of it is operator selection.** B++ is substantially closer to the phase-only correlation reference than P. Residual differences remain; the missing mixed helicity curvature and finite-scale corrections prevent an exact scalar-stiffness test.
6. **The old integer-vortex-fit rejection needs qualification.** A broader temperature search finds a .01 q=1 candidate near .3896, beyond the previous reweighting window. The same broad search finds it in the old data too. Its large logarithmic corrections and size-selection sensitivity prevent a precise BKT classification.
7. **Transport improved, but L=160 is not fully certified.** Critical-region drift remains at .01,L=160, and the production interval contains only about one or two sampled loops per walker at the largest size. The more separated regime at .3869501 is substantially more stable.

## 2. Dataset, estimators and integrity checks

### Recorded setup

The manifests, rather than the current local runner, define this analysis:

| Quantity | Recorded value |
|---|---|
| Couplings | 0 and .01 |
| Completed sizes | 64,96,128,160 |
| Pressure, K, reduced-temperature parameter | 20,1,1 |
| Recorded beta coefficients, rounded | −.516512, .942023, .987529, .879063, −1.152515 |
| Thermalization / production | 3M / 3M sweeps |
| Local updates | Gaussian, rho≈-.9, proposal mass 0 |
| PT | 200 temperatures, exchange every 5 sweeps |
| Common ladder | .37–.42; Tc=.384; window=.008; fraction=.25 |
| Scalar / spatial / profile cadence | 50 / 1,000 / 10,000 sweeps |
| Events per run | 60,000 / 3,000 / 300 |
| Initialization | Ordered positive chirality, d=z, phase zero |

These settings differ from later ladder proposals. No inference about those unrun/proposed settings is substituted for the actual recorded campaign. Each file contains the full 16-component Fourier amplitudes at five momenta, 68 local-structure/energy/histogram channels, and component-resolved κ/P/Q profiles at five temperatures.

The local operators follow the existing [measurement definitions](../../models/he3/spatial.py): n=Tr(AA†), κ=2 Im(Ax†Ay), P=Ax·Ax−Ay·Ay, B_ab=A_a·A_b, and Q=Re(AA†)−nI/3. Dots in P and B have **no complex conjugation**. Full B power contracts |Bxx|²+|Byy|²+2|Bxy|²; Q uses an orthonormal five-component basis. Thus A carries phase charge one, P/B charge two, and κ/Q are phase neutral.

Powers are averaged before ratios. The standard diagnostic is

\[
\frac{\xi_f}{L}=\frac{\sqrt{S_f(0)/\overline{S_f(k_{\min})}-1}}
{2L\sin(\pi/L)},\qquad S_f(k)=\langle|F_f(k)|^2\rangle/N.
\]

It is a second-moment finite-box quantity, not automatically an exponential decay length. Ratios below one retain an undefined ξ rather than being clipped to zero. Main errors use 30 joint delete-one blocks of 100k sweeps, with 50k–500k sensitivity checks and production halves. Connected spectra subtract the ensemble mean inside each delete; configuration-by-configuration mean subtraction is not used. Separate-run size-exponent errors add independent jackknife variance contributions rather than pretending blocks in different runs are synchronized. All errors remain conditional on the sampled histories.

### Integrity results

Every saved spatial and profile temperature-to-walker mapping matches the independently stored PT trajectory at its recorded sweep. Sample indices and sweep coordinates agree exactly. Saved spatial/profile energies equal their corresponding scalar measurements exactly.

Maximum discrepancies across all files are:

| Identity | Maximum absolute residual |
|---|---:|
| κ Fourier zero mode versus scalar chirality, per site | 1.28e-9 |
| P zero mode versus scalar complex composite, per site | 1.42e-9 |
| Qzz zero mode versus density/alignment, per site | 1.19e-9 |
| Bulk + gradient + SOC energy versus scalar energy, per site | 1.53e-9 |
| Direct measured energy versus scalar energy, per site | 3.79e-10 |
| Profile means versus matching Fourier zero modes / N | 6.78e-15 |
| Orbital + spin loss versus total residual, per site | 2.85e-16 |
| Histogram site-count sums | Exactly equal |

No nonfinite rejections, excluded-density sites, nonfinite local sites, residuals outside the roundoff allowance, or histogram under/overflow counts were recorded. Histogram density normalization, n_++n_−=n, n_+−n_−=κ, B component contractions, and rotated A/B spectral partitions also pass. The latter agree within 6.4e-12 in the reported structure-factor units.

Because the orbital result is unexpected, the actual local measurement polynomial was independently executed on synthetic CPU fields and compared with the NumPy field definitions and Fourier unpacking. The check verifies that B trace and Bxy are distinct channels, not an indexing alias. Explicit synthetic configurations also verify P=2A+·A− and B++=A+·A+.

## 3. The main new result: the charged orbital sector

### Evidence for different correlation scales

The initial orbital-tensor comparison used

\[
b_0=(B_{xx}+B_{yy})/\sqrt2,\quad b_2=P/\sqrt2,\quad b_{xy}=\sqrt2B_{xy}.
\]

Above the chirality feature, b₀ and bxy have very strong **positive real** cross coherence. At .01,L=160,T=.3869501 their normalized zero-mode cross correlation is approximately .990+.002i. This motivates the fixed diagonal basis, in which B++=(b₀+bxy)/√2 and B−−=(b₀−bxy)/√2. The charge-one amplitudes independently confirm the same direction.

| g_D=.01,T=.3869501 | L=64 | L=96 | L=128 | L=160 |
|---|---:|---:|---:|---:|
| ξ_κ/L | .428 | .272 | .200 | .154 |
| ξ_P/L | .327 | .217 | .163 | .126 |
| ξ_A−/L, all spin components | .308 | .214 | .167 | .131 |
| ξ_A+,z/L | 1.027 | 1.045 | 1.073 | 1.090 |
| ξ_B++/L | .539 | .520 | .515 | .518 |
| S_B++(0) | 112.65 | 195.15 | 296.91 | 413.07 |
| S_P(0) | 65.27 | 56.05 | 52.98 | 49.46 |

P and κ approach finite spatial scales while the coherent charged projections continue to grow. At L=160, 97.6% of the original-field zero-mode power is in the plus orbital combination; 98.7% of the combined B++/B−− zero-mode power is in B++.

![Size dependence of the relevant charged sectors above the chirality feature. The zero-SOC and finite-SOC examples are at different temperatures, each specified in its panel.](results/figures/orbital_evidence.png)

The finite-SOC result survives removing the short-ranged transverse spin contribution from A. This matters quantitatively. For L=128→160 at the stated temperature, the effective exponent from all-spin A+ is .211±.018, whereas its z component gives .1295±.0197. The transverse A+ correlation length is approximately 25 sites; its additive contribution distorts a finite-size exponent formed from the total.

Using \(\eta_{\rm eff}=2-\ln[S_{L_2}(0)/S_{L_1}(0)]/\ln(L_2/L_1)\), the charge consistency test is:

| Size pair | η_A+,z | η_B++ | η_B++−4η_A+,z, joint error |
|---|---:|---:|---:|
| 64→96 | .1305 | .6448 | .123±.060 |
| 96→128 | .1144 | .5414 | .084±.064 |
| 128→160 | .1295 | .5202 | .002±.052 |

The approach toward the charge-squared relation is evidence for a common phase in these operators. It does not prove asymptotic algebraic order from four sizes, but is a much stronger check than comparing P with a chosen jump line.

For .01,L=160 the two production halves give ξ_B++/L=.5175±.0065 and .5182±.0040, and ξ_A+,z/L=1.0893±.0115 and 1.0910±.0085. Disconnected charged-mean fractions are only .024% for A+,z and .029% for B++. This persistence therefore cannot be explained by the run simply retaining its initial global phase.

### A compatible physical explanation

The identity P=2A+·A− is exact in the full model. It immediately permits P to lose its long-distance overlap when A− becomes short ranged, even while A+ remains coherent.

A simple illustrative local form is

\[
A_+=u e^{i\phi}d,\qquad A_-=i h e^{i\phi}d,
\]

where d is real and h is a real chirality-related field. It gives

\[
\kappa=-2uh,\qquad P=2iuh e^{2i\phi}=-i\kappa e^{2i\phi},
\qquad B_{++}=u^2e^{2i\phi}.
\]

Disordering h then disorders P without necessarily disordering φ. This explains how chirality and P can have nearly identical Binder features, while a different charged channel retains phase coherence. The expression is a compatible effective description, **not** a fitted local ansatz imposed on the data or a derived complete thermal action. The measured fields have large departures from the ideal chiral manifold.

This also clarifies an earlier apparent puzzle. On an ideal chiral branch P=n e^{2iθ} is independent of σ when θ is defined through the x orbital column. That phase coordinate need not be the smooth phase that persists across chirality domains. Re-expressing the same field through A+ changes its neutral chirality factor. The claim that P must always diagnose U(1) order in the unrestricted thermal model was too strong.

There is a relevant lattice qualification. The actual forward mixed-gradient stencil does not preserve each separate mirror symmetry, as established in the [previous Hamiltonian investigation](../he3_helicity_mechanism/theory/THEORY.md). It permits diagonal orbital selection and a mixed helicity response. The observed preference for A+ is therefore not automatically a continuum-He3 phase-diagram result. Its microscopic origin and sensitivity to the discretization need a future controlled investigation. No Hamiltonian was changed here.

### Zero SOC is different on these sizes

At g_D=0,T=.3839583, ξ_P approaches approximately 13 sites, but ξ_B++/L decreases .500→.328 and ξ_A+/L decreases .627→.415 from L=64→160. The L=128→160 effective B++ exponent is 1.64±.08, rather than the finite-SOC value .52 at the other stated temperature. This is a strong downward size flow, not comparable evidence for a stable intermediate coherent regime at that zero-SOC temperature.

Closer to the zero-SOC feature, T=.3814856, ξ_B++/L≈.78 and ξ_Q/L≈.61 at L=160. Both remain box-scale quantities. A spin-screened phase-only limit is not established there. The zero-SOC thermodynamic transition and any narrower intervening regime remain unresolved.

![Temperature dependence of P versus B++, plus versus minus original-field correlations, and full Q versus connected spin tilts. Solid and dashed curves distinguish the channels described in the text.](results/figures/sector_flows.png)

## 4. Spin-nematic order, SOC and real-space profiles

### Full Q is dominated by induced alignment under SOC

For an ensemble mean Q, \(S_Q^{\rm full}(0)=S_Q^{\rm conn}(0)+N\|\langle Q\rangle\|^2\). At .01,L=160 the disconnected fraction is .988 at the cold probe, .968 at T=.3852025, and .922 at .3869501. The full ξ_Q/L can consequently grow with size even when the fluctuating spin sector has a finite scale.

The m=1 channels Qxz,Qyz are particularly useful for director tilts. Their connected second-moment lengths are:

| g_D=.01 | L=64 | L=96 | L=128 | L=160 |
|---|---:|---:|---:|---:|
| T=.3750375 | 14.9 | 14.4 | 14.3 | 15.2 |
| T=.3852025 | 16.5 | 16.6 | 18.0 | 17.8 |
| T=.3869501 | 19.1 | 23.2 | 25.8 | 25.4 |

These are compatible with finite thermal tilt scales. The cold value is close to the bare restricted benchmark √(2K/g_D)=14.1, but that numerical agreement is not a measurement of HQV confinement or string tension.

The connected axial m=0 channel can remain more extended near the ordering feature. It must not be confused with a director locking length: even a perfectly locked field has Q=n(zz−I/3), whose axial connected correlations inherit density fluctuations. The report and tables retain all five components, m=0/1/2 partitions, full and connected spectra, and their means.

At zero SOC, full and connected invariant Q are almost identical: the disconnected fraction is about .001 or less at the representative critical probes. The global alignment remains close to zero, and the five component powers are approximately isotropic. Nevertheless ξ_Q/L≈.61 near .38149 means spin correlations are still extensive on the available box. This confirms the earlier pilot concern without interpreting it as a measured spin stiffness or finite-temperature spin long-range order.

![Full and connected spin information. The lower panels show why a dimensionless full-Q ratio cannot be used alone as a director length.](results/figures/spin_alignment_and_fluctuations.png)

### Profiles strengthen the evidence for finite P scales

The profiles independently support the Fourier result. At .01,L=160,T=.3869501, define

\[
\eta_{\rm eff}(r)=-\ln[C_P(2r)/C_P(r)]/\ln2.
\]

It increases from .928±.012 at r=4 to 1.168±.027 at r=8 and 1.544±.060 at r=16. A constant exponent does not describe that interval; a finite correlation scale is consistent with this curvature. At zero SOC,T=.3839583 the corresponding values are 1.144,1.531,2.374. Large-distance estimates become noisy and periodic-image effects matter; they are not used to extract a precise exponential length.

On the colder side, the profile slopes are much smaller and flatten with increasing distance. Finite-volume plateaus and periodic geometry prevent reading a single straight-line exponent directly from them. No noisy negative tail was log-fitted or reset to zero in the numerical results.

![Measured P profiles, averaged over x/y for display. Late-distance wiggles are sampling noise; quantitative errors and the separate directions are retained in the tables.](results/figures/P_profiles.png)

Only κ/P/Q real-space profiles were recorded. There are **no real-space A+,z or B++ profiles** in these files; their evidence here comes from five Fourier modes and size scaling. This limits the ultimate algebraic-versus-large-finite-length distinction for the newly identified coherent channels.

## 5. Helicity, energy and local structure

### Response accounting reproduces the previous result

With \(D=\langle\mathcal K_x+\mathcal K_y\rangle/(2N)\), \(S=[\operatorname{Var}I_x+\operatorname{Var}I_y]/(2TN)\), and Y=D−S:

| L=160 | Temperature interval | Total Y loss | Curvature share | Screening share |
|---|---|---:|---:|---:|
| g_D=0 | .38030047–.38413969 | 1.1383±.0090 | .3260±.0025 | .6740±.0025 |
| g_D=.01 | .38349459–.38739762 | .8099±.0066 | .4229±.0034 | .5771±.0034 |

These essentially reproduce the prior 32.56% and 42.12% curvature shares. The fractions are descriptive window accounting, not independent causal percentages. Density, chirality, gradients and current screening are correlated physical quantities.

The newly available energy components permit the exact identity

\[
D=4K\langle n\rangle-\langle H_g\rangle/(2N)+K\langle W\rangle/N.
\]

Here H_g is the full gradient energy and \(W=\sum_{r,\mu}\operatorname{Re}[(\Delta_d A_{\mu x})^*\Delta_d A_{\mu y}]\), with \(\Delta_d A(r)=A(r+\hat x)-A(r+\hat y)\). It measures a signed derivative overlap, not an opposite-chirality population.

Using matching spatial-event scalar curvatures, the positive curvature losses decompose as:

| L=160 | Total loss of spatial D | Density term | Gradient term | W term |
|---|---:|---:|---:|---:|
| g_D=0 | .370981 | .317917 | .055427 | −.002362 |
| g_D=.01 | .342220 | .291731 | .051528 | −.001039 |

The identity residual is below 2e-15. In this Hamiltonian-based decomposition, density contributes about 85%, gradient roughness about 15%, and the signed interference change slightly offsets the loss. The previous “59% density” result used the exact but different product allocation D=4KnR; it is also reproduced. These allocations answer different accounting questions and are not contradictory causal measurements.

W/N is approximately -.012 near the ordering features and becomes less negative on heating. Its nonzero value rejects a globally fixed orbital chirality as exact kinematics, but its small direct contribution does not explain the main curvature collapse. Orbital changes can still affect the current screening and the other terms indirectly. Reconstructing W from the identity is a derived observable, not an independent configuration-level verification of that identity in this campaign; the previous synthetic/snapshot investigation supplied the latter verification.

### Local chiral quality remains poor

The residual is the relative squared distance to the nearest local ideal chiral field. Define \(v_\sigma=(A_x-i\sigma A_y)/\sqrt2\), \(n_\sigma=v_\sigma^\dagger v_\sigma\), and \(s_\sigma=v_\sigma\cdot v_\sigma\). Then \(\epsilon=1-\max_\sigma(n_\sigma+|s_\sigma|)/(2n)\). On the maximizing branch, the orbital loss is \(1-n_\sigma/n\) and the spin loss is \((n_\sigma-|s_\sigma|)/(2n)\). These chiral projections vσ differ from the **diagonal** fields A± used above. Reported residual means are density weighted, ⟨nε⟩/⟨n⟩; the last column below is an unweighted site fraction.

Representative density-weighted values at L=160 are:

| g_D,T | Total residual ε | Orbital loss | Nonunitary-spin loss | Site fraction ε<.1 |
|---|---:|---:|---:|---:|
| 0,.3750375 | .2886 | .2327 | .0559 | .0650 |
| 0,.3814856 | .3402 | .2737 | .0665 | .0367 |
| 0,.3839583 | .3963 | .3178 | .0785 | .0145 |
| .01,.3852025 | .3479 | .2798 | .0680 | .0331 |
| .01,.3869501 | .3898 | .3129 | .0770 | .0167 |

Orbital deviations dominate, but spin nonunitarity is appreciable. The four histogram families confirm that a small mean is not hiding a predominance of nearly ideal sites. Density exclusions and nonfinite counts are zero, so this is not a division-by-small-amplitude artifact. Local |κ|/n remains appreciable even when global chirality is small; local chirality magnitude is not domain order.

Near the zero-SOC feature the result is consistent in scale with the earlier thermal HQV study's residual ≈.338, with differences in weighting, size and history kept in mind. It is still much larger than the previous bare harmonic benchmark ≈.105. Crucially, large local residuals coexist with coherent charged channels at finite SOC: local projection failure does not imply loss of all phase coherence.

Bulk, gradient, bare dipole and SOC energies were analyzed separately. At .01,L=160,T=.3869501 the per-site values are approximately −.72778, 2.12570, and −.0013275 for bulk, gradient and SOC energy. The small SOC energy changes spin alignment substantially while the mass response remains finite. SOC has no direct derivative under the common phase twist; its effects on Y are through configurations and fluctuations.

![Response components, local-manifold residuals and energy components for L=160.](results/figures/curvature_structure_energy.png)

### Current and directional checks

The mean-current correction is at most .000254 over the new grid, far below the observed helicity fall. Selected current excess kurtoses range −.042 to .072, confirming ordinary-looking broad extensive-current distributions rather than a few singular numerical events. This does not exclude vortices.

The largest x/y helicity discrepancy in the examined critical band is 2.93 paired errors over many correlated comparisons; no systematic diagonal difference is established. Equal diagonals do not establish isotropy. Cross-current covariance is retained, but the mixed curvature Kxy was not recorded, so the full thermodynamic stiffness tensor remains unavailable.

## 6. Correlation/helicity consistency and transition fits

### The old Gaussian discrepancy is confirmed more precisely

The comparison solves the compact scalar torus response for its Gaussian coupling and uses the same winding-sector weights in the charge-two covariance. It introduces no fitted correlation exponent. The paired-period reference is conditional, not a claim that HQVs are present.

| L=160 | Measured ξ_P/L | Measured ξ_B++/L | Scalar compact reference |
|---|---:|---:|---:|
| g_D=0,T=.3814856 | .6999 | .7817 | .8255 |
| g_D=0,T=.3839583 | .0807 | .3280 | .3410 |
| g_D=.01,T=.3852025 | .6448 | .7568 | .8017 |
| g_D=.01,T=.3869501 | .1258 | .5178 | .5803 |

At the last point, the P gap is −.4546±.0048, whereas the B++ gap is −.0625±.0050. Thus choosing the appropriate charged operator removes most of that discrepancy, but does not produce exact scalar-Gaussian agreement. At zero SOC,.3839583 the B++ gap is only −.0130±.0058.

![The same scalar-helicity reference compared with P and B++. This tests operator choice without adjusting a new exponent.](results/figures/operator_and_gaussian.png)

The profile r=0 values now provide 300-event local moments rather than four snapshots. A contact background C_P=A C_G+Bδ chosen to match both S(0) and S(k_min) again fails the local-moment constraint. At zero SOC,L=160,T=.3814856 the required C_P(0) is 55.4 times the measured .3724; at .01,T=.3869501 it is 104 times the measured .2789. A correlated internal contribution is not excluded. More directly, P contains the decorrelating A− field, so a constant overlap with the remaining phase field is not a justified reduction.

The complete scalar relation η=charge² T/(2πY) remains conditional. For an anisotropic phase stiffness, √detρ replaces the scalar coupling and torus geometry also changes. A uniform plus-orbital field already has a bare stiffness tensor with diagonals 4Kn and off-diagonal 2Kn. This illustrates why the missing mixed response matters; it is not a justified substitution for its renormalized thermal value. The successful charge-one/charge-two exponent ratio avoids assuming an isotropic measured Y, but does not by itself determine that tensor.

### Updated finite-size fits

The nominal Weber–Minnhagen comparison uses

\[
Y(L,T)=\frac{2T}{\pi q^2}\left[1+\frac{1}{2\ln L+C}\right],
\]

where q is the putative vortex phase winding in units of 2π, rather than the charge of the measured composite operator. The scalar XY jump and logarithmic size correction motivate this diagnostic; applying it to this multi-component, potentially anisotropic model is an additional assumption, not a result of those references. See [Nelson and Kosterlitz](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.39.1201) and [Weber and Minnhagen](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.37.5986).

The fits use linear interpolation of the new raw temperature grid, joint 100k-sweep errors, and a broad search. They are conditional diagnostics, with interpolation and model-form systematics not included in the quoted statistical errors.

| g_D | Nominal q | Four-size fitted T | Conditional error | χ²/dof | T omitting L=64 |
|---|---:|---:|---:|---:|---:|
| 0 | .5 | .381507 | .000018 | .12 | .381519 |
| 0 | 1 | .383808 | .000028 | 84.5 | .383532 |
| .01 | .5 | .385273 | .000022 | .78 | .385320 |
| .01 | 1 | .389638 | .000037 | 4.50 | .389398 |

The half-charge fit remains favorable near the chirality/P feature, but it still does not identify a vortex charge. The newly resolved coherent channel shows why a fit to Y there is not sufficient to conclude that all phase order disappears there.

The finite-SOC q=1 candidate lies above the old reweighted interval's upper endpoint .389379. Extending the same search to the **old full raw-grid moment caches** gives T=.389665, χ²/dof=5.62. It was already compatible with a broader interpretation of the old data. The new four-size candidate has a 95% leading logarithmic correction at L=64 and about 35% at L=160, and omitting L=64 shifts T by .00024. Consequently .3896 is a useful region to examine, not a precision BKT temperature. Near that region the charge-exponent consistency is also less clean than at .38695.

Chirality/P Binder roots near the main features remain closely paired: approximately .38183 at zero SOC and .3855–.3857 at .01. Extra crossings appear farther into the hot region where the curves approach their disordered limits; all roots are listed rather than silently selecting a physical interpretation. Sub-grid differences between nearly coincident Binder roots should not be taken as independent transition splittings. Heat-capacity grid maxima at L=160 are .381951 and .385515, respectively.

The strongest current distinction is therefore **sector behavior**, not a fitted jump: at .01 the chirality/P sector loses coherence while A+,z and B++ retain it over the available scales. At zero SOC the separation is not established comparably.

### No resolved energy coexistence in the bounded check

At each run's sampled heat-capacity maximum, the production energy distribution has one resolved prominent peak. This holds for all eight cases with 100,150,200 bins and Gaussian smoothing widths .08,.12,.18 of the measured standard deviation. Peaks were required to have 10% prominence and at least .4 standard deviations of separation; these are explicit descriptive resolution choices, not a thermodynamic hypothesis test. The plot also compares the two production halves.

![Energy density distributions at each run's sampled heat-capacity maximum; full production and separate halves.](results/figures/energy_histograms.png)

This corroborates the earlier absence of resolved double-peak evidence at the measured sizes. It does not exclude weak first order: only one grid temperature per case is checked here, no equal-weight coexistence reweighting or interface-barrier scaling is performed, and sampling limitations remain. The local tensor histograms answer a different question and cannot establish energy coexistence.

## 7. Sampling quality and comparison with the earlier work

### PT improved but does not settle equilibration

| g_D | L | Minimum neighbor acceptance | Exact mean completed loop, M sweeps | Production walkers with a sampled loop / 200 |
|---|---:|---:|---:|---:|
| 0 | 64 | .900 | .132 | 200 |
| 0 | 96 | .825 | .374 | 200 |
| 0 | 128 | .743 | .815 | 197 |
| 0 | 160 | .661 | 1.422 | 171 |
| .01 | 64 | .900 | .150 | 200 |
| .01 | 96 | .834 | .380 | 200 |
| .01 | 128 | .765 | .769 | 198 |
| .01 | 160 | .692 | 1.349 | 182 |

Exact loop statistics include thermalization and condition on completed loops. Production-only loops are reconstructed from the 100-sweep history and can miss brief endpoint visits. At L=160 the median sampled completed production-loop count is one at g_D=0 and two at .01; only 184 and 195 walkers, respectively, are seen at both endpoints during production. This is an improvement over the old L=160 completed-loop means of 2.77–3.06M, but not many independent passages per walker.

![Neighbor acceptance and production last-endpoint flow. High acceptance alone does not establish mixing of internal or topological sectors.](results/figures/PT_transport.png)

At .01,L=160,T=.3856721 the two production halves show coherent drift:

| Observable | First half | Second half |
|---|---:|---:|
| Energy/site | 1.349129±.000341 | 1.351403±.000302 |
| Normalized chirality magnitude | .210943±.001650 | .198051±.001773 |
| Spin alignment | .262830±.000641 | .258947±.000506 |
| Y | .91348±.00808 | .90033±.00980 |

The chirality difference is 5.3 diagnostic errors; energy and alignment move coherently. With blocks enlarged from 100k to 500k sweeps, the full-history energy error grows .000308→.000524 and the Y error .00637→.00951. This is a substantive reason to avoid over-precise transition estimates. Signed chirality also has slow-sector differences in other runs.

The separated-sector example at .3869501 is more stable: Y halves are .6354±.0114 and .6279±.0084, and the orbital correlations agree as reported above. The main discovery is not dependent on the most visibly drifting temperature. Nevertheless there is only one ordered-start history per new case, so independent-start agreement at large L is unavailable. Short fixed-temperature autocorrelation estimates can look much faster than walker transport and miss slow tails; they are diagnostics, not equilibration certificates.

### What survives, what changes, and what remains unknown

| Earlier finding | Comparison with this campaign |
|---|---|
| Weak-SOC scalar helicity/chirality/energy curves | Reproduced within the tested conditional errors. See [old_comparison.csv](results/old_comparison.csv). |
| Curvature plus current-screening explanation of the drop | Quantitatively reproduced; energy channels now separate density, gradient and interference contributions. |
| Nearly coincident P and chirality Binder features | Reproduced, but P=2A+·A− supplies a concrete reason they need not identify the surviving phase sector. |
| Better q=.5 leading WM fit | Reproduced as a conditional fit; not upgraded to HQV evidence. |
| Poor q=1 fit | Remains poor at zero SOC. At .01 the previous temperature-window restriction missed a higher-temperature candidate already present in the old full-grid data. |
| L=64 P/helicity discrepancy | Confirmed with much more sampling and larger sizes; appropriate orbital projections explain much of it. |
| Box-scale zero-SOC spin correlations | Confirmed near the feature. A mass-only spin-screened assumption remains unestablished there. |
| Strong SOC effect on alignment, smaller effect on mass response | Confirmed and clarified by full versus connected Q and finite tilt lengths. |
| Thermal local chiral-quality failure | Confirmed by residual components and histograms; it does not imply absence of charged coherence. |
| No resolved first-order coexistence | Consistent with all eight new heat-peak energy histograms; weak first order remains unexcluded. |
| Synthetic HQV diagnostics and constrained zero-T pairs | Not contradicted, but these new Fourier/profile records contain no qualified contour topology or pair free energy. |
| Weak-SOC HQV confinement | Still not measured. Spin locking is supporting context, not a thermal string-tension determination. |
| Intermediate coupling .001 | No new spatial files; its older scalar and constrained-pair conclusions cannot be extended with this campaign. |
| Strong-SOC .1 anomaly | Not retested; no .1 data are present here. |

The existing global curve comparison is shown below. Old and new error conventions and interpolation methods differ, and any correlations from reused random-stream seeds across campaigns are not measured; the combined-error comparisons are diagnostic rather than strict independence proofs.

![New scalar curves and the old L=160 reweighted results. The 8T/pi and 2T/pi lines are references, not observed jumps.](results/figures/global_comparison.png)

## 8. What L=256 can decide, and the current limits

The first check should be the finite-SOC separated-sector point near T=.3869501, using its exact saved temperature index. If the current interpretation persists:

- P, κ, A− and connected spin-tilt lengths should remain of order 20–25 sites, so their ratios to L should decrease further, roughly toward .08–.10.
- ξ_B++/L should remain of order .5, and ξ_A+,z/L of order one, rather than both collapsing as 1/L. These are qualitative continuation predictions, not fitted confidence intervals.
- S_B++(0) and S_A+,z(0) should continue growing, with effective exponents approaching a consistent charge-squared relation. Recompute the correlated exponent difference rather than combining errors as if the two channels were independent.
- The hotter region around .3894–.3898 should be examined separately for the loss of the **plus** charged sector. Its finite-size flow, rather than a renewed P fit, should assess the higher-temperature candidate.

At zero SOC, determine whether the plus-sector flow separates from chirality on the larger box or continues downward without a stable intermediate interval. Also check whether spin scales finally become small compared with L. Do not assume the finite-SOC phase pattern persists at g_D=0.

Before using L=256 to settle small temperature differences, inspect its production halves, block-size dependence, and walker endpoint coverage. Improved acceptance at smaller L does not guarantee adequate transport at 256. The L=160 drift provides a concrete diagnostic to repeat.

Two future measurement additions would now have a specific physical purpose: real-space profiles of A+,z and B++, and the mixed twist curvature Kxy. They would test the decay shape of the actual coherent channels and complete the helicity tensor. They are recommendations only; no measurements, runners or sampling settings were changed. Thermal HQV populations and confinement still require separately qualified topological/configuration or free-energy measurements.

The measured sector separation does not determine whether domain walls, vortices or other textures dominate the current variance. The bounded energy-histogram check likewise does not settle the order or universality of either feature.

## 9. Reproducibility and output map

All new scripts and outputs are in this directory. The scripts are ordinary analysis functions, with no new simulation engine or runner infrastructure. To reproduce from the repository root:

```text
python -B exploration/he3_weak_soc_spatial_01/reduce.py
python -B exploration/he3_weak_soc_spatial_01/analyze.py
python -B exploration/he3_weak_soc_spatial_01/orbital_channels.py
python -B exploration/he3_weak_soc_spatial_01/findings.py
python -B exploration/he3_weak_soc_spatial_01/robustness.py
python -B exploration/he3_weak_soc_spatial_01/figures.py
python -B exploration/he3_weak_soc_spatial_01/energy_histograms.py
```

The reduction reads the eight completed archives one run at a time and saves compact 50k-sweep moment blocks. Its size selection remains fixed at 64,96,128,160, so a subsequently added L=256 file is left unread. Approximately 0.4 GiB of derived blocks/tables/figures are retained; no configurations were generated or duplicated.

| Output | Contents |
|---|---|
| [inventory.csv](results/inventory.csv), per-run JSON | Actual settings, provenance CRCs, mapping and identity checks, PT transport |
| [curves.csv](results/curves.csv) | All scalar responses, structure ratios, energy components, covariance derivatives and errors |
| [spatial.csv](results/spatial.csv), summary NPZ files | Full/connected spectra, directional ratios, all Q/A components, B projections and cross spectra |
| [profiles.csv](results/profiles.csv), [profile_slopes.csv](results/profile_slopes.csv) | Full/connected real/imaginary profiles, errors and finite-window slope tests |
| [histograms.csv](results/histograms.csv) | All four local-residual histogram families with errors |
| [energy_histograms.csv](results/energy_histograms.csv), [energy_modes.csv](results/energy_modes.csv) | Heat-peak energy distributions and bin/smoothing sensitivity |
| [orbital_channels.csv](results/orbital_channels.csv) | Plus/minus and spin-resolved charge-one channels; B++/B−− |
| [charge_consistency.csv](results/charge_consistency.csv), [size_flows.csv](results/size_flows.csv) | Effective size exponents and correlated charge-squared test |
| [curvature_accounting.csv](results/curvature_accounting.csv) | Old-window accounting and new exact energy decomposition |
| [gaussian_checks.csv](results/gaussian_checks.csv), [B_gaussian_checks.csv](results/B_gaussian_checks.csv) | Parameter-free reference comparisons and local-moment tests |
| [stability.csv](results/stability.csv), [orbital_stability.csv](results/orbital_stability.csv) | Block-size and production-half sensitivity |
| [wm_fits.csv](results/wm_fits.csv), [old_extended_wm.csv](results/old_extended_wm.csv), [binder_crossings.csv](results/binder_crossings.csv) | Conditional fits, broad old-data control and all candidate roots |
| [interpretation_validation.json](results/interpretation_validation.json) | Independent measurement polynomial, Fourier and synthetic orbital-identity checks |
| [summary_identity_checks.json](results/summary_identity_checks.json), [structure_quality_checks.json](results/structure_quality_checks.json) | Spectral partitions, residual range and histogram-tail checks |

The [preservation audit](preservation_after.json) confirms unchanged hashes for 115 pre-existing Python sources, unchanged size/mtime for all eight input archives, an unchanged Git branch/HEAD, and no tracked or staged diff. All seven new analysis scripts pass syntax parsing, all report links resolve, and the executed numerical checks pass. The input size/mtime check is not a new whole-file cryptographic hash; per-member archive CRCs are retained in the individual provenance records.

The prior [helicity-mechanism report](../he3_helicity_mechanism/REPORT.md), [SOC survey](../he3_soc/REPORT.md), [transition study](../he3_transition_study/REPORT.md), [spatial pilot](../he3_spatial_pilot/REPORT.md), and [HQV thermal](../he3_hqv_stage2/REPORT.md)/[relaxation](../he3_hqv_stage3/REPORT.md) reports supply the historical comparisons. The strongest new conclusion is that the chirality/P feature and the survival of charged coherence must now be analyzed through different orbital channels.
