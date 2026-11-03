# Full-field onsite landscape from g_D=.01 to .1

**The two competing onsite minima already exist at weak coupling. No new minimum or loss of their local stability is found in this interval.** SOC steadily reduces their energy difference, stiffens their soft spin modes, and increases the accessible escape barrier out of the higher-energy planar minimum. These are deterministic statements about the implemented polynomial, not finite-temperature free energies or a phase boundary.

All calculations are in this directory. No Monte Carlo, GPU work, or production edits were performed by this workstream.

## Implemented potential and validation

The full complex 3×2 field has twelve real coordinates, ordered as Re(A).ravel(), Im(A).ravel(). The pressure-20 coefficients are

\[
(-.5165123343467712,.9420228004455566,.9875290393829346,
.8790628910064697,-1.152514934539795).
\]

The potential is exactly the production \(V=-2n+\sum_i\beta_i I_i+g_DD\), using \(c_0=\mathrm{float32}(2/3)=.6666666865348816\) in the dipole invariant. Each requested g_D is also converted to its production float32 value. The quartic is represented as a sum of squared real quadratic forms; this supplies analytic full 12-dimensional gradients and Hessians without projecting onto an ansatz.

The actual `potential`, `dipole_invariant`, and `norm2` function bodies from `models/he3/update.py` were executed on CPU with their decorators removed and compared on 30 random fields at three couplings. The independent Stage-2 full periodic energy evaluator supplies an additional check. Maximum energy discrepancy is 8.9e-15; Richardson finite-difference checks of the analytic gradient and Hessian differ by 9.7e-11 and 1.7e-10. Details are in [validation.json](validation.json).

## Three structures that should not be conflated

Let \(\beta_A=\beta_2+\beta_4+\beta_5=.6685707569122314\) and \(\beta_P=\beta_1+\beta_2+(\beta_3+\beta_4+\beta_5)/2=.7825489640235901\).

The spin-z chiral minimum has \(A=\sqrt{n_A/2}\,\hat z(1,i)\), with

\[
n_A=\frac{1+g_Dc_0/2}{\beta_A},\qquad
V_A=-\frac{(1+g_Dc_0/2)^2}{\beta_A}.
\]

The real antisymmetric planar minimum has \(A_{xy}=-A_{yx}=\sqrt{n_P/2}\), other entries zero, with

\[
n_P=\frac{1+g_D(1+c_0)/2}{\beta_P},\qquad
V_P=-\frac{[1+g_D(1+c_0)/2]^2}{\beta_P}.
\]

The pure diagonal-orbital polar field has \(A=\sqrt{n_Z/2}\,\hat z(1,1)\), where \(n_Z=(1+g_Dc_0/2)/\sum_i\beta_i\). It is stationary, but has **three negative Hessian eigenvalues at every tested coupling**, with the most negative changing −3.318→−3.417. It is not an additional onsite minimum.

| Uniform structure | κ/n | Spin alignment | Local chiral residual ε | A+ power fraction | P | B++ | B−− |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chiral z | 1 | 1 | 0 | .5 | n | in magnitude n/2 | in magnitude n/2 |
| Antisymmetric planar | 0 | −.5 | .75 | .5 | 0 | n/2 | n/2 |
| Polar plus z | 0 | 1 | .5 | 1 | 0 | n | 0 |

Here \(A_\pm=(A_x\pm A_y)/\sqrt2\) and \(B_{\pm\pm}=A_\pm\cdot A_\pm\); global phases are convention dependent. For the stated chiral representative B++=+in/2 and B−−=−in/2, whereas the planar representative has equal real B++ and B−−. These signatures distinguish antisymmetric planar order from the single-plus-channel coherence found in the weak-SOC thermal data. A long-range coherent A+,z component does **not** imply that each thermal site is a homogeneous polar onsite minimum.

## Energies and local stability

| g_D | V_A | V_P | V_P−V_A | Softest positive chiral Hessian mode | Softest positive planar mode |
|---:|---:|---:|---:|---:|---:|
| .01 | −1.505716 | −1.299262 | .206454 | .019894 | .020000 |
| .02 | −1.515738 | −1.320826 | .194911 | .039579 | .040000 |
| .04 | −1.535880 | −1.364487 | .171393 | .078326 | .080000 |
| .06 | −1.556155 | −1.408858 | .147298 | .116258 | .120000 |
| .08 | −1.576564 | −1.453938 | .122626 | .153392 | .160000 |
| .10 | −1.597105 | −1.499729 | .097376 | .189744 | .200000 |

Both minima have exactly one gauge zero mode at every nonzero coupling. The antisymmetric planar field is already an onsite local minimum at zero SOC: no negative modes, four symmetry zero modes, and first positive eigenvalue 3.650005. Thus appearance of a planar minimum near .1 cannot explain a sudden onset of trapping. The exact nonnegative gradient energy also cannot linearly destabilize either homogeneous onsite minimum.

Fifteen starts at each coupling—three exact stationary states, six perturbed representatives and six unrestricted random fields—find only the chiral and planar minima. Starting exactly at the polar saddle leaves a gradient optimizer there, correctly exposed by its negative Hessian modes; it is not counted as a minimum. No energy below V_A was found. This modest search is not a global-minimum proof or an estimate of basin volumes.

The two analytic energy expressions would become equal at g_D=.17323185, outside the requested range. That extrapolated equality is not a finite-temperature transition and does not exclude other structures outside the examined interval.

![Onsite energies, Hessian stability and independently verified connecting paths. The arclength is a path coordinate, not a dynamical time or thermodynamic reaction coordinate.](landscape.png)

## Full-space paths and accessible saddles

Three 41-image strings were initialized for each coupling: a direct interpolation, a generic 12-coordinate perturbation, and a perturbed path with a different endpoint phase. Their role is to locate saddle candidates. **The strings reach their 12,000-iteration cap and are not claimed as converged minimum-energy paths.** Their approximate profiles remain in [paths.csv](paths.csv) for inspection.

The highest string images seed a stationary solver using the exact gradient and Hessian. All three preparations locate equal-energy saddles to approximately 1e-15. Each has one negative Hessian mode and two symmetry zero modes, with full gradient norm at most 1.6e-15. The saddle is an in-plane, partially chiral matrix, rather than a forced chiral-plus-planar two-amplitude interpolation. Its chirality rises .523→.591 across the interval.

Connectivity was checked independently: displace to either side of the negative mode, add a tiny generic perturbation, and minimize all twelve coordinates. One branch reaches the chiral-z minimum and the other the planar minimum in all 18 cases. The generic perturbation matters: an exactly in-plane trajectory can terminate at a spin-unstable stationary point because it lies in an invariant subspace.

The complete optimization trajectories form continuous piecewise-linear paths. On **every** segment the exact quartic energy polynomial was maximized, including all interior stationary points, so a coarse image spacing cannot hide an overshoot. The maximum connecting-path energy exceeds the refined saddle by at most 8.9e-16. These verified paths, rather than the unconverged strings, are plotted and saved in [connected_paths.csv](connected_paths.csv).

| g_D | Accessible saddle energy | Barrier above chiral minimum | Barrier above planar minimum |
|---:|---:|---:|---:|
| .01 | −1.090906 | .414810 | .208356 |
| .02 | −1.103427 | .412311 | .217399 |
| .04 | −1.128288 | .407592 | .236199 |
| .06 | −1.152889 | .403266 | .255968 |
| .08 | −1.177203 | .399360 | .276735 |
| .10 | −1.201203 | .395902 | .298526 |

These are constructive **upper bounds on the smallest onsite connecting barrier**, supported by index-one saddles and verified basin connectivity. A lower unsearched path is not excluded. SOC weakly lowers the barrier measured from the chiral minimum and raises it measured from the planar minimum as the planar energy becomes more competitive. No abrupt onsite change is seen. These numbers are not spatial nucleation barriers, interface tensions, or finite-temperature free-energy barriers.

## Why a free onsite path is not a local-update conversion path

For one changed site A in a uniform, fixed environment C, exact expansion of the implemented forward stencil gives

\[
\Delta H_g=K\left[8\|A-C\|^2+
4\operatorname{Re}((A_x-C_x)^\dagger(A_y-C_y))\right].
\]

This was checked against the full periodic 5×5 energy evaluator, with discrepancies below 8e-15. For a literal replacement between the chiral-z and antisymmetric-planar minima the cross term vanishes, giving \(8K(n_A+n_P)\): 22.3139 at .01 and 23.4396 at .1. Including onsite energy, a chiral→planar single-site replacement costs 22.5204 and 23.5370, respectively. Both are many times the MC temperature near .4.

In both fixed environments, minimizing the conditional full 12-coordinate single-site energy from the opposite structure returns to the environment's structure. Thus the isolated onsite competition is strongly opposed by neighbors; collective paths or nuclei matter. This abrupt replacement is a controlled illustration, **not** the minimum spatial activation barrier, nor a measured Gaussian acceptance rate. Gaussian proposals incorporate the quadratic environment in their proposal density, so detailed-balance acceptance must include that proposal law rather than use an unqualified exp(−ΔH/T) rule.

The results support growing structural competition and persistent local-update difficulty, while showing why free onsite barriers alone cannot explain the large-system conversion times. The old trapped populations could still differ from these ideal minima; actual spatial and tensor evidence is required to identify them.

## Conditional harmonic entropy baseline

A separate bounded calculation asks whether Gaussian fluctuations even favor the planar structure. It uses each full onsite Hessian plus the exact Fourier Hessian of the implemented gradient stencil. For \(p_\mu=e^{ik_\mu}-1\), the orbital kernel is

\[
M(k)=\begin{pmatrix}
3|p_x|^2+|p_y|^2 & 2\operatorname{Re}(p_x^*p_y)\\
2\operatorname{Re}(p_x^*p_y)&|p_x|^2+3|p_y|^2
\end{pmatrix},\qquad
H_s(k)=H_s^{\rm onsite}+2K[I_6\otimes M(k)].
\]

The Kronecker product acts in the stated Cartesian ordering: three orbital pairs for the real components and three for the imaginary components. A full-lattice finite-field perturbation check against the independent implemented-energy evaluator agrees to 2.2e-9 after Richardson cancellation of quartic terms. All nonzero modes are positive. The one common global phase zero mode at k=0 is omitted from each determinant in the same orthonormal-coordinate convention.

The Gaussian comparison is

\[
\Delta f_{\rm harm}(T)=V_P-V_A+
\frac{T}{2N}\sum_k[\log\det{}'H_P(k)-\log\det{}'H_A(k)].
\]

The full Brillouin-zone sum correctly counts the real fluctuation degrees of freedom with the prefactor 1/2. Equal omission of the global phase leaves an O(1/N) phase-volume convention; that small correction is not a substitute for controlling anharmonic errors.

| g_D | Gaussian coefficient Δf/T after subtracting ΔV, L=128 | Formal crossing Δf=0 |
|---:|---:|---:|
| .01 | −.589205 | .350394 |
| .02 | −.571168 | .341251 |
| .04 | −.536740 | .319322 |
| .06 | −.503738 | .292410 |
| .08 | −.471797 | .259912 |
| .10 | −.440736 | .220940 |

The planar minimum has the larger Gaussian entropy in every case. Its entropy advantage actually **decreases** with increasing g_D; the more rapid decrease of the bare energy gap drives the downward formal crossing. L=32,64,128 sums are numerically stable, with L=64→128 crossing changes below 1.1e-6. These small discretization differences say nothing about the physical approximation error.

This is useful qualitative support for an entropic route to a planar thermal population, but a poor quantitative phase-diagram prediction. In particular it already predicts planar preference at g_D=.01,T=.375, where the weak-SOC results instead display the chiral/plus-z ordering pattern. It therefore cannot locate the actual structural crossover simply by extrapolating the Gaussian formula to the transition region.

The expansion-size check makes the limitation concrete. At T=.38,L=128, the covariance in directions orthogonal to the local gauge mode has mean squared fluctuation divided by n of .18–.21 for chiral and .23–.31 for planar across the six couplings. These are appreciable fluctuations, not a parametrically small expansion. Cubic/quartic feedback, density renormalization, chirality domains and topological textures are omitted. Individual phase variances also grow with box size: at .01 they increase from .0411→.0551 for chiral and .0479→.0642 for planar between L=32 and 128. The free-energy-density logdet difference has an infrared limit, but the fixed-phase Cartesian expansion is not a claim of true two-dimensional long-range order.

![Formal Gaussian crossings and fluctuation-size checks. The plotted temperatures are approximation diagnostics, not inferred thermodynamic transition temperatures.](harmonic_baseline.png)

The calculation took approximately 2.4 seconds of CPU wall time and ran no sampling. [harmonic_free_energy.csv](harmonic_free_energy.csv), [harmonic_variances.csv](harmonic_variances.csv), and [harmonic_validation.json](harmonic_validation.json) retain the finite-size and variance controls. Its strongest result is the **sign** of the conditional Gaussian entropy advantage and the failure of its naive extrapolation, rather than a claimed thermal phase boundary.

## Reproduction and outputs

Run from the repository root:

```text
python -B exploration/he3_intermediate_soc/landscape/onsite.py
python -B exploration/he3_intermediate_soc/landscape/harmonic.py
```

The script uses NumPy, SciPy and Matplotlib, and reuses the established energy evaluator and local chiral-distance function. It writes only this directory. The stationary states and all 12 Hessian eigenvalues are in [stationary_states.csv](stationary_states.csv) and [hessians.csv](hessians.csv); multistarts, saddles and their endpoint connections are separately retained. [fields.npz](fields.npz) holds the small 12-coordinate image/descent paths. [fixed_environment.csv](fixed_environment.csv) documents the one-site controls. No thermodynamic statistical error is attached to these deterministic energies; the quoted tolerances concern numerical arithmetic and stationarity only.
