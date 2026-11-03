# Phase rigidity of the implemented full 3×2 GL model

This note derives static, long-wavelength and lattice statements from the actual Hamiltonian. It extends, rather than repeats, the topology and restricted string calculation in [Stage 1](../../he3_hqv_stage1/THEORY.md), the thermal qualification failure and trial pair benchmark in [Stage 2](../../he3_hqv_stage2/REPORT.md), the constrained full-field relaxation in [Stage 3](../../he3_hqv_stage3/REPORT.md), and the previous [model audit](../../he3_soc/model_audit.md). The supplied thesis was also inspected, particularly its reduced London discretization, spin-frozen chiral basis, full-field gradient, and twist derivation. Its attachment is `C:/Users/jinhao/.codex/attachments/a14e2c10-32c2-4dbb-b6ef-2e4a9aab6782/Pasted text.txt`.

The new results are an exact both-chirality gradient representation; explicit identification of the twelve quadratic modes; a small, source-checked harmonic covariance calculation; a full-tensor curvature identity; the mixed twist curvature; and examples showing why domain walls and tensor reconstruction are not equivalent to deleting phase bonds. None of these establishes the thermodynamic defect mechanism.

## 1. Exact lattice gradient and the two chiral sectors

Let the orbital columns be complex spin vectors \(x=A_x,y=A_y\), and define

\[
u=(x-iy)/\sqrt2,\qquad v=(x+iy)/\sqrt2,
\quad x=(u+v)/\sqrt2,\ y=i(u-v)/\sqrt2.
\]

The positive chiral branch has \(v=0\). With actual forward differences \(D_x,D_y\), the full gradient density summed over spin is exactly

\[
\begin{aligned}
G={}&2\sum_{a=x,y}(\|D_au\|^2+\|D_av\|^2)\\
&+2\operatorname{Re}\big[(D_xu)^\dagger D_xv-(D_yu)^\dagger D_yv
-i\{(D_xu)^\dagger D_yv+(D_yu)^\dagger D_xv\}\big].
\end{aligned}
\tag{1}
\]

No integration by parts or continuum limit was used. The cancellation at \(v=0\) is exact even for a complex, nonunitary spin vector \(u\); it does not require a real director.

For a periodic Fourier mode, write \(p_a=e^{ik_a}-1\), \(s=|p_x|^2+|p_y|^2\). The orbital matrix in \((x,y)\) is

\[
M(k)=\begin{pmatrix}
3|p_x|^2+|p_y|^2&2\operatorname{Re}(p_x^*p_y)\\
2\operatorname{Re}(p_x^*p_y)&|p_x|^2+3|p_y|^2
\end{pmatrix},\quad H_g=K\sum_k A(k)^\dagger M(k)A(k).
\tag{2}
\]

In the chiral basis its diagonal entries are \(2s\), with mixing

\[
c(k)=|p_x|^2-|p_y|^2-2i\operatorname{Re}(p_x^*p_y).
\tag{3}
\]

At long wavelength \(s=k^2+O(k^4)\), \(c=(k_x-ik_y)^2+O(k^4)\). Thus both chiralities are coupled by the gradient, but the coupling begins at two spatial derivatives. In the continuum the orbital eigenvalues are \(k^2\) and \(3k^2\), not six independent isotropic copies of an XY gradient.

There is a relevant lattice distinction. Simultaneously exchanging x and y coordinates and orbital columns is an exact symmetry of this gradient. A separate mirror is not:

\[
M_{xy}(-k_x,k_y)+M_{xy}(k_x,k_y)
=4(1-\cos k_x)(1-\cos k_y).
\tag{4}
\]

The right side is fourth order at small momentum but nonzero on the lattice. The continuum equivalence of the second and third gradient contractions after integration by parts cannot be silently imposed on each finite-spacing forward-difference contraction. In particular, equality of measured \(\Upsilon_x\) and \(\Upsilon_y\) does not establish a vanishing off-diagonal response. Such an off-diagonal response can represent a lattice anisotropy; it is not automatically an estimator error or a defect signature.

## 2. Phase, director, and nonlinear coupling

At fixed chirality, put

\[
u=\sqrt n e^{i\theta}d,\quad d\in S^2\text{ real}.
\]

In a smooth patch the continuum energy is exactly

\[
H_g=2K\int[(\nabla\sqrt n)^2+n(\nabla\theta)^2+n|\nabla d|^2],d^2r.
\tag{5}
\]

There is no quadratic phase–spin drag in this real-director metric: \(d\cdot\partial_a d=0\). At fixed n the bare mass and director stiffnesses, in the convention \(F=\frac12\int\rho(\nabla\theta)^2\), are both \(4Kn\). This is an equality of bare coefficients on the ideal branch, not a constraint on their renormalized thermal values.

On the lattice, one positive-direction bond of the same fixed-chirality field has

\[
E_{ij}=2K\|u_j-u_i\|^2
=4Kn[1-(d_i\cdot d_j)\cos(\theta_j-\theta_i)]
\tag{6}
\]

for equal densities. The smooth quadratic expansion decouples phase and director, but the full bond does not. For a small spin angle \(\alpha\), the cross term is \(-Kn\alpha^2(\delta\theta)^2\). Thermal spin gradients can therefore renormalize phase rigidity without a bilinear coupling and without unbound vortices. Conversely phase fluctuations change the cost of spin gradients. A continuum theory obtained by keeping only (5) omits this finite-spacing, higher-gradient interaction and additional interactions induced by massive fields.

There is also a precise distinction between a real director and a generic complex spin vector. At \(v=0\), write \(u=\sqrt n e^{i\theta}\zeta\), \(\zeta^\dagger\zeta=1\), and \(a_a=-i\zeta^\dagger\partial_a\zeta\). Then

\[
|\partial_a u|^2=(\partial_a\sqrt n)^2+n[(\partial_a\theta+a_a)^2+
\|\partial_a\zeta\|^2-a_a^2].
\tag{7}
\]

The decomposition has a phase convention redundancy. In a local gauge near a real director, \(\zeta\simeq d+i b\), \(d\cdot b=0\), one obtains \(a_a\simeq-2b\cdot\partial_a d\). Complex-spin fluctuations can couple phase flow and director gradients through this connection. Because b is massive on the stable chiral branch, this does not add a new harmonic massless mode; eliminating b generates nonlinear derivative interactions. Large b or large gradients invalidate the small-departure expansion.

## 3. All twelve homogeneous quadratic modes

At zero SOC, the homogeneous branch is \(u_0=\sqrt{n_0}\hat z,v_0=0\), where

\[
\beta_A=\beta_2+\beta_4+\beta_5=0.6685707569,\quad n_0=1/\beta_A=1.4957279984.
\]

Use orthonormal real coordinates \((\operatorname{Re}u,\operatorname{Im}u,
\operatorname{Re}v,\operatorname{Im}v)\), so \(\delta V=\frac12\sum_j\lambda_jq_j^2\). Direct expansion of the actual quartic invariants gives:

| Real fluctuation coordinates | Multiplicity | Onsite Hessian eigenvalue | Numerical value |
|---|---:|---:|---:|
| \(\operatorname{Im}u_z\), phase | 1 | 0 | 0 |
| \(\operatorname{Re}u_x,\operatorname{Re}u_y\), director | 2 | 0 | 0 |
| \(\operatorname{Re}u_z\), amplitude | 1 | \(8\) | 8 |
| \(\operatorname{Im}u_x,\operatorname{Im}u_y\), nonunitary spin | 2 | \(-8n_0\beta_5\) | 13.790791 |
| \(\operatorname{Re}v_z,\operatorname{Im}v_z\), opposite-chirality orbital distortion | 2 | \(8n_0(\beta_1+\beta_3)\) | 5.636103 |
| \(\operatorname{Re}v_{x,y},\operatorname{Im}v_{x,y}\) | 4 | \(4n_0(\beta_3-\beta_4-\beta_5)\) | 7.544339 |

These are static Hessian masses in the chosen dimensionless classical Hamiltonian, not frequencies of dynamical collective modes. Chirality is discrete; infinitesimal opposite-chirality fluctuations are massive, rather than an additional chirality Goldstone mode. A relative phase of u and v is not an additional Goldstone angle when the equilibrium v amplitude vanishes. There are three soft coordinates and nine massive coordinates. The eight massive directions normal to the chiral manifold exclude the radial amplitude direction.

At positive SOC the chiral-z branch remains a stationary point with \(n=(1+g_D c_0/2)/\beta_A\), \(c_0=\mathrm{float32}(2/3)\). Only the common phase remains exactly soft. The two director eigenvalues at \(g_D=.001,.01,.1\) are approximately \(.0019989,.0198943,.189744\), respectively; the leading mass is \(2g_D\), with a small reduction from mixing with massive tensor modes. Thus the restricted dipole length is a leading weak-SOC result, not an exact eigenmode formula at arbitrary coupling. The previous local-potential calculation already established the positive spectra; the new calculation identifies the modes and their gradient mixing.

The finite-momentum real Hessian is obtained without a continuum approximation by adding \(2K M(k)\) in each real/imaginary spin-row sector and transforming to the chiral basis. Eliminating the massive modes at zero SOC gives an especially simple exact quadratic kernel for \(p=\operatorname{Im}u_z\):

\[
H^{\rm eff}_{pp}(k)=4Ks-\frac{4K^2|c(k)|^2}{\lambda_{v_z}+4Ks}.
\tag{8}
\]

The phase coordinate is \(p=\sqrt{n_0}\theta\). Equation (8) has leading term \(4Kn_0 k^2\) in the phase kernel; the correction starts at \(-4K^2n_0 k^4/\lambda_{v_z}\). No harmonic integration of these stable massive fields removes the phase stiffness or creates an extra massless counterflow on this branch. They soften shorter-wavelength distortions. Amplitude adjustment to a uniform infinitesimal phase gradient starts at order \(q^2\), so its energy improvement starts at order \(q^4\), also leaving the zero-temperature uniform helicity unchanged.

This is a useful negative result: a proposed abrupt thermal helicity collapse cannot be explained by a new zero eigenvalue of this unchanged bare homogeneous Hessian. Renormalized thermal masses, strongly nonuniform configurations, alternate minima, and topological screening remain distinct possibilities.

## 4. A bounded Gaussian test of microscopic projection loss

[harmonic_checks.py](harmonic_checks.py) evaluates the exact 12×12 lattice Hessian at each momentum on L=32,64,128. It performs no sampling. The covariance is

\[
C_{ab}=\frac{T}{N}\sum_k[H(k)^+]_{ab},
\tag{9}
\]

where only the three global zero modes are omitted. To leading order about the homogeneous reference, the Stage-1 squared nearest-manifold residual is

\[
\epsilon=\frac{\|v\|^2+|\operatorname{Im}u_x|^2+|\operatorname{Im}u_y|^2}{n_0}+O(q^3).
\tag{10}
\]

The finite-grid results are in [harmonic_covariance.csv](harmonic_covariance.csv). At L=64:

| T | Harmonic mean epsilon | Opposite-chirality part | Nonunitary-spin part | Stage-2 raw mean epsilon |
|---:|---:|---:|---:|---:|
| .3500 | .09668 | .07867 | .01802 | .2305 |
| .3815 | .10538 | .08575 | .01964 | .3379 |

The harmonic result is effectively independent of L over the tested sizes. Meanwhile the phase-angle variance at T=.3815 increases from .04139 to .05547 as L grows from 32 to 128, and the sum of the two director-angle variances increases from .08194 to .11010. The normal massive fluctuations and soft angular fluctuations have different scale dependence.

The calculation establishes that nonzero microscopic manifold residual is compatible with a phase-carrying long-wavelength sector, even before defects are introduced. It also rejects the stronger quantitative claim that ordinary Gaussian fluctuations around the *bare* chiral minimum explain the observed near-transition residual: the measured mean is about 3.2 times the benchmark at .3815. The comparison deliberately uses means on both sides; the measured median .3360 is a different statistic. The expansion is not self-consistent at the observed transition, uses the bare n0 rather than the thermal density, and ignores nonlinear fluctuations, alternative local structures and defects. Its discrepancy is a failure of that quantitative bare approximation, not a count or identification of vortices.

A useful intermediate-scale prediction is available without a new simulation. If the normal modes remain short ranged, smoothing the field over a scale b well above their correlation length but below the spin/phase variation scales suppresses their variance approximately as \(b^{-2}\) in two dimensions. A large microscopic epsilon followed by this suppression is compatible with ultraviolet tensor fluctuations. Persistent excess or low-momentum growth in these channels would indicate a failure of that simple massive-mode picture. Smoothing can change topology, and this prediction should not be turned into a thermal HQV detector without independent calibration.

Locally near this branch one can write \(u=e^{i\theta}[(\sqrt n+s)d+i b]\), \(d\cdot b=0\), and \(v=e^{i\theta}\tilde v\). The charge-two polynomial obeys

\[
P=u\cdot u+v\cdot v
=e^{2i\theta}[(\sqrt n+s)^2-b^2+\tilde v\cdot\tilde v].
\tag{11}
\]

Short-ranged massive internal fluctuations may renormalize the overlap of P with \(e^{2i\theta}\) and add short-ranged noise, without changing its infrared charge or algebraic exponent, provided the overlap remains nonzero. This is a conditional infrared statement. At large residuals the local parametrization itself is not trustworthy, and composite cancellations or a different ordered tensor can remove that overlap.

## 5. Bare uniform rigidity and exact curvature identities

For any spatially uniform complex 3×2 tensor A, not only a chiral state, a small common phase gradient produces

\[
\rho^{(0)}_{ab}=2K[n\delta_{ab}+2\operatorname{Re}(A_a^\dagger A_b)].
\tag{12}
\]

Thus \(\rho_{xx}^{(0)}+\rho_{yy}^{(0)}=8Kn\), and both eigenvalues lie between \(2Kn\) and \(6Kn\). Redistributing fixed uniform density between spin or orbital components, changing chirality, or making the spin vector nonunitary cannot alone change the direction-averaged bare common-phase curvature \(4Kn\). This does not make all these tensors thermodynamically stable. It rules out a particular direct mechanism for losing the averaged bare response.

The physical thermal helicity remains

\[
\Upsilon_{ab}=\frac{\langle\mathcal K_{ab}\rangle}{N}
-\frac{\operatorname{Cov}(I_a,I_b)}{TN},\quad
I_a=\partial_{q_a}H,\quad\mathcal K_{ab}=\partial_{q_a}\partial_{q_b}H.
\tag{13}
\]

There is no direct onsite SOC derivative under the common twist; SOC changes the configurations and their correlations. A single configuration's curvature is not its thermodynamic helicity.

For the full inhomogeneous periodic lattice, define

\[
W_{\rm diag}=\sum_{r,\mu}\operatorname{Re}\big[
(A_{\mu x}(r+\hat x)-A_{\mu x}(r+\hat y))^*
(A_{\mu y}(r+\hat x)-A_{\mu y}(r+\hat y))\big].
\]

Then the exact configuration identity is

\[
\mathcal K_{xx}+\mathcal K_{yy}
=8K N_A-H_g+2K W_{\rm diag},\quad N_A=\sum_r n_r.
\tag{14}
\]

One proof differentiates the Fourier matrix (2):

\[
(\partial_{k_x}^2+\partial_{k_y}^2)M
=8\mathbf1-M+|e^{ik_x}-e^{ik_y}|^2\sigma_x.
\]

Parseval's identity then gives (14). Therefore the saved averaged curvature satisfies

\[
D\equiv\frac{\langle\mathcal K_{xx}+\mathcal K_{yy}\rangle}{2N}
=4K\langle n\rangle-\frac{\langle H_g\rangle}{2N}
+\frac{K\langle W_{\rm diag}\rangle}{N}.
\tag{15}
\]

In a globally fixed orbital chirality, \(W_{\rm diag}=0\) exactly, including arbitrary nonunitary spin textures. More generally

\[
W_{\rm diag}=\operatorname{Im}\sum_r(\Delta_{\rm diag}u)^\dagger\Delta_{\rm diag}v,
\quad\Delta_{\rm diag}A=A(r+\hat x)-A(r+\hat y).
\]

This is an interference term, not opposite-chirality weight and not a nearest-manifold residual. It can vanish despite large projection loss. Normalized curvature \(D/(4K\langle n\rangle)\) therefore mixes gradient energy and tensor interference; it is not a uniquely defined spin coherence. For arbitrary lattice fields it is not even guaranteed to lie in [0,1]. Strong density–curvature covariance is natural in (15), but no identity fixes a particular statistical fraction of a thermal curvature change.

The mixed curvature, omitted from the old scalar channel list but computable from saved fields, is

\[
\mathcal K_{xy}=2K\sum_{r,\mu}\operatorname{Re}\big[
A_{\mu x}(r+\hat x)^*A_{\mu y}(r+\hat y)
+A_{\mu y}(r+\hat x)^*A_{\mu x}(r+\hat y)\big].
\tag{16}
\]

Equivalently it is \(4K\sum_k\cos(k_x-k_y)\operatorname{Re}[A_x(k)^\dagger A_y(k)]\). It vanishes in a globally fixed orbital chirality but not for general full fields. A negative \(\operatorname{Cov}(I_x,I_y)\) cannot by itself determine \(\Upsilon_{xy}\); its curvature term is required. For a positive definite long-distance phase stiffness tensor, the scalar controlling the Gaussian exponent and logarithmic vortex energy after a spatial coordinate transformation is \(\rho_{\rm eff}=\sqrt{\det\rho}\), rather than its arithmetic mean. Finite-torus geometry transforms at the same time.

## 6. Domain walls do not generically delete phase stiffness

The thesis's reduced London construction first uses the single-chirality isotropic gradient and then discretizes a chirality variable, obtaining zero phase coupling across opposite chirality. That is a specified reduced model. It is not the result of substituting a chirality wall into the actual full gradient.

For a straight sharp wall normal to x, with equal density, common real director, and no tangential variation, opposite-chirality neighbors have

\[
E^{x}_{ij}=Kn[4-2\cos\delta\theta].
\tag{17}
\]

For a y-normal wall the corresponding expression is \(Kn[4+2\cos\delta\theta]\), whose preferred offset is pi. At their preferred offsets both have finite normal phase curvature \(2Kn\), half the same-chirality value. These are restricted sharp-wall examples, not relaxed wall energies. In two-dimensional textures the mixed three-site contributions must also be retained.

A smooth x-normal spin-frozen wall gives another explicit result. Set

\[
A_x=f(x)e^{i\theta}d,\qquad A_y=i g(x)e^{i\theta}d,
\]

with real f,g and fixed real d. Direct continuum substitution gives

\[
\begin{aligned}
G={}&3f'^2+g'^2+(3f^2+g^2)(\partial_x\theta)^2
+(f^2+3g^2)(\partial_y\theta)^2\\
&+2(fg'-gf')\partial_y\theta.
\end{aligned}
\tag{18}
\]

The local tangential phase current coefficient is \(J_y=2K(fg'-gf')\), and the local phase-stiffness trace is still \(8K(f^2+g^2)\). At a wall center g=0 with f nonzero, neither stiffness eigenvalue vanishes. A chirality zero is not a local superfluid-density zero.

There is an essential global caveat. At fixed n with \(f=\sqrt n\cos\eta,g=\sqrt n\sin\eta\), \(J_y=2Kn\eta'\). An ordinary periodic pair of opposite walls cancels its integrated current. Local wall currents alone therefore do not prove a large variance of the *total* current in (13). Amplitude variation, unequal wall profiles, geometry, nontrivial winding, other tensor modes, and their correlations can matter. The static sharp-wall-pair control used below has zero total current.

Allowing wall degrees of freedom \(a\) to respond to a twist can nevertheless soften a stable constrained branch. At a stationary configuration with positive Hessian on its allowed internal variations,

\[
\partial_q^2H_{\min}(q)=H_{qq}-H_{qa}H_{aa}^{-1}H_{aq}.
\tag{19}
\]

The second term is nonnegative and can be large for soft wall modes. At finite temperature the corresponding mechanism includes current fluctuations and entropic effects. Equation (19) does not supply a thermal helicity estimate from a static wall, nor does it prove that walls dominate the measured transition. It identifies an actual route by which structural relaxation affects phase rigidity beyond uniform-density redistribution.

## 7. Running spin stiffness, SOC, and the anomalous jump

For an isotropic smooth regime with effective charge and director stiffnesses \(\rho_c(r),\rho_d(r)\), an elementary HQV uses a pi phase winding and a director path from d to -d. With a minimal great-circle spin path,

\[
E_{\rm HQV}(R)\simeq E_{\rm core}
+\frac\pi4\int_a^R[\rho_c(r)+\rho_d(r)]\,d\ln r.
\tag{20}
\]

A neutral opposite-charge pair has twice this logarithmic coefficient, up to boundary and core terms. At the bare equal-stiffness point the pair coefficient is \(\pi\rho_c=4\pi Kn\), reproducing the Stage-2/3 reference. The spin contribution cannot be dropped merely because the phase winding is one half.

The dilute-defect scaling dimension at an intermediate scale is schematically \(x_h(r)=\pi[\rho_c(r)+\rho_d(r)]/(4T)\). If the director sector still carries appreciable running stiffness, marginality would involve the sum. Writing \(\rho_c=8T/\pi\) for such a finite-scale regime is not justified. The S2 spin sector is not an independent planar XY field; it has no separately conserved signed planar vortex charge. Its long-distance renormalization and finite correlation length can screen the spin logarithm. Finite spin correlations on an available box are evidence that this screening scale must be assessed, not proof of a particular nonzero stiffness value.

Once the spin logarithm has saturated, unconfined half phase vortices in an otherwise valid paired Gaussian sector give the familiar \(\rho_c(T_c^-)=8T_c/\pi\). The analogous polar-spinor result and its distinction between zero-temperature combined-defect energy and finite-temperature spin-disordered paired order are explicit in [Mukerjee, Xu and Moore, PRL 97, 120406 (2006), Eqs. 4–7](https://arxiv.org/pdf/cond-mat/0605102). This shared manifold supports an analogy, not an identification of the present full GL thermal theory with their model.

For a charge-two operator with nonzero overlap onto \(e^{2i\theta}\), the Gaussian exponent is

\[
\eta_P=\frac{2T}{\pi\rho_c}
\quad\text{or}\quad\frac{2T}{\pi\sqrt{\det\rho_c}}
\text{ for a stiffness tensor}.
\tag{21}
\]

The director does not contribute a second exponent to this spin-singlet operator. It *does* contribute to the HQV energy before screening. Combining the finite-scale marginality expression with the scalar (21) would give \(\eta_P=\frac14[1-\pi\rho_d/(8T)]^{-1}\), above one quarter when \(\rho_d>0\). This is only a diagnostic of the assumptions, not a new universal jump: a running S2 sector may not have any asymptotic fixed point with that residual stiffness. A large P exponent and an apparent mass-helicity crossing can therefore test different assumptions. Their mismatch cannot be repaired by adding a spin-wave exponent to P.

Positive SOC gives the restricted anisotropy \(h_D(1-d_z^2)\), with bare \(h_D=g_D n\). At coarse scales the competing length is

\[
\ell_D^{\rm eff}\sim\sqrt{\rho_d/(2h_D)},
\tag{22}
\]

which reduces to \(\sqrt{2K/g_D}\) only for the bare restricted coefficients. A frozen-amplitude spin kink has tension \(2\sqrt{2\rho_d h_D}=4n\sqrt{2Kg_D}\), reproducing Stage 1. This is not the full-field finite-temperature string tension. Thermal spin screening, anisotropy renormalization, tensor escape and entropy can alter the relevant crossover. If the locked chiral vacuum and a nonzero string tension are established, isolated HQVs are linearly confined beyond the locking scale and the asymptotic ordinary-vortex reference is \(2T/\pi\). Merely having \(L>\sqrt{2K/g_D}\) does not establish that thermal locked regime. The order of the spin-screening and anisotropy crossovers matters.

## 8. Reproducibility, independent checks, and limits

All new computation is in [harmonic_checks.py](harmonic_checks.py); results are [checks.json](checks.json), [hessian_modes.csv](hessian_modes.csv), [phase_kernel.csv](phase_kernel.csv), [harmonic_covariance.csv](harmonic_covariance.csv), and [finite_twist_controls.csv](finite_twist_controls.csv). The production scalar helper functions are extracted from their AST with CUDA decorators removed in memory, so the onsite and original-gradient arithmetic is checked independently without importing CUDA. The source SHA256 and coefficients are recorded. No old file is changed.

The bounded workload consists of 12×12 eigendecompositions on three fixed momentum grids, 12-coordinate homogeneous finite differences, and six static L=8 fields evaluated at finite common twists and gD=0,.01. It has no Monte Carlo, optimization, GPU launch, or parameter search; the final execution took about 0.62 seconds on the available CPU. The six controls are homogeneous chiral, arbitrary homogeneous tensor, homogeneous nonunitary, straight opposite-wall pair, smooth spin modulation, and an inhomogeneous full tensor.

The maximum source-potential discrepancy is 9.1×10^-13; the exact lattice curvature identity discrepancy is 4.1×10^-13; the Richardson-extrapolated onsite Hessian error is 7.2×10^-9; the analytic phase Schur complement agrees to 2.3×10^-16. Independent finite-q evaluation of the *full* Hamiltonian verifies both currents, both diagonal curvatures and the mixed curvature to 4.3×10^-9. SOC has no direct twist derivative in these controls. These numerical checks validate algebra and conventions; they are not a thermal simulation or an error bar on the mechanism.

The strongest conclusions are exact: bare homogeneous structural redistribution at fixed n cannot lower the averaged curvature; full-field textures obey (14), including the interference correction; the selected branch has only phase and director soft modes; and chirality walls retain phase coupling in the actual gradient. The harmonic residual benchmark is controlled only at sufficiently small fluctuations, while the domain-wall relaxation and HQV/SOC arguments are conditional low-energy mechanisms. Determining which mechanism dominates the existing thermal helicity drop requires the separate data analysis, not another assertion that symmetry alone fixes the transition.
