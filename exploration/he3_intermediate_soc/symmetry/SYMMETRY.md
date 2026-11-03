# Mixed stiffness, exact lattice symmetries, and orbital selection

These are deterministic algebra and CPU checks of the implemented Hamiltonian. They establish which thermal responses are allowed; they do not establish an equilibrium phase or a thermal value of the stiffness. No production file was modified and no Monte Carlo was run by this calculation.

## Main results

1. The existing mixed-curvature formula is correct. Direct finite twists of the actual production potential/gradient polynomial verify all five derivatives, including Kxy, to **3.56e-9 per site** on the tested fields.
2. The forward-difference lattice has diagonal mirrors and simultaneous inversion, but lacks axial mirrors and 90-degree rotation symmetry. **A nonzero mixed thermodynamic stiffness is allowed even in the full equilibrium ensemble.** Averaging opposite chiralities does not eliminate it.
3. A+ and A− uniform polar configurations have equal onsite energies, but they are **not symmetry-related fluctuating lattice sectors**. Explicit finite-spacing effects can distinguish them. That does not determine the preferred sign or prove the observed preference is equilibrated.
4. The nonchiral z-polar A+ structure and real antisymmetric planar structure are sharply different. Both have P=κ=0 and finite bare phase stiffness, but their spin alignment, B tensor and mixed stiffness differ.
5. A useful null result: in a deliberately restricted frozen-polar phase-wave calculation, the lattice favors **A−**, opposite to the observed coherent A+ channel. Therefore broken mirror symmetry alone is not an explanation of the observed sign; internal/amplitude fluctuations and sampling must be investigated.

## 1. Twist convention and exact response

The code twists every forward neighbor as

\[
A(r+\hat\nu)\longrightarrow e^{-iq_\nu}A(r+\hat\nu).
\]

Write Iν=∂H/∂qν and Kμν=∂²H/∂qμ∂qν at q=0. With x=A_x and y=A_y complex spin vectors, direct differentiation of the two forward mixed-gradient terms gives

\[
\mathcal K_{xy}=2K\sum_{r,\mu}\mathrm{Re}\left[
A_{\mu x}(r+\hat x)^*A_{\mu y}(r+\hat y)
+A_{\mu y}(r+\hat x)^*A_{\mu x}(r+\hat y)\right].
\]

The sign is positive with the production convention. Changing the common sign of both twist coordinates reverses both currents and leaves all second derivatives unchanged. The onsite potential, including SOC, has no common-phase twist derivative.

For a stationary equilibrium ensemble the free-energy Hessian is

\[
\Upsilon_{\mu\nu}=N^{-1}\left[\langle\mathcal K_{\mu\nu}\rangle
-T^{-1}(\langle I_\mu I_\nu\rangle-\langle I_\mu\rangle\langle I_\nu\rangle)\right].
\]

In a short or trapped trajectory the same moment combination is a response diagnostic of that sampled history. It should not silently be labeled the equilibrium thermodynamic tensor. Branch conditioning also requires branch-specific current means. Kxy and current covariance must refer to the same ensemble; sparse Kxy is usable with either synchronized moments or a clearly documented mixed-cadence estimator and joint time-block uncertainty.

## 2. Why an axial mirror is absent

For one Fourier momentum, pν=e^{ikν}−1 and the orbital gradient matrix is

\[
M(k)=\begin{pmatrix}
3|p_x|^2+|p_y|^2&C(k)\\
C(k)&|p_x|^2+3|p_y|^2
\end{pmatrix},
\]

\[
C(k)=2\mathrm{Re}(p_x^*p_y)
=2[(1-\cos k_x)(1-\cos k_y)+\sin k_x\sin k_y].
\]

A conventional axial reflection acts on the corresponding orbital column as well as on coordinates. Such a symmetry would require C(−kx,ky)=−C(kx,ky). Instead,

\[
C(-k_x,k_y)+C(k_x,k_y)=4(1-\cos k_x)(1-\cos k_y).
\]

This difference is fourth order at long wavelength, but nonzero at finite lattice spacing. The continuum leading kernel is k²I+2kkᵀ and has the larger continuum spatial/orbital symmetry. That limiting fact does not restore an exact symmetry to the actual finite-spacing simulation.

This is stronger than failure of one guessed field transformation. At k=(.7,1.1), M has eigenvalues (1.589298,4.663195), whereas the axial-mirrored matrix has (2.039105,4.213389). Hence no norm-preserving orbital unitary transformation can repair the reflection; unitary component-translation phases cannot repair it either. Conjugation only adds simultaneous momentum inversion and leaves this counterexample. A momentum-independent onsite term cannot cancel the momentum-dependent defect for every k. Shifting orbital components relative to one another would additionally spoil the generic local quartic and SOC contractions.

### Geometric symmetries that do remain

For a square periodic box, the exact geometric point group contains

- identity;
- x↔y, a diagonal reflection;
- (x,y)↦(−x,−y), a half-turn;
- (x,y)↦(−y,−x), the other diagonal reflection.

At gD>0 these geometric operations include the corresponding orbital transformation R and spin transformation with in-plane block R; choosing its z entry det R makes the spin matrix proper. Explicitly, A′(r)=R_spin A(Rᵀr)Rᵀ. This preserves the SOC contraction. At zero SOC arbitrary independent spin rotations are also permitted. Global U(1), complex conjugation/time reversal, and lattice translations remain exact. Axial mirrors and quarter-turns fail the full gradient test while preserving onsite and SOC energies.

The SOC statement also follows algebraically. For the upper 2×2 spin/orbital block C, its invariant is |Tr C|²+Tr(CC*)−c0 n, with c0 the production float32 representation of 2/3. Under a common real orthogonal in-plane rotation, C→RCRᵀ and both traces remain unchanged. SOC therefore does not restore the spatial symmetries already lost in the gradient.

The numerical random-field control at L=8 changes gradient energy by 15.896502 under an axial mirror or quarter-turn, while the onsite/SOC parts remain equal within roundoff. The true geometric operations preserve all parts. These statements concern conventional coordinate transformations and norm-preserving internal operations, not an unsupported classification of every conceivable nonlocal transformation of the partition function.

### Implication for the stiffness

Every surviving geometric operation preserves qx qy. A symmetry-averaged tensor may consequently have the form

\[
\Upsilon=\begin{pmatrix}Y&Z\\Z&Y\end{pmatrix},
\]

with Z nonzero. The diagonal equality follows only when the coordinate-exchange-related sectors are properly sampled; the absence of that equality in a trapped preparation is a separate issue.

Time reversal flips chirality and both currents, but leaves Kxy and Cov(Ix,Iy) invariant. Coordinate exchange flips chirality while preserving the xy component too. **Neither equal positive/negative chirality populations nor restoration of time reversal forces Z to vanish.** An axial mirror would change its sign, but is not an exact symmetry of this stencil. In a different mirror-symmetric discretization, a selected oriented sector could have Z≠0 while its mirror-averaged ensemble had Z=0; that argument cannot be imported here unchanged.

## 3. Uniform controls distinguish structures

For any homogeneous full matrix with n=Tr(A†A), the bare tensor is exactly

\[
\rho^{\rm bare}=2K[nI_2+2\mathrm{Re}(A^\dagger A)].
\]

Its diagonal average is 4Kn regardless of internal structure. All six fields in the table/script are independent of SOC under the common phase twist at fixed A; SOC changes which configurations are sampled.

At K=n=1:

| Uniform field | κ/n | Spin alignment | B++ magnitude | B−− magnitude | ρxx=ρyy | ρxy | Eigenvalues |
|---|---:|---:|---:|---:|---:|---:|---|
| z-spin chiral, Az=(1,i)/√2 | 1 | 1 | .5 | .5 | 4 | 0 | 4,4 |
| z-spin plus polar, Az=(1,1)/√2 | 0 | 1 | 1 | 0 | 4 | 2 | 2,6 |
| z-spin minus polar, Az=(1,−1)/√2 | 0 | 1 | 0 | 1 | 4 | −2 | 2,6 |
| Antisymmetric planar, Axy=−Ayx=1/√2 | 0 | −.5 | .5 | .5 | 4 | 0 | 4,4 |

Here B++=A+·A+, A±=(Ax±Ay)/√2, and dots have no conjugation. P=0 in all three nonchiral controls. The planar state has equal A+/A− density and orbital Bxx=Byy=1/2, Bxy=0; the plus-polar state has entirely different spin and orbital content. Thus almost-zero chirality plus finite helicity cannot identify the old .1 population with the .01 coherent A+ channel.

The polar table is a controlled response calculation at a specified field, not a claim that this field is a stable onsite minimum. The landscape calculation separately addresses that issue. Thermal mixed stiffness cannot be replaced by the bare value 2Kn of this example.

## 4. What broken mirrors do and do not explain

A+ and A− homogeneous polar fields have equal onsite energy and zero gradient energy. Their equality is an accidental restriction to uniform configurations: a 90-degree rotation would interchange these orbital orientations, but is not an exact lattice symmetry. Integrating fluctuating fields can therefore produce different free energies without spontaneous breaking of an exact A+↔A− symmetry. The observed preference might be promoted by finite-spacing terms, but this observation alone determines neither its sign nor its magnitude.

A small discriminating control makes that limitation concrete. Freeze spin, density, and polar orbital orientation and fluctuate only a small scalar phase around either polar field. Its positive harmonic kernel is proportional to

\[
\lambda_\pm(k)=2s\pm C(k),\qquad s=|p_x|^2+|p_y|^2.
\]

After removing the common zero mode, the conditional Gaussian free-energy difference is

\[
\frac{f_+-f_-}{T}=\frac1{2L^2}\sum_{k\ne0}\ln\frac{2s+C}{2s-C}.
\]

It converges to **+.1939649471** per site for L=64–256. The sign can also be seen by pairing kx and −kx: write C=2(a+b), where a=(1−cos kx)(1−cos ky)≥0 and b=sin kx sin ky. The paired plus product exceeds the minus product by a positive term proportional to sa. Thus this restricted phase entropy favors the minus orientation. It cannot explain the observed plus preference.

This is not a full-GL finite-temperature free energy: amplitudes, chirality, spin and nonunitary channels are fixed, and the polar reference need not be an onsite local minimum. Its purpose is to reject a too-simple sign argument, not to overturn the measured plus-channel result. A reliable microscopic account needs the coupled internal fluctuations or independent-start equilibrium evidence.

## 5. When a geometric-mean BKT criterion is legitimate

If one has independently established a single compact phase with a positive-definite, spatially homogeneous long-distance quadratic action

\[
F_{\rm eff}=\frac12\int d^2r\,(\nabla\theta)^T\rho(\nabla\theta),
\]

then r=ρ^(1/2)r′ makes the coefficient √detρ. It also changes the torus shape and aspect ratio. Within this effective theory, a phase-charge-m operator has exponent m²T/(2π√detρ), and a vortex with phase winding 2πq has the logarithmic energy coefficient πq²√detρ. A dilute single-species BKT criterion then uses √detρ/T=2/(πq²).

Those are conditional consequences, not evidence that a scalar phase theory or a specified q applies to the full GL data. A separate unscreened spin contribution, confining texture, critical internal mode, multiple coherent phases, or nonequilibrium branch mixture invalidates that reduction. A finite-torus helicity tensor also includes winding-sector response and need not equal the uncorrected Gaussian action coefficient.

For equal diagonals Y and off-diagonal Z, √detΥ=√(Y²−Z²), so using Y alone overestimates that candidate scalar stiffness when |Z|>0. In the bare plus-polar example the factor is √3/2≈.866. This is an illustration, not a renormalized correction to apply to the large runs with missing Kxy.

## Reproducibility

Run `python -B exploration/he3_intermediate_soc/symmetry/check_symmetry.py` from the repository root. The short script extracts the production scalar potential and gradient functions from their AST, removes CUDA decorators only in memory, and evaluates them on CPU. It compares full-energy finite twists with the derivative formulas at gD=0,.01,.1 on homogeneous chiral, plus/minus polar, planar, arbitrary uniform, and arbitrary inhomogeneous L=8 fields.

The largest full-energy difference from the established independent CPU evaluator is 1.14e-13. The [checks](checks.json) include the source SHA256 and mirrored kinetic spectra. [Finite-twist checks](finite_twist_checks.csv), [uniform controls](uniform_tensors.csv), [symmetry controls](symmetry_controls.csv), and the [conditional polar calculation](fixed_polar_phase_control.csv) contain the numerical results. No MC/GPU or source mutations were needed.
