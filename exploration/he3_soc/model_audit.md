# Model and observable audit

This supplement records definitions checked against the current repository and saved manifests. It separates algebraic properties of the implemented Hamiltonian from hypotheses about the sampled finite-temperature phases. The attached thesis is useful background on field normalization and the earlier SOC-free work; its simulation parameters and transition estimates are not measurements of the present campaign. The thesis explicitly treats bulk strong-coupling corrections applied phenomenologically to a thin-film model. The bibliography and figures were not attached, and `HE3_MODEL.md` is absent from this worktree.

## Hamiltonian and units

The field at each site is an unconstrained complex matrix (A_{\mu a}), with spin rows (\mu=x,y,z) and orbital columns (a=x,y). There are twelve real components per site. The actual onsite potential in [update.py](../../models/he3/update.py), lines 15–63, is

\[
V=-2n+\sum_{i=1}^5\beta_i I_i+g_D D,
\qquad n=\sum_{\mu,a}|A_{\mu a}|^2.
\]

Writing (x_\mu=A_{\mu x},y_\mu=A_{\mu y}), (n_x=\sum|x|^2,n_y=\sum|y|^2), (X=\sum x^2,Y=\sum y^2,Z=\sum xy,O=\sum x^*y), the code evaluates

\[
I_1=|X+Y|^2,\ I_2=n^2,\ I_3=n_x^2+n_y^2+2\mathrm{Re}(O^2),
\]
\[
I_4=n_x^2+n_y^2+2|O|^2,\qquad I_5=|X|^2+|Y|^2+2|Z|^2.
\]

For one spin row, with forward differences (a=\Delta_x A_x,b=\Delta_x A_y,c=\Delta_y A_x,d=\Delta_y A_y), the gradient density is

\[
G=3(|a|^2+|d|^2)+|b|^2+|c|^2+2\mathrm{Re}(a^*d+b^*c).
\]

The total dimensionless Hamiltonian is (H=\sum_r[V_r+K\sum_\mu G_{r\mu}]), with periodic square boundaries. Independent spin rotations are exact at (g_D=0); continuum spatial/orbital rotation symmetry should not be assumed exact for this forward-difference lattice discretization.

All sixteen manifests give (K=1), pressure 20 bar, physical `reduced_temperature=1`, and the same float32-rounded coefficients

\[
\boldsymbol\beta=(-0.5165123343,\ 0.9420228004,\ 0.9875290394,\ 0.8790628910,\ -1.1525149345).
\]

The Monte Carlo temperature changes the Boltzmann factor (e^{-H/T}), with this Hamiltonian fixed. It is **not** the same variable as `reduced_temperature`, which multiplies the phenomenological strong-coupling correction in [model.py](../../models/he3/model.py), lines 29–44. These are dimensionless simulation temperatures, not a direct calibrated physical (T/T_c) axis. The coefficient source is Regan, Wiman and Sauls, [Phys. Rev. B 101, 024517 (2020), Appendix](https://sauls.lsu.edu/publications/jas160/jas160.pdf). Their coefficients describe bulk helium; applying them to a strictly reduced thin film is an additional modeling assumption.

## Saved observables

[measure.py](../../models/he3/measure.py), lines 23–89, accumulates the following channels. Array axis 0 is measurement time; axis 1 is the **fixed temperature index**, after scattering from physical walker order.

| Saved channel | Per-configuration quantity | Reported estimator |
|---|---|---|
| `energy` | Total (H), including SOC and gradients | (\langle H\rangle/L^2) |
| `density` | (N_A=\sum_{r,\mu,a}|A_{\mu a}(r)|^2) | (\langle N_A\rangle/L^2) |
| `chirality` | (C=2\sum_{r,\mu}\mathrm{Im}(A_{\mu x}^*A_{\mu y})) | (\langle |c|\rangle, c=C/N_A) |
| `phase_real`, `phase_imag` | Real/imaginary parts of (P=\sum_{r,\mu}(A_{\mu x}^2-A_{\mu y}^2)) | (\sqrt{\langle |P|^2/L^4\rangle}) |
| `spin_alignment` | (Q_{zz}=\frac12(3N_z/N_A-1),\ N_z=\sum_{r,a}|A_{za}|^2) | (\langle Q_{zz}\rangle) |
| `current_x/y` | (I_\nu=\partial H/\partial\delta_\nu|_0) | Used in fluctuation formula below |
| `curvature_x/y` | (K_\nu=\partial^2 H/\partial\delta_\nu^2|_0) | Used in fluctuation formula below |

The twist replaces every forward neighbor in direction (\nu) by (e^{-i\delta_\nu}A(r+\hat\nu)), with the same phase for all six components. The SOC term is onsite and invariant under this common gauge phase, so it has no direct twist derivative. Its effects enter through the sampled configurations. The measured modulus is

\[
\Upsilon_\nu=\frac{\langle K_\nu\rangle-[\langle I_\nu^2\rangle-\langle I_\nu\rangle^2]/T}{L^2},
\qquad \Upsilon=(\Upsilon_x+\Upsilon_y)/2.
\]

It is essential to form the **variance** of current and to average before applying this nonlinear estimator. A single configuration does not have a helicity modulus equal to this thermodynamic response. First/second-half or branch-conditioned moduli require their own current means.

[analysis.py](../../models/he3/analysis.py), lines 29–40 and 93–117, forms

\[
U_\chi=1-\frac{\langle c^4\rangle}{3\langle c^2\rangle^2},
\qquad U_P=2-\frac{\langle p^4\rangle}{\langle p^2\rangle^2},
\quad p^2=|P|^2/L^4.
\]

Here (c) is a ratio on each configuration, not the ratio of ensemble means, and not the unweighted spatial mean of normalized local chirality. (P) is not divided by density. Consequently its magnitude responds to amplitude changes as well as phase correlations. It has gauge charge two: (A\to e^{i\theta}A\) gives (P\to e^{2i\theta}P\), and it is invariant under real spin rotations.

(Q_{zz}\) lies in ([-1/2,1]). It measures normalized population of the spin-z row, not magnetization, a spatial spin correlation function, or a direct topological observable. An isotropic equilibrium ensemble has mean zero at (g_D=0); a finite trajectory can retain a preferred global orientation. Local spin textures cannot be reconstructed from its global scalar value. The code assigns (Q_{zz}=0) if total density is exactly zero.

## Exact SOC structure and a competing local minimum

The implemented invariant is

\[
D=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\mathrm{Re}(A_{xy}A_{yx}^*)-\tfrac23 n.
\]

Its complex Hermitian quadratic form has the following exact eigenvalues per unit density:

| Field subspace | (D/n) |
|---|---:|
| (A_{xy}=-A_{yx}), other components zero | (-5/3) |
| Either component of the z spin row | (-2/3) |
| (A_{xx}=-A_{yy}), or (A_{xy}=A_{yx}) | (1/3) |
| (A_{xx}=A_{yy}) | (7/3) |

Thus SOC in the **full field space** is richer than a phenomenological term that only aligns a pre-existing chiral spin vector. A real antisymmetric planar configuration can gain more dipole energy per density than the chiral z-row configuration. (Q_{zz}\) alone cannot reconstruct (D), since fields with identical (N_z/N_A\) can have different spin/orbital interference terms.

An ideal chiral field (A=\sqrt{n/2}\,e^{i\theta}\hat d\,(1,i\sigma)), with real unit (\hat d\) and (\sigma=\pm1), obeys (c=\sigma\), (P/L^2=ne^{2i\theta}\), (Q_{zz}=(3d_z^2-1)/2), and

\[
D=n(\tfrac13-d_z^2)=-\tfrac23 nQ_{zz}.
\]

The last identity remains true after summing density-weighted ideal chiral fields, but it is not true for arbitrary saved configurations. Positive (g_D\) favors (\hat d\parallel\hat z\) within this restricted manifold.

The standalone [local_potential.py](local_potential.py) reads the saved coefficients, evaluates the original onsite polynomial in float64, compares two homogeneous ansätze, checks their twelve-dimensional real Hessians, and performs eighteen BFGS starts at each coupling. This is a small deterministic CPU calculation, not a new simulation. Tables are [local_potential.csv](results/tables/local_potential.csv) and [local_hessian.csv](results/tables/local_hessian.csv).

For the chiral z-row ansatz, (\beta_A=\beta_2+\beta_4+\beta_5=0.6685707569\), (n_A=(1+g_D/3)/\beta_A\), and (V_A=-(1+g_D/3)^2/\beta_A\). For the real antisymmetric planar ansatz, (\beta_P=\beta_1+\beta_2+(\beta_3+\beta_4+\beta_5)/2=0.7825489640\), (n_P=(1+5g_D/6)/\beta_P\), and (V_P=-(1+5g_D/6)^2/\beta_P\).

| (g_D) | (V_A) | (V_P) | (V_P-V_A) |
|---:|---:|---:|---:|
| 0 | -1.49572800 | -1.27787531 | 0.21785269 |
| 0.001 | -1.49672532 | -1.28000599 | 0.21671933 |
| 0.01 | -1.50571614 | -1.29926197 | 0.20645416 |
| 0.1 | -1.59710512 | -1.49972867 | 0.09737645 |

At (g_D=0.1\), both are locally stable homogeneous onsite minima: each has one global-phase zero mode, and the next Hessian eigenvalues are 0.189744 and 0.200000, respectively. Halving the finite-difference step from (10^{-4}\) to (5\times10^{-5}\) changes eigenvalues by less than (4\times10^{-7}\). The two ansatz energies would cross at (g_D\simeq0.173232\), outside this dataset. The multistart check found no energy below the chiral minimum; finite multistart optimization is not a proof of a global minimum, and some BFGS runs report finite-difference precision limits rather than successful termination.

These are **zero-temperature onsite statements**. They do not establish a finite-temperature planar phase, its entropy, an interface cost, or the identity of the observed branch. They do provide a concrete competing-configuration hypothesis. The planar state has (c=0,Q_{zz}=-1/2,P=0\), despite a nonzero common-phase stiffness (4Kn_P\) in a homogeneous configuration. Therefore a small measured (P\) can reflect tensor cancellation in a different internal sector rather than loss of all superfluid stiffness. Identifying such a branch requires local matrices or additional invariant channels.

## Bare dipole length and HQV interpretation

For a slowly varying chiral ansatz at fixed amplitude and phase, direct substitution gives spin energy (2Kn|\nabla\hat d|^2\), while the dipole anisotropy relative to its minimum is (g_D n(1-d_z^2)\). The associated bare length is

\[
\ell_D=\sqrt{2K/g_D}.
\]

This reproduces `dipole_length_lattice` in [model.py](../../models/he3/model.py), lines 108–115. It is measured in lattice spacings. At (K=1\), the values are 44.7214, 14.1421, and 4.47214 for (g_D=0.001,0.01,0.1\). The ranges of (L/\ell_D\) over (L=64\)–160 are 1.431–3.578, 4.525–11.314, and 14.311–35.777, respectively. At zero coupling the length diverges. The thesis used a lattice spacing equal to a GL coherence-length convention; no absolute nanometer mapping is saved. At finite temperature the effective spin stiffness, anisotropy, amplitude and departure from the chiral manifold can change the actual crossover length.

In the ideal SOC-free chiral manifold, ((\hat d,\theta)\sim(-\hat d,\theta+\pi)\) permits combined spin/phase half vortices. Independent spin rotations are explicitly broken by nonzero SOC; in the locked chiral sector, a half circulation generally involves a costly departure from the spin minimum and may carry a connecting texture. This is a theoretical motivation for confinement tests, **not a finding from thermodynamic curves**. The primary proposal by Volovik and Salomaa, [Sov. Phys. JETP 61, 986 (1985)](https://jetp.ras.ru/cgi-bin/dn/e_061_05_0986.pdf), explicitly describes the combined symmetry and dipole solitons and also shows dependence on magnetic field and confinement geometry. Its specialized experimental geometry should not be imposed on this zero-field lattice model without derivation.

The commonly tested jump (\Upsilon=2T/(\pi q^2)\) presupposes the appropriate long-distance vortex theory and stiffness normalization. Here (\Upsilon\) is the common-phase stiffness; (q=1/2\) gives (8T/\pi\), and (q=1\) gives (2T/\pi\). **The mass-phase charge alone does not determine the finite-size HQV energy when a spin texture still has stiffness.** To see this, use an isotropic long-wavelength action

\[
F=\tfrac12\int d^2r[\rho_c(\nabla\theta)^2+\rho_d(\nabla\hat d)^2].
\]

For a simple HQV, both phase and spin angle change by (\pi\) on a circle. Direct integration gives (E_{\rm HQV}\simeq\pi(\rho_c+\rho_d)\ln(R/a)/4\), whereas an integer phase vortex gives (E\simeq\pi\rho_c\ln(R/a)\). Thus a spin contribution persisting to the relevant scale changes the energetic condition; applying (\rho_c=8T/\pi\) to the mass modulus alone requires that the additional spin logarithm has been screened, along with the absence of confinement. The bare restricted GL ansatz has (\rho_c=\rho_d=4Kn\), but finite-temperature renormalized values need not remain equal.

A useful **analogy**, not a mapping of the complete GL model, is the polar spinor condensate with the same ((S^2\times U(1))/Z_2\) manifold. Mukerjee, Xu and Moore derive a spin-disordered paired-superfluid regime and an anomalous mass-helicity jump (8T/\pi\); their analysis explicitly distinguishes the zero-temperature combined spin/phase vortex energy from the finite-temperature spin-disordered theory ([Phys. Rev. Lett. 97, 120406 (2006)](https://arxiv.org/pdf/cond-mat/0605102), Eqs. 4–7). The present data contain neither spin helicity nor the spin correlation length, so they cannot show that (L=64\)–160 is beyond that crossover. Nonzero SOC introduces an additional anisotropy scale and possible strings.

A fit consistent with (q=1/2\) therefore does not observe a defect or prove its charge. Saved scalar channels contain no winding counts, spin stiffness, spatial correlations, defect locations, pairing distances or textures. They cannot demonstrate HQV confinement or determine a vortex interaction potential.

## Provenance and sampling details relevant to interpretation

All sixteen saved manifests, rather than current script defaults, establish 2,000,000 thermalization sweeps, 3,000,000 measurement sweeps, sampling every 50 sweeps, and 160 temperatures from 0.33 to 0.47. **The saved PT exchange interval is 5 sweeps**, not the 10 in the task summary or current `main/he3.py`. Walker histories have stride 100 exchange rounds, hence 500 sweeps. Exact trip counters include thermalization and are independent of history thinning.

The saved sampler is `float32_conditional_gaussian_full_hamiltonian_mh`, (\rho\simeq-0.9\), proposal mass zero. [gaussian_fast.py](../../models/he3/gaussian_fast.py), lines 54–117, implements a Gaussian autoregressive proposal with a full-Hamiltonian Metropolis–Hastings correction. Negative rho produces stochastic overrelaxation; this is not an exact microcanonical overrelaxation sweep. Field/proposal arithmetic is float32, reductions are float64. High local or PT acceptance is not an equilibration certificate. The Gaussian acceptance vectors are indexed by **physical walker**, because their counters remain in walker slots; they must not be interpreted as fixed-temperature acceptance profiles. PT acceptance vectors are indexed by neighboring temperature edges.

The seeds are (42+L\), giving 106, 138, 170, 202, and are reused across couplings at fixed (L\). Thus there are not four independently seeded production replicates of any parameter set, and cross-coupling error estimates cannot simply assert independent-run covariance. Common seeds do not imply the trajectories remain identical after different acceptance decisions.

The current initializer in [model.py](../../models/he3/model.py), lines 52–59, is homogeneous positive chirality with (\hat d=\hat z\), the same initial amplitude formula at all couplings. No source revision or saved initial field is present in the manifests, so the current implementation documents the intended initialization but cannot independently certify the historical exact source revision. This shared ordered-start protocol makes failure to escape a sector a relevant alternative to equilibrium ordering.

[parallel_tempering.py](../../core/parallel_tempering.py), lines 73–91, counts endpoint → opposite endpoint → original endpoint as a completed round trip. Interior walkers start a trip only after first visiting an endpoint. Zero completed round trips can coexist with one-way passages; sampled paths miss visits shorter than 500 sweeps. Fixed-temperature measurements can exchange physical configurations repeatedly without a full-ladder trip, while full-ladder trips can occur without relaxing a hidden internal branch. Both temperature transport and internal-sector stationarity require separate diagnostics. Finally, no raw spatial snapshots or separate dipole/quartic energy components are stored in these ten-channel NPZ files, so several physically decisive questions require new observables rather than more fitting of existing scalars.
