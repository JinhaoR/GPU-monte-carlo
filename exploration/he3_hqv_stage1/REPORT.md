# Half-quantum vortices in the full thin-film GL field: Stage 1

**Status:** analytical construction and synthetic numerical validation completed. Stages 2 and 3 below are proposals only.  
**Model:** the repository's complex 3×2 field and existing Hamiltonian.  
**Date:** 29 October 2026.

## Main result and recommendations

A useful HQV detector can be built without forcing the full GL field to be chiral. The implementation first measures its distance from a unitary, rank-one chiral matrix, and assigns a defect charge only on surrounding loops where that interpretation is reliable. It combines a local charge-two phase winding, a three-dimensional director's sign holonomy, and a branch-independent phase lift. Poor-quality or unresolved loops remain explicitly unclassified.

The main mathematical corrections to a naive detector are:

1. **A winding zero of the composite field is not sufficient evidence of an HQV.** A smooth nonchiral matrix field can produce exactly that signal. The synthetic counterexample is rejected by the full-matrix quality test.
2. **A three-dimensional director has a Z2 loop parity, not a general signed planar spin winding.** On the ideal fixed-chirality manifold the full fundamental group is Z; director parity and phase parity are linked, rather than independent charges.
3. **A single plaquette can alias an integer vortex into an apparent half charge.** The detector flags such links and checks larger loops; the joint phase lift supplies an additional resolution diagnostic.
4. **An enclosing loop measures net charge, not the number or identity of cores.** Two same-sign HQVs can give an integer net charge; a neutral HQV pair gives zero net charge on an outer loop.
5. **Full-field core escape must remain allowed.** An HQV's surrounding loop can be well classified while its interior is nonchiral and its total density never vanishes. The classifier is conditional on the surrounding ordered sector, not on an absolute topology of arbitrary nonzero matrices.

Validation passes **32 unittest methods**, **15 saved known-contour checks**, and **560 correlated noise/threshold evaluations**. Automatic candidate finding recovers the known separated integer and half-quantum defects without being given their locations. It accepts no HQV in the adversarial controls. These are finite synthetic tests, not a calibrated error rate for thermal configurations.

The next scientific step should be occasional full-resolution snapshots at zero and weak SOC, with detector validity and unresolved regions reported alongside counts. Interaction calculations should start with a **neutral HQV–anti-HQV pair**, retain all twelve real GL components during later relaxation, and verify that recognizable defects survive at several surrounding radii. A separation-energy curve without that check could measure annihilation, screening or imposed constraints instead of an HQV interaction.

## 1. Hamiltonian, symmetry and the relevant manifold

### 1.1 What has been checked in the actual implementation

The production [update.py](../../models/he3/update.py) uses spin rows mu=x,y,z and orbital columns a=x,y. Write the columns as complex spin vectors x and y, with n=|x|²+|y|². The Hamiltonian is

\[
H=\sum_r\left[-2n+\sum_{j=1}^5\beta_j I_j+g_D D+K\sum_\mu G_\mu\right].
\]

For nx=x†x, ny=y†y, X=x·x, Y=y·y, Z=x·y and O=x†y, the implemented invariants are

\[
I_1=|X+Y|^2,\quad I_2=n^2,\quad
I_3=n_x^2+n_y^2+2\operatorname{Re}O^2,
\]
\[
I_4=n_x^2+n_y^2+2|O|^2,\qquad
I_5=|X|^2+|Y|^2+2|Z|^2.
\]

The exact SOC invariant is

\[
D=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\operatorname{Re}(A_{xy}A_{yx}^*)-\frac23 n.
\]

For one spin row let a=Delta_x A_x, b=Delta_x A_y, c=Delta_y A_x and d=Delta_y A_y, using the actual forward differences. Then

\[
G=3(|a|^2+|d|^2)+|b|^2+|c|^2+2\operatorname{Re}(a^*d+b^*c).
\]

Global A→exp(i gamma)A is an exact symmetry. At g_D=0, arbitrary real spin rotations A→RA, R in SO(3), are also exact, including for this gradient stencil. Independent spin rotations are broken by nonzero SOC. Complex conjugation is an exact symmetry at either coupling; it reverses chirality and phase winding. Continuous spatial/orbital rotational symmetry is not assumed exact for the lattice stencil.

[check_algebra.py](check_algebra.py) reads the four scalar polynomial helpers from production source, removes their CUDA decorators in memory, and evaluates them as ordinary CPU functions on small random fields. It does not import the runtime, run a kernel, alter a production file or optimize a configuration. On an 11×12 periodic random field, gauge energy differences are below 5×10^−13 and conjugation differences are zero. A generic spin rotation preserves energy at zero SOC and changes it at g_D=0.001 and 0.01, as expected. These checks exercise the actual polynomial definitions, rather than a replacement Hamiltonian.

### 1.2 Chiral reference fields and combined topology

The ideal unitary rank-one chiral fields are

\[
A=a\,\hat d\,e_\sigma^T,\quad
e_\sigma=(1,i\sigma)/\sqrt2,\quad
a=\sqrt n e^{i\theta},\quad \hat d\in S^2,\quad\sigma=\pm1.
\]

The physical matrix is unchanged by (d,theta)→(−d,theta+pi). At fixed chirality and nonzero amplitude,

\[
\mathcal M_\sigma=(S^2\times U(1))/\mathbb Z_2,\qquad
\pi_1(\mathcal M_\sigma)=\mathbb Z.
\]

The cover S²×R has the generator (d,theta)→(−d,theta+pi). Its integer k gives phase circulation k pi and a director lift ending at (−1)^k d. Hence the intrinsic director parity is tied to k mod 2. It is not an additional independent signed spin charge. The same quotient topology is derived in a distinct polar-spinor system by [Mukerjee, Xu and Moore, PRL 97, 120406 (2006)](https://arxiv.org/pdf/cond-mat/0605102); its full Hamiltonian is not being substituted for this one.

In contrast, the unrestricted nonzero matrix space C6 minus the origin retracts to S11 and is simply connected. This permits nonchiral or filled cores and other escape routes. It does not erase the usual classification of an excitation by its ordered far field or fixed boundary. The numerical task is to verify that a surrounding contour supports the reference manifold interpretation, and to expose where it fails. The reference manifold is not imposed on the energy, the synthetic noise, or the entire interior of a contour.

Chirality sigma and circulation sign are different labels. A positive or negative HQV can be constructed in either chirality sector. A contour crossing between sectors cannot be classified by a single fixed-sector HQV charge.

## 2. Deriving measurable signatures from the complete matrix

### 2.1 Exact distance to a chiral field

Define

\[
z_\sigma=(x-i\sigma y)/\sqrt2,\quad
m_\sigma=z_\sigma^\dagger z_\sigma,\quad
B_\sigma=z_\sigma\cdot z_\sigma.
\]

This is a unitary orbital change of basis: n=m_++m_− and local chirality C=2 Im(x†y)=m_+−m_−.

For fixed sigma, minimize ||A−b d e_sigma^T||_F² over complex b and a real unit d. For a chosen d, b=d^T z_sigma. The optimal director is a principal eigenvector of

\[
N_\sigma=\operatorname{Re}(z_\sigma z_\sigma^\dagger),\qquad
\lambda_{\max}=\frac{m_\sigma+|B_\sigma|}{2}.
\]

The exact relative **squared** distance is

\[
\epsilon_\sigma=1-\lambda_{\max}/n
=\frac{m_{-\sigma}}n+
\frac{m_\sigma-|B_\sigma|}{2n}.
\]

The first term detects the opposite orbital chirality; the second detects complex spin structure that cannot be represented by one real director times a phase. The nearest of the two chiral sectors maximizes lambda_max. This is not generally equivalent to choosing sign(C) for a badly nonchiral field.

A large chirality alone is inadequate: z_sigma can be a nonreal spin vector with m_chi=1. The provided nonunitary control has epsilon=0.297 despite unit chirality. The detector rejects it.

The director is an unoriented line. Its normalized eigenvalue gap is |B_sigma|/n. Degeneracy makes the returned eigenvector arbitrary. With n>0 the quality identities imply

\[
\sigma C/n\ge1-2\epsilon_\sigma,\quad
|B_\sigma|/n\ge1-2\epsilon_\sigma,\quad
|P|/n\ge1-2\epsilon_\sigma,
\]

where

\[
P=x\cdot x-y\cdot y=B_++B_-.
\]

Thus a tight residual simultaneously ensures local chirality, a nonvanishing charge-two scalar and a good director eigenvalue gap. These quality checks are related, not independent physical confirmations. All are retained to explain rejection. At n=0 the code marks the geometry undefined through its validity checks.

On 132 random matrices, direct reconstruction of the fitted matrix agrees with the residual formula within 7.3×10^−16. A search over 6,000 trial real directions per matrix never improves on the analytic maximum. The residual decomposition, quality bounds and the exact fixed-chirality discrete identity G=2 sum_nu |Delta_nu z_sigma|² also pass. Results are in [algebra_validation.json](results/algebra_validation.json).

### 2.2 Composite phase winding

On the ideal manifold P=a²=n exp(2i theta). For an oriented contour,

\[
w_P=\frac1{2\pi}\sum_i\operatorname{Arg}(P_{i+1}P_i^*),\qquad
q_{\rm phase}=w_P/2.
\]

This uses the **local** polynomial P(r), not its global spatial sum saved by existing scalar measurements. Those old sums cannot reconstruct spatial defects.

Outside the manifold, P remains a polynomial but need not represent the condensate phase. The synthetic `composite_zero` fixture has nonzero matrix density everywhere, a smooth full field and wP=1, but epsilon about 0.73 on the tested loop and no chiral orbital order. Its apparent half charge is rejected. A zero of P can be a cancellation between matrix components rather than a vortex or a zero of the full order parameter.

### 2.3 Spin-director holonomy without arbitrary sign choices

Let d_i be independently signed real eigenvectors along the contour. Compute

\[
h_d=\prod_i\operatorname{sign}(d_i\cdot d_{i+1})\in\{+1,-1\}.
\]

Every local eigenvector sign appears twice and cancels. With well-resolved nearby director lines, h_d=−1 means a continuously lifted director returns to −d_0. This is the RP2 loop parity. Near-orthogonal neighboring lines are ambiguous and cause rejection.

An arbitrary three-dimensional director path has no intrinsic positive or negative planar half-turn count. Assigning an azimuthal spin angle after projecting into a chosen plane can generate spurious windings or miss an out-of-plane escape. The implementation therefore uses parity, not a planar angle fit.

On a resolved chiral loop, h_d=(−1)^wP. The main signatures are:

| Net loop type | wP | Director parity | Projected phase charge |
|---|---:|---:|---:|
| HQV / anti-HQV | +1 / −1 | −1 | +1/2 / −1/2 |
| Integer / anti-integer | +2 / −2 | +1 | +1 / −1 |
| Neutral total | 0 | +1 | 0 |

The last row does not imply an empty interior. Nor does the integer row prove a single integer core: two same-sign HQVs have the same outer signature. The code's `integer_vortex` label means integer **net** charge on the accepted loop.

### 2.4 A complementary branch-independent phase lift

The fitted amplitude is a_i=d_i^T z_sigma,i. For s_ij=sign(d_i·d_j), define

\[
U_{ij}=s_{ij}\frac{a_j a_i^*}{|a_j a_i|},\qquad
q_{\rm lift}=\frac1{2\pi}\sum_i\operatorname{Arg}U_{i,i+1}.
\]

Local changes (d_i,a_i)→(−d_i,−a_i) cancel. On short smooth links this follows phase while transporting the nearby director lift. Since product(U)=h_d, q_lift is half integral by construction; its parity is **not an independent test** of spin/phase physics. Agreement with wP/2 and the selected B_sigma winding is useful for exposing aliasing and nonchiral distortions.

An elementary plaquette around a centered integer vortex has phase steps pi/2, while P steps are pi. Principal arguments can then report wP=1, suggesting a false half charge. In the demonstration q_lift=1, the composite link is flagged, and a larger contour correctly gives wP=2. Strict acceptance requires the composite and lifted diagnostics to be mutually resolved, rather than treating the smaller-loop answer as a new defect.

![Contour signatures and deliberately misleading controls](results/contour_signatures.png)

The fitted line shown for the nonchiral composite-zero control has no reliable director interpretation; its large residual and small eigenvalue gap reject the entire contour. The figure illustrates why apparent phase/director agreement without quality checks is insufficient.

## 3. Independent implementation and synthetic construction

### 3.1 Small implementation surface

[defects.py](defects.py) provides four principal operations:

- `local_geometry(A)` computes density, chirality, the nearest-sector fit, residual decomposition, director and eigenvalue gap from an array `(Nx,Ny,3,2)`.
- `analyze_loop(A, contour)` checks a nearest-neighbor loop and returns accepted/unresolved status, net charge, parity and explicit rejection reasons.
- `plaquette_map(A)` proposes elementary candidates from **both** composite winding and lifted phase circulation, so composite doubling does not silently hide integer vortices.
- `find_candidates(A)` groups adjacent seeds and checks several surrounding loops. It requires at least two accepted contours with the same nonzero charge, and rejects a cluster if other accepted tested contours disagree.

This is an exploratory contour/cluster detector, not a production population estimator. It does not force a one-to-one core count in dense or unresolved configurations. Its reported candidate location is a lattice seed anchor, not a sublattice-accurate fitted core center; integer-vortex seeds can be displaced by a lattice site.

The default quality rules are epsilon≤0.12, density>0.01 in the chosen units, |C|/n≥0.7, |P|/n and |B_sigma|/n≥0.5, director gap≥0.5, phase increments below 3pi/4, and director-line steps below pi/3. Here epsilon=0.12 means residual norm sqrt(0.12), not a 12% residual norm. The thresholds are explicit illustrative choices, not universal constants calibrated on a thermal ensemble. With the default residual bound, several other local bounds are redundant; keeping them makes failures understandable. Density scaling requires adjusting the absolute density floor.

Geometry caching is supported to avoid repeated eigendecompositions. Supplied geometry/director overrides are trusted expert inputs; arbitrary eigenvector sign flips are supported, but an arbitrary unrelated director field is not validated as an override.

### 3.2 Known vortex fields and branch cancellation

[synthetic.py](synthetic.py) builds kinematic full matrix fields with finite-width density cores. For an open-plane neutral HQV pair at z1 and z2, define

\[
u=e^{i[\arg(z-z_1)-\arg(z-z_2)]},\qquad
v=\left(\frac{u-1}{2i},0,\frac{u+1}{2}\right).
\]

Then v†v=1 and v·v=u. Setting A=f v e_sigma^T gives P=f²u and q=±1/2 at the two centers. Equivalently v=exp(i alpha)(sin alpha,0,cos alpha), with alpha half the difference of polar angles. The formula in u is single valued and never inserts a separate half-angle branch cut into the physical matrix. Arbitrary local changes of representation d→−d and theta→theta+pi leave A unchanged to numerical precision.

An integer vortex uses A=f exp(i phi) d0 e_sigma^T and gives P=f² exp(2i phi). A smooth phase/director texture without singularities is also included. These fields are constructed with arbitrary density scale n=1; they are not asserted to minimize the GL potential.

A genuinely periodic neutral pair is built from

\[
F(X,Y)=\sin X+i(\cos X+\cos Y-1),\quad u=F/|F|,
\]

with X,Y periodic coordinates shifted so the zeros lie between lattice sites. It has two zeros of opposite index at X=0, Y=pi/2 and 3pi/2. A periodic amplitude suppresses the two cores. The fixture's opposite boundary bonds are smooth; it is not an open atan2 field wrapped onto a torus. It is also not the minimum-energy torus Green-function solution.

Core profiles make the fields continuous but are not guaranteed continuously differentiable at the exact continuum center. The tests use surrounding lattice contours; no core-energy conclusion is drawn from these unrelaxed profiles.

![Synthetic full-matrix fields and quality maps](results/synthetic_textures.png)

## 4. Validation results and counterexamples

The complete tests use the standard-library unittest runner; no new package or CUDA dependency is required. [validation.json](results/validation.json), [synthetic_validation.csv](results/synthetic_validation.csv) and [automatic_detection.csv](results/automatic_detection.csv) record the results.

| Configuration or test | Expected and observed outcome |
|---|---|
| Uniform and smooth three-dimensional spin/phase textures | Zero enclosing charge; automatic finder produces no seeds |
| Integer vortex, both signs and both orbital chiralities | Resolved loops give wP=±2, q=±1, h=+1; automatic finder recovers the isolated defects |
| Integer vortex–antivortex pair | Two resolved opposite integer charges; outer contour has zero net charge |
| HQV–anti-HQV pair, both chiralities | Individual loops give wP=±1, q=±1/2, h=−1; outer contour has zero net charge |
| Periodic HQV pair | Two opposite half charges, including after shifting a core across a periodic seam |
| Two same-sign HQVs inside one loop | Outer integer net charge; demonstrates why an integer loop does not identify one core |
| Local eigenvector sign flips; global phase; arbitrary global SO(3) rotations | Charges and parity unchanged; residual invariant |
| Production complex64 field dtype | Integer/HQV/periodic charges, rotations and automatic counts pass at the existing numerical tolerances |
| Nested loops and lattice sizes 48, 96 and 192 | Known resolved HQV charges remain consistent |
| Elementary integer-core plaquette | Rejected for composite aliasing; larger loops recover integer charge |
| Chirality wall crossing the loop | Unresolved fixed-chirality loop, not an HQV |
| Smooth nonchiral field with odd P winding and nonzero density | Rejected by manifold/director/coherence tests |
| Unit chirality but nonreal spin vector | Rejected despite chirality=1 |
| Density void on the contour | Unresolved; a good outer contour can still be classified |
| Half phase without compensating spin rotation, or half spin without phase | Physical sign seam and inconsistent/large phase links; rejected |
| Filled/nonchiral HQV core | Exterior HQV signature survives while core manifold quality fails |
| Noncontractible periodic cycle | Rejected as a local enclosed-defect contour; it belongs to global cycle bookkeeping |
| Two HQVs only two lattice sites apart | Automatic routine returns unresolved clusters, demonstrating incompleteness rather than forcing counts |

For the default radius-six known HQV loops, wP=±1, director parity is −1, residuals are below 8×10^−16 and the largest composite phase step is about 0.193 radians. The corresponding isolated integer loop gives wP=2 and maximum composite step about 0.307 radians. The composite-zero control instead has wP=1 with epsilon≈0.734 and is unresolved. These numbers are specific to the saved fixtures and contours.

The automatic finder recovers one isolated integer vortex, both members of the neutral integer pair, both HQVs of the open pair and both of the periodic pair. The adversarial fixtures produce zero accepted defects, although some produce rejected seeds. Eight additional weak-noise automatic tests locate the two known HQVs within 1.1 lattice sites of their centers. This location tolerance does not establish a spatial-resolution guarantee for generic fields.

### Noise and tolerance sensitivity

Independent complex noise is added to all six matrix components. Its parameter is the relative RMS matrix perturbation under the declared component-noise convention. Each cell below tests two contours across ten seeds; thresholds and noise levels reuse fields, so the 560 evaluations are correlated.

| Relative noise RMS | Correctly accepted at epsilon≤0.12 | Accepted with wrong charge |
|---:|---:|---:|
| 0 | 20/20 | 0 |
| 0.03 | 20/20 | 0 |
| 0.06 | 20/20 | 0 |
| 0.12 | 20/20 | 0 |
| 0.20 | 20/20 | 0 |
| 0.30 | 4/20 | 0 |
| 0.50 | 0/20 | 0 |

Across all four residual thresholds and seven noise levels, no accepted loop has the wrong known charge. Acceptance falls under large perturbations, as a conservative classifier should. At RMS=0.3 it changes from 0/20 at epsilon≤0.08 to 4/20 at 0.12 and 18/20 at 0.2. Other validity gates remain fixed. These small toy samples support a proof of concept, not a probabilistic false-positive bound or a prescription for thermal-data thresholds. Detailed results are in [noise_sensitivity.csv](results/noise_sensitivity.csv).

![Resolution and acceptance sensitivity](results/resolution_and_noise.png)

## 5. Limits that must accompany any later defect population

**Lattice sampling:** integer rounding of a phase sum is not sufficient. High winding, short separations, sharply localized SOC textures or oscillations between sites can be missed even when sampled vertex quality is good. Phase and director step limits are practical safeguards, not a theorem excluding unsampled excursions. Increasing contour size can help while also enclosing additional defects; dense clusters must remain unresolved when those effects cannot be separated. A larger contour cannot restore information absent from every sampled loop.

**Amplitude and nonchiral fluctuations:** avoid a loop through a density minimum, a degenerate director, weak composite field or a chirality-domain region. A rejected loop is not a zero-defect observation. An approximate defect density must report invalid area, rejected clusters and sensitivity to thresholds. Conversely, a nonchiral or filled interior does not invalidate a good exterior loop. No complete classification of all defects in every possible 3×2 phase is claimed.

**Branch choices:** changing the representation of d and theta together is harmless. Changing only one creates a physical seam. An eigensolver's sign jump is not a measured wall; physical matrix and energy/anisotropy maps are needed to identify a connecting texture. The director's Z2 parity has no intrinsic sign, even though the phase circulation does.

**Boundaries:** plane-angle fixtures are explicitly open. For periodic fields, seam-crossing loops use modulo indexing consistently; noncontractible torus cycles are not interpreted as enclosing a disk. The periodic fixture has total raw composite plaquette winding zero and two opposite resolved seeds. General periodic snapshots need invalid-link checks and global-cycle bookkeeping as well. A single unbalanced vortex cannot simply be inserted into an otherwise ordinary periodic scalar-composite field without compensating structure or changed boundary data.

**Physical interpretation:** accepted q is the winding of the projected chiral phase, not a separate computation of the full-field Noether current or a demonstration of quantized hydrodynamic circulation in every nonchiral state. Kinematic tests do not establish equilibrium HQV abundance, pair energetics, BKT unbinding or SOC confinement. They are the prerequisites for examining those questions.

## 6. Stage 2 proposal: minimal Monte Carlo measurements

**No snapshot hook or new production measurement has been implemented.** The source already exposes `Runtime.snapshot()` in physical walker order and `ParallelTempering.snapshot()` in fixed temperature order. A minimal later change would occasionally save selected full-resolution fields at a declared sweep/exchange event, together with the exact temperature↔walker assignments at that same event. The full proposal is [FUTURE_METHODS.md](FUTURE_METHODS.md).

The first useful snapshot set should emphasize g_D=0 and small nonzero g_D in a chiral regime, with a few temperatures on either side of the relevant ordering region and some lower-temperature control. Selection and cadence should be declared independently of the instantaneous defect count. Save:

- Complex64 A with axis/component conventions, boundary conditions and lattice spacing convention.
- L, K, all five beta values, g_D, pressure, coefficient `reduced_temperature`, Monte Carlo temperature, source revision, initialization and independent seeds.
- Sweep, exchange round, exact observation order, temperature index/value, physical walker ID and assignment maps; do not infer identity from a downsampled history.
- Matching scalar measurements when practical, schema/dtype and integrity metadata.

One L=160 field is about 1.17 MiB; all 160 temperatures are 187.5 MiB before compression. Use sparse selected-temperature saves, not spatial downsampling, which can destroy the defect information. The first offline workflow should retain candidate maps, multiscale loop results and invalid regions alongside positive/negative half and integer net-charge clusters. It should not report a population from successful loops alone.

Snapshot autocorrelation must be assessed for the defect observables themselves. PT switching at fixed temperature is different from creation/annihilation inside a physical walker, and neither supplies real-time dynamics. Populations require stationarity, blocking and independent-preparation checks. Pair-distance histograms require phase-space/Jacobian, pairing, many-body and selection corrections before they can be interpreted as an interaction free energy.

## 7. Stage 3 proposal: a controlled interaction calculation

### 7.1 Define and preserve the object being measured

Start with a neutral pair whose separate reliable loops have wP=+1 and −1 and odd director parity, with an outer neutral loop. Vary its separation R at fixed boundary conditions. Initialize branch-independent matrix fields, then later relax **all twelve real components**, including amplitude, opposite chirality and nonunitary directions. There is no minimizer or separation-energy implementation in this Stage-1 directory.

Pinning only a core value does not preserve winding. A practical proposed calculation uses small core holes/annuli or localized pins carrying admissible boundary maps, with no connecting strip fixed in advance. State the constraints explicitly. A hole calculation measures exterior constrained-defect energy, not an optimized core energy. Vary pin radius/strength, phase offsets, spin paths, pair orientation and initial texture; check numerical derivatives and minimization tolerances before interpreting energies.

Most importantly, monitor several loops around each core and look for new defects, nonchiral screening annuli or topology loss. Inner pinned maps alone may leave a trivial field outside a small screening region. A flat E(R) after that loss would not demonstrate unconfined HQVs. Conversely, forcing the whole bulk to remain chiral could prohibit precisely the escape routes the full GL calculation is meant to test. Compare constrained metastable branches with released/unrestricted outcomes and report both.

### 7.2 Zero-SOC logarithmic reference

For a smooth fixed-density chiral field, the actual gradient reduces to

\[
F_{\rm grad}=\frac12\int d^2r\,[\rho_c(\nabla\theta)^2+
\rho_d|\nabla\hat d|^2],\qquad \rho_c=\rho_d=4Kn
\]

at the bare ansatz level. A neutral planar initialization with phase charges ±q and opposite spin-angle paths ±s has the reference interaction

\[
E_{+-}(R)\simeq C+2\pi(\rho_cq^2+\rho_ds^2)\ln(R/\xi),
\qquad\xi\ll R\ll L.
\]

For q=s=1/2 the coefficient is pi(rho_c+rho_d)/2. This explicitly includes spin energy. General three-dimensional spin relaxation and full-field core structure must be allowed to challenge the reference.

Two **same-sign mass HQVs** splitting an integer vortex are a different problem: with opposite planar spin paths their separation coefficient is pi(rho_d−rho_c)/2, cancelling at bare equal stiffness. They are not interchangeable with the attractive neutral-pair benchmark. This is one reason to specify both mass charge and initialization spin path, rather than referring only to “an HQV pair.”

### 7.3 Weak SOC: evidence required for confinement

Within the chiral ansatz the excess anisotropy is g_D n(1−d_z²). A frozen-density planar spin path across a wall gives

\[
\ell_D=\sqrt{2K/g_D},\qquad
\tau_{\rm spin,chiral}=4n\sqrt{2Kg_D}.
\]

These are restricted continuum channel benchmarks, not full-GL measurements or a proof of confinement. The locked fixed-chirality vacuum remains a connected U(1) circle: d-sign lifts are not two disconnected physical vacua. A freely imposed opposite-phase wall can take another phase path. The displayed spin-wall cost is relevant only under a retained joint HQV phase/director constraint; amplitude or nonchiral escape can alter it further. A discretized admissible trial wall, later evaluated with the same discrete constraints, would provide an upper bound on that constrained minimum, not a universal tension.

The helium HQV/soliton proposal of [Volovik and Salomaa, JETP 61, 986 (1985)](https://jetp.ras.ru/cgi-bin/dn/e_061_05_0986.pdf) motivates the question but has specific field and geometry assumptions. It does not establish the answer for this zero-field two-column Hamiltonian.

For surviving identifiable defects, compare a logarithm with a crossover and a possible large-distance form C+A ln(R/xi)+tau R. A genuine confinement interpretation requires a stable positive excess force after accounting for A/R, a physical localized connecting texture, and insensitivity to box size, boundary images, pin prescription and initialization. A director sign seam is not such a texture. Merely obtaining a positive fitted tau over a short curved interval is insufficient.

The desired hierarchy is xi_core≪ell_eff≪R≪L. At K=1, bare lengths are 44.7 for g_D=0.001 and 14.1 for 0.01. The old L≤160 sizes give little clean room beyond 44.7 while remaining far from images; failing to observe a linear regime there would not refute asymptotic confinement. The bare length is not automatically the relaxed or thermodynamic crossover length.

Subtract a homogeneous bulk reference of the **same coupling**, outer boundary and geometric domain, and account separately for pin/hole self energies. A homogeneous field cannot also obey nonuniform winding inner-boundary maps. Alternatively use E(R)−E(R0) with an unchanged pin prescription and test its residual pin dependence. SOC changes bulk energy and amplitude. Use separation differences and forces to reduce unknown core constants; test several L and orientations. On a torus, use the appropriate periodic/discrete Green-function reference instead of fitting an infinite-plane logarithm near R~L. Separate core overlap, anisotropy crossover and image-dominated ranges, and retain distinct metastable branches rather than averaging them into one curve.

### 7.4 Distinct physical questions

| Procedure | Quantity addressed | Interpretation limit |
|---|---|---|
| Zero-temperature constrained minimization | Classical minimum excess energy in a stated defect/boundary sector | Does not include thermal entropy or establish equilibrium populations |
| Finite-temperature constrained/umbrella sampling | Free-energy difference or potential of mean force with a declared measure | Requires reversible, equilibrated sampling and geometric corrections |
| Real-time evolution | Motion under a justified dynamical equation and time calibration | Neither Monte Carlo sweeps nor optimizer iterations are physical time |

Stages 2 and 3 should remain unimplemented until review of the definitions, scope and synthetic validation. The most useful immediate review questions are whether the conditional chiral-loop charge is the desired operational HQV definition, how much unresolved area is acceptable in thermal data, and which boundary/pinning problem should define the later interaction energy.

## 8. Reproduction and artifacts

From the project root:

```powershell
python -B exploration/he3_hqv_stage1/check_algebra.py
python -B exploration/he3_hqv_stage1/test_stage1.py
python -B exploration/he3_hqv_stage1/demo.py
```

The environment used Python 3.12.10, NumPy 2.4.6 and Matplotlib 3.11.2. No pytest installation is needed. The tests use unittest. No existing simulation datasets are required. The `-B` flag avoids incidental bytecode writes in production directories.

| Artifact | Purpose |
|---|---|
| [THEORY.md](THEORY.md) | Full derivations, exact bounds and topology caveats |
| [FUTURE_METHODS.md](FUTURE_METHODS.md) | Detailed Stage-2 and Stage-3 designs, with no implementation |
| [defects.py](defects.py) | Geometry, contour classification and candidate proof of concept |
| [synthetic.py](synthetic.py) | Known full-matrix configurations and adversarial fixtures |
| [test_stage1.py](test_stage1.py) | Reproducible tests and tolerance/noise tables |
| [check_algebra.py](check_algebra.py) | CPU-only checks against read-only production polynomial source |
| [demo.py](demo.py) | Three selected PNG figures and synthetic fixture archive |
| [validation.json](results/validation.json) | Test counts and pass/failure status |
| [algebra_validation.json](results/algebra_validation.json) | Source-polynomial, projection and symmetry checks |
| [synthetic_validation.csv](results/synthetic_validation.csv) | Known-contour charges, quality and rejection reasons |
| [automatic_detection.csv](results/automatic_detection.csv) | Candidate finding without supplied centers |
| [noise_sensitivity.csv](results/noise_sensitivity.csv) | Reproducible acceptance sensitivity |
| [demo_loops.json](results/demo_loops.json) | Figure-contour diagnostics and integer-resolution checks |
| [synthetic_fields.npz](results/synthetic_fields.npz) | Four explicitly synthetic fields with metadata; not MC snapshots |

Everything added for this task is inside `exploration/he3_hqv_stage1/`. No production algorithm, Hamiltonian, simulation script or original dataset has been modified. No Monte Carlo campaign, pair minimization or real-time evolution has been run.
