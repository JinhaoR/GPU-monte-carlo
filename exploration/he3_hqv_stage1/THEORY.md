# Mathematical construction of a qualified HQV signature

This note derives diagnostics from the actual six-complex-component field. It does not identify defects in old scalar-only datasets, assume the field remains chiral, or implement a constrained minimization. Source definitions were checked against [update.py](../../models/he3/update.py), [measure.py](../../models/he3/measure.py), and [model.py](../../models/he3/model.py). The source is read only.

## 1. Symmetries, manifold and scope of topology

Write the two orbital columns as complex spin vectors x=A_x and y=A_y, with total local norm n=x†x+y†y. The implemented local energy is −2n+sum beta_j I_j+g_D D, and

\[
D=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\operatorname{Re}(A_{xy}A_{yx}^*)-\frac23n.
\]

Every term in the Hamiltonian, including the actual forward-difference gradient, is invariant under A→exp(i alpha)A. At g_D=0 it is also invariant under real spin rotations A→R A, R in SO(3). Independent spin SO(3) is broken by the displayed dipole term for g_D>0. Complex conjugation A→A* is an exact symmetry at all couplings: norm squares and real contractions remain unchanged. It reverses chirality and phase winding. Continuum simultaneous spatial/orbital rotations should not be assumed exact for the forward-difference lattice stencil.

The fixed-chirality unitary rank-one chiral fields can be written

\[
A=a\,\hat d\,e_\sigma^T,\qquad
e_\sigma=\frac{(1,i\sigma)}{\sqrt2},\quad
a=\sqrt n e^{i\theta},\quad \hat d\in S^2,\quad \sigma=\pm1.
\]

The parametrizations (d,theta) and (−d,theta+pi) give the same physical matrix. At fixed positive amplitude and fixed chirality the manifold is

\[
\mathcal M_\sigma=(S^2\times U(1))/\mathbb Z_2.
\]

The two chirality choices are disconnected within this ideal manifold. Crossing between them in the full GL field is allowed and produces a region that must not be assigned a single fixed-chirality defect charge.

The universal cover S²×R has deck transformation (d,theta)→(−d,theta+pi). Its deck group is Z, so pi_1(M_sigma)=Z, **not** Z×Z2. An integer k labels the combined loop: delta theta=k pi and the lifted director ends at (−1)^k times its initial value. The director parity is tied to phase parity and is not an independent topological charge. A primary derivation of the same quotient topology appears in the distinct polar-spinor model of [Mukerjee, Xu and Moore, Phys. Rev. Lett. 97, 120406 (2006)](https://arxiv.org/pdf/cond-mat/0605102), Eq. 3 and the discussion following it. That shared topology does not map the entire GL Hamiltonian to the spinor model.

The unconstrained nonzero matrix space C^6 minus the origin retracts to S^11 and is simply connected. This does **not** invalidate the usual classification of defects by their ordered far field or fixed boundary. It permits an interior field to fill a core or escape the chiral manifold without making the total amplitude zero. A classification of an arbitrary numerical contour is therefore conditional on that contour lying in a well-defined neighborhood of M_sigma. Core amplitude zeros are sufficient examples of singular cores, not a required HQV signature in the full matrix model.

## 2. An exact nearest-manifold residual, rather than a chiral assumption

Resolve the local field into its two chiral orbital combinations:

\[
z_\sigma=Ae_\sigma^*=(x-i\sigma y)/\sqrt2.
\]

The transformation is unitary in orbital space, hence

\[
n=\|z_+\|^2+\|z_-\|^2,\qquad
C=2\operatorname{Im}(x^\dagger y)=\|z_+\|^2-\|z_-\|^2.
\]

For a fixed sigma, minimize the full matrix distance over a complex amplitude b and a real unit director d:

\[
\min_{b\in\mathbb C,\,d\in S^2}\|A-b d e_\sigma^T\|_F^2.
\]

For fixed d, b=d^T z_sigma. The captured squared norm is |d^T z_sigma|²=d^T N_sigma d, where

\[
N_\sigma=\operatorname{Re}(z_\sigma z_\sigma^\dagger)
=\operatorname{Re}z_\sigma\operatorname{Re}z_\sigma^T
+\operatorname{Im}z_\sigma\operatorname{Im}z_\sigma^T.
\]

Thus a principal real eigenvector of N_sigma supplies d, and

\[
\epsilon_\sigma=\frac{\min\|A-b d e_\sigma^T\|_F^2}{n}
=1-\frac{\lambda_{\max}(N_\sigma)}{n}.
\]

This is a **normalized squared** Frobenius residual, not the residual norm. The residual norm is sqrt(epsilon_sigma). The nearest point in the union of both chiral sectors uses the sigma with the largest captured eigenvalue. Far from the manifold this choice need not equal sign(C); for example, one large but nonunitary chiral vector can have less captured real-director weight than a smaller unitary vector in the other orbital sector. Near a good chiral field the choices agree.

If m_sigma=||z_sigma||² and B_sigma=z_sigma·z_sigma without complex conjugation, the two possible nonzero eigenvalues are

\[
\lambda_\pm=\frac{m_\sigma\pm|B_\sigma|}{2}.
\]

The third eigenvalue is zero. Therefore

\[
\epsilon_\sigma=
\frac{\|z_{-\sigma}\|^2}{n}
+\frac{\|z_\sigma\|^2-|z_\sigma\cdot z_\sigma|}{2n}.
\]

The first contribution measures wrong orbital chirality. The second measures failure of the selected complex spin vector to be a phase times a real vector. They vanish together precisely on the selected unitary rank-one chiral manifold, at n>0. This decomposed diagnostic is more informative than a large chirality alone.

The director is unique only as a line, and only when the principal eigenvalue is separated. Its normalized eigengap is |B_sigma|/n. If B_sigma=0, real and imaginary spin vectors can have equal norm and orthogonal directions; a phase and one real director cannot be reliably extracted. The eigensolver returning one vector in a degenerate plane is not physical evidence of a director.

These constructions are covariant under global real spin rotations and global phase changes. The residual and eigengap are invariant. The eigenvector's arbitrary sign is handled explicitly below.

## 3. Composite field and exact quality bounds

The spatially local counterpart of the existing saved charge-two channel is

\[
P=x\cdot x-y\cdot y=B_++B_-.
\]

On the ideal manifold, P=a²=n exp(2i theta). Its phase therefore winds by twice the mass-phase angle. At a generic full GL field the same polynomial remains well defined, but it need not represent the phase of an ordered chiral condensate.

The projection residual yields useful exact bounds. Since |B_−sigma|≤||z_−sigma||² and m_sigma+|B_sigma|=2n(1−epsilon_sigma),

\[
\frac{|P|}{n}\ge 1-2\epsilon_\sigma,\qquad
\frac{\sigma C}{n}\ge 1-2\epsilon_\sigma,\qquad
\frac{|B_\sigma|}{n}\ge 1-2\epsilon_\sigma,
\]

whenever the right side is positive. Thus a tight residual bound simultaneously gives local chirality, nonvanishing composite field and a well-separated director. Separate quality maps remain useful to explain rejection and finite precision. The bounds are local; they do not resolve unsampled large changes between lattice sites.

The converse fails. A real field with only x nonzero has P=n but zero chirality and epsilon=1/2. A smooth field x=(1+z/2,0,0), y=(1−z/2,0,0), where z is a smooth complex coordinate on a disk, has P=2z and n=2+|z|²/2>0. Its composite field has a simple zero and unit winding even though the matrix is smooth and its real spin director is constant. A detector that only counts zeros or phase windings of P would incorrectly call this an HQV. The projection, chirality and director tests reject that interpretation.

## 4. Contour charges and branch-independent transport

Let the ordered list of vertices i=0,...,M−1 form a closed contour with a chosen positive orientation. Every vertex must have adequate density and the same selected chirality sigma. The residual must be acceptably small at every vertex, with a nondegenerate director and nonzero P. A low-density or nonchiral core inside the contour is allowed; a low-quality point **on** the contour makes the assigned charge unresolved.

The composite winding is

\[
w_P=\frac1{2\pi}\sum_i\operatorname{Arg}(P_{i+1}P_i^*),
\]

where principal arguments lie in (−pi,pi]. It should be near an integer numerically. This is a discrete winding of the shortest interpolating phase path, not a proof that a highly oscillatory continuum field has that winding. Links near the branch limit pi are flagged rather than rounded confidently.

Extract the real unit eigenvectors d_i without attempting to fix a global sign. The line-field holonomy is

\[
h_d=\prod_i\operatorname{sign}(d_i\cdot d_{i+1})\in\{+1,-1\}.
\]

Every arbitrary eigenvector sign occurs twice in the product, so h_d is invariant to independent local sign choices. A dot product near zero makes the shortest lift ambiguous and must cause rejection or refinement. The nontrivial value h_d=−1 means a continuous short-link lift returns to −d_0. It is the intrinsic RP² parity, not a signed half-turn count in an arbitrarily chosen plane. A three-dimensional director can leave that plane; its general loop has no well-defined signed planar spin angle.

For a smooth ideal chiral contour,

\[
h_d=(-1)^{w_P},\qquad q_{\rm phase}=w_P/2.
\]

Odd composite winding and odd director parity give an HQV-type combined loop. A unit HQV has w_P=+1 and q=+1/2; its anti-HQV has w_P=−1 and q=−1/2. An integer phase vortex has w_P=±2 and h_d=+1. Higher integers are possible; the contour measures net enclosed charge and does not count individual cores. A contour surrounding an HQV and anti-HQV has net w_P=0 and h_d=+1 despite containing two defects.

There is a useful complementary mass-phase link. Let a_i=d_i·z_sigma,i and s_ij=sign(d_i·d_j). Define

\[
U_{ij}=s_{ij}\frac{a_j a_i^*}{|a_j a_i|},\qquad
q_{\rm lift}=\frac1{2\pi}\sum_i\operatorname{Arg}U_{i,i+1}.
\]

Under (d_i,a_i)→(−d_i,−a_i), both affected sign and amplitude factors cancel. On sufficiently short smooth links this transports the actual phase after choosing the nearby director lift. Product(U)=h_d, so q_lift is half integral. Its parity relation to h_d is an **algebraic identity by construction**, not an independent physical confirmation. Agreement of 2q_lift with w_P and the selected B_sigma winding supplies a discretization/near-manifold consistency check.

Composite doubling is a particular aliasing hazard: an integer vortex centered in an elementary square can have true phase increments pi/2, but P increments pi. Principal-argument P winding can then be unstable or aliased. The lifted link can diagnose the charge on that grid, but a strict composite-based acceptance should enlarge/refine the contour before reporting a verified integer charge. All nonzero loops should be checked on more than one surrounding contour where possible.

## 5. Distinguishing other features

| Observed feature | What can be assigned safely |
|---|---|
| w_P=±1, h_d=−1, same chirality, small residual and resolved links | HQV-type net loop charge ±1/2 within the qualified chiral neighborhood |
| w_P=±2, h_d=+1, same quality conditions | Integer net phase charge; several unresolved HQVs can produce the same enclosing charge |
| Chirality changes sign along the loop | Chirality-domain or nonchiral crossing; no single-M_sigma HQV label |
| P vanishes but n stays finite | Composite cancellation is possible; inspect full tensor/projection rather than infer an amplitude core |
| n is small on the contour | Order direction/phase undefined there; move or enlarge the contour |
| Principal director eigenvalue is degenerate | Director unresolved; the eigensolver's axis is arbitrary |
| Composite/director/lift tests disagree | Nonchiral excursion, phase aliasing or unresolved link texture; not a new defect species by default |
| Zero net charge on a large contour | No net enclosed charge; neutral pairs may remain inside |
| A planar director winding assigned a positive/negative sign | That sign is meaningful only with an explicitly retained plane or extra physical constraint |

A chirality wall, skyrmion-like spin texture, smooth zero of P, or nonunitary core need not be an HQV. Conversely an HQV may have a nonchiral or core-filled interior while its surrounding loop is unambiguous. The numerical output should retain accepted, rejected and unresolved counts and the reasons, rather than silently remove difficult regions from a population estimate.

## 6. SOC and boundaries: conditional confinement benchmark

On the ideal chiral manifold the exact dipole density is D=n(1/3−d_z²). Positive g_D makes d=±z the preferred directions; their sign identification with a pi phase shift leaves an ordinary U(1) vacuum circle in each chirality sector. Its ordinary integer vortex loops map to **even** k in M_sigma. An odd HQV loop must therefore depart from that locked vacuum somewhere. A connecting spin texture or an alternative full-GL escape can carry the cost.

The original helium proposal by [Volovik and Salomaa, Sov. Phys. JETP 61, 986 (1985)](https://jetp.ras.ru/cgi-bin/dn/e_061_05_0986.pdf) describes half-circulation combined spin defects and dipole solitons, with explicit magnetic-field and geometry qualifications. It motivates the question, not a confinement conclusion for this two-column model.

For constant chirality, even the actual discrete gradient simplifies exactly for A=w e_sigma^T to

\[
G=2\sum_{\nu=x,y}\|\Delta_\nu w\|^2.
\]

For a smooth fixed-amplitude unitary w=sqrt(n) exp(i theta)d, the continuum energy is 2Kn[(grad theta)²+|grad d|²]. Thus bare phase and spin stiffnesses are both 4Kn. A frozen-amplitude one-dimensional chiral spin kink d=(sin alpha,0,cos alpha) has

\[
E_{\rm wall}/\ell=\int dx\,[2Kn(\alpha')^2+g_Dn\sin^2\alpha],
\]

width sqrt(2K/g_D) and minimum tension 4n sqrt(2Kg_D) for alpha:0→pi within that variational class. This is a useful restricted benchmark, **not the measured or proven full-GL string tension**. Allowing amplitude changes, both chiral components, nonunitarity and other tensor configurations can lower or alter the string cost. A future controlled-separation calculation must check the loop class survives its relaxation, rather than merely measure the energy of an unconstrained pair that can annihilate or escape.

An analytic polar-angle vortex or pair on a plane is not automatically periodic. A branch cut in theta and d is harmless only when their signs cancel in A itself. A seam introduced by wrapping an open-plane construction onto a torus is a physical numerical artifact. Periodic data require periodic-safe configurations, explicit link treatment across the boundary, zero total composite vorticity in a resolved nonsingular periodic field, and accounting for the two noncontractible cycle windings. A single isolated vortex needs an open/fixed-boundary problem or compensating defects; applying periodic rolls to a nonperiodic trial field is not a boundary condition.

All these signatures are geometrical diagnostics of static configurations. Monte Carlo time is a sampling coordinate; minimization iterations are an optimization coordinate; neither supplies real-time vortex dynamics.
