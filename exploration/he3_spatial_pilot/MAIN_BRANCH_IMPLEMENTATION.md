# Implementing spatial He3 measurements by hand in the main branch

This is an educational implementation guide. The working proof of concept is in this directory; **it has not been ported or merged into production**. Start with the definitions and independent reference tests, then add one observable at a time. The [pilot report](REPORT.md) contains the actual sampling results and their limitations.

## 1. What the measurements answer

Global chirality, the composite order parameter and helicity change in a small temperature interval. Their finite-size estimates do not yet distinguish one transition from two. Spatial measurements add information that cannot be reconstructed from the old global records:

| Field | Symmetry/content | Useful question |
|---|---|---|
| Local chirality kappa | Real; odd under time reversal/chirality reversal | Are chirality correlations short ranged or ordered? How does their scale change with L? |
| Local composite P | Complex; gauge charge two; invariant under real spin rotations | Are this channel's correlations algebraic or short ranged? Are its size dependence and helicity compatible? |
| Density-weighted spin tensor Q | Real, symmetric, traceless; gauge invariant | Is the spin-nematic correlation scale small compared with the box, or still important for interpreting HQVs and mass stiffness? |

No extracted director, chirality projection or fixed amplitude enters these definitions. They are meaningful for every full complex 3-by-2 matrix. Spin-nematic correlations are not magnetization. Their presence at finite L does not establish a separate spin transition.

## 2. Define the local fields directly from A

State memory has axes `(physical_walker, x, y, spin, orbital)`, dtype complex64. The orbital columns are x,y; the first two spatial indices follow the same x,y convention as the production gradients. Let N=L² and

\[
n(r)=\sum_{\mu,a}|A_{\mu a}(r)|^2.
\]

The first two fields are

\[
\kappa(r)=2\operatorname{Im}\sum_\mu A_{\mu x}^*(r)A_{\mu y}(r),\qquad
\mathcal P(r)=\sum_\mu[A_{\mu x}^2(r)-A_{\mu y}^2(r)].
\]

For x_mu=u_mu+i v_mu and y_mu=s_mu+i t_mu, implement

```text
kappa = 2 sum_mu (u_mu t_mu - v_mu s_mu)
Re(P) = sum_mu (u_mu² - v_mu² - s_mu² + t_mu²)
Im(P) = 2 sum_mu (u_mu v_mu - s_mu t_mu)
```

These formulas avoid complex temporaries. In the exploratory kernel, stored complex64 components are promoted to float64 **before** multiplication. The production global kernel currently performs local complex64 arithmetic and float64 reductions. Its zero modes therefore agree to the expected small rounding error, rather than bitwise equality. Either precision policy can be implemented deliberately; preserve a tolerance-based cross-check and record the choice.

Do not normalize kappa locally by n for the primary structure factor. The existing global normalized chirality is C/D_A, where C=sum kappa and D_A=sum n. It is not the spatial average of kappa/n. Raw kappa gives a transparent local field with C as its exact zero mode. Density-normalized variants can be studied separately.

For spin, first form

\[
M_{\mu\nu}(r)=\operatorname{Re}\sum_a A_{\mu a}(r)A_{\nu a}^*(r),\qquad
Q_{\mu\nu}=M_{\mu\nu}-\frac{n}{3}\delta_{\mu\nu}.
\]

M is real and symmetric and tr M=n, so Q is traceless. Off-diagonal entries are ordinary real dot products of the real/imaginary component vectors. There is no normalization by a small local amplitude.

### Use an orthonormal five-component representation

Define

\[
q_1=(Q_{xx}-Q_{yy})/\sqrt2,\quad
q_2=(Q_{xx}+Q_{yy}-2Q_{zz})/\sqrt6,
\]
\[
q_3=\sqrt2 Q_{xy},\quad q_4=\sqrt2 Q_{xz},\quad q_5=\sqrt2 Q_{yz}.
\]

The trace subtraction cancels from q1,q2. Thus the kernel can use Mxx,Myy,Mzz directly in these differences. The identity

\[
\sum_{a=1}^5|q_a(k)|^2
=\sum_{\mu,\nu}|Q_{\mu\nu}(k)|^2
=|Q_{xx}|^2+|Q_{yy}|^2+|Q_{zz}|^2
+2(|Q_{xy}|^2+|Q_{xz}|^2+|Q_{yz}|^2)
\]

holds at every momentum, including complex Fourier amplitudes. It gives the correct off-diagonal weights automatically. Simply summing five unweighted entries such as Qxx,Qyy,Qxy,Qxz,Qyz would omit Qzz and count off-diagonal entries incorrectly.

All components can be recovered:

\[
Q_{xx}=q_1/\sqrt2+q_2/\sqrt6,\quad
Q_{yy}=-q_1/\sqrt2+q_2/\sqrt6,\quad
Q_{zz}=-2q_2/\sqrt6,
\]

with Qxy=q3/sqrt2, etc. Under a global real spin rotation R, Q(k) transforms as R Q(k) R^T. Its Frobenius norm, and therefore the summed spectrum, is invariant on each configuration. Component powers need not be invariant. Their covariance becomes isotropic only in an appropriately sampled spin-isotropic ensemble.

A useful normalization check against the existing scalar `spin_alignment` is

\[
\sum_r Q_{zz}(r)=D_z-D_A/3=\frac23D_A\,\mathrm{spin\_alignment}.
\]

This equates a density-weighted extensive tensor entry with a density-weighted version of the existing normalized scalar. They are not the same observable.

## 3. Fourier convention and both signs of the complex field

Use the unnormalized transform

\[
O_k=\sum_{x,y}O(x,y)e^{-2\pi i(k_xx+k_yy)/L}.
\]

Store k=(0,0),(1,0),(-1,0),(0,1),(0,-1). Coordinates are integer sites and the listed wavevectors are integer Fourier indices. No factors of N or lattice spacing belong in the raw amplitudes.

For a real field, O_-k=O_k*. This applies to kappa and each q_a. For complex P it does not hold. A simple periodic test uses A(r)=A0[1+epsilon exp(2pi i x/L)] in a uniform chiral background with density n0. Then P=n0[1+epsilon exp(2pi i x/L)]², so P_(+1,0)=2 n0 epsilon N while P_(-1,0)=0. It is an effective test for a wrong sign or falsely imposed conjugacy.

One economical implementation Fourier transforms eight **real** fields:

```text
f = [kappa, Re(P), Im(P), q1, q2, q3, q4, q5]
```

Compute each at zero,+kx,+ky. For positive k, let a be the transform of Re(P) and b the transform of Im(P). Reconstruct

\[
P_{+k}=a+i b,\qquad P_{-k}=a^*+i b^*.
\]

The second expression is **not** (a+i b)*. It retains all the information in both signs without additional field passes. The exploratory [unpacking function](observables.py) implements precisely this operation.

The exact zero modes must satisfy kappa_0=C and P_0=phase_real+i phase_imag from production. These tests also catch coordinate, mapping and normalization errors.

## 4. Structure factors and finite-size correlation diagnostics

For each configuration calculate powers, then average:

\[
S_\kappa(k)=\frac{\langle|\kappa_k|^2\rangle}{N},\quad
S_P(k)=\frac{\langle|P_k|^2\rangle}{N},\quad
S_Q(k)=\frac{\langle\sum_a|q_{a,k}|^2\rangle}{N}.
\]

For a translationally invariant ensemble, S(k) is the transform of the site-averaged two-point correlation. For example G_P(r)=N^-1 sum_x <P(x+r)P*(x)>. At k=0, S_P(0)=N<p²>, where p=|P_0|/N. This connects the old global order parameter to the new spatial data. S(0)/N is the corresponding squared global magnitude, not an additional normalization of the local field.

Average the signs separately by direction:

\[
S_x=[S(+k_x)+S(-k_x)]/2,\quad
S_y=[S(+k_y)+S(-k_y)]/2,\quad \bar S_1=(S_x+S_y)/2.
\]

Keep Sx,Sy and the P sign asymmetries as diagnostics. A real field's two signs agree configuration by configuration; P's agree only under the appropriate equilibrium symmetries.

Define

\[
R=S(0)/\bar S_1,\qquad
\xi_2/L=\frac{\sqrt{R-1}}{2L\sin(\pi/L)}.
\]

Use the same formula with Sx or Sy for directional estimates. Average powers before forming this ratio; averaging single-configuration correlation lengths or directional lengths gives a different nonlinear estimator.

The formula follows the small-momentum form S(k)=S(0)[1-xi² k²+...] with lattice momentum 2 sin(pi/L). At a finite lowest momentum it is a conventional finite-size diagnostic, not an exact exponential decay length for every correlation function. In an algebraic phase it typically scales with L. In a finite chirality-ordered box the zero mode contains the ordering/Bragg contribution, so xi/L can exceed one substantially. The pilot's value near 4.3 at T=0.37 is not a correlation distance constrained by the box size.

### Disconnected contributions and invalid estimates

The default spectra are unconnected finite-volume moments. In an unbiased finite box with exact symmetry, <kappa_0>,<P_0>,<Q_0> vanish, but their squared fluctuations do not. In the ordered chirality sector, subtracting <|C|>² would change the estimator. Subtracting a noisy time-history mean can remove a slowly rotating or incompletely sampled zero mode. Preserve the raw amplitudes so both the unconnected quantity and a clearly labeled connected diagnostic can be evaluated.

For the tensor, a sample estimate of the zero-mode disconnected part is sum_a |<q_a,0>|²/N. The pilot saves its fraction of S_Q(0) as an orientation-sampling diagnostic. At nonzero SOC a physical mean tensor is allowed; a connected bulk spin correlation analysis would then require deliberate mean subtraction and potentially separate longitudinal/transverse channels.

Use explicit statuses:

| Condition | Return |
|---|---|
| Finite S0>=Sk>0 | Raw R and finite xi/L |
| S0<Sk | Raw R retained; xi/L=NaN; negative-radicand flag |
| Sk=0,S0>0 | Singular ratio; xi/L not reported as a finite physical length |
| Both zero or invalid inputs | Undefined flag |

The code also recognizes a denominator consisting only of double-precision cancellation noise, using a declared threshold 256 epsilon_machine² max(|S0|,|Sk|). It does not clamp a negative radicand. Near a hot, nearly flat spectrum, R-1 and its error are often more informative than a square root. If some jackknife deletes cross R=1, do not discard them and compute a deceptively small Gaussian xi error; keep the valid-delete count and report the ratio uncertainty instead.

## 5. GPU organization used in the prototype

Read [gpu_measure.py](gpu_measure.py) after understanding the NumPy reference in [observables.py](observables.py).

The GPU reads the current full field; it never changes it. Two kernels use the existing default stream. No stream changes, RNG calls, altered Metropolis decisions or replacement PT updates are introduced.

```text
allocate once:
    trig[L,2] = [cos(2pi x/L), -sin(2pi x/L)]
    partial[R,3,ceil(N/1024),16] float64
    output[R,8,3,2] float64

first pass, one block per (walker, mode, site tile):
    128 threads
    each thread visits at most 8 sites of a 1024-site tile
    load each site's twelve real A components
    construct the eight real fields
    multiply each by the mode cosine and negative sine
    accumulate sixteen real numbers per thread
    reduce in shared memory and write one tile sum

second pass, one block per (walker, mode):
    reduce that walker's tile sums
    destination = walker_to_temperature[walker]
    write sixteen final real numbers to the temperature slot

copy only the small output to the host
reconstruct both signs of complex P
```

There are three reduced modes: zero,+kx,+ky. Trigonometric values are precomputed once rather than evaluated at every site. Shared memory is 16 x 128 x 8 = 16,384 bytes per block. Tiling supplies enough blocks as L grows; the second pass is small and avoids floating-point atomics or nondeterministic multi-block accumulation into the final output.

This design recomputes local fields for the three modes, favoring simple reductions over a large per-thread accumulator or a persistent local-field workspace. It is practical at the tested cadence. Consumer-GPU FP64 arithmetic and repeated reads make it slower per event than the existing scalar measurement, but its measured total cadence cost is small. A one-block-per-walker design can become poorly occupied or have long serial site loops at larger L; the tiled prototype avoids that limitation.

The raw output is only 384 bytes per walker. With 47 temperatures it transfers 18,048 bytes per event, versus about 9.24 MB for all L64 fields. At L256 the full field transfer would be about 148 MB, while this output size stays the same. The derived amplitude array contains redundant conjugate channels for the real fields to keep the saved schema explicit.

## 6. PT mapping and synchronized energy

Physical field slots and site RNG states remain in walker order. PT exchanges **temperature assignments**, not field arrays. Thus the relevant map for measurement output is walker_to_temperature; temperature_to_walker is its inverse and is useful for selective snapshots.

Use the wrapper's current device map in the reduction. Do not use a stale host map, confuse the two inverses, or attach the initial temperature index to a moving walker.

The safe observation event is:

```text
perform all four local update colors
if exchange is scheduled:
    compute current full extensive H per physical walker
    perform the unchanged exchange matching and update both maps
measure global moments using the current map
measure spatial amplitudes using the same map
do not advance sweeps between these two measurements
```

The default stream orders the kernels. At multiples of 250 sweeps, with exchange interval 5, the current PT energy cache is valid. Exchanges change assignments without changing H of the physical configuration. `pt.measure()` therefore supplies the synchronized full extensive H together with the correct temperature-ordered scalar channels.

Do not reuse an energy from a previous sweep or substitute energy density in exp[-Delta(beta)H]. Save H at every spatial event. Reweight each underlying |O_k|² moment and each current/curvature moment before constructing ratios or helicity. PT temperatures are correlated; time blocks must be deleted jointly across all temperatures and measurements. Different SOC Hamiltonians cannot be combined by ordinary temperature reweighting.

## 7. A compact saved-data schema

The experimental NPZ records contain:

| Array | Shape / meaning |
|---|---|
| temperatures | (R,), nominal fixed ladder |
| global_sweeps | (Ng,), absolute completed sweeps, including burn-in |
| global_values | (Ng,R,10), production global channel order |
| spatial_sweeps | (Ns,), absolute spatial observation sweeps |
| amplitudes | (Ns,R,7,5), complex128; fields and momenta explicitly labeled |
| spatial_global_values | (Ns,R,10), exactly matching global event, including extensive H |
| temperature_to_walker | (Ns,R), map at each spatial event |
| production_start_sweep | Scalar; production uses sweep strictly greater than this |
| walker_history_indices/sweeps | Sampled PT transport in physical-walker order |

The JSON manifest records coefficients, nominal/actual float32 inverse temperatures, seeds, initialization, timing, source hashes/revision and PT counters at burn-in and final time. Snapshots have their own sweep, selected temperature IDs and physical walker IDs.

For a manual production implementation, a smaller nonredundant set has **37 additional real channels** if the existing C,P zero modes are reused: kappa's two complex nonzero modes (4), P's four complex signed modes (8), five real tensor zero modes (5), and ten complex positive tensor modes (20). The same information is also represented by eight real fields at zero,+kx,+ky: 40 nonredundant real values, three already in the existing globals. The prototype's 48-real raw buffer includes eight zero imaginary slots for simpler reduction code.

Retaining amplitudes permits connected diagnostics and component covariance checks. Saving only ensemble-averaged spectra would prevent blocking and later joint reweighting. Saving per-event spectra alone would retain some useful statistics but lose phase, mean-vector and sign/conjugacy checks.

## 8. Minimal manual implementation sequence

The simplest first main-branch version uses one common cadence, e.g. 250 sweeps, for old and new channels. Existing `Simulation`, `Runtime`, PT and `save_runs` already support named measurement arrays at a common cadence. There is no need to alter their update or exchange algorithms.

| Step | Files to consider later | Independent check before proceeding |
|---|---|---|
| 1 | New test/reference file only | Local kappa,P,Q and analytic uniform/random fields in NumPy |
| 2 | A small `models/he3/measure_spatial.py` | GPU zero modes versus NumPy and existing globals |
| 3 | Same measurement module | Positive k modes; periodic wave catches sign and P conjugacy errors |
| 4 | Same module | Five tensor components and full nine-entry Frobenius contraction; random SO(3) rotations |
| 5 | `models/he3/model.py` | Add named outputs; pass through the existing optional temperature map; reuse buffers for the active shape |
| 6 | `models/he3/analysis.py` or a small dedicated analysis helper | Ensemble powers, ratios, flags and joint blocked errors |
| 7 | `main/he3.py` settings and metadata | A short actual PT smoke; zero-mode checks after exchanges and nontrivial permutation |
| 8 | Existing workflow tests/documentation | Save/load round trip, labels, cadence, extensive H and old-file compatibility |

`utils/data_io.py` already writes arbitrary named arrays and manifests. At a common cadence, extending the names should not require changing its storage machinery. Preserve compatibility with old files that lack the spatial channels: new spatial analysis should be optional and clearly report missing data.

Start with correct allocation and names; then cache the active-shape workspace on the model or a small measurement object so each observation does not allocate partial arrays. Validate invalidation/reallocation when L or R changes. Do not cache a temperature map by value: PT updates it in place.

After the common-cadence version works, choose whether the additional engineering for global every 50 and spatial every 250 is useful. The clean small-run alternative is a sidecar observation loop like [pilot.py](pilot.py), keeping separate sweep coordinates and synchronized copies. A dual-cadence production engine would need explicit separate histories and serialization metadata; do not append arrays with different time lengths while pretending they share one sample axis. The production sampler and exchange logic need no change in either design.

No main-branch patch or merge is supplied by this investigation. Work through these stages manually and retain the independent tests at each stage.

## 9. Tests to keep

The executable [validation](validate.py) passes seventeen groups on the actual local GPU. The most useful lasting tests are:

1. Uniform chiral A: kappa0=P0=nN; S_Q(0)=2n²N/3; all nonzero modes absent within precision.
2. Uniform nonchiral/planar A: verify kappa=0 and allow P to vanish even with nonzero matrix norm.
3. Periodic complex modulation: P+ and P- differ as derived above.
4. A known density sinusoid: amplitude n epsilon N/2 at each sign.
5. Spatial spin texture and random full matrices: use independent `np.fft.fftn` and the full tensor contraction.
6. Global SO(3) and U(1): invariant scalar spectra, P amplitude transforms with exp(2i theta).
7. Exchange x/y and translate the field: modes permute and acquire the expected translation phase.
8. A nontrivial walker-to-temperature permutation and real PT exchanges: output must follow the current assignments.
9. Read-only behavior: measurements leave GPU fields unchanged.
10. R<1, zero field and a uniform singular field: preserve invalid flags instead of clamping.

Use scale-aware absolute and relative tolerances. Compare GPU and CPU on the **same stored complex64 input**; comparing a rotated high-precision field against its separately rounded complex64 version introduces a legitimate extra rounding difference. Avoid testing a tiny absent mode with relative tolerance alone.

## 10. Performance and statistical use

On the local RTX 4070 SUPER, with 47 replicas and warm JIT:

| L | Existing cached globals / event | New GPU reductions / event | New measurement incl. small host read | Meter workspace |
|---:|---:|---:|---:|---:|
| 64 | 0.168 ms | 0.343 ms | 0.439 ms | 0.091 MB |
| 128 | 0.603 ms | 1.106 ms | 1.210 ms | 0.309 MB |
| 160 | 0.947 ms | 1.910 ms | 2.143 ms | 0.472 MB |
| 256 | 2.372 ms | 4.756 ms | 4.953 ms | 1.177 MB |

The repeated L64 update/measurement comparison gives about 0.98% overhead at spatial cadence 250. L128's short comparison gives about 0.23%, while the isolated-event cadence estimate is about 0.71%; the difference is within the limits of short wall-clock measurements and clock/desktop variation. Treat these as feasibility measurements rather than precise speedups. L160/L256 entries are **measurement-only tests**, not simulated thermodynamic histories. Total campaign memory/time also scales with the number of temperatures, RNG/state arrays and I/O, not just this small meter.

For statistics, average per-event powers first. Use joint time-block jackknife/bootstrap across all PT temperatures and sectors. The pilot uses 5,000-sweep blocks and checks 1,250–20,000, with only five blocks at the longest setting. A short ACF can miss slow spin/sector modes; even apparently high ESS and reasonable swap rates do not establish equilibrium. Compare independent hot/ordered starts, halves, component anisotropy and actual endpoint transport.

For phase-diagram work, add sizes with useful ratios, compare xi_kappa/L and xi_P/L flows and inspect real-space shapes. Do not import an XY universal crossing value without validating the effective field and conventions. BKT logarithmic corrections and finite-volume effects matter even in simpler models ([Hasenbusch](https://arxiv.org/abs/cond-mat/0502556)). The spin-nematic scale supplies an independent check on assumptions used in a mass-only HQV jump. A spin-disordered paired-superfluid theory is a useful comparison, not a demonstrated reduction of this GL model ([Mukerjee, Xu and Moore](https://arxiv.org/abs/cond-mat/0605102)).

At fixed L, these measurements can reveal correlation scales and test a proposed effective description. Establishing a thermodynamic intermediate phase requires two sectors with different persistent large-L behavior over a nonzero temperature interval. One L64 pilot cannot establish that interval. Spin-twist stiffness and the full charge-two orbital tensor remain future possibilities requiring their own derivations and tests.
