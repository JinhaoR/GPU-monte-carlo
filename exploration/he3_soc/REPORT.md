# Thin-film helium-3 with dipole spin–orbit coupling: an independent investigation

**Dataset:** `soc_scan_02`, pressure 20 bar, four couplings and four sizes.  
**Analysis date:** 29 October 2026.  
**Scope:** existing scalar measurements and saved parallel-tempering histories; no new Monte Carlo runs. All numerical results below concern the implemented dimensionless lattice model.

## Executive assessment and ranked findings

The most consequential result is a sampling diagnosis with a specific physical implication: **the strong-SOC data contain two long-lived spin/chirality populations, but their relative weights are not equilibrated.** Their apparent low-temperature transition is entangled with a slowly depleting inventory of configurations. A separate, substantial common-phase stiffness persists in the almost nonchiral population. This deserves investigation as competing order, but the present trajectories cannot establish an equilibrium phase diagram there.

### Robust numerical observations

1. **Strong SOC produces almost conserved configuration branches.** During production, the numbers of high-spin-alignment configurations across the complete temperature ladder change only from 33→31, 44→43, 50→49 and 55→53 for increasing size. All six resolved conversions are high→low; no reverse conversion is observed. Tracking physical walker identities confirms that ordinary switching at a fixed temperature is mostly PT exchanging different branches. This directly challenges an equilibrium reading of the apparent transition (§5).
2. **Extreme negative chirality Binders are rare-event moment ratios.** The minima range from −485 to −2548. Only 13, 5, 2 and 2 records out of 60,000 contribute more than 99.96% of the fourth moment. Removing those records as an influence check leaves a Binder near zero. These are valid values of the measured moment ratio, with an unresolved tail probability; their magnitude does not establish a first-order transition (§5.3).
3. **Weak SOC changes spin alignment much more than it changes the main thermodynamic temperature scale.** At the lowest saved temperature and L=160, mean Qzz changes from −0.0053(61) at zero SOC to 0.57526(24) at 0.001 and 0.624496(38) at 0.01. Density and helicity change much less. Errors in parentheses are within-history block errors, not independent-run uncertainties (§4).
4. **The weak-SOC helicity size dependence favors the tested half-charge WM form over the tested integer-charge form.** With all four sizes, reduced chi-squares are 1.50, 0.365 and 0.295 for q=1/2, versus 108, 30.0 and 5.62 for q=1. This is evidence about a finite-size functional form, not identification of HQVs or proof of BKT criticality (§6).
5. **At strong SOC, substantial stiffness belongs to the low-Q, almost nonchiral population.** At T=0.442331, all four files have zero high-Q records but helicity about 1.18–1.19. Including or excluding the small measured-current-mean correction does not explain that stiffness. The composite phase observable can vanish in a competing internal configuration even while the common-phase stiffness remains finite (§5.4).

### Promising but unresolved phenomena

1. **A competing nonchiral superfluid configuration.** The actual SOC quadratic form favors an antisymmetric planar field more strongly per unit density than a chiral z-row field. A deterministic onsite calculation finds both to be local minima at g_D=0.1. This provides a concrete alternative to interpreting everything through chiral HQVs, but global scalar measurements cannot identify the sampled low-Q tensor configuration (§2.3).
2. **Strong spin locking with a modest shift of symmetry-even observables.** The q=1/2 WM diagnostic shifts by approximately 0.000354 and 0.003598 at g_D=0.001 and 0.01. Their ratios to coupling are similar. This is compatible with smooth changes in accessible bulk curves while the spin sector changes markedly; it does not determine whether weak SOC is singular at infinite distance (§4, §6).
3. **A higher-temperature stiffness loss at strong SOC.** The q=1 form fits near 0.46376 and a heat-capacity maximum lies near 0.46675, whereas the spin/chirality population front is near 0.356–0.371. These distinguish measured temperature scales in the sampled ensemble, but their equilibrium meanings remain open (§5–6).

### Potential artifacts and convergence problems

1. **The strong-SOC spin front cannot safely be extrapolated as an equilibrium transition.** Its location depends on the sampled branch population and moves while that population depletes.
2. **High swap acceptance conceals poor configuration-sector transport.** Strong-SOC mean acceptance is 46–76%, but there are no exact completed round trips. Sampled one-way passages do occur, all cold→hot. The largest flow obstruction near the spin front is distinct from the global acceptance minimum (§3.1).
3. **Weak-SOC results also have unresolved slow sectors.** The largest runs complete relatively few PT loops. At g_D=0.001, L=160, signed chirality drifts coherently at low temperature even while energy and absolute chirality are much steadier. A short autocorrelation estimate misses this slow symmetry-sector tail (§3.3).
4. **Small fitted statistical errors omit larger analysis and equilibration uncertainties.** Changing interpolation moves weak-SOC WM temperatures by roughly 0.000025–0.000079; thinning the ladder or dropping a size can shift them further. At strong SOC both charge forms fit well at different temperatures (§6.2).

## 1. Data, provenance and methods

### 1.1 What is actually available

All 16 requested files are present under `data/he3/P=20_gd={0,0.001,0.01,0.1}_gaussian/L={64,96,128,160}.npz`. Each contains ten real scalar measurement channels, arrays of shape (60,000,160), a JSON manifest, and histories of every physical walker's temperature assignment. No spatial fields, vortex configurations, separate SOC energies or spin correlation functions are saved.

The manifests agree on 2,000,000 thermalization sweeps, 3,000,000 production sweeps, measurements every 50 sweeps, and 160 temperatures spanning 0.33–0.47. **The actual PT exchange interval is 5 sweeps, rather than the 10 in the supplied summary.** Histories are recorded every 100 exchange rounds, or 500 sweeps: 10,001 snapshots including initialization and the end. PT counters include thermalization. The sampler is the saved float32 conditional-Gaussian full-Hamiltonian Metropolis–Hastings implementation, with rho approximately −0.9 and proposal mass zero. Negative rho implements stochastic overrelaxation; it is not a separate exact microcanonical sweep.

At fixed size, the same seed is reused across the four couplings: 106, 138, 170 and 202 for increasing L, with PT seeds one larger. There is one production history per parameter set, not an independently seeded ensemble of replicate runs. Shared seeds do not imply that different Hamiltonians generate identical trajectories; they also do not justify assuming independent cross-coupling errors.

The current production defaults differ from the saved campaign. Therefore numerical settings in this report come from manifests, while observable interpretations come from inspecting current source. No historical source revision is stored in the NPZ files. `HE3_MODEL.md`, mentioned in the request, is absent. The attached thesis supplies useful SOC-free derivations and normalization context; its figures and bibliography were not needed to compute current results. The detailed source audit is [model_audit.md](model_audit.md).

**Traceability:** [inventory.csv](results/inventory.csv) records paths, sizes, seeds, parameters and ZIP member CRC32 values. CRC32 and file size provide a practical source-change check, not a cryptographic provenance guarantee. Data were opened read-only throughout.

### 1.2 Reduction and uncertainties

[summarize.py](summarize.py) processes one file at a time and only the necessary raw channels together. It stores 100-measurement block means of sufficient moments in `results/blocks/`; subsequent comparisons operate on these compact summaries. All samples are checked for finiteness, normalized chirality bounds and spin-alignment bounds. The analysis does not treat the 160 temperatures as independent samples.

The main thermodynamic error bars use a joint delete-one-block jackknife with **2,000 measurements per block**, corresponding to 100,000 sweeps and 30 blocks per full history. Nonlinear Binder, phase and helicity estimates are recomputed after each block deletion. Sensitivity checks use 100, 500 and 5,000 measurements per block; first and second halves are evaluated separately. Strong-SOC selected diagnostics extend to 10,000 measurements per block, leaving only six full-history blocks.

The uncertainty of fitted WM parameters uses 240 resamples of the 30 blocks. All channels and temperatures are resampled jointly within a size, preserving their measured covariance; different-size histories are resampled separately. Crossing and grid-peak checks use 400 resamples. These procedures assume the chosen blocks adequately separate the fluctuations being estimated and estimate variability **conditional on the configurations visited**. They cannot include missing branches, persistent initialization bias, independent-run variability or the validity of a scaling hypothesis. WM resampling holds central error weights fixed and searches a local temperature interval of approximately ±0.006 around each optimum. Its tail quantiles have only about six observations per tail and are not high-precision confidence limits.

FFT autocovariances provide descriptive integrated times using adjacent-lag pairs, truncation at the first nonpositive pair, and a running minimum of retained pairs. With tau_int=1/2+sum rho(t), the reported ESS is n/(2 tau_int), capped at n. This is an initial-monotone-sequence diagnostic. The reversible-chain assumptions behind its usual justification do not directly apply to alternating PT updates, and stationarity is independently uncertain. Accordingly these ESS values are **finite-history diagnostics**, not certified counts of independent equilibrium configurations. The method follows the paired-sequence construction in [Geyer, Statistical Science 7, 473–483 (1992)](https://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/GeyerStatSci1992.pdf).

Half differences are divided by the quadrature sum of separately blocked half errors as a drift indicator. The halves need not be independent, especially for slow modes, and many temperatures/observables are inspected. Isolated large ratios are not discoveries; coherent drift supported by raw time dependence and block-size sensitivity is more persuasive.

## 2. What the observables and SOC actually measure

### 2.1 Implemented model and temperature

The site field A is an unconstrained complex 3×2 spin/orbital matrix, with twelve real components. Writing n_r=tr(A A†), the Hamiltonian is

\[
H=\sum_r[-2n_r+\sum_{j=1}^{5}\beta_j I_j(A_r)+g_D D(A_r)+K G_r],
\]

with periodic boundaries, the five implemented quartic invariants, and forward-difference gradients. The coefficient vector in every file is

\[
(-0.5165123343,\ 0.9420228004,\ 0.9875290394,\ 0.8790628910,\ -1.1525149345),\qquad K=1.
\]

The saved physical coefficient parameter `reduced_temperature=1` stays fixed. The plotted Monte Carlo temperature changes exp(−H/T); it is not directly an experimentally calibrated T/Tc. The strong-coupling coefficient source is [Regan, Wiman and Sauls, Phys. Rev. B 101, 024517 (2020)](https://sauls.lsu.edu/publications/jas160/jas160.pdf); applying bulk coefficients to the strictly reduced film is a modeling assumption inherited from the project.

### 2.2 Normalizations that matter

Let N=L², D_A=sum |A|², C=2 sum Im(A_mu,x* A_mu,y), and P=sum(A_mu,x²−A_mu,y²), with sums over sites and spin rows. The report uses

| Quantity | Exact estimator/meaning |
|---|---|
| Energy and density | <H>/N and <D_A>/N; density is order-parameter norm, not particle number density |
| Chirality | m=C/D_A on each configuration; plotted order is <abs(m)> |
| Chirality Binder | 1−<m⁴>/(3<m²>²), using the signed normalized m |
| Composite phase | sqrt(<abs(P)²>/N²), with no division by density |
| Composite Binder | 2−<abs(P)⁴>/<abs(P)²>² |
| Spin alignment | Qzz=(3 D_z/D_A−1)/2 per configuration, then averaged; range [−1/2,1] |
| Helicity | Upsilon_nu=[<K_nu>−(<I_nu²>−<I_nu>²)/T]/N; average x and y |

I_nu and K_nu are the first and second derivatives of H when every forward neighbor in direction nu is multiplied by exp(−i delta_nu), with the same phase applied to every component. Helicity therefore measures **common condensate phase stiffness**. The dipole term is onsite and contributes indirectly through configurations, with no direct twist derivative. The current variance must be formed after averaging; a single configuration does not have this thermodynamic modulus.

Qzz measures spin-z norm population, not magnetization or local director correlations. P carries gauge charge two and is invariant under real spin rotations. Its magnitude contains amplitude and internal-tensor information as well as phase correlations. Neither quantity directly observes vortices.

For comparison of characteristic temperatures, the auxiliary responses are

\[
c_V=\mathrm{Var}(H)/(NT^2),\quad
X_\chi=N[\langle m^2\rangle-\langle|m|\rangle^2]/T,
\]
\[
X_P=N[\langle p^2\rangle-\langle p\rangle^2]/T,\qquad p=|P|/N.
\]

The last two are fluctuation diagnostics of the chosen magnitudes; X_chi is not the signed zero-field susceptibility N Var(m)/T. Peaks are not assigned universal exponents.

### 2.3 SOC admits more than spin reorientation within a chiral state

The code's exact invariant is

\[
D=|A_{xx}+A_{yy}|^2+|A_{xx}|^2+|A_{yy}|^2
+2\mathrm{Re}(A_{xy}A_{yx}^*)-\tfrac23 n.
\]

Its quadratic eigenvalues per norm are −5/3 for the antisymmetric planar combination A_xy=−A_yx, −2/3 for the z row, +1/3 for two other planar combinations, and +7/3 for A_xx=A_yy. Within the ideal chiral ansatz A=sqrt(n/2) exp(i theta) d_hat (1,i sigma), D=n(1/3−d_z²), so positive SOC favors d along z. **That restricted spin-locking picture does not cover the full simulated field space.**

[local_potential.py](local_potential.py) independently evaluates the onsite polynomial, compares homogeneous ansätze and checks their real twelve-component Hessians. At g_D=0.1 the optimized chiral and real antisymmetric planar onsite energies are −1.59710512 and −1.49972867 per site. Both are local minima; apart from the phase zero mode, their lowest Hessian eigenvalues are 0.189744 and 0.200000. Halving the differentiation step changes eigenvalues by less than 4×10^−7. Their energy difference narrows from 0.21785269 at zero coupling to 0.09737645 at 0.1. The two ansatz energies cross only near 0.173232, outside this campaign. Eighteen minimization starts per coupling found no lower energy than the chiral ansatz, without proving global optimality.

These are deterministic, zero-temperature **onsite** calculations, not finite-temperature free energies. They establish an allowed metastable competitor, not its equilibrium stability or the identity of the sampled branch. Importantly, the ideal planar configuration has Qzz=−1/2, m=0 and P=0, yet common-phase stiffness 4Kn in a homogeneous configuration. Thus a small saved P need not mean absence of all U(1) stiffness. Scalar signatures also permit textured or domain-based alternatives. Numerical tables and full derivations are linked in [the audit](model_audit.md).

## 3. Statistical reliability across the campaign

### 3.1 Temperature mobility and hidden configuration sectors

[pt_diagnostics.py](pt_diagnostics.py) checks every saved history row is a permutation of the 160 temperatures and reconstructs sampled endpoint visits, temperature coverage and last-endpoint flow. Exact trip bookkeeping requires endpoint→opposite endpoint→original endpoint; walkers starting inside the ladder first have to reach an endpoint. Zero loops is therefore a different statement from zero one-way travel. Sampled histories can miss short visits; exact cumulative trip counters are authoritative.

| g_D | L=64: acceptance / exact loops | L=96 | L=128 | L=160 |
|---:|---:|---:|---:|---:|
| 0 | 0.76918 / 1920 | 0.66149 / 542 | 0.56111 / 183 | 0.46976 / 65 |
| 0.001 | 0.76910 / 1747 | 0.66127 / 493 | 0.56097 / 150 | 0.46959 / 38 |
| 0.01 | 0.76788 / 1787 | 0.65922 / 596 | 0.55792 / 252 | 0.46570 / 98 |
| 0.1 | 0.76312 / 0 | 0.65300 / 0 | 0.55225 / 0 | 0.46208 / 0 |

Counts cover all five million sweeps, including thermalization. At fixed L, similar acceptance conceals radically different transport. At L=160, only 61, 35 and 84 walkers complete any loop at g_D=0, 0.001 and 0.01. Mean completed-loop times are approximately 3.06, 2.97 and 2.77 million sweeps, conditional on completion and comparable to the entire production interval. Nonzero loops improve the evidence for mobility but do not certify spin or defect equilibration.

At strong SOC, sampled full-run cold→hot passages number 14, 6, 5 and 3; none is hot→cold. Completions during production number 2, 1, 1 and 2, matching the branch losses. One of the two L=160 passages began before production. Branch conversion and endpoint arrival need not coincide: for L=128 conversion precedes the observed hot-endpoint visit substantially.

The steepest interior last-endpoint-flow drop coincides with the strong-SOC Qzz front. Acceptance at those edges is still 0.707, 0.643, 0.552 and 0.456. The global acceptance minimum is instead near T=0.465949, falling to 0.1215 at L=160. Every walker is observed on both sides of each spin-front edge during production. The problem is therefore sustained transport between **configuration sectors**, not a permanently uncrossed geometric edge. Broad local random walking can coexist with persistent branch identity.

![Swap acceptance and endpoint-labelled transport](results/pt_acceptance_flow.png)

The last-endpoint statistic is descriptive: walkers without a recorded endpoint visit are excluded, labels can miss short visits, and nonstationarity prevents interpreting the profile as an equilibrium diffusivity. [pt_findings.md](pt_findings.md) contains full definitions, coverage, entropy and time-window checks. [The L=160 histories](results/pt_walker_histories_L160.png) show the difference across all couplings.

### 3.2 Autocorrelation, blocking and first/second-half checks

The following minima summarize the finite-history ESS across temperatures, separately for absolute chirality and Qzz. The error ratio is the maximum over temperatures of SE(block 5000)/SE(block 100) for absolute chirality. Each row is traceable to `statistics_gd=..._L=....csv`; extreme values need not occur at the same temperature.

| g_D | L | Minimum ESS, abs chirality | Minimum ESS, Qzz | Maximum error ratio |
|---:|---:|---:|---:|---:|
| 0 | 64 | 6724 | 8657 | 1.59 |
| 0 | 96 | 1952 | 2698 | 1.55 |
| 0 | 128 | 780 | 1186 | 1.67 |
| 0 | 160 | 593 | 451 | 1.68 |
| 0.001 | 64 | 6862 | 7603 | 1.46 |
| 0.001 | 96 | 1989 | 2057 | 1.63 |
| 0.001 | 128 | 690 | 623 | 1.78 |
| 0.001 | 160 | 440 | 476 | 1.60 |
| 0.01 | 64 | 6119 | 7206 | 1.65 |
| 0.01 | 96 | 2531 | 3050 | 1.53 |
| 0.01 | 128 | 1201 | 1441 | 1.48 |
| 0.01 | 160 | 611 | 774 | 1.77 |
| 0.1 | 64 | 23 | 23 | 7.09 |
| 0.1 | 96 | 51 | 51 | 6.66 |
| 0.1 | 128 | 22 | 22 | 6.92 |
| 0.1 | 160 | 313 | 313 | 3.86 |

The 60,000 records clearly do not give 60,000 independent order-parameter measurements near the weak-SOC ordering region. At L=160, energy/absolute-chirality/Qzz ESS at the main grid peak are 698/593/451 for g_D=0, 552/440/476 for 0.001, and 810/611/774 for 0.01. Corresponding order-parameter times are a few thousand sweeps. These are much shorter than production but do not bound still slower unvisited modes.

At the representative strong-SOC mixed points, increasing block length gives no convincing plateau before very few blocks remain. At L=64 the branch-fraction error grows from 0.01084 at block 200 to 0.02411, 0.05473 and 0.08062 at blocks 1000, 5000 and 10000. At L=128 the sequence is 0.01141, 0.02521, 0.05660 and 0.07976. Treating only the short-scale fluctuations gives substantially overconfident uncertainties. Nonstationarity means that simply choosing the largest error does not repair equilibrium inference.

![Finite-history effective sample size and block sensitivity](results/statistical_reliability.png)

### 3.3 A slower symmetry sector survives in a weak-SOC run

An independent check finds coherent low-temperature signed-chirality drift at g_D=0.001, L=160. At T=0.3504518, the first-half mean signed m is 0.293047 and the second-half mean is 0.153412. The difference −0.139635 is 7.80 times the combined block-2000 half-error; it remains 6.99 and 6.05 times that diagnostic error when coarsened to blocks 5000 and 10000. The latter has only three blocks per half and is itself imprecise. Energy, density and absolute chirality are much steadier. This is consistent with slow exploration of the two chirality signs, rather than comparable drift in the magnitude of order.

The finite-history autocorrelation estimate at that point gives about 1123 effective signed-chirality records despite the long-block drift. It is a concrete reason to avoid equating a respectable ESS with equilibrium. The signed-sector imbalance can leave symmetry-even moments much less affected; it does not automatically invalidate every weak-SOC curve. Conversely, an apparently steady even observable does not certify defect or sign-sector sampling. The current initializer prepares positive-chirality z-row fields, making residual preparation memory a plausible explanation, although the historical source revision is not recorded. [validation.json](results/validation.json), [weak_sector_drift.csv](results/weak_sector_drift.csv), and the [time-block figure](results/weak_chirality_sector_drift.png) document this additional check.

![Signed and absolute chirality have different relaxation behavior](results/weak_chirality_sector_drift.png)

No independent-run error, replica-based R-hat, true hysteresis loop, or exact production-only loop count is available. The saved arrays cannot reconstruct unsaved thermalization order-parameter histories. Gaussian sampler acceptance counters are indexed by physical walker, not fixed temperature; they have not been misused as a temperature acceptance profile.

## 4. Weak SOC: large spin response, nearby bulk ordering scales

### 4.1 Fixed-size comparison and low-temperature spin alignment

![Across-coupling comparison at L=160](results/soc_overview.png)

At L=160 and T=0.33, all values use 2,000-measurement jackknife blocks:

| g_D | Qzz | Density | abs chirality | Helicity |
|---:|---:|---:|---:|---:|
| 0 | −0.00528 ± 0.00610 | 1.065354 ± 0.000021 | 0.694056 ± 0.000015 | 2.81763 ± 0.00336 |
| 0.001 | 0.575260 ± 0.000242 | 1.066120 ± 0.000019 | 0.694528 ± 0.000011 | 2.81221 ± 0.00262 |
| 0.01 | 0.624496 ± 0.000038 | 1.073324 ± 0.000012 | 0.698850 ± 0.000009 | 2.86029 ± 0.00206 |
| 0.1 | 0.700053 ± 0.000008 | 1.140510 ± 0.000011 | 0.733557 ± 0.000008 | 3.22220 ± 0.00161 |

The strongest row describes the sampled low-temperature chiral branch; its tiny local error cannot establish its equilibrium weight. Full values are in [low_temperature.csv](results/low_temperature.csv).

For g_D=0.001, Qzz grows from 0.49336(97) at L=64 through 0.55418(54) and 0.56959(43) to 0.57526(24). At 0.01 it is already approximately size saturated: 0.62337(7), 0.62442(7), 0.62453(3), 0.62450(4). At zero SOC, mean Qzz remains consistent with an isotropic average within its slower orientation fluctuations. SOC therefore removes orientational freedom strongly before producing a comparable change in the measured low-temperature phase stiffness. Qzz need not approach one because thermal fluctuations and nonideal field components remain.

For a smooth ideal chiral texture, the bare dipole length is sqrt(2K/g_D): 44.721, 14.142 and 4.472 lattice spacings for increasing nonzero coupling. The studied L/ell_D ranges are 1.43–3.58, 4.53–11.31 and 14.31–35.78. The residual size dependence at 0.001 and saturation at 0.01 are compatible with an anisotropy crossover. They do not measure a renormalized spin correlation length: amplitude, spin stiffness and anisotropy renormalization are absent from these global diagnostics. Neither a universal collapse against L/ell_D nor a thermodynamic singularity at zero coupling is established.

### 4.2 Ordering temperatures: what coincides, what remains distinct

At each weak coupling, energy rises and density decreases rapidly through the ordering region; absolute chirality and composite phase amplitude decrease there, with the expected strong size dependence. The raw helicity remains broadly size independent well below that region and decreases with increasing L through it. The curves are smooth enough to compare, but their rapid curvature makes interpolation important.

![Finite-size stiffness, chirality Binder and spin alignment](results/finite_size.png)

The table compares deliberately different finite-size diagnostics. Binder crossings use L=128 and 160 and select the first ordered-side crossing in a narrow stated window. The quoted crossing errors are bootstrap standard deviations conditional on that root selection, not thermodynamic critical errors.

| g_D | q=1/2 WM diagnostic | Chirality Binder crossing | Composite Binder crossing | L=160 grid maxima of cV, Xchi and XP |
|---:|---:|---:|---:|---:|
| 0 | 0.381493 | 0.381655 ± 0.000038 | 0.381668 ± 0.000049 | 0.382211 |
| 0.001 | 0.381847 | 0.382298 ± 0.000116 | 0.382361 ± 0.000163 | 0.382211 |
| 0.01 | 0.385091 | 0.385479 ± 0.000098 | 0.385500 ± 0.000098 | 0.385436 |

The chirality and composite Binder crossings are extremely close compared with the ladder spacing and finite-size ambiguity. Their bootstrap fluctuations are correlated because the observables are measured together. At 0.001, 14 of 400 composite-crossing resamples fail the selected-root criterion and are excluded from its conditional interval. Other size pairs and noisy disordered-region intersections give additional crossings; the full collection is retained rather than silently declaring one universal crossing. See [crossings.csv](results/crossings.csv) and [binder_crossing_bootstrap.csv](results/binder_crossing_bootstrap.csv).

All 400 L=160 bootstrap resamples select the same grid point for each of the three response maxima at a given weak coupling. **This does not imply zero temperature uncertainty.** Adjacent temperatures are about 0.00064–0.00065 apart near these peaks. Bootstrap stability identifies the largest saved grid value; it cannot locate a continuous maximum. Neighboring grid points give a practical resolution interval under a smooth, single-peak interpolation assumption; narrower unsampled features are not excluded. The pointwise peak heights also have ordinary block errors: cV is 73.85±2.20, 81.80±3.46 and 67.29±1.55 for the three weak couplings at L=160.

![Composite ordering and energy fluctuations](results/weak_ordering.png)

These measurements support a **common narrow region of rapid ordering changes** at each weak coupling. They do not establish coincident thermodynamic transitions, a resolved intermediate phase, or a numerical bound on its width. The modest offset between a WM fit and a Binder crossing is a difference between finite-size estimators, not by itself a separation between U(1) and Z2 critical temperatures. Peak heights increase with size, but four sizes over a factor 2.5 and a coarse grid do not justify critical-exponent or latent-heat fits.

The q=1/2 WM shifts relative to zero coupling are 0.00035382 at 0.001 and 0.00359795 at 0.01. Their coupling ratios, 0.354 and 0.360, explain the preliminary impression of approximate linearity. Without knowing cross-coupling covariance, the sums of the individual conditional bootstrap standard deviations, about 0.000048 and 0.000057, are conservative upper bounds for the statistical standard deviation of each difference within this resampling model. They do not cover interpolation, finite-size bias or unknown equilibration error. Two nonzero weak couplings cannot establish a perturbative law, especially when analysis sensitivity reaches the smaller shift.

## 5. Strong SOC: separate the measured populations from equilibrium claims

### 5.1 Branch inventory is the decisive diagnostic

Define b=1 when Qzz>0.3 and B(t)=sum_T b(t,T). PT swaps permute configurations and leave B unchanged; changes in B therefore detect actual branch conversion by local dynamics. Thresholds 0.2 and 0.4 give the same initial/final inventories and conversion directions. A classifier abs(m)>0.1 agrees with the Qzz classifier on all 9.6 million records per size except 1, 0, 1 and 2 transitional records.

| L | High-Q inventory, start→end of production | Resolved high→low / low→high | Largest mean-Qzz step bracket |
|---:|---:|---:|---:|
| 64 | 33→31 | 2 / 0 | 0.355607–0.356420 |
| 96 | 44→43 | 1 / 0 | 0.364834–0.365461 |
| 128 | 50→49 | 1 / 0 | 0.367933–0.368546 |
| 160 | 55→53 | 2 / 0 | 0.370379–0.370990 |

At exactly matching history/measurement sweeps, physical walker labels identify all six losses independently. Unobserved short excursions between saved samples remain possible. The strong claim is absence of **resolved branch regeneration**, corroborated by slowly changing total inventories and zero exact PT loops, rather than mathematical proof that no reverse event ever occurred.

![Population inventory and time dependence](results/strong_branch_inventory.png)

The front shifts substantially with L and the retained branch population. For L=64, 96 and 128 the high-branch occupancy near its middle falls from 0.683→0.355, 0.618→0.399 and 0.594→0.283 between production halves. L=160 changes only 0.258→0.223 because both losses occur early in production. Its smaller half difference does not restore bidirectional branch exploration. A seemingly stable mean can be the mean of a frozen population.

### 5.2 Joint distributions reveal competing configurations, with drifting weights

The high-Q population is strongly chiral, denser and lower in energy than the low-Q population at the representative mixed temperatures. For example, at L=128 and T=0.368546:

| Conditional measurement | High Q | Low Q |
|---|---:|---:|
| Qzz | 0.61524 | −0.26680 |
| abs normalized chirality | 0.64676 | 0.00335 |
| Density | 1.05930 | 1.02933 |
| Energy/site | 0.92401 | 0.93896 |
| Helicity from conditional current moments | 2.60142 | 2.46450 |

These conditional values describe the observed records, not equilibrated pure-phase properties. Selecting a branch changes the ensemble, and a formal branch-restricted thermodynamic derivative can also require selection-boundary terms. Sparse conditional estimates are especially unsafe; figures mask branch curves below 0.5% occupancy.

![Energy, spin and chirality populations](results/strong_joint_branches.png)

Qzz–absolute-chirality Pearson correlations exceed 0.9998 at all four representative fronts, mainly because the two clusters are so separated. They are not evidence of a universal continuous spin–chirality relation. Energy distributions overlap much more than the chirality and density populations do. The difference in chirality magnitude rules out interpreting the two populations as only different spin orientations of an otherwise identical uniformly chiral state.

An antisymmetric planar competitor, a more general nonchiral tensor state, or textured/domain configurations are alternatives. Their entropy and free-energy barriers are not determined by the onsite calculation. **Bimodal pooled histograms plus unidirectional branch losses are not a measurement of equilibrium coexistence.** No hysteresis protocol was run, so apparent history dependence should not be called a measured hysteresis loop. [strong_findings.md](strong_findings.md) records all conditional tables and event brackets.

### 5.3 Why the Binder becomes enormous and negative

| L | T of minimum | Uchi | <m²> | <m⁴> | abs(m)>0.1 records | Their share of <m⁴> |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.368546 | −485.20 | 1.63855e−4 | 3.91612e−5 | 13 | 99.9606% |
| 96 | 0.373441 | −1008.53 | 6.66447e−5 | 1.34515e−5 | 5 | 99.9753% |
| 128 | 0.375291 | −1662.87 | 3.20815e−5 | 5.13749e−6 | 2 | 99.9789% |
| 160 | 0.377781 | −2548.21 | 2.51827e−5 | 4.84990e−6 | 2 | 99.9903% |

There are 60,000 measurements at each listed temperature. Deleting just these records changes Uchi to +0.00244, −0.00674, +0.00114 and −0.01087. This is an influence diagnostic, **not a prescription to discard physical samples**. The problem is that their equilibrium probability has not been established.

For a background of small approximately Gaussian m plus rare large-chirality records, the fourth moment can be almost entirely a tail statistic while the second moment still includes a substantial background. The ratio <m⁴>/<m²>² can then be enormous. Narrower background fluctuations at larger L can deepen the minimum without determining transition order. Here the branch population also drifts, making equilibrium tail-weight inference still less secure.

![Rare events, moment concentration and influence](results/strong_binder_rare_events.png)

At L=128 the first/second-half Binders at the selected minimum are −1673.59 and −0.00108; at L=160 they are −2242.39 and −0.02342. Both rare L=160 records occur in one 200-measurement block, giving a block-jackknife error about 2540 for a central value −2548. A Gaussian error bar is a poor summary of an almost unmeasured tail. Selecting the most extreme temperature after looking across the ladder adds selection effects. Neither a first-order exponent nor a latent heat should be inferred from these minima.

### 5.4 Common-phase stiffness survives loss of global chirality and composite amplitude

At T=0.442331 the raw helicities are 1.19391, 1.17989, 1.18561 and 1.18607 for increasing L, with no high-Q samples in any file. At T=0.388056 they are already approximately 2.19–2.20 in almost entirely low-Q populations. Thus the high-temperature stiffness does not come from the small residue of the low-temperature chiral branch. At 0.442331, the <I_nu>²/(TN) correction is only about 10^−7–1.5×10^−4; it cannot account for a modulus near 1.18. The two directional estimates are comparable.

![Helicity components and sampled-current correction](results/strong_helicity_components.png)

The recorded ensemble has at least three distinct features: the lower spin/chirality population front, a smooth interval with substantial stiffness and almost zero global chirality/P, and a higher-temperature stiffness loss. This is a valuable hypothesis generator. It does not establish a stable intermediate equilibrium phase. Identifying the low-Q manifold is essential because the saved charge-two P can cancel in a planar tensor configuration independently of phase stiffness.

The heat-capacity maximum for L=160 occurs at saved T=0.466753, height 15.363±0.105, close to the upper edge of the ladder. It should not be compared as an equilibrium peak scaling law to the lower front's enormous order-parameter fluctuations: those arise from drifting mixtures. A high-temperature branch-restricted analysis could be meaningful, but the present evidence alone cannot exclude a missing competitor or preparation dependence.

## 6. HQV/BKT tests: what the helicity does and does not establish

### 6.1 Direct comparison of the two charge hypotheses

The tested form is

\[
\Upsilon_L(T_*)=\frac{2T_*}{\pi q^2}\left[1+\frac{1}{2\ln L+C}\right],
\qquad 2\ln L_{\min}+C>0.
\]

At any trial temperature only the four size values enter; fitting Tstar and C leaves **two nominal degrees of freedom**, not 160×4 observations. This adapts the leading logarithmic finite-size relation of [Weber and Minnhagen, Phys. Rev. B 37, 5986 (1988)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.37.5986) to the chosen charge hypothesis. It does not establish that this effective theory applies to the full helium model.

The diagonal weights use linearly interpolated pointwise errors rather than a full neighboring-temperature covariance calculation. Reported chi-squares are therefore comparative residual diagnostics with nominal degrees of freedom; calibrated goodness-of-fit probabilities are not claimed. The q=1 weak-coupling fits also require large leading corrections, for example C≈−7.762 versus the lower allowed value −2 ln(64)≈−8.318 at zero SOC. Such large corrections further weaken an asymptotic interpretation.

[bkt_checks.py](bkt_checks.py) profiles every saved temperature, refines candidate minima, and retains the positive correction constraint. This independent diagnostic was necessary: the existing helper's five uniformly spaced initial guesses can miss a narrow low-residual region and settle on a very poor local optimum. The direct comparison is saved in [wm_optimizer_audit.csv](results/wm_optimizer_audit.csv). Production analysis code is unchanged. Fits below are from the independently profiled calculation.

| g_D | q | Tstar | Conditional bootstrap SD(Tstar) | C | chi²/dof |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 0.381493 | 0.000025 | −4.397 | 1.501 |
| 0 | 1 | 0.383814 | 0.000028 | −7.762 | 107.729 |
| 0.001 | 0.5 | 0.381847 | 0.000023 | −3.376 | 0.365 |
| 0.001 | 1 | 0.384322 | 0.000037 | −7.661 | 29.992 |
| 0.01 | 0.5 | 0.385091 | 0.000031 | −2.708 | 0.295 |
| 0.01 | 1 | 0.389665 | 0.000034 | −7.238 | 5.623 |
| 0.1 | 0.5 | 0.443355 | 0.000655* | +25.557 | 0.897 |
| 0.1 | 1 | 0.463757 | 0.000091 | −6.188 | 0.329 |

*Strong-SOC q=1/2 SD excludes 10 of 240 resamples hitting the local fit boundary or vanishing correction; its C distribution is extremely broad and skewed. The quoted SD is not a reliable equilibrium error. Likewise, tiny SDs for rejected q=1 weak-SOC fits do not rescue a model with large residuals.*

![Charge-dependent profile residuals and raw stiffness](results/bkt_charge_comparison.png)

Raw jump-line intersections, listed in [crossings.csv](results/crossings.csv), drift with L. These intersections neglect the logarithmic correction and are not independent estimates of vortex charge. At strong SOC a nearly size-independent smooth stiffness curve passes through the q=1/2 jump near 0.44 before its more pronounced higher-temperature fall. A large positive C makes the fitted correction very small. A good q=1/2 fit there therefore does not supply distinctive evidence for half vortices. The integer form also fits, at a different temperature near the stiffness loss.

### 6.2 Sensitivity is larger than a single quoted fit error

All details are in [wm_sensitivity.csv](results/wm_sensitivity.csv), [wm_bootstrap.csv](results/wm_bootstrap.csv), and [wm_interpolation.csv](results/wm_interpolation.csv).

- **Size selection:** weak q=1/2 leave-one-size-out temperatures range 0.381395–0.381531, 0.381832–0.381861 and 0.385072–0.385111. The zero-SOC shift after excluding L=64 is about four conditional bootstrap SDs. Three sizes leave only one fit degree of freedom, making small residuals especially weak evidence. At g_D=0.01, one three-size q=1 subset also fits very well despite the worse four-size result.
- **Time halves:** weak q=1/2 first/second fits differ by about 0.000029, 0.000072 and 0.000071. Strong q=1/2 shifts from 0.443442 in the first half to 0.441388 in the second. This is much larger than weak-coupling effects of the same test.
- **Block length:** weak central fits change little over blocks 100–5000; this is useful stability of the sampled even-sector estimator, not a certificate for hidden sectors. At strong SOC both central Tstar and especially C are more sensitive.
- **Interpolation:** replacing piecewise-linear interpolation by PCHIP or a cubic spline shifts weak q=1/2 Tstar upward by 0.000025–0.000079, comparable to or larger than the bootstrap SD. Retaining only every other ladder temperature shifts the zero-SOC result by as much as −0.000265 and the 0.001 result by −0.000152. These are sensitivity checks, not independent estimates to average.
- **Temperature windows:** windows containing the narrow weak minimum return the same optimum; deliberately excluding it either forces a boundary/vanishing-correction solution or gives a poor fit. Because this method evaluates the size relation at one fitted temperature, changing the allowed window is not equivalent to fitting a broad critical curve. At strong SOC, exclusion of L=64 leaves the q=1/2 optimum at vanishing correction, without a finite identifiable C.

The results reproduce the supplied provisional weak-SOC numbers closely while exposing the limits of their apparent precision. Interpolation, size selection, unknown equilibrium bias and scaling-form error should be reported separately from the small resampling SD. No justified single error bar combines all of them.

### 6.3 Half circulation, spin energy and SOC confinement

In an ideal chiral sector, (d_hat,theta) and (−d_hat,theta+pi) describe the same matrix field. This permits a combined spin/phase half vortex. However, an HQV can carry a logarithmic **spin** energy as well as a logarithmic mass-phase energy. For long-wavelength energy (rho_c/2)(grad theta)²+(rho_d/2)(grad d_hat)², an ideal half texture has logarithmic coefficient pi(rho_c+rho_d)/4 while both stiffnesses remain active. The sole mass-stiffness jump 8T/pi requires appropriate long-distance screening of the spin contribution and an unconfined defect theory. A primary example with the same (S²×U(1))/Z2 structure is [Mukerjee, Xu and Moore, Phys. Rev. Lett. 97, 120406 (2006)](https://arxiv.org/pdf/cond-mat/0605102). It motivates this caveat; it is not evidence that the present finite sizes reach that regime.

Nonzero SOC explicitly locks spin in the chiral manifold. A half-vortex texture may then require a costly departure from a dipole minimum and a connecting texture, changing the asymptotic defect theory. The original helium HQV proposals discuss combined symmetry, textures and dipole solitons, with important geometry/field dependence: [Volovik and Salomaa, Sov. Phys. JETP 61, 986 (1985)](https://jetp.ras.ru/cgi-bin/dn/e_061_05_0986.pdf). Their specific experimental setting should not be transferred unchanged to this zero-field lattice model.

The present scalar data establish neither the active spin stiffness nor the spatial spin correlation length. They contain no circulation maps, half-phase/spin windings, pair-separation distributions, wall tension or imposed-defect interaction free energies. Consequently:

1. The good weak q=1/2 finite-size fit is compatible with an HQV-related picture, but does not identify HQVs.
2. Weak SOC can strongly alter the spin sector while leaving accessible thermodynamic curves similar; this is compatible with a singular long-distance perturbation, without demonstrating one.
3. No measured result here directly demonstrates HQV confinement, a logarithmic HQV pair interaction, or a linear confining term.
4. Before assigning a charge to strong-SOC stiffness loss, the low-Q order-parameter manifold itself must be identified.

## 7. Evidence priorities and questions for follow-up

The most informative next work should address missing physics or demonstrated sampling failures. Increasing the present run length solely to shrink a within-run error bar would not answer the main questions.

| Priority and physical question | Why these data are insufficient | Targeted observable/calculation | What different outcomes would imply |
|---|---|---|---|
| **1. What is the low-Q strong-SOC configuration?** | Global Qzz, chirality and P do not reconstruct a 3×2 matrix or distinguish uniform order from domains | Save representative local fields in both branches; measure local quartic/SOC invariants, singular values, spin population tensor and spatial chirality; compare to chiral/planar reference tensors | Uniform planar-like order changes the relevant defect theory; domain/textured chiral states point to interfaces or spin/orbital textures instead |
| **2. Are the strong-SOC branches equilibrium competitors?** | Six losses and no observed regeneration leave unknown weights and preparation dependence | A few deliberately independent starts from identified competing configurations near the measured front; follow total branch inventory and both conversion directions; if barriers persist, compute a constrained free-energy profile | Agreement of preparations with stationary reversible weights supports equilibrium competition; persistent distinct branches identifies the remaining barrier rather than a critical temperature |
| **3. Are HQVs present in the SOC-free and weak-SOC chiral sector?** | Mass stiffness and charge-two order do not locate defects | Extract local chiral director/phase with a consistent Z2 identification; map combined pi phase and pi director windings, test gauge/branch-cut invariance, and record defects, cores and separations | Verified half defects support an HQV mechanism; integer vortices or other textures require an alternative account of the successful WM fit |
| **4. Does SOC confine those identified defects?** | No defect interaction, separation or connecting-wall measurement is saved | Impose/identify separated HQV pairs and measure constrained free energy versus distance for g_D=0 and small nonzero g_D; resolve spin texture and any connecting wall, with sizes spanning the observed crossover | A logarithmic regime followed by a reproducible linear contribution and resolved wall supports confinement; persistent logarithmic behavior over larger distances constrains the crossover, without proving its absence asymptotically |
| **5. What are the renormalized spin length and stiffness?** | The bare sqrt(2K/g_D) and global Qzz do not determine spin screening | Spin correlation functions/tensor correlations, spin-twist response, and anisotropy response in an identified branch; compare their length scale with defect separations and L | Screened spin logarithms strengthen the mass-only half-jump assumption; active long-range spin costs or locking require a coupled theory |
| **6. Are U(1) and Z2 transitions distinct at weak SOC?** | Four sizes, discrete temperature spacing, related finite-size estimators and slow sign-sector sampling leave no controlled thermodynamic separation | After sector sampling checks, measure spatial charge-two and chirality correlations/correlation-length ratios, with targeted ladder refinement in the common ordering region and enough independent preparations to test bias | Consistent distinct limiting scales support an intermediate phase; joint convergence limits the split and tests the earlier interpretation |
| **7. What controls the high-T strong-SOC stiffness loss?** | Both q fits can succeed and the low-Q manifold is unknown; the energy peak lies near the ladder top | First identify that manifold, then measure its allowed defects and spatial correlations around 0.46; verify preparation independence there before extending sizes | Integer-defect unbinding, another composite defect, or a crossover become distinguishable; a good WM fit alone cannot decide |

For future bookkeeping, exact one-way endpoint counters, production-start counter snapshots, separate dipole energy and occasional configurations would be particularly informative. Those additions would clarify diagnostics and enable physics measurements; none were made to production code in this investigation.

## 8. Reproduction and numerical evidence map

Run from the project root with the existing NumPy/SciPy/Matplotlib environment. The executed environment was Python 3.12.10, NumPy 2.4.6, SciPy 1.18.1 and Matplotlib 3.11.2. The `-B` flag avoids incidental bytecode writes in production directories.

```powershell
python -B exploration/he3_soc/summarize.py
python -B exploration/he3_soc/compare.py
python -B exploration/he3_soc/ordering_checks.py
python -B exploration/he3_soc/bkt_checks.py
python -B exploration/he3_soc/interpolation_checks.py
python -B exploration/he3_soc/pt_diagnostics.py
python -B exploration/he3_soc/strong_soc.py
python -B exploration/he3_soc/local_potential.py
python -B exploration/he3_soc/validate.py
```

`summarize.py` reuses existing complete per-file reductions after checking source member CRCs and file size. Append `--force` to rebuild all reductions after changing an analysis formula. It never rewrites source NPZs. Derived block caches can be regenerated from the data and need not be copied into a publication archive. Later scripts overwrite only their own derived outputs under this exploratory directory. The analysis is CPU based and launches no GPU or Monte Carlo job.

| Scientific result | Reproduction script | Numerical evidence |
|---|---|---|
| Source settings and provenance | `summarize.py`, `pt_diagnostics.py` | `inventory.csv`, `inventory_gd=...json`, `pt_summary.csv` |
| Thermodynamics, block errors and time halves | `summarize.py` | `thermo_gd=..._L=....csv`, `statistics_gd=..._L=....csv`, `results/blocks/` |
| SOC and size comparisons, responses | `compare.py` | `low_temperature.csv`, `response_peaks.csv`, `crossings.csv` |
| Crossing/root and grid-peak sensitivity | `ordering_checks.py` | `binder_crossing_bootstrap.csv`, `peak_bootstrap.csv` |
| Charge fits, size/windows/halves, fit resampling | `bkt_checks.py` | `wm_sensitivity.csv`, `wm_bootstrap.csv`, `wm_profiles.csv` |
| Interpolation/ladder resolution | `interpolation_checks.py` | `wm_interpolation.csv` |
| PT coverage, flow and endpoint events | `pt_diagnostics.py` | `pt_summary.csv`, `pt_edges.csv`, `pt_walkers.csv`, `pt_endpoint_events.csv`, `pt_time_windows.csv` |
| Strong branch distributions and rare-event influence | `strong_soc.py` | `strong_anomalies.csv`, `strong_branch_events.csv`, `strong_temperature.csv`, `strong_rare_events.csv`, `strong_uncertainties.csv`, `strong_correlations.csv`, `strong_classification.csv` |
| Competing onsite configurations | `local_potential.py` | `results/tables/local_potential.csv`, `results/tables/local_hessian.csv` |
| Independent estimator validation and weak signed drift | `validate.py` | `validation.json`, `weak_sector_drift.csv` |

Independent direct-raw calculations at selected temperatures in g_D=0,L=64 and g_D=0.1,L=160, including Binder-minimum points, agree with the new reductions and the repository's original blocked-jackknife function. Maximum observable and error differences are 2.89×10^−10 and 4.86×10^−11, respectively; these validate arithmetic rather than equilibration. The deterministic local-potential script also checks its invariant/SOC algebra against 100 random fields and repeats Hessians at two step sizes.

The supplementary [model audit](model_audit.md), [PT study](pt_findings.md), [strong-SOC study](strong_findings.md), and [validation study](validation_findings.md) provide derivations and more detailed event tables. The report's central distinction is between reproducible properties of the recorded trajectories, conditional finite-size inferences, and physical hypotheses requiring spatial or independently equilibrated evidence.
