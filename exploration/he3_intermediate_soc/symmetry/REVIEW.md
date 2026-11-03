# Independent review of transport and pilot interpretation

This review reads the root pilot/analysis scripts and the transport reconstruction without modifying them. It supplements the independent mixed-curvature checks in [SYMMETRY.md](SYMMETRY.md).

## Code and estimator checks

The pilot records scalar measurements in fixed-temperature order and stores the matching temperature-to-physical-walker permutation. Scattering those rows back with `w[time, mapping] = values` correctly reconstructs physical walker trajectories. Runtime snapshots are physical-walker ordered; indexing them by that same mapping correctly aligns Kxy, local antisymmetric weight, and Fourier records. The every-fifth scalar sample corresponds to the 100-sweep spatial cadence. The source uses synchronized sparse diagonal curvatures and both currents when estimating the full tensor, so it does not mix incompatible ensembles within that estimator.

The Fourier indices for A+,z, A−,z, B++/B−− and the antisymmetric planar scalar (Axy−Ayx)/√2 agree with the established field order. The local antisymmetric squared projection is normalized by the matching total density. The chiral and planar initial densities use their respective actual quartic coefficients and dipole coefficients, and initial energy is independently compared with the full CPU evaluator. No production sampler/PT modification is involved.

The twelve-block jackknife formulas have the correct prefactor. The existing 16k- and 32k-sweep designs divide both scalar/spatial cadences evenly in full and half intervals. These are conditional history errors: short blocks, slow population drift, or different preparations need not be represented by them. The dense scalar Y and sparse tensor average can differ within finite sampling; use matched moments to test estimator identities.

The sustained-visit duration originally counted n samples as 20n sweeps. The observed first-to-last span is 20(n−1); this was reported to root and corrected. Even the corrected threshold establishes persistence only at sampled times, not uninterrupted microscopic occupation between them. Visits over the entire run include the designated burn interval unless explicitly restricted.

Q>.25 together with |χ|>.25 is a chiral-like descriptor. Q<−.1 together with |χ|<.15 is a low-Q/low-global-chirality descriptor; it does not uniquely identify the antisymmetric planar local minimum. Locally chiral domains can cancel global χ, and small-box fluctuations can enter either region. Therefore label counted events as structural-region visits, or corroborate the planar assignment using synchronized antisymmetric projection and local residual distributions. Bidirectional visits plus round trips are necessary encouraging evidence, but do not by themselves prove independent-start agreement or stationarity.

## What the fixed-inventory sorting calculation means

The exact constrained assignment distribution for M high-labeled configurations among R temperature slots is

\[
\Pr(b_1,\ldots,b_R\mid M)\propto
\mathbf1_{\sum_i b_i=M}\prod_i r(\beta_i)^{b_i},
\qquad r(\beta)=Z_H(\beta)/Z_L(\beta).
\]

Its dynamic-programming marginal formula is correct and checked against toy enumeration. For internally equilibrated restricted basins,

\[
\frac{d\ln r}{d\beta}=-[E_H(\beta)-E_L(\beta)].
\]

A locally constant conditional energy difference therefore motivates log r≈a−βΔE. A temperature-independent **dimensionless offset in log r** cancels at fixed M. A temperature-independent physical energy/free-energy difference does not generally cancel, because temperatures differ. The original loose comment about a free-energy offset was reported to the transport author for correction.

The assignment test fixes M from the data and calibrates ΔE from the measured branch energies near the observed front. Thus the front location is mostly supplied by the inventory: in the sharp-sorting limit, the coldest M slots are high labeled irrespective of ΔE magnitude. Excellent location agreement is not an independent equilibrium prediction. Width and partial-occupancy probabilities provide the less tautological test. A same-M sharp-step baseline and an early-gap/late-occupation check were recommended to the transport author.

The author added both controls. The same-inventory sharp step has maximum occupancy residuals .23–.29, versus .015–.020 for the energy-based model. Measured/predicted 10–90% widths agree closely, and first-half energy gaps predict second-half occupation with maximum residuals .012–.021 while still conditioning on second-half inventory. These additional checks support the finite-width sorting explanation and address the main circularity concern; they do not supply the missing equilibrium inventory probabilities.

Even successful held-out occupancy prediction does not determine the equilibrium distribution of M, the free-energy difference between physical structures, or a transition temperature. It establishes that swaps can efficiently distribute a **trapped inventory** over temperatures. The relevant equilibrium information would require actual reversible changes of that inventory, internally stationary branch distributions, and independent preparations. This distinction strengthens, rather than weakens, the diagnosis of the old failure: a persuasive temperature front can arise from sorting without equilibrating physical structures.

Finally, the lack of resolved reverse conversions is limited by the available sampling cadence. It is strong evidence against observed interconversion, not proof that arbitrarily brief opposite events never happened. Conversely, the thousands of fixed-temperature label changes do not count as physical conversions.

## Consequences for the scientific claims

- The broad-hot-endpoint small pilots test whether a path for structural visits exists; they do not locate a thermodynamic crossover.
- L=8 results can remain preparation dependent even with many PT loops and bidirectional structural visits. That is a substantive null result, not an invitation to pool the preparations.
- Nearly equal diagonal helicities do not imply isotropy. A positive resolved Kxy-derived response is symmetry allowed, but noisy or negative tensor eigenvalues in short hot histories should not be turned into a geometric-mean BKT estimate.
- Old gD=.1 branch-conditioned helicity is descriptive of trapped populations. Its survival does not identify those configurations with the new .01 z-polar coherent orbital sector.
- A full-GL onsite barrier is a deterministic internal-energy result. It is not a macroscopic conversion barrier or thermal nucleation rate; spatial paths and entropy can change both.

## Final synthesis check

The complete report's mixed-response derivation, distinction between bare and thermal tensors, and conditional geometric-mean discussion are consistent with the independent checks. The Gaussian lattice comparison uses the correct 12 real coordinates and full-zone factor 1/2; the common omitted gauge mode is appropriately qualified as a finite-volume convention rather than an anharmonic correction. The report distinguishes the newly measured small-box planar-like projection from the unavailable microscopic structure of the old large-box population.

Three targeted corrections were sent to root: the four old cold-to-hot counts sum to 28, not 26; preparation-dependent cold populations should be called different rather than declaring one wrong without equilibrium evidence; and the finite stationary-state search establishes that no new minimum was **found**, not completeness of all possible minima. The current investigation supplies strong sampling and structural evidence while preserving the unresolved equilibrium and topology questions.
