# Strong-SOC sampling: physical conversions versus temperature sorting

This read-only calculation independently rereads the four original `data/he3/P=20_gd=0.1_gaussian/L=*.npz` archives. It uses no new Monte Carlo and changes no production files. Reproduce with `python -B exploration/he3_intermediate_soc/transport/reconstruct.py`.

The strongest new result is quantitative: **the apparent low-temperature spin/chirality front is predicted very accurately by temperature sorting of an almost conserved branch inventory, without fitting an equilibrium transition temperature.** This reinforces the earlier sampling diagnosis and explains why stable-looking curves and frequent swaps are insufficient.

## Recorded setup and estimator conventions

All four runs contain 160 temperatures from .33 to .47, 2 million burn sweeps and 3 million production sweeps. Exchanges occur every 5 sweeps. Scalars are recorded every 50 sweeps; all physical walker identities are recorded every 500 sweeps, including burn. Gaussian proposals use rho≈−.9 and zero proposal mass. The data contain no configurations, spatial fields, B tensor, local chiral residual or mixed curvature.

The scalar called `spin_alignment` is

\[
s_z=\frac12\left(3\frac{\sum_r n_z(r)}{\sum_r n(r)}-1\right),
\]

not an unnormalized local Qzz. Following the old empirical classifier, a configuration is labelled high if s_z>.3. Varying this threshold to .2 or .4 gives the same initial/final inventories and the same number and direction of inventory changes. The alternative |global normalized chirality|>.1 disagrees in only 1,0,1,2 of the 9.6 million measurements per size. Thus high/low label two sharply distinct recorded populations, but not identified microscopic phases.

Production measurements are in temperature order. The PT engine keeps fields and site random streams in physical walker slots and only exchanges their inverse temperatures. The script therefore maps each measurement back using the saved walker-to-temperature permutation at the **exact same sweep**, and verifies every permutation and alignment. No interpolation of physical identity is used.

## Direct raw-data findings

| L | High inventory, start→end | Resolved walker high→low / low→high | Fixed-temperature label changes at 500-sweep cadence | Exact complete loops, full 5M sweeps | Sampled cold→hot / hot→cold passages, full run |
|---:|---:|---:|---:|---:|---:|
|64|33→31|2 / 0|14,938|0|14 / 0|
|96|44→43|1 / 0|12,087|0|6 / 0|
|128|50→49|1 / 0|8,497|0|5 / 0|
|160|55→53|2 / 0|6,180|0|3 / 0|

The large fixed-temperature switch counts mostly reflect replacement by another physical configuration. They are not basin conversions. All six resolved conversions are losses of the high-spin, chiral population. Independent 50-sweep inventory changes establish the same six losses; events.csv records both that finer time bracket and the 500-sweep physical-walker bracket. Events between snapshots could be missed, so this is not an exact upper bound on all microscopic recrossings. Nevertheless the complete absence of exact PT loops is independently established by the saved GPU counters.

Production-only constant labels persist in 158,159,159,158 of the 160 walkers. The unchanged high cohorts have median temperature-index spans 44,51,54,56, whereas unchanged low cohorts span 133,120,113,109. Their occupation curves are almost flat within their respective available temperature ranges, then sharply suppressed across the population front. Broad diffusion within either population is therefore compatible with almost no exploration between them.

No saved production configuration at the cold endpoint has a low label; no saved production configuration at the hot endpoint has a high label. At L=160, the unchanged high walkers never exceed T=.377156 in the saved history, while unchanged low walkers never go below T=.366703. They overlap around the front but do not connect the full ladder. A conversion can allow a one-way escape: the L=128 event precedes its first observed hot-endpoint arrival by about 184,000 sweeps, so conversion and reaching the endpoint are not interchangeable events.

![Physical inventory losses and corresponding physical walker trajectories; dots mark first low-label snapshots.](physical_conversions.png)

The old branch-table means were recomputed directly from raw density, chirality, alignment, energy, currents and curvatures. All tested old values agree within 1e−8; the actual maxima are retained in [raw_validation.csv](raw_validation.csv). These checks validate arithmetic, not equilibration.

## A constrained-population model predicts the false front

Suppose physical configurations equilibrate reasonably *within* each of two basins but the number M in the high basin is effectively conserved. The canonical distribution of labels among PT temperatures is then

\[
\mathcal P(\{b_i\}\mid M)\propto
\delta_{\sum_i b_i,M}\prod_i w_i^{b_i},\qquad
w_i=Z_H(\beta_i)/Z_L(\beta_i).
\]

For nearly constant conditional total-energy difference ΔE=E_H−E_L across the narrow front,

\[
\ln w_i\simeq c-\Delta E(\beta_i-\beta_*).
\]

The temperature-independent additive constant c in **log(Z_H/Z_L) cancels at fixed M**. A constant physical energy difference does not cancel; it changes the slope with inverse temperature. Thus a measured M and one measured conditional energy gap predict the front without supplying an equilibrium coexistence temperature. The script evaluates the exact finite-160-label constrained marginals by forward/backward dynamic programming, rather than using a grand-canonical logistic approximation. A five-label example is checked against explicit enumeration to <1e−13, and every prediction satisfies Σp_i=M to <1e−7.

For each real run, ΔE is measured at the temperature whose high occupancy is closest to one half; it is not fitted to the occupation curve. The full-production prediction averages the few integer-M predictions with their observed durations.

| L | Measured total ΔE at front | Maximum absolute probability residual over 160 temperatures | Mean absolute error |
|---:|---:|---:|---:|
|64|−113.668|.0152|.000623|
|96|−178.569|.0174|.000508|
|128|−244.963|.0160|.000349|
|160|−302.725|.0205|.000328|

The agreement is also strong separately within constant-inventory time intervals:

| L | M sequence | Measured T at p=.5 | Predicted T at p=.5 |
|---:|---|---|---|
|64|33,32,31|.357479,.356709,.355925|.357550,.356769,.355971|
|160|55,54,53|.371889,.371268,.370675|.371907,.371295,.370684|

These interpolated locations describe the sampled sorting front, not a transition estimate. Small model residuals are expected because the conditional energy gap varies with temperature, the restricted populations need not be perfectly internally stationary, and the trajectories are finite. Errors above are descriptive residuals, not a statistical goodness-of-fit claim. There is no fitted threshold temperature; inventory and conditional energy are measured inputs. The supplied inventory already largely fixes the front location, so that agreement alone is not an independent prediction. Its width provides an additional test:

| L | Observed / predicted 10–90% width in T | Sharp-step maximum probability residual | First-half energy gap predicting second-half maximum residual |
|---:|---:|---:|---:|
|64|.004768 / .004745|.2913|.0149|
|96|.002970 / .002938|.2725|.0147|
|128|.002163 / .002165|.2287|.0115|
|160|.001596 / .001649|.2331|.0205|

The sharp-step control fills exactly the M coldest temperatures with high labels, using the same measured inventory mixture. It gets the broad location right but misses the finite width. The second-half control calibrates ΔE from the first half only and retains the second half's measured inventory. These controls support the energy-driven sorting explanation without claiming an independent equilibrium test.

![Top: temperature occupation of unchanged physical populations. Bottom: observed front and the independently specified fixed-inventory prediction.](sorting_and_occupation.png)

This gives a concrete interpretation of the original size drift: different surviving inventories sort to different parts of the common ladder. It does **not** establish that no true transition exists in that range. The missing log-partition normalization determines the unconstrained high/low probabilities; conditioning on M removes that information. The restricted-label model therefore cannot recover the equilibrium branch weights or a coexistence temperature.

At L=160 the final M=53 interval lasts 2.916 million sweeps, almost all production. First/second-half agreement can consequently coexist with complete failure to regenerate the high branch. This is stronger evidence than merely observing a slow scalar autocorrelation.

## The hot endpoint is close to the upper feature, but is already nearly phase-disordered

| L | Y(T=.47), 100k-block conditional error | s_z(.47) | Minimum swap acceptance |
|---:|---:|---:|---:|
|64|.10815±.00947|−.01042|.5599|
|96|.05679±.00868|−.00616|.3665|
|128|.01178±.00765|−.00496|.2220|
|160|.00521±.00778|−.00442|.1215|

The minimum-acceptance edge is centered at .465949. At L≥96 the largest sampled fluctuation heat peak is at .466753, only .003247 below the endpoint. L=64 instead has a slightly larger fluctuation peak at its lower population front; such a variance in a nonstationary restricted mixture is not automatically an equilibrium heat capacity.

The hot-end Y decreases strongly with size and is consistent with zero at L=128,160. Therefore **retention of a large common-phase stiffness at the hot endpoint is not the demonstrated cause of missing reversibility**. A modest extension to, for example, .55–.60 on small boxes is a reasonable controlled test of faster structural forgetting, but may simply destroy high configurations more efficiently without helping their regeneration on cooling. It must be judged by bidirectional physical conversions and independent-start agreement, not just hot-end Y or more accepted swaps.

The phase-stiff low branch is verified below that upper feature. At T=.442331, Y≈1.18–1.19 in all four sizes, with no high-label sample; at T=.388056 it is ≈2.19–2.20 and essentially all samples are low. At the population fronts both conditional responses remain large: high/low Y=2.825/2.623,2.676/2.520,2.601/2.464,2.568/2.434. Conditional responses are descriptive, not thermodynamic derivatives of independently equilibrated phases. Global scalars cannot identify whether the low branch is planar, coherent A+, a domain texture, or another structure. In particular its negative alignment differs from the predominantly z-coherent weak-SOC channel; spatial tensor measurements are needed to connect them.

## What the sampler and ladder can, and cannot, repair

The inspected Gaussian updater proposes all twelve real components **at one lattice site**, using the fixed-neighbor quadratic conditional covariance and a correlated autoregression. Its actual precision eigenvalues are 10K+mass and 6K+mass. It includes the full Hamiltonian change and reverse/forward proposal-density ratio in MH acceptance; it does not impose the chiral ansatz or forbid the planar basin. Four-color scheduling is used. This review identifies no correctness defect requiring a production change.

The unchanged high walkers have full-run mean local acceptance .584,.588,.590,.591; unchanged low walkers have .646,.649,.651,.652. These averages include burn and different temperature occupations, so their difference is not an equal-temperature mobility comparison. Their finite values do show that fields are not simply frozen by rejection of every local move. Rebuilding a spatially coherent structure may still require a collective nucleation/texture rearrangement inaccessible to ordinary local steps on these times. The records do not measure that free-energy barrier or identify its saddle.

Adjacent swap improvement near .466 would help the upper transport bottleneck, but the lower front edges already accepted about .46–.71 in the old diagnostics. Neither a denser ladder nor an exchange interval below the already small five sweeps changes the inability of local evolution to regenerate a basin. Faster exchanges can also shorten local residence at favorable temperatures; their net benefit must be measured rather than presumed.

Scientifically justified next tests are small independent chiral/random/planar starts, with both original and broadened hot endpoints if budget allows. Keep physical identities and classify conversion on configurations. Require agreement in energy and structural distributions, several conversions in both directions, and stationarity across multiple windows before averaging apparent competing populations as equilibrium.

If ordinary pilots fail, two potential enhanced methods deserve separate implementation review:

1. **Tempering in SOC as well as temperature.** Write H_g=H0+g𝒟, with 𝒟 the bare dipole invariant. For replicas i,j, the swap log ratio is `(βi−βj)(H0_i−H0_j)+(βi gi−βj gj)(𝒟_i−𝒟_j)`. This preserves the joint target with the corresponding MH decision. Auxiliary weak-coupling replicas may connect structures that temperature alone fails to regenerate, but multiply memory/cost and need overlap plus reversible physical conversion tests. They are not automatically an improvement.
2. **A structural umbrella or explicit reversible collective proposal.** An umbrella in chirality/alignment must include its full bias change in every acceptance ratio, retain overlapping windows, and reweight before reporting unbiased distributions. A collective map must specify forward/reverse density and any Jacobian, then accept using the full target; a heuristic replacement of a configuration by a desired structure is invalid. Global spin rotations alone preserve chirality and generally cannot transform a chiral field into a planar one. A poor umbrella coordinate can leave hidden barriers unresolved.

Nothing here implements these methods, tunes the production sampler, or treats raw elapsed Monte Carlo sweeps as real-time physical dynamics. The strongest supported diagnosis is **well-developed motion within structurally restricted sectors, one-way losses, and an inventory-controlled population front**, with the equilibrium basin weights and microscopic conversion barrier still unknown.

## Output map

`summary.csv` gives transport, inventory, hot response and sorting residuals; `walkers.csv` resolves unchanged/converted cohorts and physical coverage; `events.csv` gives dual-resolution conversion brackets; `endpoint_passages.csv` gives sampled one-way endpoint events; `inventory_segments.csv` records constant-population intervals and observed/predicted fronts; `temperature_profiles.csv` gives all curves; `threshold_checks.csv` and `raw_validation.csv` document validation. Compact trace NPZs and both figures are reproducible from the script. No fields were inferred from unavailable scalar data.
