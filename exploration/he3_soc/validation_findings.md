# Independent arithmetic validation and weak-SOC sector drift

Run `python exploration/he3_soc/validate.py` after `summarize.py`.
`results/validation.json` records successful checks against selected raw channels
from g_D=0,L=64 and g_D=0.1,L=160, including the minimum-chirality-Binder
temperature in each file. It recomputes the normalized moments directly and
compares both 100- and 2,000-measurement block estimates and errors with the
repository's original `utils.Analysis.statistics.blocked_jackknife`. The largest
observable discrepancy is 2.89×10^-10 and the largest jackknife-error discrepancy
is 4.86×10^-11. These are arithmetic checks, not equilibration checks.

The same script makes `results/weak_chirality_sector_drift.png` and
`weak_sector_drift.csv` from the compact block cache. They identify an
observable-specific reliability limitation beyond incomplete PT loops: at
L=160 and T=0.3504517922 the signed chirality decreases during production while
its absolute value remains steady. For g_D=0.001:

| Quantity | First half | Second half | Difference ± block-2000 half-difference SE |
|---|---:|---:|---:|
| Signed m_chi | 0.2930470 | 0.1534119 | −0.1396351 ± 0.0179047 |
| Absolute m_chi | 0.6368330 | 0.6368687 | 0.0000358 ± 0.0000270 |
| Density | 1.0172003 | 1.0172333 | 0.0000329 ± 0.0000272 |
| Energy density | 0.9064078 | 0.9063237 | −0.0000841 ± 0.0000442 |
| Qzz | 0.5099416 | 0.5096991 | −0.0002424 ± 0.0004843 |

For the signed channel the first/second-half contrast remains large under
coarsening: SE=0.0199661 for 5,000-measurement blocks and 0.0230635 for
10,000-measurement blocks. These correspond to descriptive ratios of −6.99 and
−6.05 respectively, but the latter has only three blocks per half. The point
was selected after an exploratory scan; it is not a preregistered significance
test. Many temperatures and channels were inspected, they share PT walkers,
and neither Gaussian z probabilities nor independent multiple-testing counts
are justified. The coherent drift across a broad low-temperature range and
its persistence under coarse blocking are the evidence, rather than one
isolated p value. Isolated 3–5 standard-error contrasts in other channels should
not be promoted to separate discoveries from the thousands of comparisons.

The controls g_D=0 and 0.01 at the identical L and temperature also show a
positive signed-chirality excess decreasing in the block traces. The current
initialization in `models/he3/model.py:52`–`:59` creates every replica in the
same positive-chirality d_z state. Residual initialization memory is therefore
a plausible explanation; the production manifests do not record a source
revision that would prove the historical initialization. Chirality-reversal
symmetry makes the two signs equivalent for absolute chirality, even Binder
moments, energy, density and common-phase stiffness. Unequal sign populations
thus diagnose slow symmetry-sector relaxation without automatically proving
bias in every even observable.

The FFT tau code pairs rho(0)+rho(1), rho(2)+rho(3), …, stops at the first
nonpositive pair, and replaces the retained pair sequence by its cumulative
minimum. Its precise name is an **initial-monotone sequence** estimate with a
biased FFT autocorrelation function and tau clamped to at least 1/2 saved
sample. This ensures estimated ESS≤60,000; it does not certify equilibrium.
At the g_D=0.001 point above tau≈26.7 saved samples and nominal ESS≈1,123 coexist
with a visible long-time sign drift and growth of the blocking error. Short-lag
autocorrelation estimates can miss slower modes. The alternating PT schedule
is not itself reversible, so reversible-chain theoretical guarantees for this
sequence estimator should not be invoked as a proof here.

For the WM analysis, raw-charge conversion and logarithmic denominators are
implemented consistently. Bootstrap resampling preserves cross-temperature and
cross-channel correlations inside each L's 2,000-measurement blocks. Important
limits remain: only 30 such blocks exist; independence of these blocks is an
assumption; interpolation weights do not explicitly use neighboring-temperature
covariance; the nominal four-size/two-parameter chi-square has only two degrees
of freedom; and the bootstrap is confined to the central local fitting basin
and holds the error weights fixed. Its reported uncertainty omits alternative
basins, model misspecification and unknown equilibration bias. The 10 rejected
strong-SOC q=0.5 replicates make its displayed quantiles conditional on accepted
replicates, not an unconditional 95% confidence interval. With 240 replicates
the 2.5% tails themselves contain only about six order statistics.

Finally, the same local and PT seeds recur across couplings at fixed L. Those
are neither independent replicas at one parameter set nor a measured paired
covariance. Adding errors in quadrature for cross-coupling differences is an
independence approximation; it is not provably conservative when the covariance
between campaigns is unknown.
