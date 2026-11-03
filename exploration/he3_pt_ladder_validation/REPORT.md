# Controlled validation of narrower He3 PT ladders

2 November 2026. L=96 GPU pilots; g_D=0 and 0.01; all Hamiltonian, sampler and exchange kernels unchanged.

## Main conclusion

**The narrower ladders improve measured early temperature transport, but an improvement in equilibrium sampling has not yet been established. Neither narrow ladder is sufficiently validated for L320 production.** All 12 bounded L96 tests completed with the actual Gaussian sampler, PT exchanges and existing measurements. They show a promising reduction in cost per endpoint passage, including faster traffic between common interior temperatures. However, nearly all walkers' first round trips remain unfinished within the comparison window, and initialization dependence or time-block drift remains visible in physical observables.

The strongest reason to withhold a production recommendation is not low acceptance. Minimum acceptance is already 0.83–0.90 at L96. At g_D=0.01, C160 retains an ordered/random difference in squared chirality of **−0.0113 ± 0.0033** at the critical probe. B200 has better agreement between those whole-run means, but its ordered-start helicity changes by **−0.204 ± 0.051** between the two comparison halves. These are conditional block errors from short runs, not calibrated significance tests or final equilibrium uncertainties.

Use **B200 as the conservative narrow candidate for the next controlled test**, with C160 retained as the cheaper alternative. The next useful screen is L96, 0.5M diagnostic burn plus 1M comparison sweeps for each of the 12 cases, provisionally about **2.5 aggregate GPU-hours including a timing allowance** at the saved campaign speed. That timing is conditional on Kraken reproducing the saved performance; the same test would take about 8.5 hours on the measured local GPU. It has **not** been launched.

## Measured comparisons

All rows below are L96, with 20k diagnostic burn plus a 40k comparison window. `Passages` counts arrivals at the opposite endpoint during that window, including journeys begun earlier. `Interior` uses the same physical temperature surfaces for every ladder, defined below. `Loops` requires the whole round trip inside the window. `Both` counts walkers visiting both endpoints within it. Seconds include all 60k simulated sweeps, including burn-in, but exclude setup and saving.

| g_D | Ladder | Start | Seconds | Minimum acceptance | Passages | Interior | Loops | Both / walkers |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | A200 | Ordered | 109.7 | 0.826 | 24 | 39 | 0 | 9/200 |
| 0 | A200 | Random | 108.8 | 0.834 | 34 | 58 | 0 | 13/200 |
| 0 | B200 | Ordered | 109.6 | 0.897 | 72 | 89 | 2 | 33/200 |
| 0 | B200 | Random | 108.2 | 0.896 | 70 | 82 | 1 | 37/200 |
| 0 | C160 | Ordered | 87.4 | 0.875 | 61 | 72 | 2 | 38/160 |
| 0 | C160 | Random | 86.2 | 0.868 | 54 | 66 | 2 | 29/160 |
| 0.01 | A200 | Ordered | 108.0 | 0.833 | 37 | 47 | 0 | 14/200 |
| 0.01 | A200 | Random | 105.7 | 0.842 | 33 | 49 | 0 | 15/200 |
| 0.01 | B200 | Ordered | 108.3 | 0.893 | 54 | 70 | 1 | 35/200 |
| 0.01 | B200 | Random | 105.4 | 0.893 | 58 | 83 | 1 | 34/200 |
| 0.01 | C160 | Ordered | 86.2 | 0.871 | 49 | 61 | 1 | 31/160 |
| 0.01 | C160 | Random | 84.8 | 0.860 | 59 | 74 | 2 | 29/160 |

Combining the two initializations descriptively gives:

| g_D | Ladder | Endpoint passages / walker / M sweeps | Seconds / endpoint passage | Seconds / interior passage |
|---|---|---:|---:|---:|
| 0 | A200 | 3.63 | 3.77 | 2.25 |
| 0 | B200 | 8.88 | 1.53 | 1.27 |
| 0 | C160 | 8.98 | 1.51 | 1.26 |
| 0.01 | A200 | 4.38 | 3.05 | 2.23 |
| 0.01 | B200 | 7.00 | 1.91 | 1.40 |
| 0.01 | C160 | 8.44 | 1.58 | 1.27 |

Thus B/C produce **2.46/2.49 times as many endpoint passages per measured second at g_D=0**, and **1.60/1.93 times at g_D=0.01**, compared with A. The common interior surfaces also improve by approximately **1.6–1.8 times per second**, so the result is not explained solely by defining closer endpoints. C's cost per sweep is about 20% lower than B's. At zero SOC their passage rates per walker per sweep are almost identical: C's advantage is mostly lower cost, not a demonstrated change in the slow physical dynamics. These ratios are finite-window descriptions, without reliable seed-level confidence intervals.

![Measured endpoint and common-interior traffic](figures/02_passages.png)

### Censoring, flow and physical memory

Only **0–2 strict comparison-window loops** finish in any case. Between 152 and 197 walkers per run begin a first loop without finishing it; another 0–9 never reach either endpoint during that window. Neither the first-passage nor first-loop survival curve reaches its median in any case. Reporting the mean of the few completed trips as an equilibrium round-trip time would therefore be misleading.

The inverse passage-rate scales are 0.43–0.67M sweeps for A and 0.21–0.30M for B/C. They describe traffic in a transient 40k interval, not stationary waiting times. The previously measured long-run L96 reference scale is about 0.42M; at L256 it is **7.64M / 5.19M for g_D=0 / 0.01**. This explains both why local L96 pilots can screen an idea and why they cannot settle large-size equilibration.

![Censored passage and loop distributions](figures/06_censoring.png)

Flow curves and their labelled-walker coverage are shown together below. The minimum coverage across temperatures is 0.83–0.96, but a known previous endpoint does not imply a completed journey between endpoints. Short-window flow differences should not be used to adapt another ladder before testing stationarity.

![Walker flow and endpoint-label coverage](figures/05_flow_coverage.png)

Near the heat peak, configurations last visiting the cold endpoint have larger squared chirality than those last visiting the hot endpoint in **every** run. The difference is approximately **0.040–0.050 at g_D=0** and **0.031–0.041 at g_D=0.01**, with conditional 5k-block errors of about 0.003–0.007. Energy, composite and spin conditional results are also saved in [endpoint_conditioned.csv](results/endpoint_conditioned.csv). This confirms persistent configuration memory. Such endpoint conditioning can occur in a stationary process with slow reactive trajectories; it is not by itself proof of initialization bias. The independent starts and time-block comparisons below address that separate question.

At a 50-sweep observation interval, the fraction of nonzero consecutive index displacements that reverse near the critical region is approximately **0.69–0.71 for A, 0.43–0.46 for B, and 0.50–0.53 for C**. An acceptance-only chain with the correct alternating matching parity predicts approximately **0.44, 0.32–0.33 and 0.35–0.37**, respectively. The residual excess is consistent with configuration-dependent transport rather than independent exchange decisions. These are coarse displacement reversals, not individual rejection probabilities.

Sampled critical-region residence and recrossing records are in [residence.csv](results/residence.csv) and [summary.csv](results/summary.csv). Pooling only uncensored episodes for a descriptive comparison, median residence is 50 sweeps for A and 100 for B/C; the 90th percentiles are 150, 250 and 200 sweeps. More densely placed critical replicas naturally change these temperature-path residence statistics. A crossing of this temperature band is not necessarily a physical chirality conversion, and these short episodes do not measure the slowest structural lifetime.

### Independent starts and stationarity

The critical probes are T=0.3821053 at g_D=0 and T=0.3856721 at g_D=0.01, taken from the previous L96 analysis. Values are compared at these same temperatures using joint moment interpolation, rather than comparing different nearby grid points. The following differences are ordered minus random, with conditional 5k-block errors:

| g_D | Ladder | Squared chirality difference | Binder difference | Helicity difference |
|---|---|---:|---:|---:|
| 0 | A | +0.0071 ± 0.0063 | +0.0171 ± 0.0253 | +0.0523 ± 0.0696 |
| 0 | B | +0.0026 ± 0.0029 | +0.0100 ± 0.0112 | +0.0129 ± 0.0471 |
| 0 | C | −0.0005 ± 0.0034 | +0.0006 ± 0.0154 | +0.0341 ± 0.0371 |
| 0.01 | A | −0.0108 ± 0.0040 | −0.0575 ± 0.0180 | −0.1565 ± 0.0853 |
| 0.01 | B | +0.0011 ± 0.0048 | −0.0064 ± 0.0213 | −0.0463 ± 0.0613 |
| 0.01 | C | −0.0113 ± 0.0033 | −0.0467 ± 0.0196 | −0.0098 ± 0.0526 |

Zero-SOC critical means agree reasonably between starts, but the cold spin alignment has differences **+0.0493 ± 0.0118, +0.0334 ± 0.0137 and +0.0357 ± 0.0152** for A/B/C at T=0.375. The random-start values are closer to the long zero-SOC spin reference. Slow spin rotation can therefore remain hidden behind agreement in scalar chirality or helicity.

At nonzero SOC, C does not resolve the start dependence already visible in A. B's average agreement is better, but its ordered-start helicity shifts by **−0.204 ± 0.051** between the first and second 20k comparison halves; Binder shifts by **−0.064 ± 0.028**. This is why B is a candidate for a longer controlled experiment, not a validated winner. None of these isolated comparisons is a multiple-testing-adjusted hypothesis test; their practical message is that the short-run errors cannot certify equilibrium.

![Critical observables against the long saved reference](figures/03_initialization_reference.png)

![Early relaxation and comparison-window variation](figures/04_relaxation.png)

### Precision and effective samples per cost

To avoid interpreting extra interpolation averaging as improved mixing, this table uses each ladder's **single nearest native temperature** to the critical probe. Its offset is at most 0.000022. Ranges span the two starts, not confidence bounds. Errors use eight 5k blocks; ESS per hour includes the time spent on all 60k sweeps.

| g_D | Ladder | Binder standard error | Binder FFT ESS in 40k | Binder block ESS / total GPU-hour | Helicity standard error |
|---|---|---:|---:|---:|---:|
| 0 | A | 0.0155–0.0200 | 119–246 | 3,665–5,289 | 0.0364–0.0593 |
| 0 | B | 0.0071–0.0123 | 604–775 | 9,253–23,846 | 0.0323–0.0499 |
| 0 | C | 0.0120–0.0154 | 676–743 | 10,321–16,645 | 0.0260–0.0359 |
| 0.01 | A | 0.0092–0.0155 | 314–335 | 9,400–13,762 | 0.0385–0.0761 |
| 0.01 | B | 0.0123–0.0199 | 339–643 | 5,267–12,087 | 0.0462–0.0728 |
| 0.01 | C | 0.0100–0.0175 | 355–384 | 8,723–14,388 | 0.0426–0.0509 |

At zero SOC the narrower placement improves short-window Binder precision. At nonzero SOC the precision comparison is mixed: B's higher traffic does not produce uniformly smaller Binder or helicity errors. Native fixed-temperature integrated autocorrelation estimates for energy, squared chirality and composite intensity span approximately **25–126 sweeps**, while helicity influence estimates are near the **25–31 sweep sampling floor**. These noise-dominated, temperature-slot correlations can be short while the same physical walker retains memory for much longer. They are not equilibration-time estimates.

Block-equivalent ESS is the instantaneous influence-function variance divided by the jackknife variance of the estimator. It is a precision proxy, not a literal count of independent equilibrated configurations. Some helicity values exceed the 800 retained samples because of anticorrelation or noisy block-variance estimates. The FFT estimator instead imposes its usual sampling floor. Neither estimator includes unvisited sectors or undetected slow tails; four 10k blocks are too few to establish reliable error saturation. Thus **cost per useful equilibrium-independent sample is unresolved**. Tables retain the conditional estimates rather than presenting them as certified efficiency gains.

[native_statistics.csv](results/native_statistics.csv) gives the native-temperature values, variances, ESS and variance-times-cost. [statistics.csv](results/statistics.csv) gives joint-interpolated 1k/5k/10k block results and half-run contrasts. The figure below shows that block-size sensitivity at the common probes; its vertical ranges are sensitivity ranges, not confidence intervals.

![Conditional precision per total simulated hour](figures/08_conditional_efficiency.png)

### Are the endpoints sufficient?

**Cold:** T=0.375 is clearly ordered by chirality at L96 for both couplings. At zero SOC, squared chirality is about 0.255 and Binder about 0.666. This supports its role as a cold reference, but the spin-start differences above prevent claiming full cold-state equilibration. Adequacy at L320 remains an extrapolation.

**Hot:** same-walker residual correlations remove the mean and variance at each temperature, and require the walker to stay above T=0.400 at **every recorded five-sweep exchange** between the two measurements. At lag 500, absolute chirality and composite correlations have magnitude at most about 0.034; scalar spin alignment reaches 0.059. This is encouraging for the global channels. However, random-start C retains energy correlations **0.178 / 0.204 at lag 500** and **0.164 / 0.192 at lag 1,000**, for g_D=0 / 0.01. At lag 500, three walkers contribute approximately 90% / 78% of that covariance. Matched seeds across couplings mean these are not independent replications. The signal could reflect a few selected slow histories; its cause is not established.

The full five-component spin tensor has only **33–36 qualifying hot-stay pairs for B** and **10–45 for C** at lag 1,000, versus about 1,600 for A. These pairs are themselves correlated. Narrow-ladder tensor correlations are therefore poorly determined; one case is omitted for having fewer than 21 pairs. The narrow hot interval also reduces continuous residence above 0.400, so this deficit is not alone proof that 0.405 is too cold. It does prevent asserting adequate tensor-memory loss.

Only **one physical walker across all 12 pilots** completes an eligible sampled cold→hot→cold excursion with |chirality|>0.1 at both cold visits; its sign changes. That is insufficient to measure chirality reset probability. No spatial field configurations were saved, so winding, HQV populations and topology-changing events cannot be reconstructed from these global/Fourier measurements. **T_max=0.405 has not been validated as a topological reset temperature.** The old broad-ladder reference and the present small global correlations cannot establish that missing result.

![Physical-walker energy and spin memory during continuous hot stays](figures/07_physical_memory.png)

The plotted correlations are descriptive and their pairs are not independent. The narrow critical band has too few uninterrupted stays of 500 sweeps or more for these conditional-correlation estimates; missing entries in [memory.csv](results/memory.csv) are not zero correlations. Endpoint-conditioned means provide the better-supported critical-memory diagnostic here.

### Are algorithm changes warranted?

The excess reversals and endpoint-conditioned physical memory support slow structural rearrangement as an additional transport obstacle. High acceptance alone misses it. They do not isolate whether rho=−0.9 or an exchange interval of five is responsible, and there is no correctness finding requiring either implementation to change.

First complete the fixed-kernel A/B/C comparison. If its longer histories still show poor conversion, test **one factor at a time**: either occasional existing rho=0 Gaussian updates at a fixed ladder, or exchange interval 10 versus 5 with the same local-sweep budget and initializations. A rho=0 proposal need not remove collective or topological barriers; less frequent exchange need not improve conversion. Both require measured changes in physical mixing and cost, not an acceptance argument. No refresh schedule, exchange change or new sampler was implemented here.

## Comparison design and execution

The comparison uses the reference specified in the request and saved campaign, not the current contents of `run.py`. That file presently contains a different 502-replica configuration and exchange interval 10; it was **not changed or executed**. All pilots used the requested exchange interval **5**, Gaussian rho=−0.9, proposal mass=0, pressure 20, K=1 and the same pressure-corrected full 3×2 GL Hamiltonian as the saved datasets.

| Ladder | Replicas | Range | Placement |
|---|---:|---|---|
| A | 200 | 0.37–0.42 | Existing function: Tc=0.384, window=0.008, fraction=0.25 |
| B | 200 | 0.375–0.405 | Common energy-fluctuation density described below |
| C | 160 | 0.375–0.405 | The **same density and endpoints as B**, with fewer quantiles |

Thus A→B compares the narrower range **and improved placement at fixed replica count**. It does not isolate range from placement. B→C isolates replica count at fixed density and range. This resolves the previous confounding of replica count with temperature coverage without adding a fourth GPU comparison.

For a correct, equilibrated PT chain, changing the endpoints does not change the canonical marginal at a temperature shared by both ladders. “Physically sufficient endpoints” therefore means sufficient mixing within the available time. A persistent difference at a shared temperature would be a sampling, statistical or implementation issue, not an expected consequence of a different equilibrium ensemble.

The candidate density is the common maximum of sqrt(C_V)/T for the two couplings, using the previous L256 energy envelope, peak-location allowance and smoothly tapered L320 stress-growth scenario. It is frozen before any pilots and used unchanged at L96; no pilot outcomes were used to adapt it. This tests the actual proposed larger-size geometry, rather than optimizing separately for each small size. The minimum spacings are A: 0.000154153, B: **0.0000490743**, C: **0.0000614201**. Exact temperatures are in [ladder_A.txt](ladder_A.txt), [ladder_B.txt](ladder_B.txt) and [ladder_C.txt](ladder_C.txt).

The previous investigation already compared energy and flow-density mixtures. Pure flow feedback could create poor-overlap gaps; a 25% flow contribution was promising but not validated. This pilot keeps that extra choice fixed at zero. It tests the smallest necessary A/B/C comparison instead of spending its short budget on another algorithmic factor. Likewise, neither rho nor exchange frequency was varied.

Each ladder/coupling has one ordered and one random start, for **12 final histories**. Ordered and random runs use different MC/exchange seeds; corresponding ladders use matched seeds. The random field consists of independent full complex Gaussian components with mean density equal to the homogeneous chiral reference density. It is not restricted to a chiral ansatz. Seeds do not make trajectories at different temperatures or replica counts identical, and these are only **one seed per initialization and coupling**, not a replicated performance trial.

The existing `He3GaussianRuntime`, `ParallelTempering`, `Simulation.run()` and `save_runs()` perform the simulation. Each history has **60,000 sweeps**, with all early measurements retained. Analysis designates sweeps 1–20,000 as diagnostic burn-in and the last **40,000** as the comparison window. These are explicitly short diagnostic intervals, not established equilibration lengths. Metadata records both this mask and the fact that the producer saved the complete transient.

The final 12-case batch took **20.8 minutes including setup and saving**. Including the three preserved preliminary cases and the bounded preflight checks, local testing remained roughly half an hour. All GPU work has ended.

Scalar measurements are every 50 sweeps; the existing spatial measurements are every 1,000. The full five-component spin tensor is reused from those spatial channels. No new measurement kernel or recording framework was introduced. Real-space profiles were unnecessary for this test and were not enabled. No L128 or L160 pilot was started: the known L96 slow scale already exceeds the local window, so larger-size local runs would provide weaker evidence at higher cost.

### Hardware and the recording-resolution check

The available local GPU is an **RTX 4070 SUPER, 12 GB**, with approximately 31.85 GiB host RAM. Kraken's RTX 5090 is the separate target specified by the user. The synchronized preflight benchmark used the real Gaussian updates, PT, scalar and spatial measurements. At L96 it measured **1.821 ms/sweep at R200** and **1.465 ms/sweep at R160**. Core device arrays occupy approximately 155/124 MiB; CUDA reported about 10.64 GiB free after setup. Desktop/context allocations are additional.

A preliminary 50-sweep walker history recovered only four of nine exact full-run loops in the zero-SOC ordered B case. The preliminary process was stopped, its three completed archives preserved under `results/coarse_history/`, and all final cases repeated with **history_stride=1: every five-sweep exchange is recorded**. The read-only recording overhead in the preflight was below 1%. Every saved measurement array in the three repeated cases is bitwise identical to its preliminary counterpart. Thus the discrepancy was missed endpoint visits, not altered physics.

Final endpoint histories are checked against the exact per-walker round-trip counters. Sparse recording can disproportionately miss fast visits on the narrow ladder, so comparing uncorrected sampled counts would have biased this investigation. Future long runs need either sufficiently fine histories or the small exact-counter export described in the checkpoint plan.

![The three frozen temperature ladders](figures/01_ladders.png)

## Methods and interpretation of the diagnostics

Physical fields and RNG states remain in walker slots. Only the temperature assignment moves. Every scalar sample is mapped back to its physical walker using the permutation saved at the exact same sweep. Spatial walker IDs are independently checked against the history. Therefore a changed configuration at a fixed temperature is not mistaken for conversion of the same physical configuration.

Endpoint passages are counted over the comparison window, allowing journeys begun during the saved transient to finish in it. Strict production loops require a start and return within the last 40k sweeps. First-passage/first-loop survival curves retain unfinished trajectories and separately count walkers that never start at an endpoint. These are descriptive finite-window distributions; correlated walkers and state-dependent starts do not justify iid survival error bars.

The passage-flux scale is 2R D/N_pass. It is an inverse traffic rate, not an observed uncensored round-trip mean. In these short transient histories it must **not** be interpreted as a stationary mean waiting time. The plotted passage-rate errors describe variation across eight 5k time blocks; they do not measure unvisited tails or seed-to-seed uncertainty.

For a comparison that does not benefit merely from moving the endpoints closer, I also count passages between the same physical temperature surfaces: **T≤0.379 and T≥0.400 at zero SOC; T≤0.383 and T≥0.400 at g_D=0.01**. Both surfaces lie inside every ladder. Event counts are normalized per physical walker and per million sweeps. Cost per passage uses actual elapsed simulation time; tables distinguish production-only cost from total cost including diagnostic burn-in.

Scalar autocorrelations use the existing FFT/initial-monotone estimator. Binder and helicity use moment influence functions. The corresponding fixed-temperature estimator ESS is descriptive: alternating PT is not generally reversible as a whole, and a short trajectory can miss a slow tail. Block-jackknife checks use **1k, 5k and 10k** blocks, leaving 40, 8 and 4 blocks. An apparent improvement in an ESS proxy is not accepted as more useful equilibrium information if the means remain initialization-dependent.

The fixed comparison probes are 0.375, the previous L96 heat-peak temperature and ±0.0003 around it, 0.402 and 0.405. For cross-ladder means and uncertainties, primitive moments are interpolated onto those probes **within the joint time blocks**, then Binder/helicity are calculated. Temperature slots are not resampled independently. Interpolation error is outside the quoted errors. Physical-walker memory is evaluated at actual saved temperatures, not on fictitious interpolated configurations.

The long saved L96 runs, with 3M burn plus 3M production, provide a reference distribution. Their existing 100k joint blocks supply reference errors. Agreement with this reference is reassuring but does not make it exact equilibrium truth. The L128/L160/L256 histories supply the previously measured slow-scale and energy-growth context; no short-pilot speedup is silently extrapolated to those sizes.

## Candidate larger-size settings and their limits

**B is the conservative narrow candidate for the next controlled validation; C is the economical alternative whose reduction in replica count still needs justification. Neither is certified for production.** Keeping R=200 in B makes the first comparison to A easier to interpret. B did not pass every stationarity check, so this choice is an experimental control, not a declaration that it equilibrates. Both use one common energy-based placement for g_D=0 and 0.01. No separate-coupling pilot establishes that common placement is optimal; the previous offline comparison found only a few percent acceptance-only cost benefit from separate energy ladders at L320, insufficient reason to add another local comparison now.

The concrete candidate is simply:

```python
temperatures = np.loadtxt("exploration/he3_pt_ladder_validation/ladder_B.txt")
# R=200, T_min=0.375, T_max=0.405
# Gaussian rho=-0.9, proposal_mass=0.0; exchange_interval=5
# The R=160 comparison uses ladder_C.txt.
```

This is a proposal for later controlled work. The existing temperature-ladder generator remains unchanged. A small precomputed file represents the two critical peaks directly and avoids a new ladder API or campaign system.

### L320 and L384 planning scenarios

Predicted acceptance uses the maximum energy width sampled across each edge. The low/high scenarios hold the previous per-site C_V envelope fixed or allow its critical peak to grow as area, with a smooth return to the background. For transport, the previous measured L256 traffic scale is multiplied by the acceptance-only transport ratio, with either a fixed extra memory penalty or one growing by L/256. These are **uncertainty scenarios, not statistical confidence intervals**. The new short-pilot improvement is not used as an assumed large-size speedup.

| Size, coupling, candidate | Minimum predicted acceptance range | Traffic-scale scenario | First diagnostic burn + production | Estimated elapsed time |
|---|---:|---:|---:|---:|
| L320, g_D=0, **B200** | 0.66–0.69 | 6.37–8.53M | 2M + 4M | 12.3–13.7 h |
| L320, g_D=0.01, **B200** | 0.66–0.68 | 4.40–5.84M | 2M + 4M | 12.3–13.7 h |
| L384, g_D=0, **B200** | 0.54–0.63 | 8.14–14.26M | 4M + 8M | 35.4–43.3 h |
| L384, g_D=0.01, **B200** | 0.55–0.63 | 5.63–9.67M | 4M + 8M | 35.4–43.3 h |
| L320, g_D=0, C160 | 0.58–0.61 | 5.55–7.48M | 2M + 4M | 9.8–11.0 h |
| L320, g_D=0.01, C160 | 0.59–0.61 | 3.84–5.13M | 2M + 4M | 9.8–11.0 h |
| L384, g_D=0, C160 | 0.44–0.54 | 7.24–12.95M | 4M + 8M | 28.3–34.7 h |
| L384, g_D=0.01, C160 | 0.44–0.54 | 5.03–8.80M | 4M + 8M | 28.3–34.7 h |

The temperatures above are the **same frozen B and C files**, including for L384; these predictions differ slightly from reoptimizing a new density separately at L384. B200 has more overlap, but costs about 25% more per sweep and has longer acceptance-only travel times. A200 retains the widest temperature coverage, but its stress-case minima fall to 0.20/0.28 at L320 and 0.066/0.118 at L384.

The diagnostic lengths are initial observation allocations, **not adequate equilibrium production lengths established by this study**. They should follow the smaller-size validation below. Lengths must be extended or rejected based on independent-start agreement, block stability and physical conversion, not just completing the nominal allocation.

Timing uses the actual saved campaign costs, bracketed by L² scaling and the measured recent exponent 2.50, with linear replica-count scaling. Those manifests do not identify their GPU, so their applicability to Kraken's RTX 5090 remains an assumption. The new direct timing is on a 4070 SUPER and is labelled separately. Do not convert between GPUs using an assumed hardware-specification speed ratio. Reserve at least 20% of the 72-hour allocation for timing error and I/O.

At R160 the core field/RNG/counter budget is **1.34 GiB at L320 and 1.93 GiB at L384**. Retaining every five-sweep permutation adds about **0.72 GiB for the 6M L320 diagnostic** and **1.43 GiB for the 12M L384 diagnostic** on the GPU, before workspace/context. Those histories are twenty times larger than 100-sweep recording. The measurement payload, allowing the existing production scalar/spatial/profile cadences, is approximately **2.12 GiB and 4.26 GiB**. A simple array-duplication/history budget reaches roughly **7.1 GiB and 14.2 GiB host memory**, excluding Python/JSON/compression overhead. Plan at least 32 GB host RAM for these diagnostic allocations and prefer 64 GB for longer continuations; Kraken's host allocation is unspecified.

For B200, the corresponding L320/L384 core GPU budgets are **1.68/2.42 GiB**, five-sweep histories **0.89/1.79 GiB**, measured-array payloads **2.63/5.28 GiB**, and approximate host array peaks **8.8/17.7 GiB**, before overhead. Long fine-resolution histories stored inside JSON can become a substantial memory/I/O cost; the exact traffic-counter export and binary checkpoint plan avoid treating that representation as a permanent architecture.

As a transparent duration scenario, two traffic scales for burn-in plus three for production would put **B200** at about **32–43M / 22–29M sweeps at L320 for g_D=0 / 0.01**, and **41–71M / 28–48M at L384**. This is a budgeting convention, not an equilibrium theorem. At the recent timing trend these cost approximately **73–98 h / 50–67 h at L320**, and **147–257 h / 102–175 h at L384**, before the wall-time reserve. C160 would reduce the corresponding cost scenarios to **51–68 h / 35–47 h** and **105–187 h / 73–127 h**, if its sampling is validated. Thus continuation should be planned for L320 and is necessary for a reliable L384 program. Nothing here supports an L512 recommendation.

Detailed alternative A/B/C budgets, history storage and assumptions are in [larger_size_scenarios.csv](results/larger_size_scenarios.csv).

## Minimal checkpoint plan

[CHECKPOINT_PLAN.md](CHECKPOINT_PLAN.md) specifies the existing objects' state, a minimal serialization/resume path and bounded exact-continuation tests. **No checkpointing was implemented.** The essential state is the full field, all local/exchange RNG arrays, both temperature maps, actual betas, next matching parity, sweep/counter state, complete partial-trip bookkeeping, energy-cache validity, accumulated measurements and history, and the model's measurement counter.

Two easy-to-miss issues are explicit in the plan. Repeated calls to `Simulation.run()` currently restart its local sample-index metadata; spatial record indices also assume one accumulated scalar history. Both must resume consistently to avoid duplicate or missing events. Additionally, exact endpoint passages can be recovered from the existing per-walker bookkeeping as **2 × completed trips + visited-opposite flag**. Exporting that small state at block boundaries would avoid relying on enormous fine-resolution histories for passage rates, without changing the exchange kernel.

## Smallest useful next experiment and interim choice

The smallest next **screen** is a matched L96 A/B/C comparison at both couplings and both initializations with **0.5M diagnostic burn plus 1M production per history**. That is 12 histories, about **2.08 hours total** at the saved campaign timing, or roughly **2.5 hours with a 20% timing allowance**. On the measured local 4070 SUPER it would take about 8.5 hours and is outside this investigation's authorization; it was not launched. The nominal 0.5M burn remains a hypothesis to check, not a guarantee.

Proceed only if different starts agree at the common cold/critical/hot probes, estimated errors stabilize with increasing block size, endpoint and interior-surface traffic are adequately sampled, and no structural-memory problem worsens. Require effect sizes and uncertainties across independent seeds before declaring B or C superior. If the histories still drift, extend the comparison instead of selecting the fastest-looking transient.

If that screen passes, compare A with the selected narrow ladder at **L128**, again at both couplings and both starts, provisionally **2M + 3M sweeps**. The eight histories cost approximately **9.48 aggregate hours for A+B**, or **8.53 hours for A+C**, before a 20% timing allowance, at the saved campaign timing. The L128 rows in the cost table are alternatives; they do not prescribe running all three ladders. This is a proposed follow-up, not a launched campaign. L160 or a focused L256 endpoint check should then be chosen according to the observed size trend; L96 success alone cannot certify L320 topological equilibration.

In the meantime, retain **A200, 0.37–0.42, rho=−0.9, mass=0, exchange interval 5** as the reference for controlled comparisons at the already studied sizes. This recommendation does not claim that A's existing L256 histories are fully equilibrated, and does not change the current runner. Do not launch an L320 equilibrium production campaign on the strength of these short pilots.

## Reproduction and preservation

All new code, pilot archives, tables and figures are in this directory. The six small scripts reuse the existing simulation and analysis utilities. No production engine, Hamiltonian, sampler, PT kernel, existing measurement module, runner, Slurm script, previous analysis or input dataset was modified. No cluster job was submitted and no currently running simulation was stopped; the only interrupted process was this investigation's preliminary pilot.

The final raw records have the usual `save_runs()` format. `diagnostic_burn_sweeps=20000` defines the analysis mask, while the stored scalar samples deliberately include all 60k sweeps. Do not interpret the producer's zero thermalization value as a claim that the initial field was equilibrated.

```text
python -B exploration/he3_pt_ladder_validation/ladder_design.py
python -B exploration/he3_pt_ladder_validation/pilot.py --benchmark
# The following is the bounded GPU experiment; it refuses to overwrite its files:
python -B exploration/he3_pt_ladder_validation/pilot.py --sweeps 60000
python -B exploration/he3_pt_ladder_validation/analyze.py
python -B exploration/he3_pt_ladder_validation/resources.py
python -B exploration/he3_pt_ladder_validation/figures.py
python -B exploration/he3_pt_ladder_validation/validate.py
```

Validation passed for all 12 archives: finite measurements; exact scalar and spatial sampling coordinates; temperature/walker inverse permutations; float32 beta assignments; expected Gaussian and exchange attempt counts; no nonfinite rejections; matching scalar/spatial energy and order-parameter channels; and reconstructed per-walker full-run trip counts equal to the exact PT counters. The three recording-resolution repeats also have bitwise-identical observable arrays. [validation.json](results/validation.json) records the checks.

The preservation audit confirms unchanged hashes for **147 previous Python files**, unchanged size and modification time for **18 original datasets**, unchanged Git branch and HEAD, and empty tracked and staged diffs. Previous untracked exploratory work remains present. Dataset contents were not independently rehashed; preservation is checked by size/mtime and by making no writes to them. New files were confined to this investigation.

## Explicit decision

**Is the narrower ladder sufficiently validated for L320? No.** Faster transient traffic is measured, but slow physical modes and independent-start/block stability are not resolved.

**Smallest additional experiment:** the costed 12-history L96 A/B/C screen, 0.5M diagnostic burn plus 1M comparison sweeps each, with the same kernels and independent starts. Treat its adequacy as a result to assess. Advance to the costed L128 A-versus-selected-narrow comparison only if the L96 evidence supports it; larger-size checks remain necessary before claiming L320 equilibration.

**Interim configuration:** retain the requested A200 reference, 0.37–0.42, Gaussian rho=−0.9, mass=0 and exchange interval 5 for controlled comparisons. B200, 0.375–0.405 using [ladder_B.txt](ladder_B.txt), is the conservative next candidate; C160 is the cost-saving comparison. Do not treat either as an authorized or validated L320 production setup. No runner, production setting, engine, checkpoint implementation or Slurm job was changed or launched.
