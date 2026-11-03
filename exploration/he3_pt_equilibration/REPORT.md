# Equilibration and parallel-tempering scaling in thin-film He3

2 November 2026. Read-only investigation of 18 completed archives at pressure 20, using all available sizes through L=256 at g_D=0 and 0.01. **Both L=256 archives are now available.** L=64 is included in transport/scaling diagnostics as requested here; it remains excluded from every transition crossing comparison.

## Main findings

The main limitation is **slow configuration-dependent transport through the ordering region**, beyond what neighboring swap acceptance predicts. Energy fluctuations sharpen with size, but an acceptance-only model substantially underestimates the observed slowdown. Physical configurations arriving from opposite ends of the ladder have different energy, chirality and composite order even at the same temperature. This is direct evidence of memory beyond temperature index; it does not by itself prove that the stationary distribution has been sampled incorrectly.

Completed-trip averages understate the planning problem. At L=256 they are 3.18M and 3.02M sweeps, but most walkers have no complete production loop. An endpoint-passage-rate estimate that includes the exposure of all walkers gives **7.64M and 5.19M sweeps per walker round-trip equivalent**, respectively. These are finite-window traffic scales under a stationarity assumption, not measured uncensored mean waiting times or rigorous future bounds.

The recorded global observables lose memory in the hot region. There is no evidence here that raising T_max beyond 0.42 is the first remedy. Reducing the temperature range, improving placement without increasing replica count, and comparing independent initial states are more promising. Whether vortices or winding sectors also forget their history cannot be determined from the saved data.

On the specified **RTX 5090**, GPU capacity is unlikely to be the first constraint. Run duration and reliable continuation are more restrictive. A conditional cost estimate puts an L=320, 160-replica, 8M+8M diagnostic allocation near 26–29 hours. That allocation would still be too short to certify equilibration using the present transport rates. L=384 generally needs continuation across allocations; L=512 is not justified yet.

No simulation was launched. No production source, run configuration, sampler, PT implementation, Hamiltonian, measurement module or Slurm script was changed.

## 1. What actually ran

The old campaign is stored under `data/he3/`, despite its manifest name `soc_scan_02`. The new campaign is under `data/weak_soc_spatial_simple_01/`. The exact files, settings, timing and transport counters are in [inventory.csv](results/inventory.csv); compact per-run manifest extracts accompany it.

| Saved setting | Old: soc_scan_02 | New: weak_soc_spatial_simple_01 |
|---|---:|---:|
| Sizes at each coupling | 64,96,128,160 | 64,96,128,160,256 |
| Replicas | 160 | 200 |
| Temperature range | 0.33–0.47 | 0.37–0.42 |
| Thermalization + production | 2M + 3M | 3M + 3M |
| Gaussian rho / proposal mass | −0.9 / 0 | −0.9 / 0 |
| Exchange interval | **5 sweeps** | **5 sweeps** |
| Scalar cadence / production events | 50 / 60,000 | 50 / 60,000 |
| Walker-history cadence, including burn-in | 500 | 100 |
| Spatial / real-space profile cadence | unavailable | 1,000 / 10,000 |
| New-run initialization | — | ordered, positive chirality, d along z, phase zero |

The new ladder dictionary is T_min=0.37, T_max=0.42, n_T=200, Tc=0.384, tc_window=0.008, tc_fraction=0.25. Its minimum spacing is 0.000154153. These are saved settings, irrespective of the present runner. The new runs share the recorded model parameters; the Gaussian acceptance counts and exact PT trip counters include thermalization.

The implementation alternates even and odd matchings every five sweeps: **a particular edge is attempted every ten sweeps**. Physical fields and their local RNG states stay with their walker; only temperature assignments move. Scalar measurements are saved in temperature order. This analysis reconstructs physical-walker measurements using the saved inverse permutation at exactly matching sample times. Spatial walker labels were checked against that permutation. A change at a temperature slot is therefore never automatically counted as physical conversion.

## 2. Transport, censoring and energy overlap

### Completed trips are a selected subset

Times below are in millions of sweeps. Old/new completed means are exact counters over the entire saved run, including burn-in. Production coverage uses sampled histories and requires visits or complete loops within the 3M production window.

| g_D | L | Old completed mean | New completed mean | New passage-flux scale | Both endpoints in production | Production-loop walkers |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 0.393 | 0.132 | 0.168 | 200/200 | 200/200 |
| 0 | 96 | 1.191 | 0.374 | 0.415 | 200/200 | 200/200 |
| 0 | 128 | 2.200 | 0.815 | 0.919 | 199/200 | 197/200 |
| 0 | 160 | 3.061 | 1.422 | 1.813 | 184/200 | 171/200 |
| 0 | 256 | — | **3.179** | **7.643** | **97/200** | **41/200** |
| 0.01 | 64 | 0.422 | 0.150 | 0.183 | 200/200 | 200/200 |
| 0.01 | 96 | 1.103 | 0.380 | 0.422 | 200/200 | 200/200 |
| 0.01 | 128 | 2.081 | 0.769 | 0.879 | 199/200 | 198/200 |
| 0.01 | 160 | 2.771 | 1.349 | 1.546 | 195/200 | 182/200 |
| 0.01 | 256 | — | **3.023** | **5.195** | **127/200** | **64/200** |

The previously reported zero-SOC numbers are reproduced. Its L=256 history contains 43 strict production loops from 41 walkers, while exact counters contain 123 loops over all 6M sweeps. For g_D=0.01 these are 67 loops from 64 walkers and 171 exact full-run loops. The sampled full histories recover 122/123 and 171/171 loops, respectively. This agreement checks that sampling resolution is adequate for the largest-size comparison.

For the passage-flux scale, count alternating endpoint-to-opposite-end arrivals **during production**, including journeys begun before production. If N_pass passages arrive in duration D with R walkers, define

\[
\tau_{\rm flux}=\frac{2RD}{N_{\rm pass}}.
\]

At L=256, N_pass=157 and 231. Six 500k time blocks give conditional jackknife errors of approximately **0.18M and 0.21M**. Those errors describe fluctuations of the observed rate, not uncertainty from unvisited slow sectors or the stationarity assumption. A small rate error does not imply that individual walks have completed many independent cycles. At small L, missed endpoint visits at the saved cadence inflate this scale; the small-size flux values should not be used to fit a precise exponent.

Each 3M burn-in or production period is only about 0.39/0.58 of these L=256 traffic scales. The present 3M+3M allocation supplies useful measurements, but cannot be certified as adequate thermalization and production from its completed-trip averages.

First-production-loop survival curves explicitly retain right-censored walks. Their median is **not reached** at either L=256 coupling: 159/200 and 136/200 first loops are censored. At L=160 the descriptive medians are 1.49M and 1.39M. The pooled opposite-end first-passage medians rise from 0.60M/0.55M at L=160 to 2.79M/1.77M at L=256. These Kaplan–Meier curves are descriptive: starts have different states, repeated passages and walkers are correlated, and finite windows restrict the tail. No iid survival confidence intervals or uncensored tail means are claimed.

![Transport scales and censoring](figures/02_transport_censoring.png)

### Acceptance improves, but does not explain transport

At old L=160, minimum acceptance was 8.7%/13.0% for g_D=0/0.01. It is 66.1%/69.2% in the new L=160 runs, and 43.9%/49.8% at new L=256. The old-to-new comparison includes changed endpoints, replica count, burn-in and measurement workload; it is not a controlled experiment on one ladder parameter.

At the L=256 minimum-acceptance edge:

| g_D | Temperature midpoint | Recorded acceptance | Product-energy prediction | Gaussian prediction | Histogram overlap integral |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.381874 | 0.439 | 0.436 | 0.449 | 0.573 |
| 0.01 | 0.385594 | 0.498 | 0.492 | 0.507 | 0.619 |

The product prediction pairs time-offset energy histories to estimate independent marginal draws. It is checked against simultaneous-sample predictions as well. The Gaussian formula is

\[
A_{i,i+1}\simeq\operatorname{erfc}\!\left(\frac{|\Delta\beta|\,\sigma_H}{2}\right),\qquad
\sigma_H=L T\sqrt{C_V}.
\]

Across the full new L=256 ladders, Gaussian-minus-product mean absolute errors are 0.0013 and 0.0011; maximum errors are 0.0133 and 0.0145, near the narrow dip. This supports its use for cautious spacing predictions. It does not validate Gaussian dynamics or future-size equilibrium. Histogram overlap \(\int\min(p_i,p_{i+1})dE\) and swap acceptance are different quantities. Acceptance counters are cumulative whereas energy histograms are production-only.

### The acceptance-only random walk is much too fast

I solved a Markov chain on **(temperature index, next matching parity)** using the measured edge acceptance rates. Including parity matters: deterministic alternating matchings can support persistent motion, whereas a symmetric nearest-neighbor walk omits it. At acceptance one the implemented null gives the exact ballistic round-trip limit 2R times the exchange interval.

| New L=256 transport scale, M sweeps | g_D=0 | g_D=0.01 |
|---|---:|---:|
| Alternating acceptance-only model | 0.169 | 0.173 |
| Symmetric nearest-neighbor model | 0.565 | 0.569 |
| Markov model fitted to observed 500-sweep index transitions | 1.319 | 1.328 |
| Actual completed-trip mean | 3.179 | 3.023 |
| Observed passage-flux scale | **7.643** | **5.195** |

Thus neither ladder length and mean edge acceptance, nor even the empirical one-step coarse transition matrix, reproduce global transport. The flux scales exceed the alternating null by factors **45 and 30**. Correlated reversals and persistent configuration state are consistent explanations. This comparison cannot uniquely partition the slowdown into chirality, spin, vortices and other collective coordinates.

The last-endpoint flow f(T) supplies the location: approximately **80% and 64%** of its drop is within a width 0.002 centered on the respective specific-heat peak. The corresponding fractions at L=64 are only 27%/27%. The sharpest large-size flow edges coincide with the acceptance bottlenecks.

![Acceptance, endpoint flow and specific heat](figures/01_acceptance_flow_heat.png)

At the L=256 specific-heat grid peak, the measured index-displacement variance divided by twice a 500-sweep lag is 0.00452/0.00628, against acceptance-only values 0.244/0.238: suppression by factors about 54/38. These lag-dependent diagnostics are not infinitesimal diffusivities; at long lags finite ladder boundaries also constrain the variance.

Residence data show frequent short recrossings of the narrow critical band, alongside long stays on either side. Define the band as T_peak±0.0003. The 99th percentiles of **completed** cold-side/hot-side residence episodes at L=256 are approximately 207k/379k sweeps at zero SOC and 236k/202k at g_D=0.01. Some hot-side episodes span the entire production window and are censored. Brief residence inside the narrow band therefore does not imply efficient conversion across it. Raw censor flags and full distributions are saved in [residence.csv](results/residence.csv).

![First passages and sampled residence distributions](figures/08_passages_residence.png)

## 3. Physical memory, autocorrelation and uncertainty

### Configurations remember their arrival side

At a fixed temperature, separate configurations by the last endpoint visited, preserving the physical-walker identity. Near the bottleneck:

| g_D, T | Quantity | Cold-label mean | Hot-label mean | Difference ± conditional SE |
|---|---|---:|---:|---:|
| 0, 0.381796 | energy/site | 1.31105 | 1.32316 | −0.01211 ± 0.00076 |
| | squared normalized chirality | 0.08695 | 0.05723 | +0.02972 ± 0.00195 |
| | squared composite amplitude/site normalization | 0.03445 | 0.02215 | +0.01231 ± 0.00079 |
| 0.01, 0.385515 | energy/site | 1.33839 | 1.34800 | −0.00961 ± 0.00052 |
| | squared normalized chirality | 0.06726 | 0.04357 | +0.02369 ± 0.00127 |
| | squared composite amplitude/site normalization | 0.02596 | 0.01656 | +0.00940 ± 0.00054 |
| | recorded spin alignment | 0.27978 | 0.26354 | +0.01624 ± 0.00083 |

The squared-chirality differences are about 0.90 and 0.77 instantaneous standard deviations. Errors use 30 joint 100k time blocks, not independent walker samples. Such last-visited-endpoint conditioning selects different reactive histories even in equilibrium. It demonstrates slow structural conversion; it is not an equilibrium-bias test on its own.

![Suppressed motion and endpoint-conditioned structure](figures/03_diffusion_structural_memory.png)

Physical-walker memory was also calculated after subtracting temperature-dependent mean values and standardizing at each temperature. For configurations continuously sampled within T_peak±0.0005, the 500-sweep spin-alignment residual correlation is **0.965 at zero SOC**, versus 0.191 at g_D=0.01. Chirality-magnitude and composite-power residual correlations are about 0.35–0.38 and 0.32. Conditioning on staying in the band biases these correlations; they are not canonical integrated autocorrelation times. The pooled zero-SOC spin residual still has correlation about 0.069 at 100k sweeps. Pooling all temperatures otherwise hides much of the critical memory.

### What the hot region does and does not establish

During continuously sampled visits with T≥0.40, the 500-sweep correlations of recorded chirality, composite-power residual and spin residual are all close to zero (spin: 0.0034/0.0076). This resolves relaxation only down to the 500-sweep observation cadence, for these global quantities.

For cold→T≥0.40→cold excursions on the **same** walker, requiring |m|>0.1 at both cold observations:

| L=256 | g_D=0 | g_D=0.01 |
|---|---:|---:|
| Eligible excursions / distinct walkers | 20 / 18 | 47 / 43 |
| Opposite chirality, first eligible excursion per walker | 8/18 | 23/43 |
| Opposite-sign probability from independent before/after marginals | 0.506 | 0.502 |
| Mean normalized spin-tensor dot product, first excursion | 0.048 | 0.994 |

The sign results are compatible with substantial chirality forgetting among successful excursions. The sample is small and selected; almost no walker contributes many such excursions. The near-unity spin overlap under SOC is expected from the preferred axis and **must not be interpreted as trapping**. The zero-SOC tensor overlap is close to zero. Residual spin-component correlations across these small selected samples remain inconclusive. Raising the hot threshold to 0.41 leaves the qualitative conclusion unchanged.

Saved global moments, a few Fourier modes and averaged real-space profiles do **not** reconstruct winding, HQV populations, vortex annihilation or chirality-domain topology. Structural barriers are demonstrated; topological trapping is an untested hypothesis. Occasional actual configurations or validated topological summaries would be needed in a separately authorized future run.

### Fixed-temperature statistics and their limits

The existing FFT/initial-monotone autocorrelation utility was reused. Alternating PT is not generally reversible as a whole, so this truncation is a descriptive diagnostic, not a guaranteed variance bound. Binder and helicity use their moment influence functions, rather than treating instantaneous ratios as measurements of the nonlinear estimator. At the measured specific-heat peak:

| g_D | L | energy tau | m² tau | Binder influence tau | composite-power tau | spin tau |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 59 | 70 | 64 | 65 | 50 |
| 0 | 96 | 157 | 180 | 158 | 168 | 143 |
| 0 | 128 | 302 | 359 | 316 | 332 | 237 |
| 0 | 160 | 508 | 641 | 547 | 598 | 361 |
| 0 | 256 | **2,604** | **3,478** | **2,286** | **3,237** | **2,077** |
| 0.01 | 64 | 63 | 72 | 71 | 67 | 63 |
| 0.01 | 96 | 117 | 134 | 123 | 128 | 114 |
| 0.01 | 128 | 237 | 292 | 259 | 274 | 245 |
| 0.01 | 160 | 381 | 487 | 403 | 455 | 400 |
| 0.01 | 256 | **1,111** | **1,585** | **1,139** | **1,444** | **1,149** |

All times are sweeps. Signed chirality and the composite real component are also included in [fixed_temperature_acf.csv](results/fixed_temperature_acf.csv). The m² effective sample counts N/(2 tau), for the 3M production window, fall to approximately **431 and 946** at L=256. These are descriptive, observable-specific counts within the visited ensemble. PT swaps help temperature-slot decorrelation without requiring the same physical configuration to change sector.

The helicity influence has a large fast-noise contribution and a much shorter descriptive tau, about 29/26 sweeps at L=256. Its nominal ESS of 52k/59k is not evidence that the ordered configurations or topological sectors equilibrate that quickly. Direct long-block variance estimates give zero-SOC helicity ESS near 29k with 100k or 250k blocks, illustrating sensitivity to a small slow component. Raw-moment/block checks are in [long_block_ess.csv](results/long_block_ess.csv).

At zero-SOC L=256 the Binder SE at the heat peak is 0.0103, 0.0124 and 0.0088 using 100k, 250k and 500k blocks. Energy SE is 0.00063, 0.00087 and 0.00057. Only 30, 12 and 6 blocks remain. The smaller error with six blocks does not establish convergence or justify selecting the smallest error. For g_D=0.01, L=160, the near-critical Binder at T=0.385829 changes from 0.4065 to 0.3668 between production halves, a shift −0.0397 with combined conditional SE 0.0077. Stationarity concerns therefore persist in a source size used for L=256 crossings.

![Autocorrelation growth and block sensitivity](figures/04_acf_blocking.png)

## 4. Scaling: sharper physics and growing memory

The new zero-SOC sampled heat peak grows from 46.6 at L=64 to 86.6 at L=160 and 114.8 at L=256; with SOC the corresponding values are 44.9, 70.5 and 84.4. The full width above half the raw peak shrinks from 0.001109 to 0.000666 between L=160 and 256 at zero SOC, and from 0.001411 to 0.000952 at g_D=0.01. These are descriptive, grid-interpolated widths, not fitted critical exponents.

Total-energy widths at the peak grow from 569 to 1,048 and from 518 to 907. Thus simple sigma_H proportional to L understates the increase: C_V is growing too. Local energy reweighting using the existing WHAM solver places the L=256 peaks near **0.381918 and 0.385559**, with peak C_V about 116.8 and 85.9. Fine-grid interpolation does not create independent information, and these peak estimates inherit sampling limitations.

The heat peak, the chirality variation, the flow bottleneck and endpoint-conditioned structural differences occur in the same shrinking region. This is evidence for a critical structural limitation beyond an extensive-volume penalty. It is not evidence sufficient to select a continuous-transition universality class or a first-order barrier law.

Between new L=160 and 256, the passage-flux time grows by factors 4.22 and 3.36. Writing that single step as a power gives effective exponents **3.06 and 2.58**. The completed-trip means give much smaller apparent exponents because the long tail is increasingly censored. Neither should be promoted to an asymptotic law. The available sizes and single-start histories cannot distinguish a power law from the onset of faster barrier growth.

Measured elapsed simulation cost per sweep grows by a factor 3.24 over the same size step, equivalent to a local timing exponent 2.50. Combined with endpoint traffic, the time per walker round-trip equivalent worsens by about **14-fold/11-fold**. At L=256 this traffic scale costs roughly 10.0/6.8 hours at the recorded timing, even though the entire saved 6M run took only 7.86 hours. Precision per GPU hour therefore deteriorates much faster than site count alone suggests.

## 5. Ladder and exchange-frequency options

### Offline prediction procedure

All candidate temperatures are analysis artifacts only. [design.py](design.py) makes the calculation reproducible; [ladder_predictions.csv](results/ladder_predictions.csv) contains both couplings and all scenarios.

For each coupling, I refine the critical C_V curve using local WHAM, add twice its raw-grid 100k-block jackknife SE, and broaden the critical envelope over ±0.0003 in temperature. This is an engineering allowance for unresolved peak shape/location, **not a confidence bound**. Predictions use the maximum sigma_H over 17 points across each proposed edge, preventing a narrow peak from disappearing between edge midpoints.

Two future scenarios are evaluated: unchanged per-site C_V, and a stress case whose peak C_V grows as (L/256)². The latter enhancement tapers smoothly to the background with exp[−((T−T_peak)/0.0015)^4]. It is deliberately stronger than the observed last size step, but is not a theorem or an identified first-order law.

The energy-based common ladder uses temperature density proportional to max_g sqrt(C_V,g)/T. Flow feedback uses the measured smoothed endpoint-flow drop to estimate the usual inverse-square-root diffusivity density. The feedback theory motivates optimizing traffic rather than constant acceptance, but assumes a meaningful diffusivity that can itself change when the ladder changes. See [Katzgraber, Trebst, Huse and Troyer](https://arxiv.org/abs/cond-mat/0602085).

I also tested **15 small parameter changes through the existing, unchanged make_temperature_ladder function**: center at the midpoint of the two measured peaks, windows 1.2/1.6/2 times their separation, and fractions 0.25/0.35/0.45/0.55/0.65. Selection minimizes the worst-coupling acceptance-only round-trip time at fixed R=200. This is a transparent surrogate objective, not a measured physical optimum.

### What changes are worth testing

The following uses the L=320 stress scenario. Paired values are g_D=0 / 0.01. Times are acceptance-only round trips, deliberately **not** claimed actual mixing times.

| Candidate | R | Minimum predicted acceptance | Null round trip, k sweeps | Fixed-diffusivity flow-time ratio |
|---|---:|---:|---:|---:|
| Current 0.37–0.42 | 200 | 0.202 / 0.277 | 284 / 273 | 1 / 1 |
| Retuned existing ladder function | 200 | 0.334 / 0.405 | 268 / 266 | 0.765 / 0.862 |
| Common energy-based placement | 200 | 0.541 / 0.546 | 259 / 264 | 0.589 / 0.714 |
| 75% energy + 25% flow density | 200 | 0.482 / 0.480 | 268 / 274 | 0.472 / 0.616 |
| 50% energy + 50% flow density | 200 | 0.357 / 0.354 | 292 / 296 | 0.414 / 0.562 |
| Same 50% blend, more replicas | 256 | 0.462 / 0.460 | 321 / 326 | 0.413 / 0.562 |
| Flow density alone | 200 | 0.044 / 0.043 | 579 / 572 | 0.381 / 0.537 |
| Common energy placement, **0.375–0.405** | 160 | **0.580 / 0.587** | **132 / 137** | 0.410 / 0.488 |

The flow ratios hold inferred diffusivity fixed. They are optimistic geometric diagnostics, not validated speedup factors. In particular, the apparent excellent flow ratio for pure feedback conflicts with its new low-overlap gaps. Retain an energy-spacing contribution instead of using flow alone.

The best L=320 simple-function candidate in that small scan is:

```python
temperature_ladder = dict(
    T_min=0.37, T_max=0.42, n_T=200,
    Tc=0.3837385536, tc_window=0.0058250853, tc_fraction=0.25,
)
```

This is a **proposal, not an active setting**. Its minimum spacing is 0.00011065. The analogous L=384 stress-case scan selects fraction 0.35, minimum spacing 0.00007793. The single smooth enhancement can cover both transitions, although a small precomputed energy-based file resolves the two distinct peaks more efficiently. There is no need to change the ladder function or build ladder infrastructure.

Separate per-coupling energy ladders reduce the L=320 null times to 250k/256k, only about 3–4% below the common energy ladder. Sharing one well-placed ladder between these two couplings is therefore a reasonable convenience at this level. At smaller sizes it is deliberately inefficient. These calculations do not newly validate g_D=0.001; that coupling was outside the requested dataset focus.

Adding 56 replicas to the 50% blend improves minimum acceptance, but increases both null travel time and replica-sweep work by about 40%. High acceptance is not an adequate reason for doing it. Likewise, raising the hot endpoint has no demonstrated benefit for the recorded global memory. The narrower 0.375–0.405 range is promising because it retains an ordered cold region for both couplings and extends above the region where recorded hot memory decays. It needs an L≤256 comparison against the wide range before being trusted for production, especially for topology.

![Ladder tradeoffs at L=320](figures/06_ladder_tradeoffs.png)

### Exchange every 5 or 10 sweeps

Every analyzed run used interval 5. There is **no measured interval-10 comparison**. Holding independent edge acceptance fixed, doubling the interval doubles the null transport time in sweeps. In the real chain, additional local evolution might reduce correlated reversals; less frequent energy reduction/exchange may also save wall time. Neither effect can be quantified from the current counters alone.

Keep 5 as the evidence-based baseline. If testing 10 later, compare matched runs with the same ladder, local kernel and sweep budget, and report flow, configuration memory and block error per elapsed hour. A bounded L=128 or 160 comparison can reject an obviously bad change; it cannot establish an L=256 improvement if its window contains too few passages. The broader reason to compare exchange frequency against physical relaxation rather than swap acceptance is discussed by [Bittner, Nußbaumer and Janke](https://arxiv.org/abs/0809.0571). No such simulation was run here.

## 6. Gaussian sampler assessment

[gaussian_fast.py](../../models/he3/gaussian_fast.py) proposes all twelve real components at a site using a conditional Gaussian autoregression, followed by **full-Hamiltonian Metropolis–Hastings correction including the reverse/forward proposal ratio**. The two reference orbital precisions are 6K+mass and 10K+mass. The four-color composition preserves the target in exact arithmetic, although the entire ordered sweep need not satisfy detailed balance as one reversible kernel. The existing float32 arithmetic remains a numerical limitation, not a new defect established by this investigation.

At zero SOC the recorded mean local acceptance is 70.26% at L=64 and 70.38% at L=256. At g_D=0.01 it is about 69.61–69.66%. These are cumulative **walker-indexed** counters, not temperature-resolved acceptance profiles. Falling local acceptance is therefore not the observed source of size-dependent slowdown.

Negative rho is stochastic overrelaxation. A useful exact toy calculation for a fixed-neighbor Gaussian AR process gives

\[
\tau_x=\frac{1+\rho}{2(1-\rho)},\qquad
\tau_{x^2}=\frac{1+\rho^2}{2(1-\rho^2)}.
\]

For rho=−0.9 these are 0.0263 and 4.763 updates, versus 0.5 and 0.5 for rho=0. The injected noise variance is 1−rho²=0.19 of the reference variance. Thus anticorrelation can help a linear mode while preserving even-amplitude memory; good acceptance or fast phase decorrelation does not guarantee rapid rearrangement of domains. These formulas do not predict the nonlinear He3 relaxation times.

The smallest justified local-update experiment would compare the current kernel against occasional **state-independent rho=0 refresh sweeps**, keeping the same full MH correction. A state-independent mixture of invariant kernels remains invariant; changing proposal mass similarly requires retaining the actual proposal-density ratio. Do not use state-dependent tuning during measurement without a justified stationary kernel. There is no present evidence selecting an optimal refresh fraction or mass, and no alternative was implemented.

Global symmetry rotations alone cannot remove chirality-domain or winding barriers, and a cluster algorithm cannot be assumed valid for this full complex tensor Hamiltonian. If modest ladder/endpoint/local-refresh changes fail, the simplest next candidate is a fixed umbrella bias in energy or squared chirality, selected from the observed bottleneck. A dimensionless bias W would sample exp[−beta H−W], include −delta W in the acceptance ratio, and recover canonical averages with weights exp[W]. This is a proposed auxiliary sampling distribution, not a change to the physical Hamiltonian. It first needs overlapping histograms, acceptable reweighting effective sample size, independent-start agreement and evidence that spin/winding barriers do not remain hidden orthogonal to the biased coordinate. There is insufficient evidence to build it now.

## 7. Implications for transition results

The [previous L=256 report](../he3_transition_L256/REPORT.md) correctly treats the leading half-charge Weber–Minnhagen estimate as less robust. Its primary four-size fit has chi²/dof=10.07/2, while direct-grid errors and production-half choices alter fit quality. Neither the new transport evidence nor a direct jump crossing repairs that model sensitivity.

Here I rechecked **raw-grid, linearly interpolated** crossings using the existing crossing function. The primary bootstrap uses 30 joint 100k blocks and 600 resamples; 500k blocks and separate production halves use 200 resamples. All temperatures and channels from a run share each resampled block. Different sizes are resampled independently, and the Binder-minus-helicity contrast retains their shared L=256 fluctuations. Interpolated temperatures are not independent observations.

The modulus convention remains

\[
Y=\frac{\langle K_x+K_y\rangle-[\mathrm{Var}(I_x)+\mathrm{Var}(I_y)]/T}{2L^2},
\qquad Y_c=\frac{2T_c}{\pi q^2}.
\]

The direct half-charge line is therefore 8T/pi. The table concerns finite-size proxies, not thermodynamic critical temperatures:

| g_D | Binder (160,256), raw-grid | Direct q=1/2, L=256 | Correlated difference |
|---:|---:|---:|---:|
| 0 | 0.381803 ± 0.000027 | 0.381809 ± 0.000008 | **−0.000006 ± 0.000023** |
| 0.01 | 0.385546 ± 0.000015 | 0.385463 ± 0.000011 | **+0.000082 ± 0.000014** |

At zero SOC, the difference remains compatible with zero using 500k blocks (SE 0.000026) and either production half. The earlier WHAM Binder value, 0.381818±0.000021, is a different interpolation estimator and agrees at this precision; it has not been silently replaced. Agreement of finite-size proxies does not establish coincident thermodynamic transitions.

At g_D=0.01, the new L=256 data produce a nonzero **finite-size proxy difference** conditional on the sampled ensemble. However, the Binder proxy moves from **0.385488±0.000033 to 0.385599±0.000025** between halves; the direct half-charge proxy moves from **0.385440±0.000014 to 0.385488±0.000013**. The L=160 source also shows the drift described above. These facts prevent interpreting the pooled small statistical error as a secure thermodynamic separation. The first-half difference includes zero within its 95% bootstrap interval.

The integer-charge direct crossings are also saved, but a crossing of 2T/pi is not a validation of integer-vortex BKT scaling. No new WM fit or defect-population inference is claimed here.

![Crossing sensitivity to blocks and production halves](figures/05_transition_stability.png)

Four uncertainties must remain distinct:

1. **Conditional statistical error:** joint time-block resampling within visited configurations. Existing data can improve its assessment through block, half and interpolation checks.
2. **Slow-mode/equilibration uncertainty:** unknown bias from initial conditions and sectors with few or no conversions. Longer trajectories and independent ordered/disordered starts are needed; a bootstrap cannot invent these visits.
3. **Finite-size corrections:** four sizes do not determine a reliable correction exponent or prove disappearance of the negative Binder dip.
4. **BKT model uncertainty:** charge assumption, coupling to spin/chirality and subleading finite-size terms. Longer sampling can clarify discrepancies but cannot make an inappropriate scaling form correct.

The zero-SOC conclusion survives as **statistical compatibility of finite-size crossings, with unresolved equilibrium and scaling systematics**. It should not be upgraded to a proof of one transition.

For a precision-only illustration, reducing the raw (160,256) zero-SOC Binder crossing SE from 2.7e−5 to 1e−5 would require roughly 7.3 times the production data if both source sizes were extended and the variance continued to scale inversely with length: about 22M production sweeps per run. The direct half-charge crossing already has conditional SE below 1e−5 in 3M. This difference explains why a seemingly precise helicity crossing can coexist with unresolved slow equilibration; it is not a reason to accept a shorter helicity run without checking the configurations.

## 8. A costed route beyond L=256

### Timing and memory assumptions

The recorded new zero-SOC simulation times are 1,161, 2,675, 5,120, 8,729 and 28,287 seconds at L=64,96,128,160,256. The L=256 SOC timing is virtually identical. These are elapsed `simulation_seconds`, not an isolated kernel benchmark, and exclude final archive writing. The manifests do not identify the GPU. The user specified **RTX 5090**; its official capacity is [32 GB GDDR7](https://www.nvidia.com/en-au/geforce/graphics-cards/50-series/rtx-5090/). Timing predictions below assume comparable hardware and workload; no new 5090 benchmark was performed.

I bracket cost with area scaling and continuation of the measured L160→256 exponent 2.50, and assume linear scaling in replica count. Future-size throughput, memory bandwidth, final saving and contention can invalidate that bracket. Reserve at least 20% of the 72-hour allocation rather than using the limit as the estimated simulation budget.

| Illustrative **8M burn + 8M production** | Elapsed estimate | Core GPU arrays | Raw measured-array payload | Approx. arrays at host save peak |
|---|---:|---:|---:|---:|
| L320, R160 | **26.2–29.3 h** | 1.34 GiB | 4.23 GiB | 8.84 GiB |
| L320, R200 | 32.7–36.6 h | 1.68 GiB | 5.25 GiB | 10.99 GiB |
| L384, R160 | **37.7–46.2 h** | 1.93 GiB | 4.26 GiB | 8.90 GiB |
| L384, R200 | 47.1–57.8 h | 2.42 GiB | 5.28 GiB | 11.04 GiB |
| L384, R256 | 60.3–74.0 h | 3.09 GiB | 6.72 GiB | 14.04 GiB |
| L512, R256 | **107–152 h** | 5.50 GiB | 6.77 GiB | 14.15 GiB |

Core GPU arrays are 88 R L² bytes: complex64 field, per-site RNG and Gaussian counters. Workspace, CUDA context and walker-history storage are additional. These figures are a component budget, not a measured peak. Host estimates include duplicate arrays/history but not all Python-object, JSON and compressor overhead. A 32 GB host allocation is a reasonable minimum planning choice for the 16M cases; prefer 64 GB for longer runs and verify the actual allocation. Host RAM was not specified.

At the existing cadences, output for 8M production is roughly 3.4 GiB compressed at R160 or 4.3 GiB at R200, extrapolating current compression; profiles and compression ratios can change. Production length doubles measurement storage, and total length increases PT-history storage. The measurement code retains arrays until saving, so host usage matters even when GPU capacity is ample. Complete scenarios are in [resource_scenarios.csv](results/resource_scenarios.csv).

![Runtime and GPU core-memory estimates](figures/07_resources.png)

### Transport-based length planning

For a transparent scenario calculation, multiply the actual L256 passage-flux scale by the ratio of future/current acceptance-only round-trip times. One scenario keeps the additional physical-memory penalty fixed; another multiplies it by L/256 to allow continued memory growth. These are **planning scenarios, not confidence bounds**; a newly emerging barrier could be worse. The heat-capacity stress scenario gives:

| Candidate | g_D=0 flux scale | g_D=0.01 flux scale |
|---|---:|---:|
| L320, current 200, 0.37–0.42 | 12.9–16.1M | 8.2–10.2M |
| L320, common energy 200, 0.37–0.42 | 11.7–14.6M | 7.9–9.9M |
| **L320, common energy 160, 0.375–0.405** | **6.0–7.5M** | **4.1–5.1M** |
| **L384, common energy 160, 0.375–0.405** | **8.6–12.9M** | **5.9–8.8M** |

At L384 the unchanged 200-replica ladder has predicted minimum acceptance only 0.066/0.118 in the stress case, compared with 0.221/0.297 if C_V stops growing. The narrower energy-based R160 candidate gives 0.476/0.486 in the stress case. Hence blind reuse of the current placement is increasingly risky, while hundreds of additional replicas are not a necessary first response.

A useful **budgeting rule**, not an equilibrium theorem, is to allow roughly two measured transport scales for burn-in and three for production, then require independent-start agreement, stable block errors, endpoint coverage and stable observables before accepting results. Even five traffic scales need not mix an unobserved topological sector.

For zero-SOC L320 with a *validated* narrow R160 ladder, this suggests initially planning around **12–15M thermalization plus 18–23M production**, with extension determined by measured convergence. A 30–38M total costs approximately 55–69 hours using the recent timing trend; 20% wall-time reserve makes the upper part unsuitable for one allocation. An 8M+8M run can be a diagnostic allocation with useful visited-ensemble statistics, but should not be labelled equilibrated in advance. With the current wide ladder, the analogous transport budget is about 64–80M sweeps, approximately 147–184 hours at R200.

At L384, even the narrow candidate suggests a zero-SOC total around 43–65M sweeps under that rule, roughly 125–187 hours before extra overhead. It should follow measured L320 performance and have continuation available. L512 magnifies both transport uncertainty and cost; postpone it.

The present simulation saves the completed result at the end and does not provide a full restart of the evolving Markov chain. Therefore **checkpoint/continuation becomes necessary for a reliable larger-size program** unless a measured optimization substantially shortens these scales. A future minimal checkpoint must preserve the actual field, local/exchange RNG states, temperature mappings, matching parity, sweep counts and necessary sampler/measurement state. Restarting from saved observables or arbitrary snapshots is not equivalent. No checkpoint implementation was added here.

### Ranked practical recommendations

| Priority | Action | Expected benefit | Cost and principal limitation |
|---:|---|---|---|
| 1 | Compare independent ordered/disordered starts at L≤256 and monitor passage flux, flow and physical memory | Detect initial-state/sector dependence that present error bars cannot cover | Additional reference runs; necessary evidence, not an algorithmic speedup |
| 2 | Test narrower 0.375–0.405 coverage with 160–200 well-placed replicas against the wide reference | Largest supported opportunity to reduce temperature travel and replica work | Requires verification of equilibrium observables and hot forgetting; topology remains unmeasured |
| 3 | Retune the existing ladder function or use one small energy-based table; try only a modest flow contribution | Improves the known critical bottlenecks without increasing replica count | Offline predictions support placement, not guaranteed physical conversion |
| 4 | Add minimal full-state continuation before committing long L320/384 runs | Makes required lengths compatible with the 72-hour limit and protects completed work | Small, separate future implementation; not done here |
| 5 | If needed, isolate interval 5 vs 10 or occasional rho=0 refresh in bounded comparisons | Could reduce reversal/memory cost per hour | Benefit at small L is not proof at L256; preserve the full MH target |
| 6 | Defer L512 and elaborate generalized-ensemble/cluster machinery | Avoids spending the next allocation on an unvalidated scaling assumption | Reconsider only after L320 transport and independent-start checks |

## Reproduction and validation

All new work is confined to this directory. There are seven standalone analysis scripts; none is a runner or simulation engine. Reused utilities include the previous autocorrelation estimator, WHAM solver, crossing/interval helpers, and the existing temperature-ladder function. The large NPZ files in `results/` are disposable derived caches, not new simulation data.

Run from the repository root, in this order, using the existing NumPy/SciPy/Matplotlib environment:

```text
python -B exploration/he3_pt_equilibration/extract.py
python -B exploration/he3_pt_equilibration/transport.py
python -B exploration/he3_pt_equilibration/physical.py
python -B exploration/he3_pt_equilibration/conditioned.py
python -B exploration/he3_pt_equilibration/design.py
python -B exploration/he3_pt_equilibration/checks.py
python -B exploration/he3_pt_equilibration/figures.py
```

Extraction checks every saved walker permutation and exact alignment of scalar/spatial histories. Additional bounded checks cover Markov-chain stationarity and the acceptance-one limit, known censored endpoint trajectories, temperature/walker round-trip mapping, and raw scalar moments against cached blocks. [validation.json](results/validation.json) records that **140 pre-existing Python sources retain their hashes, all 18 input archives retain size and modification time, branch/HEAD are unchanged, and tracked/staged diffs are empty**. Dataset preservation checks are size/mtime checks, not full-file cryptographic hashes; per-member CRCs are recorded from each archive. Analysis scripts pass syntax checks, and figures are supplied as PNG and PDF.

The strongest next step is to measure whether a simpler, shorter ladder actually reduces the **physical-memory penalty**, while checking independent initial states. The current data already show why acceptance alone is an inadequate basis for launching a much larger lattice.
