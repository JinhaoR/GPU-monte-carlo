# Cluster validation of the He3 PT ladders

3 November 2026. Input: `data/pt_ladder_test/`. All twelve L96 histories analyzed; no new simulations run.

## Conclusion

**The narrower ladders genuinely improve measured transport at L96. C160 is now the best-supported candidate for further validation. Full equilibrium sampling, particularly of signed chirality, is not yet established, and these data do not validate L320 production.**

Compared with A200, C160 produces **2.32 times as many endpoint passages per measured GPU-second at g_D=0 and 1.95 times at g_D=0.01**. Traffic between the same interior temperature surfaces improves by **2.20 and 1.89 times**, respectively. Every B/C walker completes a round trip inside the 1M comparison window. This is a substantial advance over the short pilots, where nearly all first trips remained censored.

The critical-region Binder, squared chirality and helicity means agree between independent starts within about 1.7 conditional standard errors for all ladders. C also generally gives better precision per cost. However, cold signed-chirality errors continue increasing with block length, and some finite-window imbalances remain. A completed trip is not an independent equilibrium sample.

The previous report preferred B200 as the conservative next candidate because the local runs were too short to justify fewer replicas. **The new evidence favors C160:** B and C have similar traffic per walker per sweep, while C costs less. B's higher acceptance does not consistently improve Binder or helicity precision.

## What was actually run and checked

All archives report **NVIDIA GeForce RTX 5090**, pressure 20, the same full 3×2 GL Hamiltonian, Gaussian rho=−0.9, proposal mass=0, and exchange interval 5. There are three frozen ladders, two couplings and two independent initializations:

| Ladder | Replicas | Temperature range | Placement |
|---|---:|---|---|
| A | 200 | 0.370–0.420 | Original reference function |
| B | 200 | 0.375–0.405 | Common energy-based density |
| C | 160 | 0.375–0.405 | The same density as B, fewer quantiles |

A→B changes range and placement at fixed replica count; B→C changes replica count at the same density and range. The frozen ladders were not adapted to these results.

Each history contains 1.5M sweeps. The first 500k are retained diagnostic burn-in; the comparison uses `sample_sweeps > 500000`, leaving 1M sweeps. Producer metadata correctly records zero discarded thermalization because the transient was saved. There are **30,000 scalar samples, 1,500 spatial samples and 300,001 walker-history rows per run**, including the transient. Production retains 20,000 scalar and 1,000 spatial samples. Positions are recorded at every exchange, every five sweeps.

Ordered/random update seeds are 20261112/20261115, exchange seeds are one higher, and random fields use host seed 20261118. Seeds are matched across ladders/couplings, so those comparisons are not independent replicated performance trials. The two initializations are independently seeded.

All twelve files pass checks of finite channels, sampling coordinates, spatial/scalar consistency, temperature/walker permutations, beta assignments, attempt counts and absence of nonfinite proposal rejections. Reconstructed full-run trips agree **per walker** with the exact GPU trip counters. Energy-component and Fourier/scalar checks use the existing validated spatial-analysis function. [Validation table](results/validation.csv), [audit](results/audit.json).

The total recorded simulation time is **2.048 GPU-hours**, close to the previous 2.08-hour estimate. This excludes saving, setup and queue time; first-case simulation timing includes any initial kernel compilation. A/B cost about 0.452–0.458 ms/sweep, C about 0.319–0.320 ms/sweep. C's measured 30% reduction in time per sweep is hardware/workload specific; it should not be assumed at larger L.

## Transport, censoring and cost

The table uses the 1M comparison window. Passage arrivals can finish journeys begun before the window; a listed loop must start and finish inside it. “Walkers with loop” is the number completing at least one such loop. The traffic scale is

\[
\Phi=\frac{2R\,D}{N_{\rm opposite\ endpoint\ arrivals}},\qquad D=10^6.
\]

It is derived from traffic, not from the selectively observed durations of completed loops. Errors below are conditional delete-one-100k-block jackknife errors; they do not include uncertainty between additional independent seeds.

| g_D | Ladder | Start | Passages | Complete loops | Walkers with loop | Φ (k sweeps) | Simulation seconds |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | A200 | Ordered | 1,027 | 456 | 196/200 | 389.5 ± 9.5 | 686.7 |
| 0 | A200 | Random | 1,034 | 464 | 194/200 | 386.8 ± 8.4 | 684.0 |
| 0 | B200 | Ordered | 2,031 | 939 | 200/200 | 196.9 ± 3.1 | 683.1 |
| 0 | B200 | Random | 2,039 | 952 | 200/200 | 196.2 ± 2.2 | 677.1 |
| 0 | C160 | Ordered | 1,669 | 780 | 160/160 | 191.7 ± 5.4 | 480.2 |
| 0 | C160 | Random | 1,680 | 790 | 160/160 | 190.5 ± 4.2 | 480.1 |
| 0.01 | A200 | Ordered | 1,028 | 446 | 192/200 | 389.1 ± 10.6 | 677.4 |
| 0.01 | A200 | Random | 1,050 | 468 | 198/200 | 381.0 ± 4.9 | 682.7 |
| 0.01 | B200 | Ordered | 1,791 | 831 | 200/200 | 223.3 ± 5.0 | 681.8 |
| 0.01 | B200 | Random | 1,821 | 841 | 200/200 | 219.7 ± 5.9 | 681.3 |
| 0.01 | C160 | Ordered | 1,414 | 661 | 160/160 | 226.3 ± 5.8 | 479.5 |
| 0.01 | C160 | Random | 1,446 | 671 | 160/160 | 221.3 ± 5.1 | 478.5 |

![Measured transport scale and cost](figures/02_transport_cost.png)

Pooling the two starts descriptively:

| g_D | Ladder | Φ (k sweeps) | Seconds / endpoint passage | Seconds / common-interior passage |
|---|---|---:|---:|---:|
| 0 | A | 388.2 | 0.665 | 0.572 |
| 0 | B | 196.6 | 0.334 | 0.305 |
| 0 | C | 191.1 | 0.287 | 0.260 |
| 0.01 | A | 385.0 | 0.655 | 0.551 |
| 0.01 | B | 221.5 | 0.377 | 0.329 |
| 0.01 | C | 223.8 | 0.335 | 0.292 |

Costs include the full 500k+1M simulation, charging burn-in against production information. Common interior passages use T≤0.379 to T≥0.400 at zero SOC, and T≤0.383 to T≥0.400 at g_D=0.01. Their improvement shows that the result is not just a consequence of closer endpoint definitions. [Detailed transport](results/summary.csv), [pooled comparison](results/pooled_transport.csv), [100k traffic blocks](results/traffic_blocks.csv).

All walkers reach an endpoint during the comparison. First-loop medians are approximately 312–331k for A, 167–208k for B and 154–196k for C. Unlike the local pilots, the first-passage and first-loop medians are observed. A still has 2–8 censored first loops per run; B/C have none. Later unfinished journeys remain in the event table and the survival calculation.

![First-passage and first-loop survival, including censoring](figures/07_censoring.png)

Acceptance remains a poor stand-alone predictor. Minimum neighboring acceptance is 0.824–0.834 for A, 0.898–0.903 for B and 0.873–0.880 for C. Yet the measured traffic scale is roughly **6–8 times** the correctly alternating acceptance-only chain's prediction. Critical-region reversal fractions are still about 0.69–0.71, 0.43–0.46 and 0.50–0.53, respectively, above the corresponding independent-acceptance predictions. Denser placement helps; it does not eliminate configuration-dependent transport.

![Swap acceptance and physical-walker flow](figures/01_acceptance_flow.png)

## Physical observables and statistical efficiency

The critical probes remain the previous L96 values, T=0.3821053 and 0.3856721 for g_D=0 and 0.01. Primitive moments are interpolated onto common temperatures **before** nonlinear Binder/helicity estimation and inside joint time blocks. Neighboring temperatures are not independent observations. Spatial comparisons between initializations use the same native temperature within each ladder.

For normalized signed chirality m=κ_total/density_total, Binder is 1−〈m⁴〉/(3〈m²〉²). Helicity uses the existing centered-current estimator averaged over x/y, with its existing 1/L² normalization. Blocks of 5k, 25k, 50k, 100k and 200k sweeps test error sensitivity. The main errors use 100k blocks, giving ten blocks per history. The largest block leaves only five blocks, so its error is itself noisy.

At the central critical probe, ordered-minus-random differences are:

| g_D | Ladder | Δ〈m²〉 | ΔBinder | Δhelicity |
|---|---|---:|---:|---:|
| 0 | A | −0.00233 ± 0.00135 | −0.00870 ± 0.00579 | −0.0242 ± 0.0176 |
| 0 | B | −0.00038 ± 0.00097 | −0.00237 ± 0.00387 | −0.0039 ± 0.0112 |
| 0 | C | +0.00093 ± 0.00089 | +0.00491 ± 0.00360 | −0.0109 ± 0.0087 |
| 0.01 | A | +0.00078 ± 0.00110 | +0.00279 ± 0.00538 | +0.0003 ± 0.0123 |
| 0.01 | B | −0.000003 ± 0.00121 | +0.00122 ± 0.00554 | −0.0069 ± 0.0144 |
| 0.01 | C | −0.00099 ± 0.00083 | −0.00420 ± 0.00374 | −0.0092 ± 0.0113 |

The short-pilot C160 SOC discrepancy in squared chirality was −0.0113 ± 0.0033. Its substantial reduction is encouraging. Comparisons with the saved 3M+3M L96 reference are mostly compatible, but not uniformly within two conditional errors. For example, the random A SOC energy at the critical probe is higher by 0.00141 ± 0.00038. That reference also has finite-history uncertainty and is not exact equilibrium truth. Tables retain all probes, including exceptions; no selection of only favorable comparisons is used.

![Critical means; black line and grey band are the previous long reference](figures/03_critical_means.png)

![25k-block squared-chirality histories; dotted line marks the comparison boundary](figures/04_relaxation.png)

For efficiency, the following uses a single native temperature nearest the critical probe, avoiding the extra averaging introduced by interpolation. Offsets are at most 0.000022. Ranges span the two starts, not confidence intervals.

| g_D | Ladder | Binder SE | Binder conditional ESS / GPU-hour | Helicity SE |
|---|---|---:|---:|---:|
| 0 | A | 0.00372–0.00443 | 11,360–17,870 | 0.00964–0.01476 |
| 0 | B | 0.00218–0.00309 | 22,088–44,848 | 0.00877–0.00937 |
| 0 | C | 0.00221–0.00383 | 25,378–69,988 | 0.00620–0.01165 |
| 0.01 | A | 0.00351–0.00409 | 18,936–26,272 | 0.00742–0.00980 |
| 0.01 | B | 0.00360–0.00459 | 14,471–24,771 | 0.00981–0.01399 |
| 0.01 | C | 0.00237–0.00304 | 44,706–69,111 | 0.00731–0.01126 |

ESS here is the instantaneous influence-function variance divided by the block-jackknife estimator variance. It describes conditional precision, not the number of independent equilibrium configurations. Some helicity estimates exceed the 20,000 measurements because of anticorrelation or noisy block variances. FFT helicity times remain near the 25-sweep sampling floor; those short estimates do not measure structural equilibration.

An alternative directly measurable efficiency proxy is estimator variance × total simulation seconds; lower is better. Averaging that quantity over the two starts, **C's gain over A** is approximately:

| g_D | Binder gain, varying 50k–200k blocks | Helicity gain, varying 50k–200k blocks |
|---|---:|---:|
| 0 | 1.3–2.4× | 1.8–2.6× |
| 0.01 | 2.1–2.8× | 1.05–1.53× |

These are sensitivity ranges, not confidence bounds on a universal speedup. B's SOC Binder and helicity efficiency is less consistent; its improved traffic alone would have selected the wrong winner. [Native estimates](results/native_statistics.csv), [efficiency gains](results/precision_gains.csv).

![Conditional precision per total GPU-hour; vertical ranges vary block length](figures/08_precision_cost.png)

## Remaining slow sectors and a necessary error-bar correction

**Signed chirality is the clearest remaining limitation.** Its cold-end uncertainty has not reached a convincing block-size plateau. At g_D=0, C ordered has 〈m〉=0.0405: its error grows from 0.0070 at 5k blocks to 0.0184 at 200k. C random has 〈m〉=0.0173, with error increasing from 0.0076 to 0.0203. In B's random SOC run the error grows from 0.0080 to 0.0289. The finite-volume ensemble has no imposed chirality bias; persistent signed imbalances and long error tails warrant more sampling even when sign-even observables agree.

For A random at zero SOC and T=0.375, signed chirality changes from −0.0017 to −0.0807 between the comparison halves. The difference is **−0.0790 ± 0.0167** with 100k blocks, and remains substantial with 25k–50k blocks. This behavior is not unique to narrow temperature coverage, and does not establish a deterministic initial-state bias; a long fluctuation in a stationary process is also possible.

Cold spin alignment also has a residual C ordered/random difference at zero SOC, **0.0143 ± 0.0058** using 100k blocks. The block-size dependence matters more than attaching a formal significance level to any one of these comparisons.

![Block-size sensitivity; spin and signed chirality use T=0.375](figures/05_blocking.png)

A useful caution emerged from the helicity checks. For zero-SOC C ordered, the critical helicity shifts by **−0.0281** between halves. Using five 100k blocks per half gives an error 0.00546, apparently a 5.1-standard-error effect. Recomputing from primitive moments gives errors **0.01575, 0.01608 and 0.01388** with 5k, 25k and 50k blocks: only about **1.7–2.0 standard errors**. Two 250k blocks per half give an even smaller, unreliable error. It would be wrong to claim a robust “5-sigma equilibration failure” from that unstable variance estimate. The shift is retained as a diagnostic, with its block sensitivity.

The previous B ordered SOC helicity shift also becomes smaller: **−0.0418**, with errors 0.0162–0.0210 across 25k–100k blocks, versus −0.204 ± 0.051 in the short pilot. Several other pointwise discrepancies remain; the tests are numerous and correlated, and these ratios are not multiple-testing-adjusted p-values. [Split sensitivity](results/split_sensitivity.csv), [burn-cut sensitivity](results/burn_sensitivity.csv).

Same-walker signed-chirality correlations reveal the physical slow scale directly. At a 100k lag they are about **0.069 for A versus 0.014–0.027 for B/C at zero SOC**, but still **0.106–0.125 for B/C at g_D=0.01**. Energy and scalar spin residual correlations are much smaller at that lag. The SOC signed-chirality tail falls substantially by 250k but demonstrates why a short temperature-slot ACF is insufficient.

These correlations pool each ladder's actual temperature itinerary. Their amplitudes also reflect different temperature-occupancy weights, so they are trajectory-memory diagnostics, not matched-temperature measurements of an intrinsic local relaxation time.

![Correlations of the same physical walker, after subtracting temperature-dependent means](figures/06_walker_memory.png)

Configurations arriving from opposite endpoints remain different near the heat peak. Cold-origin minus hot-origin 〈m²〉 is about **0.039–0.047 at g_D=0**, and **0.032–0.038 at g_D=0.01**, with 100k-block errors around 0.0006–0.0015. The narrow ladders therefore improve travel without removing reactive configuration memory. Endpoint conditioning can distinguish two subensembles in a stationary chain; it is not independently a proof of biased equilibrium means. [Conditioned measurements](results/conditioned.csv).

## Endpoint adequacy and additional spatial measurements

T=0.375 remains clearly ordered: in C, 〈m²〉 is about 0.2550 at zero SOC and 0.2795 at g_D=0.01, with Binder approximately 0.66563 and 0.66603. This supports the cold reference region, while the signed/spin qualifications above remain.

At T≥0.400, requiring uninterrupted residence at **every five-sweep exchange**, narrow-ladder correlations at lag 500 have magnitude below about 0.010 for energy, 0.008 for signed chirality and 0.004 for composite intensity; scalar spin reaches 0.022. There are 41,618–79,389 qualifying pairs per narrow run, though these pairs are correlated. The previous random C energy correlation of approximately 0.18–0.20 is absent at this lag in the longer comparisons.

The full five-component spin tensor has 284–907 qualifying narrow-ladder pairs at lag 1,000, with correlations −0.036 to +0.015. This is a large improvement over the 10–45 C pairs in the short tests. However, no narrow run provides an uninterrupted 5,000-sweep hot stay at the spatial measurement endpoints. Missing long-lag entries mean insufficient residence, not zero memory.

![Energy and full spin-tensor memory during continuous hot stays](figures/09_hot_memory.png)

There are now hundreds of eligible cold→hot→cold excursions. Restricting to the first eligible return per walker avoids counting prolific walkers repeatedly: narrow-ladder chirality-flip fractions range **0.450–0.563**, with 160–200 contributing walkers per case. These are compatible with the approximately one-half expectation from the measured marginal signs at the available block-error resolution. This supports loss of global chirality-sign memory on typical hot excursions. Return selection is still conditional on completion; it cannot prove all possible slow states are sampled. [Excursion summary](results/excursion_summary.csv).

Additional spatial checks compare the zero-wavevector powers of the full charge-two tensor B, original-field A and connected spin tensor Q, local deviation from the chiral manifold, and bulk/gradient/SOC energy components between starts. They show no additional broad, reproducible initialization split across ladders. Some individual energy-component differences reach 2–3 conditional standard errors. The results are retained in [spatial contrasts](results/spatial_contrasts.csv), not interpreted as independent tests of each other. Tensor return dot products are also exported; under SOC, a nonzero preferred alignment makes zero an inappropriate universal target for the uncentered dot product.

The saved Fourier/global channels do **not** determine winding sectors, HQV populations or actual topology-changing events. Thus the evidence supports 0.405 as a useful hot endpoint for the measured L96 global fields, not as a proven topological reset temperature at all sizes.

## Recommended next step

1. **Keep C160, 0.375–0.405, as the leading common-ladder candidate.** Retain the exact frozen temperatures, rho=−0.9, mass=0 and exchange interval 5. There is no demonstrated correctness problem requiring a sampler/PT rewrite, refresh schedule or exchange change.
2. **For stronger L96 equilibrium validation, run only C next**, at both couplings and both starts with fresh seeds: provisionally **1M diagnostic burn + 3M comparison sweeps**. This targets the remaining signed-chirality/block-length issue without repeating all twelve cases. The measured RTX 5090 cost is approximately **1.42 aggregate GPU-hours**, or **1.71 hours with a 20% allowance**. Use the extra samples to test 100k–500k blocks, sign balance, half-run stability and reproducibility against this independent seed pair. These lengths are a diagnostic allocation, not a guarantee.
3. If that check is satisfactory, the previously proposed **L128 A-versus-C comparison**, both couplings and both starts, remains the appropriate size-scaling test: provisionally 2M+3M per history. The new L96 RTX 5090 timing gives **7.65–8.84 aggregate hours**, bracketing area scaling and the previous measured exponent 2.50; reserve about **10.6 hours including allowance**. Larger-size actual timings are still needed.

The 500k burn plus 1M comparison now supports useful L96 comparisons of critical sign-even quantities. It does **not** establish uniformly reliable errors for every sector. Choosing C is a measured efficiency decision; certifying its equilibrium distribution is a stronger requirement.

The proposed longer histories would be fresh runs: these NPZ files contain measurements and diagnostics, not a restartable final field/RNG checkpoint. The current `test_run.py` settings have not been increased.

**Do not apply the measured approximately twofold L96 gain to L320/384 automatically.** The earlier L256 traffic scales of 7.64M/5.19M and their growth with L remain relevant. The [previous larger-size scenarios](../../exploration/he3_pt_ladder_validation/results/larger_size_scenarios.csv) remain conditional planning estimates, not validated predictions. This investigation adds no larger-L numerical evidence and does not remove the need for checkpoint/continuation planning. [Costed follow-ups](results/next_test_costs.csv).

## Reproduction and preservation

Three small CPU analysis scripts reuse the existing moment/jackknife, autocorrelation, PT-event and spatial-analysis functions:

```text
python -B analysis/pt_ladder_test/analyze.py
python -B analysis/pt_ladder_test/compare.py
python -B analysis/pt_ladder_test/checks.py
```

`analyze.py` reads the raw archives one at a time and saves numerical tables; optional NPZ filenames select individual cases. `compare.py` produces comparison tables and nine PNG/PDF figure pairs. `checks.py` independently reconstructs the moment blocks for split/burn sensitivity and checks agreement with the first reduction. The small moment-block files are retained for further offline analysis. Original large archives are not copied into this output directory.

The audit confirms all twelve consistency checks, unchanged input sizes/mtimes, unchanged hashes for 147 previous Python source files and an empty tracked Git diff. The simulation engine, `run.py`, `test_run.py`, frozen ladders and earlier investigations were not modified. No GPU computation, new simulation, Slurm submission or checkpoint implementation was performed.

**Decision:** the narrow-ladder transport benefit is established at L96, and C160 is the preferred next candidate. Signed-sector uncertainty still prevents claiming complete equilibrium validation. Complete the focused C-only L96 check before treating the method as established for larger systems; L320 production is not yet supported.
