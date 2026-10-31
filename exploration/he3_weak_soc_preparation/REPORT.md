# One common He3 temperature ladder

The current [run.py](../../run.py) uses **one 502-temperature ladder** for every size `[64, 96, 128, 256]` and coupling `[0, .001, .01]`. `make_temperature_ladder` is unchanged; `temperatures` and tracked walker IDs are constructed once before the original simulation loops. The dictionary is:

```python
temperature_ladder = {
    "T_min": 0.365, "T_max": 0.425, "n_T": 502,
    "Tc": 0.384, "tc_window": 0.012, "tc_fraction": 0.43,
}
```

The sampling settings found at the start of this change were **2,000,000 thermalization and 4,000,000 production sweeps**, scalar/exchange intervals 10, spatial interval 500, and profile interval 10,000. All remain unchanged. No observables, Hamiltonian, PT, Simulation, initialization, seed or storage code was changed. The size list alone was changed to the four requested sizes. No simulation, GPU benchmark or Slurm job was launched for this ladder update.

The minimum spacing is **0.00005201891**; the maximum is **0.00020268924**. There are **152 temperatures within .380-.388**. [Exact temperatures](ladders/common_L256.txt), [numerical summary](results/common_ladder.json), and [all 1,503 coupling/edge checks](results/common_ladder_edges.csv) are saved. The text file is an audit copy; the runner uses the original function directly.

The design uses the actual L160 energy histories, existing block moments, and the validated fine reweighted energy curves for all three couplings. Raw and cached heat capacities agree within 1.84e-9. Reweighting matters because the sharp peak falls between the old sampling temperatures:

| g_D | Largest sampled Cv | Reweighted peak Cv | Peak temperature | Old minimum swap acceptance |
|---:|---:|---:|---:|---:|
| 0 | 73.85 | 88.93 | .382005 | .08686 |
| .001 | 81.80 | 89.09 | .382350 | .09903 |
| .01 | 67.29 | 72.59 | .385580 | .12968 |

Here Cv is per site, and H is extensive: `Var(H) = T^2 L^2 Cv`. For each coupling the L160 envelope takes the larger of the sampled and reweighted curves, adds twice the largest jackknife error from 100k/200k/300k-sweep blocks, and includes the reweighting bin/source and first/second-half variants. This is a design allowance, not a simultaneous confidence bound or proof of equilibration.

To protect against a shifted or narrower dip, its maximum is applied throughout .380-.388 and padded by .0007 on each side (one old critical-grid gap, rounded upward). Thus the protected region is **.3793-.3887**. The L256 design uses

```
Cv_design_256(T) = 1.10 * (256/160)^2 * Cv_envelope_160(T).
```

This deliberately allows area growth of Cv, including outside the critical region. It is more conservative than the observed L128-to-L160 peak increases of about 13-15%, versus 56% for area growth over that size change. It avoids assuming BKT, Ising or another asymptotic universality class. The extra factor 1.10 allows further extrapolation uncertainty. The resulting protected Cv levels are 259.64, 262.61 and 211.16. These are stress-test envelopes, not measured L256 heat capacities or rigorous bounds on unseen behavior.

For each proposed edge, the calculation maximizes `sigma_H = T*256*sqrt(Cv_design)` over the **whole interval**, including a 1e-6-spaced grid and both endpoints. It requires `q = |Delta beta| max(sigma_H) <= .65` for every coupling. This retains the previous validated L64 pilot's overlap target, while substantially strengthening the size extrapolation and critical-region coverage.

`Tc=.384` is the midpoint of the protected critical band, not a fitted transition temperature. A bounded search over focus widths .010-.020 in .0005 increments and focus fractions .20-.65 in .01 increments selected .012/.43 with the fewest replicas in that search; ties favor the larger overlap margin. This wider focus window compensates for the original function's smooth shoulders. At this shape 501 replicas fail (`max q=.651006`), while 502 pass (`max q=.649699`). No modification of the ladder function is needed. This is a practical search within a readable parameter family, not a claim of globally optimal PT transport.

The cold/hot endpoints remain .365/.425. Interpolation of the old L160 records gives normalized absolute chirality about .575-.588 at the cold end and .00584-.00590 at the hot end. These retain an ordered control and a disordered control. The hot SOC spin-alignment mean remains induced (about .0011/.0103 at g_D=.001/.01); endpoint decorrelation of all internal modes is not established.

| g_D | Minimum predicted energy overlap, full ladder | Minimum within .380-.388 | Minimum predicted swap acceptance, full ladder |
|---:|---:|---:|---:|
| 0 | .74529 | .76483 | .64594 |
| .001 | .74683 | .76354 | .64801 |
| .01 | .75623 | .78734 | .66065 |

These use the **local equal-variance Gaussian approximation**: neighboring mean energies differ by approximately `Delta beta * sigma_H^2`, giving overlap `integral min(p_i,p_j) dH = erfc(q/(2 sqrt(2)))` and average swap acceptance `erfc(q/2)`. Overlap and swap acceptance are different quantities. The conservative variance envelope reduces these estimates, but does not make either a rigorous lower bound for non-Gaussian or poorly equilibrated energy distributions. The worst full-ladder edges are around .374, rather than the narrow critical dip; the whole protected region is checked, including its shoulders.

![Common ladder spacing and conservative overlap predictions](results/common_ladder.png)

**Resource estimates at the unchanged 2M/4M settings, per coupling:**

| L | Field storage, GiB | Estimated live GPU arrays, GiB | Estimated loop hours on the earlier RTX 4070 SUPER |
|---:|---:|---:|---:|
| 64 | .092 | .286 | 3.2 |
| 96 | .207 | .501 | 7.2 |
| 128 | .368 | .801 | 12.8 |
| 256 | 1.471 | 2.864 | 51.4 |

GPU accounting includes 48 bytes/site for A, 16 for site RNG, 24 for Gaussian counters, spatial workspaces, and the original PT history buffers. The last consume about 116 MiB at every size; the L256 meter adds about 56 MiB. Allow CUDA context, allocator and setup headroom beyond the table (roughly 4 GiB total budget at L256). The existing PT kernel already strides over edges; 502 replicas require no change to it. Each case performs 600,000 alternating exchange rounds, 300,000 attempts per edge, or 150.3 million neighboring-pair attempts. All 502 walkers are tracked at the unchanged 100-sweep history cadence.

The time estimates scale the previous actual L128/R47 Gaussian+PT timing by R*L^2, then conservatively add all current scalar/spatial/profile costs using the earlier L256 static measurements. They include some double-counted old measurement work. They exclude initialization, compression and final I/O, and have not been benchmarked at R502. Treat them as planning estimates with at least factor-two uncertainty, not runtime guarantees. All twelve cases total roughly **224 GPU-hours** in this estimate, about nine days sequentially before saving overhead.

**Host memory is the limiting resource on this workstation.** Each case retains 400,000 scalar events, 8,000 spatial events and 400 profile events. Numeric measurement arrays total about **21.84-21.88 GiB per case**, plus PT metadata. The existing Simulation temporarily holds both lists and stacked scalar arrays, giving approximately **37 GiB of array payload at peak**, before Python objects, allocator retention and JSON metadata conversion. Use a machine with **at least 64 GiB RAM**; the current workstation reports only 31.85 GiB total (about 17.1 GiB available during inspection). The GPU itself has about 12 GiB. No storage or cadence changes were made to hide this cost. Reserve approximately **300 GiB of disk** for twelve cases and metadata; actual compressed sizes remain unmeasured for this ladder.

Local overlap does not establish good round trips. The old L160 completed-trip means were 2.77-3.06M sweeps, conditioned on completed trips. The larger replica count and narrower temperature range compete with the improved bottleneck resolution; a reliable new round-trip time cannot be inferred from acceptance alone. The unchanged 2M/4M budget is not newly certified for L256. This distinction is also central to [feedback-optimized PT](https://arxiv.org/abs/cond-mat/0602085); no adaptive algorithm was introduced here.

[Read-only validation](results/common_ladder_validation.json) checked all 1,503 edges with independent interval maxima, strict ordering and distinct float32 inverse temperatures, exact agreement with the audit text file, numerical Gaussian overlap/swap integrals, unchanged sampling settings, and a single ladder call before the loops. Eighteen existing CPU checks also passed. The small test/benchmark scripts were updated only to reference the one dictionary; their simulations and GPU benchmarks were not run. Previous live-test results remain evidence for the unchanged measurement code, not for R502/L256 transport.

---

# Earlier measurement and prelaunch report (historical)

The material below records the previous L<=96 per-coupling design and its validation. Its old ladders, defaults and resource counts are superseded by the common-ladder report above; earlier results and exploratory work are preserved.


The entrypoint is [run.py](../../run.py). It retains the original settings, runtime selection, size → pressure → SOC loops, model/runtime/PT/Simulation construction, `simulation.run()`, metadata and `save_runs()` sequence. The separate `main/he3.py` was restored to its pre-task version. No production simulation or Slurm job was launched.

The requested prelaunch check is complete: the final runner passed actual Gaussian-update/PT/save-load tests at all three couplings. Compact real-space profiles are now enabled every 10,000 production sweeps at five selected temperatures. **The 500k thermalization / 1M production settings remain unchanged, but are not established as adequate for equilibrium inference at L96.** The data-based recommendation below is 2M / 4M per independent start as an initial allocation, subject to transport, stationarity and blocking checks.

## What remains, and what was removed

The existing PT, Simulation, Runtime, Gaussian MH update, Hamiltonian and `utils/data_io.py` are unchanged. The existing `make_temperature_ladder` implementation and interface are also unchanged.

The changes are a small optional addition to `He3Model.measure`, three focused measurement/reference/analysis modules, optional analysis switches, and the necessary settings and calls in `run.py`. Spatial sampling is off by default on a new model. The runner enables Fourier/structure measurements every 50th scalar measurement and compact profiles every 1,000th. After the original Simulation finishes, one helper attaches those records to the original run dictionary before `save_runs()`.

Removed: the custom PT subclass, exact-counter extension, replacement observation loop, chunked recording/loader, campaign generator, encoded seeds, runtime source hashing, expanded-size campaign settings, resource-projection scripts and obsolete per-size ladder files. Earlier exploratory directories and results were preserved. Current evidence is in `simple_*`, `prelaunch_*`, and `live_prelaunch_results.json`; older tiny-test results remain historical evidence.

## Reused and new measurements

The [spatial pilot](../he3_spatial_pilot/MAIN_BRANCH_IMPLEMENTATION.md) supplied the definitions and tiled reduction design for chirality kappa, the charge-two composite P, the five orthonormal components of Q, Fourier signs, and invalid correlation-length flags. The retained GPU implementation extends those reductions; it does not project A onto a chiral state. Local observable products and reductions use float64 on the stored complex64 field.

| Quantity | Retained implementation and interpretation |
|---|---|
| Original ten scalar channels | Unchanged energy, density, signed chirality, complex composite zero mode, both currents/curvatures and normalized spin alignment. Absolute chirality and current-mean-corrected helicity remain reconstructible. |
| kappa, P, Q | Raw amplitudes at zero and both signs of the smallest x/y momenta. Q uses an orthonormal five-component basis, preserving the full Frobenius contraction. |
| Full charge-two B=A^T A | Bxx, Byy and Bxy; contraction weights are 1,1,2. A planar state has P=0 while B remains nonzero. |
| Charge-one A | All six complex entries. Their contracted spectrum measures combined spin/orbital/phase correlations, not pure phase rigidity. |
| Local chiral distance | Density-weighted residual, orbital and spin contributions, density exclusions and histograms. This is local structure, not an order parameter or an HQV classifier. |
| Energy components | Bulk, gradient, bare dipole invariant and g_D-weighted dipole energy, evaluated with the existing Hamiltonian's device functions. |

For fixed chirality sign, write v=(A_x-i sigma A_y)/sqrt(2). Optimizing the phase/amplitude z at fixed real unit d leaves the retained norm |d·v|². Its maximum is the largest eigenvalue of Re(v v†), namely (v†v+|v·v|)/2. Thus the implemented distance is

    epsilon = 1 - max_sigma(n_sigma + |v_sigma dot v_sigma|)/(2 n).

On the maximizing branch, `orbital_loss=1-n_sigma/n` and `spin_loss=(n_sigma-|v·v|)/(2n)` sum to epsilon. Ideal unitary chiral states give zero; the normalized planar example gives 3/4; a maximally chiral state with v·v=0 gives 1/2. Chirality alone therefore misses spin nonunitarity.

Sites with n<=1e-12 are excluded from the local normalized diagnostics. Their counts and density are saved. Analysis reports both valid-density and total-density denominators; the latter is a partial weighted mean if exclusions are nonzero. The missing contribution is bounded by 3/4 times the excluded density fraction. Zero denominators remain NaN. Histograms retain underflow/overflow, including small negative residuals from roundoff; nothing is clipped into range.

The exact spin normalization check is sum Qzz=(2/3)(sum n) spin_alignment. The existing scalar `spin_alignment` is not the raw tensor component.

The analysis helper averages powers before ratios, retains x/y separately, and reports full and connected spectra. Connected subtraction uses ensemble means and is recomputed within each joint block deletion. A large unconnected Q correlation-length ratio under SOC can reflect an induced mean tensor; it does not establish a soft spin mode. Singular/negative-radicand estimates remain flagged, with NaN lengths and valid-delete counts. No HQV detection, pinning or spin-helicity estimator was added.

## Storage and analysis

The original NPZ format is retained. Scalar arrays have 100,000 production events at the default settings. The four added arrays have 2,000 spatial events:

| Array | Axes |
|---|---|
| spatial_amplitudes | event, temperature, 16 fields, 5 momenta |
| spatial_structure | event, temperature, 20 raw diagnostics plus 4×12 histogram bins |
| spatial_walkers | event, temperature; physical walker ID |
| spatial_energy | event, temperature; synchronized extensive H |

Metadata stores field/component names, histogram edges, spatial indices into the scalar history, and absolute spatial sweep numbers. These sample axes are deliberately distinct. Legacy files load unchanged; absent spatial channels are reported rather than fabricated.

`analysis/he3.py` retains its existing folder/pattern loading and analysis switches. Set `run_spatial=True` to save spectra, connected estimates, local distributions and joint blocked errors. `spatial_block_size=200` means 100,000 sweeps at the default spatial cadence; ten blocks are only a starting uncertainty assessment. The original scalar analysis still uses all scalar samples. No scaling fit is imposed on the new channels.

The measurement-call counter assumes a fresh model for each Simulation, as in `run.py`, and no additional manual `measure()` calls inserted into that Simulation. The requested compact profiles reuse the existing NumPy FFT reference. On each rare event, the measurement function copies only the five selected walkers, transforms one at a time, and saves profiles rather than fields. PT and Simulation are unchanged.

`realspace_profiles` has axes (event, selected temperature, 7 components, x/y, r=0..L/2). The seven components are kappa, P, q1..q5, with `G_c(r)=mean_x f_c(x+r) conj(f_c(x))`. Their normalization is per site, not per event density or G(0). P remains complex; negative displacements follow by conjugation. Summing the five Q profiles gives the full Frobenius contraction. `realspace_means`, `realspace_walkers` and `realspace_energy` retain the seven local means, physical IDs and synchronized extensive H. Metadata gives exact scalar indices, absolute sweeps, probe indices/temperatures, fields and displacements. At the defaults there are 100 profile events, beginning at absolute sweep 510,000.

The actual nearest ladder temperatures are:

| g_D | Selected temperatures (rounded here only) |
|---:|---|
| 0 | .374848, .381638, .382554, .384097, .414692 |
| .001 | .374947, .382015, .382616, .384130, .415189 |
| .01 | .374740, .385201, .385528, .386842, .414954 |

These span a cold control, the observed critical region and a hot control. `realspace_interval=0` disables them. `run_realspace=True` in the existing analysis script enables `summarize_realspace`; its default block size of ten events means 100k sweeps. It computes full and homogeneous connected profiles, subtracting the **ensemble** mean squared, independently in every joint block deletion. Subtracting each configuration's spatial mean would remove real long-wavelength fluctuations. Separate real/imaginary errors are retained, and the raw Q components support further SOC channel analysis. A connected estimate still requires equilibrated component means; slow spin rotation or chirality-sector trapping cannot be repaired by subtraction.

Profiles are worthwhile because two Fourier scales cannot determine the decay shape. Compare intermediate-distance behavior resembling `a r^(-eta)` with `a r^(-p) exp(-r/xi)`, while retaining possible plateaus and varying fit windows and L. Exclude contact/short-distance lattice effects; periodic images matter near L/2. Keep x/y separate until their agreement is established. With xi comparable to L, exponential decay can mimic algebraic decay over the available range; 100 correlated events are not 100 independent observations. Do not log-transform noisy negative estimates or fit away a chirality/SOC tensor plateau. No decay fit or phase classification is automated. Even the XY benchmark requires finite-size and logarithmic corrections ([Hasenbusch](https://arxiv.org/abs/cond-mat/0502556)); its scaling formulas are not automatically valid for the full GL fields.

## Ladder and PT choices

The audit read the twelve existing weak-SOC files at L=64,96,128,160, their manifests, energy histories and walker histories. The L64/L96 block moments and validated transition-study curves determine the final grids. Their provenance matched the current datasets. Old files contain scalar/PT data, not the newly added spatial or local-structure measurements.

Stored H is extensive; the heat capacity used here is per site. Therefore sigma_H=T L sqrt(Cv). The design checks |Delta beta_MC| sigma_H against approximately 0.65 using a conservative L96 envelope of the L64/L96 curves, with a 10% critical-peak allowance. Broad critical regions cover chirality, composite, heat-capacity and helicity features rather than a single fitted temperature.

The original ladder function uses a smooth focus window. Its parameters were adjusted to cover those regions and balance central resolution against the tails. Replica counts were then chosen from the worst predicted edge spacing. The `Tc` argument is only a focus center.

| g_D | Replicas | Focus center | Focus width / fraction | Critical region checked | Maximum predicted edge metric |
|---:|---:|---:|---:|---:|---:|
| 0 | 101 | .382 | .008 / .25 | .380–.384 | .648 |
| .001 | 107 | .38225 | .009 / .27 | .380–.3845 | .644 |
| .01 | 94 | .38525 | .007 / .22 | .3835–.387 | .650 |

All three span .365–.425. Exact audit copies are [gd=0](ladders/gd=0.txt), [gd=.001](ladders/gd=0.001.txt), and [gd=.01](ladders/gd=0.01.txt); [simple_ladders.json](results/simple_ladders.json) records source paths and spacing checks. The runner constructs them directly through the unchanged `make_temperature_ladder`, so launch does not depend on old data files. Tests verify agreement with the audit copies. One common grid per coupling keeps comparisons and loading simple; L80 has no size-matched old dataset, and L>96 is not covered by this preparation.

The old 160-replica .33–.47 campaign showed critical acceptance/flow bottlenecks near .382 for g_D=0/.001 and near .3858 for .01. At L96 its minimum acceptance was about .39/.36/.39, and completed round trips averaged about 1.19/1.26/1.10 million sweeps. At L160 the mean completed trips approached three million sweeps. The old 500-sweep walker histories also show why fixed-temperature sign/spin changes cannot automatically be interpreted as changes in the same configuration.

The colder endpoint remains well inside the ordered region. Near .425, the existing L160 data have absolute normalized chirality about .0058–.0059, composite magnitude about .0078–.0079, and helicity consistent with zero. These support a hotter control, while finite SOC still induces a tensor mean. They do not prove decorrelation of every slow mode.

The original exchange algorithm and cumulative diagnostics are retained. Exchange interval remains the requested 10; the old campaign used 5, but there is no controlled comparison demonstrating which gives better transport per unit cost here. `history_stride=10` now records every 100 sweeps. Exact round-trip counters still include thermalization; sampled production passages have finite time resolution. Improved spacing is a prediction, not a measured improvement in round trips or equilibrium.

The [prelaunch transport review](results/prelaunch_transport_review.json) rechecked the actual old L96 manifests and histories, including the first 500k and 1.5M sweeps:

| g_D | Mean completed round trip | 500k / mean trip | 1M / mean trip | Walkers with a sampled loop by 500k / 160 | By 1.5M / 160 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,190,721 | .420 | .840 | 17 | 97 |
| .001 | 1,257,865 | .398 | .795 | 10 | 90 |
| .01 | 1,102,642 | .453 | .907 | 13 | 97 |

The means are exact counters over the old **2M burn + 3M production** histories, conditioned on completing a trip. They exclude unfinished trips. The cutoff counts are lower bounds from 500-sweep snapshots, beginning at old sweep zero; they are not production-only statistics. Only 79/67/69 walkers had visited both endpoints by 500k. Even by 2M only 121/123/129 had completed a sampled loop. This supports treating 500k/1M as an initial diagnostic run at L96, not a justified precision campaign.

Narrowing the range and reducing replicas could improve this. The earlier L64, g_D=0 spatial pilot actually measured about 68k/71k-sweep trips for ordered/hot starts, but used 47 temperatures over .370–.415 and exchanges every five sweeps. It retained start differences and unresolved blocking dependence. That result cannot be transferred to the current 94–107-temperature L96 ladders. Even a crude unbiased-walk estimate `time proportional to exchange_interval * R^2` gives only .80/.89/.69 times the old cost for the three current ladders; this is an illustration, not a transport prediction. Local overlap and global flow are different diagnostics, as emphasized by [Katzgraber et al.](https://arxiv.org/abs/cond-mat/0602085).

**Recommendation, not enabled:** initially allocate 2M burn + 4M production sweeps per independently seeded ordered and hot start at L96. This provides approximately 1.6–1.8 old mean trip times for burn and 3.2–3.6 for production, enough to assess transport more credibly but not to guarantee equilibration. L80 has no matched old data; use the L96 allocation conservatively. At L64, old mean trips were .393/.432/.422M sweeps, so 1M burn + 3M production per start is a reasonable initial allocation. If only one scalar budget is wanted for every size, use 2M/4M manually; the code still contains 500k/1M. These recommendations require more host memory at the unchanged scalar cadence.

Accept adequacy only after ordered/hot estimates agree within stable blocked uncertainties for energy, chirality, P, full/connected Q and component means; first/second production halves show no persistent drift; and walker histories show repeated endpoint passages without a stranded population. Compare 100k/200k/400k-sweep blocks, extending the histories if errors keep rising or too few blocks remain. Round trips are necessary transport evidence, not independent sample counts or proof of spin/topological equilibration. Use production-only history slices as well as cumulative exact counters. Keep exchange interval 10 until a controlled comparison of 5 versus 10 can assess passages and decorrelation per wall time; higher acceptance alone is insufficient. No such campaign or exchange experiment was run here.

## Validation and cost

[25 deterministic checks](results/simple_validation.json) passed: independent NumPy/eigenvalue references, analytic and unrestricted random matrices, gauge/spin transformations, tensor contractions, Fourier signs, zero/near-zero fields, direct-pair versus FFT correlations, old-estimator/twist checks, and legacy serialization. The two new checks cover profile-only recording/cadence and ensemble connected subtraction with joint block errors and an incomplete trailing block. Maximum GPU/FFT amplitude discrepancy was 4.55e-13. Energy-part differences from the original total reduction were at most 5.74e-5 in the synthetic tests, consistent with the original float32 local Hamiltonian arithmetic; no constant shift was introduced.

[Static integration checks](results/simple_integration.json) still pass at all three couplings, preserving the earlier tests with local updates frozen. The new [live check](results/live_prelaunch_results.json), implemented in [live_check.py](live_check.py), invokes the final `run.main()` with temporarily reduced settings: L8, six replicas, 40 burn + 160 production sweeps, sample/exchange interval 10, spatial interval 20, profile interval 40. All Gaussian kernels, exchanges, measurements and saving execute normally. Test hooks inspect the original calls without replacing the simulation or update algorithms.

Each saved file contains 16 scalar, eight spatial and four profile events. All 18 named arrays were checked exactly against the pre-save measurements; all ten scalars were independently checked against NumPy Hamiltonian/contraction/twist references. Sparse scalar indices are 1,3,...,15 and 3,7,11,15; corresponding sweeps are 60,80,...,200 and 80,120,160,200. Every mapping matched the saved PT history. Gaussian counters are in physical walker order; measurement arrays are in temperature order. Profiles use only their explicitly saved probe indices.

| g_D | Gaussian accepted / attempted | PT accepted / attempted |
|---:|---:|---:|
| 0 | 50,017 / 76,800 | 28 / 50 |
| .001 | 50,218 / 76,800 | 29 / 50 |
| .01 | 49,746 / 76,800 | 30 / 50 |

The field and local RNG advanced, while each measurement left fields, both RNG streams, maps and cached energy/version unchanged. Maximum live discrepancies were 4.94e-6 for scalar references (including finite-difference curvatures), 7.95e-15 for Fourier modes and 5.56e-16 for direct pair profiles, including a full-matrix Q contraction. Connected estimates and errors matched direct block deletions. There were 600 completed test sweeps across three ladders; an initial test-harness attribute typo stopped an additional L8/g_D=0 attempt after 50 sweeps, before saving, and was corrected. These tiny runs establish correctness, not equilibration or full-size transport.

The [updated static benchmark](results/prelaunch_benchmark.json) used an RTX 4070 SUPER, 107 replicas, and 24 timed repetitions per component after warm-up:

| L | Low modes/event | Structure + energies/event | Five-profile added cost/event | Profile raw storage per 1M sweeps |
|---:|---:|---:|---:|---:|
| 64 | 1.31 ms | .90 ms | 22.32 ms | 3.76 MB |
| 80 | 1.93 ms | 1.33 ms | 36.25 ms | 4.65 MB |
| 96 | 2.40 ms | 1.70 ms | 59.07 ms | 5.55 MB |

The profile timings include GPU-to-host copies and CPU FFTs through the actual measurement path, subtracting the separately measured scalar cost. At interval 10k they add 2.23/3.63/5.91 microseconds per production sweep, or 2.23/3.63/5.91 seconds per million sweeps. Low-mode/structure work at interval 500 adds 4.41/6.53/8.18 microseconds per sweep. These are measured observation costs, not total-MC overhead percentages. They exclude final compression/I/O and do not reuse the earlier pilot's 1% claim. Meter workspaces are about 0.96–1.92 MB. At 107 temperatures, raw scalar plus spatial arrays total about 1.25 GB per size, plus at most 5.55 MB for profiles; existing in-memory stacking requires additional temporary RAM. Allow several GB of free host memory. Like the original script, saving occurs after each size; interrupted runs have no exact restart checkpoint. Enlarging the settings requires revisiting memory and sampling adequacy.

## Final settings and manual entrypoint

```python
pressures = [20.0]
g_d_values = [0.0, 0.001, 0.01]
sizes = [64, 80, 96]
K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0
sampler = "gaussian"
gaussian_rho = -0.9
gaussian_proposal_mass = 0.0
thermalization_sweeps = 500_000
measurement_sweeps = 1_000_000
sample_interval = 10
exchange_interval = 10
spatial_interval = 500
realspace_interval = 10_000  # five nearest temperature probes; 0 disables
history_stride = 10
```

The P=20 constructor still applies the pressure correction once, giving approximately (-.51651233, .94202280, .98752904, .87906289, -1.15251493). It saves these resolved coefficients. `reduced_temperature=1` remains distinct from MC temperature. Original seed arithmetic (`seed_base+L`, `pt_seed_base+L`) and ordered initialization are retained; different couplings share seeds and are not independent repeat histories.

Manual launch from the checkout is `python -B run.py`. It uses the new `weak_soc_spatial_simple_01` output folder and refuses an existing case file before starting it. No custom CLI or job launcher is provided. Longer independent starts and useful factor-two sizes are recommendations for later, contingent on diagnostics; they are not enabled. The current budgets do not certify equilibrium, BKT behavior, HQVs or separated transitions.

The [final audit](results/prelaunch_audit.json) verified the actual Git main worktree against its earlier hashes/status, all 244 protected exploratory files, all twelve old weak-SOC datasets, and the unchanged simulation/update/serialization code. No merge, reset, commit, push, production simulation or Slurm submission was performed.
