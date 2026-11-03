# Large-size He3 campaign preparation — 3 November 2026

**Runner readability update:** `large_run.py` now follows SETTINGS / SIMULATION / SAVE, with a simple main loop and a small `run_case()`. All settings present at the start of this refactor were retained, including the current **sample_interval=100**; the preparation benchmarks below used 50 and remain historical measurements. Matching completed cases are skipped, compatible checkpoints resume, and orphan partials, conflicting histories or incompatible settings raise before any sweeps or writes. The simulation, measurement and checkpoint modules were not changed by this refactor. [All 81 focused tests pass](results/runner_refactor_tests.log), including graceful stopping and file-conflict checks. [Four tiny case comparisons](results/runner_refactor_compatibility.log) establish bitwise agreement between the old runner, fresh refactored runs and continuation from old checkpoints, for both couplings and both preparations (excluding elapsed-time metadata). No production run was launched.

The requested campaign is prepared for a **controlled L256 reference**, pending a short validation on Kraken's RTX 5090. Nothing has been submitted to Slurm or run as production. The default [large_run.py](../../large_run.py) runs one L256, g_D=0, ordered case. Its independent random partner and g_D=.01 are explicit settings; L320 and conditional L384 are not enabled. The existing `run.py` and `test_run.py` remain unchanged.

The Hamiltonian, Gaussian proposal/acceptance kernels and PT exchange acceptance kernel are unchanged. Changes are confined to measurements, checkpoint state and diagnostics in the existing objects, and analysis needed for the requested comparisons. Earlier exploration was preserved.

## Configuration and scientific priorities

All cases use P=20 bar, K=1, the existing pressure-corrected coefficients, coefficient temperature 1, full complex 3×2 matrices, Gaussian rho=-.9 and proposal_mass=0, and exchanges every five sweeps. The exact [frozen C160 file](../he3_pt_ladder_validation/ladder_C.txt) is loaded once. Its endpoints are .375 and .405, its smallest spacing is **0.000061420089**, and its largest spacing is .000336694807. No ladder is regenerated or adapted.

| L | Burn sweeps | Production sweeps | Status |
|---|---:|---:|---|
| 160 | 1,000,000 | 3,000,000 | Planned comparison, both couplings and preparations |
| 256 | 3,000,000 | 6,000,000 | First controlled reference |
| 320 | 3,000,000 | 5,000,000 | Requires cluster restart validation and L256 review |
| 384 | 4,000,000 | 8,000,000 | Conditional follow-up |

Scalar cadence is 50 sweeps, spatial Fourier/structure cadence 1,000, and selected real-space profiles 20,000. Ordered/random preparations have different update, exchange and initialization seeds, also differing between sizes and couplings. Random starts fill all six complex components independently with the ordered state's expected density. No restricted phase or chiral parametrization is introduced.

The [L256 transition report](../he3_transition_L256/REPORT.md) motivates this campaign: at g_D=0, the (160,256) chirality crossing was .381818 ± .000021 and the direct half-charge L256 crossing .381809 ± .000008. Their agreement did **not** rescue the four-size leading WM fit: chi²/dof=10.07/2. The integer fit was much worse. Those errors were conditional statistical errors, not certificates of equilibrium or BKT behavior.

The [charged-observable report](../he3_weak_soc_spatial_01/REPORT.md) found coherent A+,z and B++ channels above the SOC chirality feature, while P lost overlap with them. It also identified the forward mixed-gradient stencil's diagonal orbital preference. This makes the full helicity tensor and those particular charged channels higher priorities than another vortex classifier.

![Prior transition evidence, with the existing conditional errors](figures/prior_transition_diagnostics.png)

## Measurements and independent physics checks

All ten previous scalar histories remain. `curvature_xy` is the eleventh and is measured in the same reduction, at the same sample and temperature assignment as Ix, Iy, Kxx and Kyy. For the implemented uniform link twists, each forward neighbor is multiplied by exp(-i delta_nu). Writing X_mu=A_mu,x(r+x), Y_mu=A_mu,y(r+x), U_mu=A_mu,x(r+y), V_mu=A_mu,y(r+y), the added extensive curvature is

\[
K_{xy}=2K\sum_{r,\mu}\mathrm{Re}(X_\mu^*V_\mu+Y_\mu^*U_\mu).
\]

It follows from the two mixed forward-gradient contractions, not from a continuum replacement of the stencil. The onsite quartic and SOC terms have no direct derivative under this U(1) link twist. SOC nevertheless changes the equilibrium curvature and current distributions. The reconstructed tensor is

\[
\rho_{ij}=\frac{\langle K_{ij}\rangle-[\langle I_iI_j\rangle-\langle I_i\rangle\langle I_j\rangle]/T}{L^2}.
\]

The cross-current covariance is essential. Equal diagonal entries do not imply rho_xy=0. For a uniform real plus-orbital state of density n, the independent check gives rho_xx=rho_yy=4Kn and rho_xy=2Kn. The tests differentiate an independent complex128 implementation of the full GL contractions, with the actual forward stencil and g_D=0/.01, using Richardson central differences. Random, correlated and uniform fields are covered, including nonunit K. The diagnostic mixed derivative discrepancy at h=.01 is below **1.7e-7** in the tested fields; unit tests allow the existing complex64 local-product rounding. Tests check all five first/second twist derivatives, not just Kxy.

Spatial recording reuses the validated 16 fields at k=0, ±kx,min and ±ky,min: kappa, P, five orthonormal Q components, Bxx/Byy/Bxy and all six original A components. The existing manifold-distance, histogram and energy-component records are retained. **No redundant Fourier channels need to be saved.** [charged_amplitudes](../../models/he3/spatial.py) and the spatial summaries expose the requested combinations before taking absolute squares:

\[
A_\pm=(A_x\pm A_y)/\sqrt2,\quad
B_{++}=(B_{xx}+B_{yy}+2B_{xy})/2,\quad
B_{--}=(B_{xx}+B_{yy}-2B_{xy})/2,
\]
\[
B_{+-}=(B_{xx}-B_{yy})/2=P/2.
\]

These are **real diagonal orbitals**, not circular chiral projections. The A channels retain each spin component and their contraction. Full B uses weights (1,1,2); this Frobenius norm is preserved under the orbital transformation. The fixed x/y orbital channels remain accessible. Tests compare explicit site sums at all five signed momenta against the transformed GPU-compatible basis, and check the B norm identity. Fourier sums are unnormalized; S(k)=<|f_k|²>/L². The existing ratio helper returns xi/L, not xi itself.

The real-space records retain kappa/P/Q and add only A+,z, A−,z, B++ and B--. They save x/y profiles through L/2 at the nearest ladder slots to (.3815,.38182,.3830) for g_D=0 and (.3852,.3856,.3870) for .01. FFTs now compute these 11 requested fields without unused radial binning. Normalization is G(r)=mean_x f(x+r)f(x)*. Explicit displaced products and a known phase wave validate the result. Charge-one and charge-two phases have their expected different wave numbers. Connected Q subtracts the ensemble mean inside each statistical resample; no snapshot mean is removed from the charged channels.

![Independent mixed-twist and charged-profile checks](figures/physics_validation.png)

## Exact restart and economical traffic diagnostics

The implementation follows the existing [checkpoint plan](../he3_pt_ladder_validation/CHECKPOINT_PLAN.md). Runtime saves the field, actual site RNG array, beta assignments and energy-cache state. The Gaussian runtime also saves all per-site counters and its sweep count. PT saves both inverse mappings, exchange RNG, next parity, absolute counters, energy buffer and all five unfinished-trip rows. Simulation saves completed progress, accumulated scalar histories, sample coordinates, spatial/profile records and their measurement counter. Scratch Fourier buffers are rebuilt; they contain no physical state.

Checkpoints occur only between completed sweeps, after scheduled exchanges and measurements. A pickle-free NPZ is written to a sibling temporary file, flushed, read back for archive integrity and atomically replaced. Invalid settings, shapes/dtypes, permutations, parity, beta assignments, counters and sample indices are rejected before restoring live state. The GPU/Python/Numba/Numba-CUDA/NumPy/CUDA versions are recorded and checked. Keep the same source and software environment throughout a trajectory; this is not a promise of bitwise continuation across code or hardware changes.

`Simulation.run()` on an existing Simulation now continues its unfinished allocation; it does not restart its burn/sample clock. A fresh Simulation preserves the original one-shot behavior. Checkpoint loading deliberately requires the same planned allocation and cadences. Changing a completed run's target length is not an automatic extension feature.

Every walker has a position saved every 50 sweeps, with an additional **one-byte exact last-visited endpoint label** (-1 unknown, 0 cold, 1 hot) obtained from the existing trip bookkeeping. Thus visits between position samples do not corrupt sampled flow labels. Only the history-recording kernel changes; the exchange kernel does not. Exact swap/trip counters are saved at zero, the burn boundary, each 50k boundary and completion. Endpoint passages obey Npass=2 Nroundtrips+visited_other. Counter differences give exact interval traffic, including journeys spanning a boundary.

Full duration/path detail is still limited by the 50-sweep history. A strict production-only journey and a journey completed during production but begun in burn are different statistics. Retain right-censored walks; never estimate an uncensored waiting-time mean from completed journeys alone. The new `endpoint_flow` uses the saved exact labels, while the prior PT tools remain usable for sampled trajectories.

`save_runs(..., binary_metadata=True)` stores diagnostic arrays as binary NPZ members instead of enormous JSON lists. `load_runs()` reads both old v1 files and this v2 representation into the same logical run dictionary. The default writer behavior for other callers remains v1.

Restart validation compares uninterrupted runs to interruptions at sweeps 19,20,24,25,29,30,35,37,40,80 and two interruptions at 37/59, at both couplings. This covers burn, sample/profile boundaries, both parities, invalid/valid energy caches and history chunk rollover. Fields, both RNG arrays, sampler/PT counters, unfinished trips, histories, mappings and **every saved measurement are bitwise equal**. Failed replacement preserves the old checkpoint; truncated, incompatible and malformed files are rejected. An independent every-exchange endpoint tracker verifies the sparse saved labels. The final runner itself was tested through partial output, checkpoint loading and completion with actual Gaussian/PT updates and all measurements enabled at L8/R160.

## Bounded costs and resource requirements

The actual local device is an **RTX 4070 SUPER**, not the cluster's RTX 5090. [benchmark.py](benchmark.py) uses the frozen 160-replica ladder, g_D=.01, 20 compilation/warmup sweeps, 100 timed sweeps and one final measured sweep per process. It times measurements separately, writes/loads a checkpoint and saves an output archive. No such short evolving run measures equilibration. Final results are in [results](results/) and [resources.csv](results/resources.csv).

| L | Local PT ms/sweep | Total measurement overhead at chosen cadences | Measured host peak, GiB | Minimal-history checkpoint write / restore, s |
|---|---:|---:|---:|---:|
| 160 | 3.95 | 2.16% | .90 | .59 / .35 |
| 256 | 9.80 | 2.19% | 1.71 | 1.49 / .90 |
| 320 | 15.69 | 2.20% | 2.46 | 2.34 / 1.36 |
| 384 | 22.26 | 2.10% | 3.37 | 3.33 / 2.16 |

The overhead is total scalar+spatial+profile cost, not an isolated Kxy speed comparison. Timing windows are short, on Windows with normal clock/scheduling variation, and exclude initialization/compilation. Checkpoint timings include read-back verification but only minimal accumulated data and local disk I/O. Shared-filesystem performance and full-history saving are not benchmarked.

| L | Known device arrays + full history, GiB | Raw final data, GiB | Full checkpoint estimate, GiB | Conservative host budget, GiB | Historical GPU-hours/case estimate |
|---|---:|---:|---:|---:|---:|
| 160 | .40 | 1.68 | 2.02 | 6.0 | 1.3 |
| 256 | 1.01 | 3.39 | 4.25 | 10.5 | 9.4 |
| 320 | 1.49 | 2.84 | 4.19 | 10.4 | 13.1–14.6 |
| 384 | 2.15 | 4.55 | 6.48 | 15.0 | 28.3–34.7 |

Device figures sum actual owned arrays and projected history; they exclude the CUDA context, driver allocations and other applications. Host budgets allow simultaneous history/stacked arrays and checkpoint copies. Output figures are byte-count projections before compression; a one-event compressed archive is not a reliable full-run compression ratio. L320 stores less data than L256 because its production allocation is shorter.

The last column uses the **older recorded 200-replica L160/L256 timings**, scaled by 160/200 and the requested allocations. Above L256, it spans area scaling and the observed L160-to-L256 timing exponent. It is not a fresh 5090 benchmark or a confidence interval; the older extracts do not identify their GPU. The recent L96 C160 pilots do explicitly identify an RTX 5090 and run at about .319–.320 ms/sweep, but extrapolating their hardware utilization to large L is not established. Re-measure L256 on Kraken before allocating L320. Four preparations/couplings per size would cost roughly 95–102 historical GPU hours for L160/256/320, before saving, overhead and any scientifically required extensions.

The [Slurm example](reference.slurm) requests one GPU, four CPUs, 32 GiB host RAM and 12 hours. Select Kraken's actual RTX 5090/account/partition names locally. The runner checkpoints hourly and stops after 11.5 hours or a stop signal, leaving 30 minutes for output. Re-running resumes the same case; there is no automatic requeue or submission. Allow at least 20 GiB free disk per L256 case for checkpoint replacement plus partial/final data, and about 30 GiB for conditional L384. Retained checkpoints and partial files are not automatically deleted.

The warning signal targets the `srun` job step, as specified in the [Slurm sbatch manual](https://slurm.schedmd.com/sbatch.html#OPT_signal); its delivery can be up to 60 seconds early. The wall-clock stop is also checked independently between sweeps.

![Measured costs and explicit full-allocation projections](figures/resources.png)

## Equilibration gate and analysis handoff

The [old PT analysis](../he3_pt_equilibration/REPORT.md) found L256 completed-trip means of 3.18M/3.02M sweeps (g_D=0/.01), but traffic scales 2 R duration/Npass of **7.64M/5.19M**. Only 41/200 and 64/200 walkers had a strict sampled production loop. Completed-trip means omit the slow censored tail. At L160 the old completed means were already 1.42M/1.35M. Thus the requested 1M or 3M burns cannot be declared sufficient from those data.

The [longer L96 comparison](../../analysis/pt_ladder_test/REPORT.md) supports C160: roughly twice A's endpoint traffic per GPU second, with all walkers completing a production loop in B/C. However, signed-chirality errors continued increasing with block length and SOC walker chirality memory remained visible at long lags. This is evidence for a candidate ladder, not proof of large-L equilibration. The earlier conservative C scenarios give traffic scales around 5.55–7.48M at L320/g_D=0 and 3.84–5.13M at .01; their uncertainty is not a measured bound. The 5M production allocation is a diagnostic window, not several assured independent passages per walker.

![Frozen spacing and the old large-size transport warning](figures/ladder_and_prior_transport.png)

Before accepting or extending a reference, examine production traffic in blocks, completed and censored journeys, exact flow labels, signed-chirality/Q/current/energy memory in physical-walker order, and independent-start agreement. Compare the first/second production halves and several later cutoffs. Recompute errors with 50k,100k,200k,500k and, where enough blocks remain, 1M blocks. An apparent plateau with only a handful of blocks is weak evidence. Good swaps, matching means or one completed round trip alone are insufficient. No numerical gate can rule out an unobserved slow sector.

[analysis_ready.py](analysis_ready.py) prepares joint time-block curves for Binder, diagonal/full helicity, kappa/P/Q correlation-length ratios, connected Q and the charged orbital/spin channels. Each run resamples the same blocks across all its temperatures and scalar/spatial channels; distinct runs must use independent bootstrap seeds. It reuses the existing crossing and WM routines. The crossing routine now ignores undefined/flat regions outside the requested window. WM now checks input weights, selected sizes/windows and flat or vanishing-correction optima; the existing ten WM tests pass.

For each coupling and separately for each preparation:

1. Compare Binder and xi/L crossings across size pairs. Multiple/missing bootstrap roots remain failures, not a selected convenient branch. Compare their differences within the joint bootstrap.
2. Compute direct intersections with rho=2T/(pi q²): **2T/pi for q=1, 8T/pi for q=.5**. Report these finite-L proxies separately from WM estimates. Use the historical diagonal average for continuity; use sqrt(det rho) only for a positive-definite tensor and a physically justified single anisotropic phase action. The helper returns undefined effective stiffness outside the positive-definite domain; it cannot establish the physical single-phase assumption. Anisotropic coordinate rescaling also changes the finite torus geometry. Soft spin modes, winding sectors and coupled transitions can invalidate that reduction and its universal-jump interpretation.
3. Fit the leading WM form by passing q²rho and q²sigma to the existing fitter, refitting each time-block bootstrap. Inspect chi²/dof, profile width, boundaries and failures, not merely an optimizer temperature. Temperature interpolation creates no new independent observations.
4. Fit T_L=T_BKT+A/[ln(L/L0)]² to the direct proxies with bootstrap refits. A free L0 is refused with only three sizes: that would give zero fit degrees of freedom. Fixed-L0 sensitivity is available. With only four sizes the free fit still has one degree of freedom and may be ill-conditioned. Bounds on L0 and leading-order truncation are systematic assumptions.
5. Compare charged S(0) scaling, ratios and real-space decay above/below chirality. For S(0)~L^(2-eta), the pairwise eta helper retains bootstrap errors. Test power-law versus exponential descriptions over several distance/temperature windows with periodic finite-size effects; sparse profiles alone do not establish asymptotic algebraic order.

Repeat size cuts Lmin=96,128,160,256 where the remaining sizes permit a fit. **L64 is excluded.** Existing smaller data are supplementary; no L64/L96 rerun is requested. Old and new ladders differ: the new multi-size WM helper refuses mismatched grids, so use the earlier validated interpolation/WHAM analysis for those comparisons rather than silently treating unrelated slots as identical. Keep conditional statistical errors separate from size/window/fit-form systematics, equilibration/start disagreement and theoretical jump-charge assumptions.

## Validation status and next execution

**All 72 focused tests pass**, covering the new campaign, independent GL energy checks, PT, caches, data mappings, crossings and BKT fits. See [focused_tests.log](results/focused_tests.log). The whole repository suite was also run. Its unchanged HEAD baseline already had 3 failures and 31 errors, including removed runner interfaces, missing `analyze`, and unrelated Ising/XY analysis expectations. After correcting the relevant WM/crossing issues, the full suite runs 117 tests with 1 remaining failure and 27 errors, all also present in that baseline. See [final_regression.log](results/final_regression.log) and [baseline_regression.log](results/baseline_regression.log). The repository-wide suite is not presented as green.

Ready locally: the new measurement normalization, charged transforms/profiles, synchronized binary output, exact continuation on the local stack, bounded cost measurements, explicit reference configuration and analysis helpers. Still unvalidated: continuation on RTX 5090/Linux, cluster filesystem checkpoint cost, sustained large-L timing, large-L C160 transport/equilibrium and any BKT universality claim. L320 is not yet scientifically cleared.

In a short Kraken GPU allocation, from the repository root, run:

```bash
python -m unittest tests.test_he3_campaign tests.test_pt_walkers tests.test_bkt tests.test_crossings -v
python exploration/he3_large_size_preparation/benchmark.py --L 256 --output analysis/large_C160_preflight
```

Retain that log and GPU/software versions. If these pass with adequate memory/I/O headroom, the next **manually launched** long calculation is `python -u large_run.py`, with the default single L256 reference. Follow with its independent random partner and the other coupling; inspect those results before enabling L320. The `-u` flag only makes log output unbuffered. No long calculation or Slurm job was launched during this preparation.
