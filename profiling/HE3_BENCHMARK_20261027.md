# He3 benchmark comparison — 27 October 2026

The updated engine is substantially faster for the larger lattices. An
alternating same-session L64 comparison measured **1.66x throughput**, with
39.7% less elapsed time per sampled sweep. Simulation source and production
datasets were unchanged during benchmarking. Only the profiling harness was
extended to accept ladder parameters and record the active cached kernel.

The GPU is the same RTX 4070 SUPER, driver 610.88, Windows WDDM. Tests used
100 replicas, 16x16 thread blocks, K=1, pressure 20 bar, coefficient temperature
1, proposal width .15, and PT exchanges and measurements every ten sweeps.
One sweep updates **all 100 replicas**. Initialization and compilation are
excluded. Normal desktop applications and driver clock management remained active.

**Comparison with the previous benchmark.** To isolate the code changes from
the revised temperature ladder, the main comparison preserves the historical
.15–.45 ladder focused on .257. Each case uses 1,000 warmup sweeps and five
windows of 3,000 measured sweeps; values are medians.

| L | Previous sampled ms/sweep | Updated sampled ms/sweep | Throughput ratio | Less elapsed time |
|---|---:|---:|---:|---:|
| 32 | .205216 | .199653 | 1.03x | 2.7% |
| 48 | .229855 | .183814 | 1.25x | 20.0% |
| 64 | .373944 | .240163 | 1.56x | 35.8% |
| 80 | .524055 | .317248 | 1.65x | 39.5% |

L32's small change is within the scale of timing variation. Local-only timings
at L32/L48 show no improvement, although the L48 sampled loop benefits from the
faster energy/measurement paths. Small lattices remain sensitive to host dispatch
and Windows scheduling. This is not evidence that the smaller kernels alone
became slower.

| L | Previous updates | Updated updates | Previous updates+PT | Updated updates+PT |
|---|---:|---:|---:|---:|
| 32 | .151131 | .166939 | .163237 | .166103 |
| 48 | .136978 | .140702 | .149476 | .160599 |
| 64 | .233263 | .156559 | .261787 | .160094 |
| 80 | .323367 | .256459 | .384183 | .257237 |

All entries are milliseconds per sweep. Mode windows run sequentially on evolving
states, so subtraction of their times is not a controlled overhead measurement.
The sampled benchmark includes host measurement copies but excludes growing
history lists, final history stacking, and compressed output.

Raw results: [previous](results/he3_baseline.json),
[updated](results/he3_20261027_baseline.json).

**Two checks confirm that the L64 gain exceeds historical timing noise.**

- After 100,000 warmup sweeps, the previous sampled median was .336499 ms,
  and the new median is **.207154 ms: 1.62x throughput**. New windows span
  .205531–.207600 ms. Compare
  [previous long warmup](results/he3_thermalized_L64.json) and
  [updated long warmup](results/he3_20261027_thermalized_L64.json).
- A separate same-session comparison loaded the old He3 and Runtime source from
  git HEAD `0bfd8ad` into isolated in-memory modules. Neither version replaced
  workspace files. Six alternating old/new windows gave medians **.350918 ms
  versus .211477 ms: 1.66x throughput**. All six paired comparisons favored the
  new engine (1.58–1.75x). See [paired results](results/he3_20261027_paired_L64.json).

**NVIDIA timeline comparison.** The Nsight Systems captures repeat the previous
capture settings. At L64, both traces contain 4,000 update kernels, 100 energy
kernels, 100 exchanges, and 100 measurement kernels. Mean durations are:

| Kernel | Previous | Updated | Updated fraction of GPU kernel time |
|---|---:|---:|---:|
| One Metropolis color | 53.23 us | 36.96 us | 81.9% |
| Full measurement / cached measurement | 694.34 us | 300.24 us | 16.6% |
| PT energy | 378.35 us | 23.02 us | 1.3% |
| Exchange/bookkeeping | 3.38 us | 3.32 us | 0.2% |

The combined changes are direct gradient differences, mixed-precision local
energy/measurement arithmetic, and cached energy in measurements. These results
do not isolate the speedup of caching alone. The standalone PT energy kernel
still executes; its large speedup primarily reflects the arithmetic change.

At L80, update/measurement/energy means changed from 76.28/1033.98/581.23 us
to 58.98/487.35/35.92 us. The exchange kernel remained about 3.53 us.

Update execution is now the main remaining GPU cost, about 82% of kernel time.
PT energy is a much smaller target than it was previously. Actual host transfers
remain negligible. The fraction of elapsed time covered by recorded CUDA work
fell from about 93% to 87% at L64 and 95% to 91% at L80 because the kernels finish
sooner and submission/scheduling gaps occupy a larger share. These are timeline
activity fractions, not measured SM occupancy.

Open the new captures in Nsight Systems:
[L64](results/he3_20261027_nsys_L64.nsys-rep),
[L80](results/he3_20261027_nsys_L80.nsys-rep).
Adjacent CSV and SQLite exports retain the timing evidence.

**Compiled resources support the observed improvement.**

| Kernel | Previous registers/thread | Updated registers/thread |
|---|---:|---:|
| Metropolis color | 71 | 64 |
| PT energy | 72 | 56 |
| Full / cached measurement | 156 | 94 |

All three still report zero local-memory bytes. Register-based theoretical warp
occupancy ceilings rise from 50% to 66.7% for updates/energy and from 16.7% to
33.3% for measurement. Actual occupancy was not measured. The update PTX has
97 scalar global float-load instructions, versus 157 previously; this static
instruction count is not a DRAM-traffic measurement. The retained RNG path
still contains 13 FP64 conversion/multiply sequences.

The previous hardware-counter attempt was denied by NVIDIA driver permissions;
no driver settings were changed or new counter-based utilization claims made.

**Runtime projection and your current temperature ladder.** With the matched
historical ladder, applying PT timing to 100,000 thermalization sweeps and
sampled timing to 500,000 measurement sweeps at each size gives **12.71 minutes
previously versus 9.08 minutes now**, excluding setup/history assembly/saving.
That is 1.40x aggregate throughput, or 28.5% less elapsed loop time.

A separate check used your actual revised ladder: .32–.44, focus .39, window
.04, fraction .70. Its sampled medians were .189633/.189790/.229929/.343680 ms
for L32/48/64/80. The projected loop time is **9.21 minutes**. Adding the previous
saving estimate of about 47 seconds gives **roughly ten minutes overall**,
plus setup and history assembly. This is a projection, not a timed full
production run. Saving performance was not rebenchmarked because the storage
code and sample count are unchanged.

See [current-ladder results](results/he3_20261027_current_ladder.json),
[comparison summary](results/he3_20261027_comparison.json), and
[comparison figure](results/he3_20261027_comparison.png).

**Reproduction**, from the project root in PowerShell:

```powershell
$he3Python = 'C:\Users\jinhao\AppData\Local\Programs\Python\Python312\python.exe'
& $he3Python -B profiling/profile_he3.py --sizes 32 48 64 80 --warmup 1000 --sweeps 3000 --repeats 5 --output profiling/results/new_comparison.json

# Revised production ladder:
& $he3Python -B profiling/profile_he3.py --sizes 32 48 64 80 --T-min .32 --T-max .44 --Tc .39 --tc-window .04 --tc-fraction .70 --warmup 1000 --sweeps 3000 --repeats 5 --modes pt production --output profiling/results/new_production_ladder.json
```
