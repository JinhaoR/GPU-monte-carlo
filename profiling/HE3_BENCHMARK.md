# He3 performance investigation — 26 October 2026

The current He3 simulation was benchmarked on an NVIDIA GeForce RTX 4070 SUPER
(12 GB, driver 610.88, Windows WDDM), using Python 3.12.10, Numba 0.68.0,
NumPy 2.4.6, and NVIDIA Nsight Systems 2025.6.3. Nsight Compute 2026.1.1
was also attempted. The simulation implementation and existing datasets were
left unchanged; only this profiling harness and its results were added.

The configuration matches `main/he3.py`: pressure 20 bar, fixed coefficient
temperature 1, K=1, proposal half-width .15, 100 temperatures from .15 to .45
with the existing ladder focused on .257, and exchanges and measurements every
10 sweeps. One sweep here updates **all 100 replicas**.

**Unprofiled throughput.** Initialization, RNG setup, JIT compilation, and a
1,000-sweep warmup were excluded from timing. Each entry below is the median
of five 3,000-sweep windows, synchronized before and after each window.
The sampled mode runs the normal measurement kernel, host copy, and per-channel
copies. It excludes growing history lists, their final stacking, and file saving.

| L | Local updates, ms/sweep | Updates + PT, ms/sweep | Updates + PT + sampling, ms/sweep | Sampled sweeps/s |
|---|---:|---:|---:|---:|
| 32 | 0.1511 | 0.1632 | 0.2052 | 4,873 |
| 48 | 0.1370 | 0.1495 | 0.2299 | 4,351 |
| 64 | 0.2333 | 0.2618 | 0.3739 | 2,674 |
| 80 | 0.3234 | 0.3842 | 0.5241 | 1,908 |

A separate L64 check used the production 100,000-sweep warmup. Its medians were
0.2275 ms for local updates, 0.2576 ms with PT, and **0.3365 ms with sampling**
(2,972 sweeps/s). The five sampled windows ranged from .33596 to .33728 ms/sweep.
This check shows that the initial L64 timing was somewhat conservative.
Raw results: [baseline](results/he3_baseline.json),
[longer warmup](results/he3_thermalized_L64.json).

Clocks were left under normal driver control, with the normal desktop running.
The short-window comparison is descriptive rather than a controlled paired
optimization experiment: modes run sequentially, states evolve, and Windows
scheduling and clocks vary. For example, the initial L64 sampled windows span
.3416–.4008 ms/sweep. Differences between modes should not be interpreted as
precise isolated overhead measurements.

**Where NVIDIA's timeline puts the time.** Nsight Systems captured warmed,
bounded sampled runs, excluding compilation and initialization. Kernel-time
fractions in the L64 trace were:

| Operation | Launches / 1,000 sweeps | Mean duration | Fraction of kernel time |
|---|---:|---:|---:|
| One Metropolis color | 4,000 | 53.23 us | 66.4% |
| Full ordered measurement | 100 | 694.34 us | 21.7% |
| Energy calculation for PT | 100 | 378.35 us | 11.8% |
| Replica exchange and bookkeeping | 100 | 3.38 us | 0.1% |

The two reductions together consume about one third of kernel time. The exchange
decision itself costs very little. The 4,300 launches match the requested schedule.
L32 and L80 captures used 2,000 sweeps, with correspondingly doubled counts.

| L | Trace window | Time in recorded CUDA kernels/copies | Fraction of window |
|---|---:|---:|---:|
| 32 | 408.49 ms | 160.74 ms | 39.3% |
| 64 | 344.45 ms | 320.65 ms | 93.1% |
| 80 | 980.09 ms | 934.18 ms | 95.3% |

These fractions describe whether recorded CUDA work is running, **not SM
occupancy or arithmetic utilization**. L32 has a substantial host/launch/scheduling
limit: its median gap between adjacent update kernels is about 25 us, versus a
12.8 us update kernel. At L64/L80 the corresponding gap is about 1.2 us,
with 53/76 us kernels. Larger lattices are primarily limited by kernel execution.
Without hardware counters, the balance between arithmetic, memory transactions,
and instruction latency remains unresolved.

The copy API must be interpreted carefully. In the L64 trace,
`cuMemcpyDtoH` spends 178.3 ms in blocking CPU calls, but the actual 100 GPU
copies take only **0.1046 ms total**, transferring 720 kB. Most of the API duration
waits for queued GPU work. PCIe bandwidth is not the measured bottleneck.
Allocation and freeing together take only .863 ms in that trace.

Open these files in Nsight Systems:
[L32](results/he3_nsys_L32.nsys-rep),
[L64](results/he3_nsys_L64.nsys-rep),
[L80](results/he3_nsys_L80.nsys-rep).
The adjacent CSV and SQLite exports retain the numerical evidence.

**Compiled-kernel findings and optimization candidates.** These observations
come from the compiled PTX, CUDA resource metadata, and source inspection; they
do not substitute for hardware stall/cache counters.

| Kernel | Registers/thread | Local bytes/thread | Static shared bytes/block |
|---|---:|---:|---:|
| Metropolis color | 71 | 0 | 0 |
| PT energy | 72 | 0 | 2,048 |
| Ordered measurement | 156 | 0 | 18,432 |

There is no evidence of register spills. The GPU has 65,536 registers and at most
48 resident warps per SM. With 256-thread blocks, the resource ceilings are three
blocks (50% theoretical warp occupancy) for update/energy and one block (16.7%)
for ordered measurement. Actual occupancy was not measured.

The most useful next optimization experiments are:

1. **Simplify the update energy difference.** The update evaluates all three
   affected gradient cells twice. Compiled PTX retains both sets of neighbor
   loads: 144 neighbor float loads plus 12 center float loads per proposal,
   before cache reuse. Computing the quadratic gradient change directly can
   avoid repeated terms and stencil work. The existing independent local-energy
   tests should be retained when evaluating this change.
2. **Investigate the reduction precision and duplicate scans.** Energy and full
   measurements use complex128/float64 throughout their local algebra, not only
   in the final sums. On sampling sweeps, energy is recomputed for PT and then
   included again in the full measurement on the same physical configurations.
   Reusing or combining this work is a concrete target; removing the entire
   energy kernel would cap the benefit at roughly 12% of current GPU kernel
   time. Any mixed-precision change needs accuracy checks, especially the
   cancellation in helicity near the transition.
3. **Investigate field layout and RNG instructions.** Adjacent update lanes
   access a given field component with 96-byte spacing in the current layout.
   The old engine packs colors/components contiguously. Hardware counters are
   needed to quantify the remaining transaction/cache cost. The compiled update
   also retains 13 FP64 conversions and multiplies in the nominally float32 RNG
   path; their execution cost has not been isolated.
4. **Reduce launch/submission overhead for small lattices.** L32 is a stronger
   candidate for CUDA graphs or another reduction in Python/driver dispatch
   than L64/L80, where recorded kernels already fill most of the window.

An 8x8 block experiment removed padding for L48/L80 but did **not** improve
performance overall. Its sampled timings were .1790/.2489/.4065/.5821 ms for
L32/48/64/80, versus .2052/.2299/.3739/.5241 ms at 16x16. The aggregate projection
was slower. The timing variability also argues against choosing a new default
from the L32 improvement alone. See [raw 8x8 results](results/he3_threads8.json).

**Runtime and saving.** Applying the measured PT rate to 100,000 warmup sweeps
and the sampled rate to 500,000 measurement sweeps gives about **12.7 minutes**
for all four sizes, excluding setup, history assembly, and output. Substituting
the longer-warmup L64 measurement gives 12.4 minutes. These are extrapolations,
not a timed full production run.

Saving one actual L64 dataset through `save_runs` took **11.67 seconds**:
360 MB of sample arrays became a 337.9 MB compressed NPZ. It was written to a
temporary file and removed; the original dataset was untouched. All four sizes
have the same number and shape of sample channels, suggesting about another
47 seconds for compression/writing. An overall expectation is roughly
**13–14 minutes** on this machine under similar conditions.
See [saving benchmark](results/he3_save_benchmark.json).

**Nsight Compute limitation.** Hardware-counter collection returned
`ERR_NVGPUCTRPERM`, including when retried outside the sandbox. Driver settings
were not changed. Consequently there are no measured SM-utilization, DRAM
throughput, cache-efficiency, or warp-stall results. The small initial
`he3_ncu_update_L64.ncu-rep` contains kernel identification only, not counters.

**Reproduction.** Run from the repository root in PowerShell:

```powershell
$he3Python = 'C:\Users\jinhao\AppData\Local\Programs\Python\Python312\python.exe'
& $he3Python -B profiling/profile_he3.py --sizes 32 48 64 80 --warmup 1000 --sweeps 3000 --repeats 5 --output profiling/results/new_baseline.json

nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --stats=false --output=profiling/results/new_L64 $he3Python -B profiling/profile_he3.py --sizes 64 --warmup 1000 --sweeps 1000 --repeats 1 --modes production --cuda-profile-api --output profiling/results/new_L64_timing.json

# Requires NVIDIA performance-counter access in the driver:
ncu --profile-from-start off --target-processes all --set basic --clock-control none --kernel-name regex:metropolis_color --launch-count 1 --export profiling/results/new_ncu_L64 $he3Python -B profiling/profile_he3.py --sizes 64 --warmup 500 --sweeps 1 --repeats 1 --modes probe --cuda-profile-api
```

Profiler replay times and failed counter attempts are excluded from throughput
estimates. The benchmark does not establish statistical equilibration or mixing.
