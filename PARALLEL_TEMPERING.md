# Parallel tempering implementation and validation

Implemented on 2026-10-20. Main implementation: `core/parallel_tempering.py`.

## Usage and architecture

Wrap the existing Runtime and pass the wrapper to Simulation:

```python
from core.parallel_tempering import ParallelTempering
pt = ParallelTempering(runtime, exchange_interval=10, seed=43)
run = Simulation(pt, thermalization_sweeps=10000,
                 measurement_sweeps=100000, sample_interval=50).run()
```

Ordinary Runtime-based simulations retain the same calls and behavior. Entry points do not enable PT automatically. Runtime retains the physical GPU state, site RNG, local updates, and model measurement launches. The wrapper owns only exchange scheduling and GPU exchange bookkeeping. Its constructor allocates the energy vector, inverse maps, independent exchange RNG, swap counters, and basic round-trip state once.

An exchange interval is measured in completed full local sweeps. Each round recomputes current total Hamiltonian energies in a dedicated energy-only GPU reduction. Neighboring temperature labels are paired in alternating even/odd disjoint matchings. With two walkers the sole pair is attempted every interval. Accepted exchanges swap beta assignments and both inverse maps; configurations and site RNG do not move. The default CUDA stream orders updates, energy reduction, exchange, and observation. Measurement kernels write directly into fixed temperature order, retaining the existing single CPU result transfer and adding no mapping transfer.

## Acceptance and correctness

For configurations a,b at inverse temperatures beta_a,beta_b, the new/old joint canonical weight ratio is exp((beta_a-beta_b)*(E_a-E_b)). The kernel uses a float64 log comparison and the actual float32 beta values used by local Metropolis. Each disjoint matching satisfies detailed balance; the deterministic alternating schedule preserves the equilibrium distribution, although the full composition need not be reversible. Energies are total energies and count each periodic bond once. [Theory reference](https://arxiv.org/html/physics/0508111).

Sixteen GPU tests passed with:

```powershell
python -B -m unittest discover -s tests -v
```

Coverage includes forced acceptance and rejection, sign and freshness of total energy, both inverse maps and beta assignments, all observation channels after swaps, unchanged configurations/site RNG during exchange, odd/even pairing, 259 walkers exceeding one thread per pair, exchange timing across thermalization/sample boundaries, endpoint and interior round-trip semantics, invalid ladders/intervals, nonfinite-energy handling, saved metadata round trips, and ordinary multi-size simulation/analysis/plotting. Scoped and repository git diff whitespace checks pass.

Independent equilibrium experiments used periodic L=4, 5,000 burn-in sweeps plus 50,000 measured sweeps, sampling every five sweeps. Blocked means and jackknife used 100-sample blocks (500 sweeps, 100 blocks). Ordinary and PT used independent fixed seeds. Ising was compared against enumeration of all 65,536 configurations at the effective float32 beta ladder; XY was compared against independent ordinary Metropolis, including helicity and Binder. Each mode produced 10,000 measurements per temperature.

| Model and check | Largest deviation in standard errors |
|---|---:|
| Ising ordinary versus exact enumeration | 1.19 |
| Ising pt versus exact enumeration | 0.84 |
| IsingModel PT versus ordinary, combined errors | 1.39 |
| XYModel PT versus ordinary, combined errors | 1.72 |

Ising temperatures were 1.8,2.4,3.2; XY temperatures were 0.6,0.9,1.3. The PT runs completed 2,044 Ising and 1,033 XY walker round trips. These checks establish agreement at the tested sizes and precision; they do not establish faster decorrelation for larger lattices.

## Wall-time overhead

RTX 4070 SUPER, i7-13700KF, Windows WDDM, driver 610.88, Python 3.12.10, NumPy 2.4.6, Numba 0.68.0. Five interleaved shuffled trials per candidate, each 3,000 sweeps and 60 measurement points per temperature. Compilation was warmed and state/RNG were reset outside timing. PT timings include final diagnostics collection. Startup and plotting are excluded. The display GPU causes visible variation in the raw trials.

All cases used 25 temperatures and 16x16 local-update blocks. XY used its current 0.78-1.02 ladder; Ising used a separate validation ladder 1.8-3.2 spanning its transition. Sampling was every 50 sweeps. These benchmark ladders do not change either entry point.

| Model | L | Ordinary seconds | PT every 10 seconds | Overhead | PT every 50 seconds | Overhead |
|---|---:|---:|---:|---:|---:|---:|
| XYModel | 32 | 0.4140 | 0.4538 | +9.6% | 0.4200 | +1.4% |
| XYModel | 48 | 0.3920 | 0.4240 | +8.1% | 0.4104 | +4.7% |
| XYModel | 64 | 0.3724 | 0.3940 | +5.8% | 0.3779 | +1.5% |
| IsingModel | 32 | 0.3378 | 0.4034 | +19.4% | 0.3722 | +10.2% |
| IsingModel | 64 | 0.3272 | 0.3613 | +10.4% | 0.3389 | +3.6% |

At L=64, XY overhead was 5.8% for exchange interval 10 and 1.5% for interval 50. Ising overhead was 10.4% and 3.6%, respectively. Across the tested cases, interval-10 overhead was 5.8-19.4%; interval-50 overhead was 1.4-10.2%. These are short-run wall-time measurements, not a claim of effective-sample speedup. No completed L=64 round trips occurred within these short 3,000-sweep benchmark trials; longer production diagnostics are needed to assess mixing and the chosen interval.

Nsight Systems captured a separate L=64 XY run with 1,000 sweeps and 100 exchange rounds, deliberately excluding sampling and diagnostics from the capture:

- 4,000 local update kernels: 15.675 ms of GPU execution.
- 100 total-energy kernels: 4.136 ms, approximately 41.36 microseconds each.
- 100 exchange/diagnostic kernels: 0.336 ms, approximately 3.36 microseconds each.
- Zero memory-copy events, allocation calls, or per-exchange synchronization calls in the captured range. Only the explicit capture-boundary context synchronizations occur.

This confirms that energy evaluation and exchanges remain GPU resident. Sampling retains the existing small observable-result transfer; explicit diagnostic collection copies only small maps/counters.

## Changes outside core/parallel_tempering.py

| File | Change |
|---|---|
| `core/model.py` | Optional non-abstract GPU total-energy hook and optional measurement destination indices. |
| `core/runtime.py` | Forwards energy launches and optional ordered observations; preserves old model.measure(state) when ordering is omitted. |
| `core/simulation.py` | Adds PT diagnostics/settings under metadata.parallel_tempering after the run. |
| `models/xy/model.py` | Launches energy-only and ordered measurement entrypoints. |
| `models/xy/measure.py` | Dedicated total-energy reduction and shared full-measurement helper with ordinary/ordered entrypoints. |
| `models/ising/model.py` | Same launch integration for Ising. |
| `models/ising/measure.py` | Same energy/order integration for Ising, preserving int8 spins and float64 sums. |
| `models/xy/analysis.py` | Repairs stale utils.statistics import to utils.Analysis.statistics; removes one existing whitespace-only line. |
| `models/ising/analysis.py` | Repairs stale utils.binder import to utils.Analysis.binder. |
| `tests/test_parallel_tempering.py` | Ten deterministic GPU regression tests. |
| `tests/test_simulation.py` | Updates the ordinary XY integration check to supply the minimum twenty blocks required by the existing statistics utility. |
| `README.md` | Usage, interfaces, diagnostics scope, assumptions, and ladder limitations; preserves UTF-16 encoding. |
| `PARALLEL_TEMPERING.md` | This implementation/validation report. |

The current Model/Runtime contracts preserve ordinary models without PT; opting into PT requires the GPU energy and ordered-measurement hooks. Dedicated ordinary and ordered CUDA entrypoints share device reduction code because this Numba version cannot marshal a host None kernel argument. The existing ordinary kernel signatures remain available.

Entry points, update kernels, temperature-ladder utility, data IO format, saved simulation data, and unrelated user changes were not modified by this task. No commits or pushes were made.

## Remaining assumptions and concerns

- All temperatures share one temperature-independent Hamiltonian and model parameters. Temperature-dependent He3 coefficients would require generalized reduced-energy differences, not this energy-only swap rule.
- Use the wrapper exclusively after construction; the mappings assume a fixed ladder and unmodified underlying beta assignments. Existing default-stream ordering is required.
- Local updates keep their previous float32 arithmetic. PT acceptance promotes the effective betas and total energies to float64 but does not change that local numerical approximation.
- Diagnostics are cumulative from wrapper construction, including thermalization. An unattempted edge reports rate zero together with an attempt count of zero. Mean round-trip duration is None before completion.
- Nonfinite total energies reject affected swaps; the error is raised at diagnostics collection/end-of-run rather than synchronizing after every exchange.
- Saved diagnostics/results are not GPU checkpoint/resume state.
- The pre-existing main_ising.py ladder is 0.78-1.02, centered at 0.89, despite J=1 Ising Tc approximately 2.269. It needs a model-appropriate hot endpoint for useful PT mixing. That configuration was reported and left unchanged.
- Larger-lattice effective-sample gains depend on ladder overlap, round trips, and observable autocorrelation. Adaptive ladders, reweighting, graphs, and unrelated optimizations were not implemented.

## Reproducibility artifacts

Temporary artifacts: `C:\Users\jinhao\AppData\Local\Temp\gpu-monte-carlo-pt-20261020`.

- `validate_benchmark.py`: reproducible equilibrium, timing, and profiler modes.
- `equilibrium.json`: every equilibrium mean, blocked error, exact target, and comparison.
- `benchmark.json`: every trial, medians, overheads, acceptance rates, and round trips.
- `exchange_only.nsys-rep`, `exchange_only.sqlite`, `profile_summary.json`: transfer and execution evidence.
- `source_hashes_*.json`: source hashes verified unchanged during each benchmark/validation mode.
