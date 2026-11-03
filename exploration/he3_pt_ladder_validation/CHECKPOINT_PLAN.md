# Minimal checkpoint/continuation plan — not implemented

The existing Runtime, ParallelTempering and Simulation should remain the three main objects. Add serialization and resume state around their current responsibilities; do not introduce another runner or sampling engine.

## State required

| Owner | Save and restore |
|---|---|
| Runtime | Full complex64 lattice field; per-site xoroshiro RNG array; current float32 inverse temperatures; dimensions and model parameters; configuration/energy version counters; valid cached energy buffer and validity state |
| Gaussian updater | Per-site attempt, acceptance and nonfinite counters; rho and proposal mass |
| PT | Both inverse temperature/walker maps; temperature ladder and canonical float32 beta values; exchange RNG array; next matching parity; absolute sweep and exchange-round counters; all swap counters; all five rows of `_trips` |
| PT history | Tracked walker IDs, history stride, all recorded positions and their sweep/round coordinates; used row count within the final history chunk |
| Simulation | Thermalization/production phase; completed absolute and phase-local sweeps; next sampling event; accumulated scalar samples and their absolute sweep numbers |
| Model measurements | `_measurement_count`; spatial and real-space cadences and selected temperatures; accumulated spatial/profile records, including sample indices, walker maps and sweep numbers |

Restore the actual RNG arrays, not merely their initial seeds. Restore matching parity, not just the temperature permutation. The partially completed round trip includes its origin endpoint, first-start sweep and visited-opposite flag; resetting these would corrupt continuation diagnostics even if the equilibrium kernel were unchanged.

The energy buffer is small. Preserving its valid/invalid state avoids accidentally changing which measurement path executes. If reconstructing it instead, demonstrate bitwise agreement with uninterrupted execution on the same hardware/software stack.

## Integration and write point

1. Checkpoint only between completed local sweeps, after any scheduled exchange and measurements for that sweep. Synchronize the existing CUDA stream before copying arrays.
2. Let Runtime and PT expose small state dictionaries containing their arrays/scalars. Keep Simulation responsible for progress and measurements. A single NPZ plus simple metadata is sufficient for the first version.
3. Write a temporary sibling file, finish and close it, validate its required entries, then atomically replace the destination. A half-written file must never replace the last usable checkpoint.
4. On restore, allocate the existing objects, replace their state from the checkpoint, verify the inverse maps and beta assignment, and continue at the **next uncompleted event**. Constructor-created RNG/field values must not replace restored values.

`Simulation.run()` currently counts sampling steps relative to each call. Calling it again on the same PT object is not a complete resume protocol: its scalar metadata would start over. Likewise, `add_spatial_samples()` assumes spatial event indices refer to the scalar sample array of that Simulation. A minimal first implementation should restore and append the existing accumulated histories with consistent absolute event indices. Segment-wise output can be considered later, only if memory requires it.

For a conservative first version, retaining accumulated measurements in the checkpoint is simpler than introducing an output journal. It costs host RAM and I/O; the larger-size report budgets those explicitly. Loading must check model parameters, shapes, dtypes, temperatures and counter consistency before any update. Bitwise continuation across different GPU models, compiler versions or numerical libraries is not promised.

## Exact endpoint traffic without an enormous position history

The current PT bookkeeping already contains sufficient information. For each walker, the number of alternating endpoint passages since its first endpoint visit is

\[
N_{\mathrm{pass}}=2N_{\mathrm{round\ trips}}+\mathrm{visited\_other}.
\]

The visited-opposite flag is the existing `_trips[_VISITED_OTHER]` row. It becomes one on reaching the opposite endpoint and resets on returning to the origin. Saving this small vector with round-trip counts at the analysis boundary and subsequent block boundaries gives exact passage counts by subtraction. This needs no change to the exchange acceptance algorithm and no new GPU counting kernel. It is a proposed small diagnostic export, **not implemented here**.

The validation pilots record every exchange instead. Their initial 50-sweep histories missed fast visits, although all observable arrays were bitwise unchanged. For long runs, combining occasional position histories with exact passage bookkeeping would be much more economical than placing every five-sweep permutation in JSON. Position histories would still be required for flow and path analysis; sparse histories must retain their resolution limitation.

## Bounded validation before use

Use a small L=8, R=8 system and compare a single uninterrupted run with a checkpointed/restored run using exactly the same seeds and initial field. Test interruptions:

- At sweep 37, between exchange events.
- At sweeps 35 and 40, with opposite next-matching parities.
- Immediately before and after scalar and spatial measurement boundaries.
- During thermalization and production, including their boundary.
- Twice in one trajectory, to catch history-index and buffer-append errors.

Require equality of final fields, all RNG arrays, beta assignments, maps, parity, counters, energy-cache state, round-trip bookkeeping, sampled trajectories and every saved observable. Require strictly increasing, unique absolute sample indices with no missing expected events. Separately test truncated checkpoint rejection and failed-write preservation. Confirm deterministic equality on the same GPU/software stack before claiming exact continuation.

No checkpoint code, production changes or Slurm changes were made in this investigation.
