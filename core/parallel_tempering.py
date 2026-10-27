"""GPU replica exchange for a shared, temperature-independent Hamiltonian.

Physical configurations and site RNG stay in their original walker slots.
Only beta assignments and the two inverse temperature/walker maps change.
All kernels use Runtime's existing default CUDA stream, so the local sweep,
fresh total-energy reduction, exchanges, and observations execute in order.
"""

import math
import operator

import numpy as np
from numba import cuda, float64
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float64,
)


# Rows of the GPU round-trip bookkeeping array, indexed by physical walker.
_ORIGIN = 0
_VISITED_OTHER = 1
_START_SWEEP = 2
_ROUND_TRIPS = 3
_ROUND_TRIP_SWEEPS = 4


@cuda.jit
def _record_walkers_kernel(walker_to_temperature, walker_ids, history, row):
    column = cuda.grid(1)
    if column < walker_ids.size:
        history[row, column] = walker_to_temperature[walker_ids[column]]


@cuda.jit
def _exchange_kernel(
    energies, betas, walker_to_temperature, temperature_to_walker,
    rng_states, swap_counts, trips, parity, sweep,
):
    """One block: disjoint pairs, then a barrier before walker diagnostics."""
    tid = cuda.threadIdx.x
    R = energies.size

    # Striding within ONE block also supports ladders larger than a block.
    for left in range(parity + 2 * tid, R - 1, 2 * cuda.blockDim.x):
        right = left + 1
        a = temperature_to_walker[left]
        b = temperature_to_walker[right]
        ea = energies[a]
        eb = energies[b]
        swap_counts[0, left] += 1

        if not math.isfinite(ea) or not math.isfinite(eb):
            swap_counts[2, left] += 1
            continue

        # New/old joint Boltzmann weight = exp((beta_a-beta_b)*(Ea-Eb)).
        # Use the same float32 betas as local updates, promoted before subtracting.
        log_ratio = (float64(betas[a]) - float64(betas[b])) * (ea - eb)
        u = xoroshiro128p_uniform_float64(rng_states, left)
        if log_ratio >= 0.0 or u == 0.0 or math.log(u) < log_ratio:
            temperature_to_walker[left] = b
            temperature_to_walker[right] = a
            walker_to_temperature[a] = right
            walker_to_temperature[b] = left
            beta_a = betas[a]
            betas[a] = betas[b]
            betas[b] = beta_a
            swap_counts[1, left] += 1

    # All mapping writes must finish before any walker checks its endpoint.
    cuda.syncthreads()
    for walker in range(tid, R, cuda.blockDim.x):
        temperature = walker_to_temperature[walker]
        if temperature != 0 and temperature != R - 1:
            continue

        origin = trips[_ORIGIN, walker]
        if origin < 0:
            # Interior walkers start a trip only on their first endpoint visit.
            trips[_ORIGIN, walker] = temperature
            trips[_START_SWEEP, walker] = sweep
        elif temperature != origin:
            trips[_VISITED_OTHER, walker] = 1
        elif trips[_VISITED_OTHER, walker]:
            trips[_ROUND_TRIPS, walker] += 1
            trips[_ROUND_TRIP_SWEEPS, walker] += (
                sweep - trips[_START_SWEEP, walker]
            )
            trips[_START_SWEEP, walker] = sweep
            trips[_VISITED_OTHER, walker] = 0


class ParallelTempering:
    """Wrap a Runtime and pass this object directly to Simulation.

    Temperatures must be strictly increasing. A sweep performs all four local
    colors before any exchange. Matchings alternate (0,1),(2,3),... and
    (1,2),(3,4),...; two walkers attempt their sole pair every exchange interval.
    Each matching satisfies detailed balance. Alternating matchings preserves
    the joint canonical distribution, without making the full schedule reversible.

    Use the wrapper exclusively after construction. Diagnostics are cumulative
    from construction, including thermalization and any repeated Simulation runs.
    The wrapped model must provide GPU total energies and ordered measurements.

    tracked_walkers enables optional GPU trajectory recording for those walker
    IDs. history_stride counts exchange rounds, not local sweeps. diagnostics()
    exports sampled positions, their sweep/round coordinates, and the current
    final position. Exact round-trip counters are independent of this stride.
    """

    def __init__(self, runtime, exchange_interval=10, seed=None,
                 tracked_walkers=None, history_stride=10):

        if isinstance(exchange_interval, (bool, np.bool_)):
            raise ValueError("exchange_interval must be a positive integer")
        try:
            interval = operator.index(exchange_interval)
        except TypeError:
            raise ValueError("exchange_interval must be a positive integer") from None
        if interval < 1:
            raise ValueError("exchange_interval must be a positive integer")
        if runtime.R < 2:
            raise ValueError("Parallel tempering requires at least two temperatures")
        if isinstance(history_stride, (bool, np.bool_)):
            raise ValueError("history_stride must be a positive integer")
        try:
            history_stride = operator.index(history_stride)
        except TypeError:
            raise ValueError("history_stride must be a positive integer") from None
        if history_stride < 1:
            raise ValueError("history_stride must be a positive integer")
        walker_ids = None
        if tracked_walkers is not None:
            try:
                values = list(tracked_walkers)
                if any(isinstance(value, (bool, np.bool_)) for value in values):
                    raise ValueError("tracked_walkers must contain integer walker IDs")
                walker_ids = np.asarray([operator.index(value) for value in values],
                                        dtype=np.int64)
            except (TypeError, OverflowError):
                raise ValueError("tracked_walkers must contain integer walker IDs") from None
            if (walker_ids.ndim != 1 or not len(walker_ids)
                    or np.any(walker_ids < 0) or np.any(walker_ids >= runtime.R)
                    or len(np.unique(walker_ids)) != len(walker_ids)):
                raise ValueError("tracked_walkers must be distinct IDs in range(R)")
            walker_ids = walker_ids.astype(np.int32)

        # One initialization read validates the assignments of the wrapped Runtime.
        beta = runtime.betas.copy_to_host()
        expected = (1.0 / runtime.temperatures).astype(np.float32)
        if not np.array_equal(beta, expected):
            raise ValueError("Runtime betas must initially match its temperature order")
        if not np.all(np.isfinite(beta) & (beta > 0)) or not np.all(np.diff(beta) < 0):
            raise ValueError("PT temperatures must have distinct, finite positive float32 betas")

        self.runtime = runtime
        self.model = runtime.model
        self.L = runtime.L
        self.R = runtime.R
        self.temperatures = runtime.temperatures
        self.seed = runtime.seed
        self.exchange_interval = interval
        self.exchange_seed = operator.index(runtime.seed + 1 if seed is None else seed)
        self.sweeps = 0
        self.exchange_rounds = 0
        self._parity = 0
        self._temperature_betas = beta.copy()

        self.energies = cuda.device_array(self.R, dtype=np.float64)
        # Fail at construction if the model lacks the optional GPU energy path.
        runtime.measure_energy(self.energies)
        identity = np.arange(self.R, dtype=np.int32)
        self.walker_to_temperature = cuda.to_device(identity)
        self.temperature_to_walker = cuda.to_device(identity)
        self._rng_states = create_xoroshiro128p_states(
            self.R - 1, seed=self.exchange_seed
        )
        # Per neighboring edge: attempted, accepted, nonfinite-energy attempts.
        self._swap_counts = cuda.to_device(np.zeros((3, self.R - 1), dtype=np.int64))
        trips = np.zeros((5, self.R), dtype=np.int64)
        trips[_ORIGIN] = -1
        trips[_START_SWEEP] = -1
        trips[_ORIGIN, 0] = 0
        trips[_ORIGIN, -1] = self.R - 1
        trips[_START_SWEEP, [0, -1]] = 0
        self._trips = cuda.to_device(trips)

        # Optional device-side trajectory capture. The stride counts exchange
        # rounds; the exact sweep/round coordinates accompany each saved row.
        # Chunks grow as needed, so repeated Simulation runs cannot overflow a
        # fixed recording buffer. No history is copied to the CPU in sweep().
        self._tracked_walkers = walker_ids
        self._history_stride = history_stride
        self._history_chunk_rows = 1024
        self._history_chunks = []
        self._history_sweeps = []
        self._history_rounds = []
        self._tracked_walkers_gpu = (
            None if walker_ids is None else cuda.to_device(walker_ids)
        )
        if walker_ids is not None:
            self._record_walker_history()

    def _record_walker_history(self):
        row = len(self._history_sweeps) % self._history_chunk_rows
        if row == 0:
            self._history_chunks.append(cuda.device_array(
                (self._history_chunk_rows, len(self._tracked_walkers)), dtype=np.int32,
            ))
        blocks = (len(self._tracked_walkers) + 127) // 128
        _record_walkers_kernel[blocks, 128](
            self.walker_to_temperature, self._tracked_walkers_gpu,
            self._history_chunks[-1], row,
        )
        self._history_sweeps.append(self.sweeps)
        self._history_rounds.append(self.exchange_rounds)

    def _walker_history(self, final_walker_to_temperature):
        count = len(self._history_sweeps)
        rows = []
        for chunk in self._history_chunks:
            used = min(self._history_chunk_rows, count)
            rows.append(chunk[:used].copy_to_host())
            count -= used
        positions = np.concatenate(rows, axis=0)
        sweeps = np.asarray(self._history_sweeps, dtype=np.int64)
        rounds = np.asarray(self._history_rounds, dtype=np.int64)
        if rounds[-1] != self.exchange_rounds or sweeps[-1] != self.sweeps:
            # Include the current position without changing recording state.
            positions = np.concatenate((positions,
                final_walker_to_temperature[self._tracked_walkers][None, :]), axis=0)
            sweeps = np.append(sweeps, self.sweeps)
            rounds = np.append(rounds, self.exchange_rounds)
        return {
            "walker_ids": self._tracked_walkers.copy(),
            "sweeps": sweeps,
            "exchange_rounds": rounds,
            "temperature_indices": positions,
            "stride": self._history_stride,
            "stride_units": "exchange_rounds",
        }

    def sweep(self):
        self.runtime.sweep()
        self.sweeps += 1
        if self.sweeps % self.exchange_interval == 0:
            self.attempt_exchanges()

    def attempt_exchanges(self):
        """Measure current TOTAL energies, then attempt the next GPU matching.

        This method does not advance local sweeps. It is also useful for explicit
        exchange schedules; normal Simulation use only needs sweep().
        """
        self.runtime.measure_energy(self.energies)
        _exchange_kernel[1, 128](
            self.energies, self.runtime.betas,
            self.walker_to_temperature, self.temperature_to_walker,
            self._rng_states, self._swap_counts, self._trips,
            self._parity, self.sweeps,
        )
        self.exchange_rounds += 1
        if self.R > 2:
            self._parity = 1 - self._parity
        if (self._tracked_walkers is not None
                and self.exchange_rounds % self._history_stride == 0):
            self._record_walker_history()

    def measure(self):
        """Return CPU observables in the original, fixed temperature order."""
        return self.runtime.measure(temperature_indices=self.walker_to_temperature)

    def snapshot(self):
        """Explicit CPU snapshot in temperature order; Runtime stays walker ordered."""
        state = self.runtime.snapshot()
        return state[self.temperature_to_walker.copy_to_host()]

    def diagnostics(self):
        """Copy small diagnostic arrays to CPU; no transfers occur during exchanges."""
        counts = self._swap_counts.copy_to_host()
        if np.any(counts[2]):
            raise FloatingPointError("Nonfinite energies encountered during replica exchange")
        trips = self._trips.copy_to_host()
        w_to_t = self.walker_to_temperature.copy_to_host()
        t_to_w = self.temperature_to_walker.copy_to_host()

        attempts, accepted = counts[:2]
        rates = np.divide(
            accepted, attempts, out=np.zeros(self.R - 1), where=attempts > 0
        )
        completed = int(trips[_ROUND_TRIPS].sum())
        mean_trip = (
            float(trips[_ROUND_TRIP_SWEEPS].sum()) / completed
            if completed else None
        )
        result = {
            "exchange_interval": self.exchange_interval,
            "exchange_seed": self.exchange_seed,
            "sweeps": self.sweeps,
            "exchange_rounds": self.exchange_rounds,
            "next_parity": self._parity,
            "diagnostics_scope": "since_wrapper_creation_including_thermalization",
            "temperature_betas": self._temperature_betas.copy(),
            "attempts": attempts.copy(),
            "accepted": accepted.copy(),
            "acceptance_rates": rates,
            "walker_to_temperature": w_to_t,
            "temperature_to_walker": t_to_w,
            "walker_round_trips": trips[_ROUND_TRIPS].copy(),
            "walker_round_trip_total_sweeps": trips[_ROUND_TRIP_SWEEPS].copy(),
            "mean_round_trip_sweeps": mean_trip,
        }
        if self._tracked_walkers is not None:
            result["walker_history"] = self._walker_history(w_to_t)
        return result
