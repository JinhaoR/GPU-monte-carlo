
"""XY measurements

For each configuration and replica we return
    energy, mx, my, twist_cos_x, twist_sin_x
"""
import math

from numba import cuda, float64


MEASURE_THREADS = 256
MEASUREMENT_NAMES = (
    "energy", "mx", "my", "twist_cos_x", "twist_sin_x"
)

@cuda.jit(device=True)
def _xy_measure(state, output, J, temperature_indices, cached_energies):
    """One block per replica, one global output per channel and replica."""
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    L = state.shape[1]

    # Each CUDA block gets its own shared array.
    shared = cuda.shared.array((5, MEASURE_THREADS), dtype=float64)

    e = 0.0
    mx = 0.0
    my = 0.0
    cx = 0.0
    qx = 0.0

    # Each thread visits its own strided sequence of lattice sites.
    for k in range(tid, L * L, cuda.blockDim.x):
        i = k // L
        j = k % L

        theta = float64(state[r, i, j])
        theta_x = float64(state[r, (i + 1) % L, j])

        dx = theta - theta_x
        cos_x = math.cos(dx)

        if cached_energies is None:
            theta_y = float64(state[r, i, (j + 1) % L])
            dy = theta - theta_y
            cos_y = math.cos(dy)
            e += -J * (cos_x + cos_y)
        mx += math.cos(theta)
        my += math.sin(theta)
        cx += cos_x
        qx += math.sin(dx)

    shared[0, tid] = e
    shared[1, tid] = mx
    shared[2, tid] = my
    shared[3, tid] = cx
    shared[4, tid] = qx
    cuda.syncthreads()

    # Binary-tree reduction inside this block, for all five channels.
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            for channel in range(5):
                shared[channel, tid] += shared[channel, tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        destination = r if temperature_indices is None else temperature_indices[r]
        if cached_energies is None:
            output[0, destination] = shared[0, 0]
        else:
            output[0, destination] = cached_energies[r]
        for channel in range(1, 5):
            output[channel, destination] = shared[channel, 0]


@cuda.jit
def xy_measure_kernel(state, output, J):
    """Measurements in physical walker order."""
    _xy_measure(state, output, J, None, None)


@cuda.jit
def xy_ordered_measure_kernel(state, output, J, temperature_indices):
    """Measurements scattered into fixed temperature order on the GPU."""
    _xy_measure(state, output, J, temperature_indices, None)


@cuda.jit
def xy_cached_measure_kernel(state, output, J, cached_energies):
    """Measurements with cached energies in physical walker order."""
    _xy_measure(state, output, J, None, cached_energies)


@cuda.jit
def xy_ordered_cached_measure_kernel(state, output, J, temperature_indices, cached_energies):
    """Scatter measurements and cached walker energies into temperature order."""
    _xy_measure(state, output, J, temperature_indices, cached_energies)


@cuda.jit
def xy_energy_kernel(state, output, J):
    """Total Hamiltonian energy only, without transferring data to the CPU."""
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array(MEASURE_THREADS, dtype=float64)

    energy = 0.0
    for k in range(tid, L * L, cuda.blockDim.x):
        i = k // L
        j = k % L
        theta = float64(state[r, i, j])
        theta_x = float64(state[r, (i + 1) % L, j])
        theta_y = float64(state[r, i, (j + 1) % L])
        # Each periodic bond is counted once, as in xy_measure_kernel.
        energy += -J * (
            math.cos(theta - theta_x) + math.cos(theta - theta_y)
        )

    shared[tid] = energy
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        output[r] = shared[0]
