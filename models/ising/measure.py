"""GPU sums of Ising energy and magnetization for each replica."""

from numba import cuda, float64


MEASURE_THREADS = 256
MEASUREMENT_NAMES = ("energy", "magnetization")


@cuda.jit(device=True)
def _ising_measure(spins, output, J, temperature_indices):
    """One block per replica, with float64 measurement accumulation."""
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    L = spins.shape[1]
    shared = cuda.shared.array((2, MEASURE_THREADS), dtype=float64)

    e = 0.0
    m = 0.0
    for k in range(tid, L * L, cuda.blockDim.x):
        i = k // L
        j = k % L
        s = spins[r, i, j]
        down = spins[r, (i + 1) % L, j]
        right = spins[r, i, (j + 1) % L]

        # Count each periodic nearest-neighbor bond once.
        e += -float64(J) * s * (down + right)
        m += s

    shared[0, tid] = e
    shared[1, tid] = m
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            for channel in range(2):
                shared[channel, tid] += shared[channel, tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        destination = r if temperature_indices is None else temperature_indices[r]
        for channel in range(2):
            output[channel, destination] = shared[channel, 0]


@cuda.jit
def ising_measure_kernel(spins, output, J):
    """Measurements in physical walker order."""
    _ising_measure(spins, output, J, None)


@cuda.jit
def ising_ordered_measure_kernel(spins, output, J, temperature_indices):
    """Measurements scattered into fixed temperature order on the GPU."""
    _ising_measure(spins, output, J, temperature_indices)


@cuda.jit
def ising_energy_kernel(spins, output, J):
    """Total Hamiltonian energy only, without transferring data to the CPU."""
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    L = spins.shape[1]
    shared = cuda.shared.array(MEASURE_THREADS, dtype=float64)

    energy = 0.0
    for k in range(tid, L * L, cuda.blockDim.x):
        i = k // L
        j = k % L
        s = spins[r, i, j]
        down = spins[r, (i + 1) % L, j]
        right = spins[r, i, (j + 1) % L]
        # Each periodic bond is counted once, as in ising_measure_kernel.
        energy += -float64(J) * s * (down + right)

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
