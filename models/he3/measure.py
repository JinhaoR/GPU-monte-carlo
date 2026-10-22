"""Total energy, density, chirality, charge-two phase, and exact twist derivatives."""

from numba import cuda, complex128, float64

from .update import norm2, potential, gradient


MEASURE_THREADS = 256
MEASUREMENT_NAMES = (
    "energy", "density", "chirality", "phase_real", "phase_imag",
    "current_x", "curvature_x", "current_y", "curvature_y",
)


@cuda.jit(device=True)
def _measure(state, output, betas, K, temperature_indices):
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array((9, MEASURE_THREADS), dtype=float64)
    psi = cuda.local.array((3, 2), dtype=complex128)
    energy, density, chirality, pr, pi = 0.0, 0.0, 0.0, 0.0, 0.0
    ix, kx, iy, ky = 0.0, 0.0, 0.0, 0.0
    for site in range(tid, L * L, cuda.blockDim.x):
        i, j = site // L, site % L
        for mu in range(3):
            psi[mu, 0], psi[mu, 1] = state[r, i, j, mu, 0], state[r, i, j, mu, 1]
        energy += potential(psi, betas)
        for mu in range(3):
            x, y = psi[mu, 0], psi[mu, 1]
            xp_x = complex128(state[r, (i + 1) % L, j, mu, 0])
            xp_y = complex128(state[r, (i + 1) % L, j, mu, 1])
            yp_x = complex128(state[r, i, (j + 1) % L, mu, 0])
            yp_y = complex128(state[r, i, (j + 1) % L, mu, 1])
            a, b, c, d = xp_x - x, xp_y - y, yp_x - x, yp_y - y
            energy += K * gradient(a, b, c, d)
            density += norm2(x) + norm2(y)
            chirality += 2 * (x.conjugate() * y).imag
            phase = x * x - y * y
            pr += phase.real
            pi += phase.imag
            # Differentiate forward neighbors exp(-i*delta_nu) Psi(i+nu).
            zx = (3 * a.conjugate() + d.conjugate()) * xp_x + (b.conjugate() + c.conjugate()) * xp_y
            zy = (c.conjugate() + b.conjugate()) * yp_x + (3 * d.conjugate() + a.conjugate()) * yp_y
            ix += 2 * K * zx.imag
            kx += 2 * K * (3 * norm2(xp_x) + norm2(xp_y) - zx.real)
            iy += 2 * K * zy.imag
            ky += 2 * K * (norm2(yp_x) + 3 * norm2(yp_y) - zy.real)

    shared[0, tid], shared[1, tid], shared[2, tid] = energy, density, chirality
    shared[3, tid], shared[4, tid] = pr, pi
    shared[5, tid], shared[6, tid] = ix, kx
    shared[7, tid], shared[8, tid] = iy, ky
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            for channel in range(9):
                shared[channel, tid] += shared[channel, tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        destination = r if temperature_indices is None else temperature_indices[r]
        for channel in range(9):
            output[channel, destination] = shared[channel, 0]


@cuda.jit
def he3_measure_kernel(state, output, betas, K):
    _measure(state, output, betas, K, None)


@cuda.jit
def he3_ordered_measure_kernel(state, output, betas, K, temperature_indices):
    _measure(state, output, betas, K, temperature_indices)


@cuda.jit
def he3_energy_kernel(state, output, betas, K):
    """Current total walker energies for PT, without a host transfer."""
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array(MEASURE_THREADS, dtype=float64)
    psi = cuda.local.array((3, 2), dtype=complex128)
    energy = 0.0
    for site in range(tid, L * L, cuda.blockDim.x):
        i, j = site // L, site % L
        for mu in range(3):
            psi[mu, 0], psi[mu, 1] = state[r, i, j, mu, 0], state[r, i, j, mu, 1]
        energy += potential(psi, betas)
        for mu in range(3):
            energy += K * gradient(
                complex128(state[r, (i + 1) % L, j, mu, 0]) - psi[mu, 0],
                complex128(state[r, (i + 1) % L, j, mu, 1]) - psi[mu, 1],
                complex128(state[r, i, (j + 1) % L, mu, 0]) - psi[mu, 0],
                complex128(state[r, i, (j + 1) % L, mu, 1]) - psi[mu, 1])
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
