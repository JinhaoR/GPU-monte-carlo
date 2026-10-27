"""Total energy, density, chirality, charge-two phase, and exact twist derivatives.

Local fields use complex64 arithmetic; lattice sums and cached energies use
float64. Cached energy stays in physical walker order until output scattering.
"""

from numba import cuda, complex64, float64

from .update import norm2, potential, gradient


MEASURE_THREADS = 256
MEASUREMENT_NAMES = (
    "energy", "density", "chirality", "phase_real", "phase_imag",
    "current_x", "curvature_x", "current_y", "curvature_y",
)


@cuda.jit(device=True)
def _measure(state, output, betas, K, temperature_indices, cached_energies):
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array((9, MEASURE_THREADS), dtype=float64)
    psi = cuda.local.array((3, 2), dtype=complex64)
    energy, density, chirality, pr, pi = 0.0, 0.0, 0.0, 0.0, 0.0
    ix, kx, iy, ky = 0.0, 0.0, 0.0, 0.0
    for site in range(tid, L * L, cuda.blockDim.x):
        i, j = site // L, site % L
        for mu in range(3):
            psi[mu, 0], psi[mu, 1] = state[r, i, j, mu, 0], state[r, i, j, mu, 1]
        if cached_energies is None:
            energy += potential(psi, betas)
        for mu in range(3):
            x, y = psi[mu, 0], psi[mu, 1]
            xp_x = state[r, (i + 1) % L, j, mu, 0]
            xp_y = state[r, (i + 1) % L, j, mu, 1]
            yp_x = state[r, i, (j + 1) % L, mu, 0]
            yp_y = state[r, i, (j + 1) % L, mu, 1]
            a, b, c, d = xp_x - x, xp_y - y, yp_x - x, yp_y - y
            if cached_energies is None:
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
        # Energy either comes from the reduction or the PT cache.
        if cached_energies is None:
            output[0, destination] = shared[0, 0]
        else:
            output[0, destination] = cached_energies[r]

        # All remaining observables use the regular reduction.
        for channel in range(1, 9):
            output[channel, destination] = shared[channel, 0]


@cuda.jit
def he3_measure_kernel(state, output, betas, K):
    _measure(state, output, betas, K, None, None)


@cuda.jit
def he3_ordered_measure_kernel(state, output, betas, K, temperature_indices):
    _measure(state, output, betas, K, temperature_indices, None)

@cuda.jit
def he3_cached_measure_kernel(state, output, betas, K, cached_energies):
    _measure(state, output, betas, K, None, cached_energies)


@cuda.jit
def he3_ordered_cached_measure_kernel(
    state, output, betas, K, temperature_indices, cached_energies
):
    _measure(
        state, output, betas, K,
        temperature_indices, cached_energies,
    )

@cuda.jit
def he3_energy_kernel(state, output, betas, K):
    """Current total walker energies for PT, without a host transfer."""
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array(MEASURE_THREADS, dtype=float64)
    psi = cuda.local.array((3, 2), dtype=complex64)
    energy = 0.0
    for site in range(tid, L * L, cuda.blockDim.x):
        i, j = site // L, site % L
        for mu in range(3):
            psi[mu, 0], psi[mu, 1] = state[r, i, j, mu, 0], state[r, i, j, mu, 1]
        energy += potential(psi, betas)
        for mu in range(3):
            energy += K * gradient(
                state[r, (i + 1) % L, j, mu, 0] - psi[mu, 0],
                state[r, (i + 1) % L, j, mu, 1] - psi[mu, 1],
                state[r, i, (j + 1) % L, mu, 0] - psi[mu, 0],
                state[r, i, (j + 1) % L, mu, 1] - psi[mu, 1])
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
