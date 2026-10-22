"""Total energy, chirality, doubled phase, spin tensor, and phase-twist sums."""

import math

from numba import cuda, float64


MEASURE_THREADS = 256
MEASUREMENT_NAMES = (
    "energy", "chirality", "px", "py", "spin2", "twist_cos_x", "twist_sin_x",
)


@cuda.jit(device=True)
def _measure(state, output, J, temperature_indices):
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array((12, MEASURE_THREADS), dtype=float64)
    e, chirality, px, py, cx, qx = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xx, yy, zz, xy, xz, yz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for k in range(tid, L * L, cuda.blockDim.x):
        i, j = k // L, k % L
        theta, tau = float64(state[r, i, j, 0]), float64(state[r, i, j, 1])
        dx, dy, dz = float64(state[r, i, j, 2]), float64(state[r, i, j, 3]), float64(state[r, i, j, 4])
        chirality += tau
        px += math.cos(2 * theta)
        py += math.sin(2 * theta)
        xx += dx * dx - 1 / 3
        yy += dy * dy - 1 / 3
        zz += dz * dz - 1 / 3
        xy += dx * dy
        xz += dx * dz
        yz += dy * dz
        for direction in range(2):
            ni = (i + 1) % L if direction == 0 else i
            nj = j if direction == 0 else (j + 1) % L
            dot = (dx * float64(state[r, ni, nj, 2]) + dy * float64(state[r, ni, nj, 3])
                   + dz * float64(state[r, ni, nj, 4]))
            weight = (1 + tau * float64(state[r, ni, nj, 1])) * dot
            angle = theta - float64(state[r, ni, nj, 0])
            cosine = weight * math.cos(angle)
            e -= J * cosine
            if direction == 0:
                cx += cosine
                qx += weight * math.sin(angle)

    shared[0, tid], shared[1, tid] = e, chirality
    shared[2, tid], shared[3, tid] = px, py
    shared[4, tid], shared[5, tid], shared[6, tid] = xx, yy, zz
    shared[7, tid], shared[8, tid], shared[9, tid] = xy, xz, yz
    shared[10, tid], shared[11, tid] = cx, qx
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            for channel in range(12):
                shared[channel, tid] += shared[channel, tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        destination = r if temperature_indices is None else temperature_indices[r]
        for channel in range(4):
            output[channel, destination] = shared[channel, 0]
        # D=(dd^T-I/3)/sqrt(3); off-diagonal tensor entries occur twice.
        output[4, destination] = (shared[4, 0]**2 + shared[5, 0]**2 + shared[6, 0]**2
            + 2 * (shared[7, 0]**2 + shared[8, 0]**2 + shared[9, 0]**2)) / 3
        output[5, destination], output[6, destination] = shared[10, 0], shared[11, 0]


@cuda.jit
def kawamura_measure_kernel(state, output, J):
    _measure(state, output, J, None)


@cuda.jit
def kawamura_ordered_measure_kernel(state, output, J, temperature_indices):
    _measure(state, output, J, temperature_indices)


@cuda.jit
def kawamura_energy_kernel(state, output, J):
    r, tid = cuda.blockIdx.x, cuda.threadIdx.x
    L = state.shape[1]
    shared = cuda.shared.array(MEASURE_THREADS, dtype=float64)
    energy = 0.0
    for k in range(tid, L * L, cuda.blockDim.x):
        i, j = k // L, k % L
        theta, tau = float64(state[r, i, j, 0]), float64(state[r, i, j, 1])
        for direction in range(2):
            ni = (i + 1) % L if direction == 0 else i
            nj = j if direction == 0 else (j + 1) % L
            dot = 0.0
            for component in range(2, 5):
                dot += float64(state[r, i, j, component]) * float64(state[r, ni, nj, component])
            energy -= J * (1 + tau * float64(state[r, ni, nj, 1])) * dot * math.cos(
                theta - float64(state[r, ni, nj, 0]))
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
