"""Read-only tiled measurements, extending the validated spatial pilot.

Float64 local observable arithmetic and reductions; Hamiltonian contributions
call the unchanged production device functions with their float32 inputs.
No RNG, energy-cache write, projection or sampler operation occurs here.
"""
import math
import numpy as np
from numba import cuda, float32, float64, complex64
from .update import potential, gradient, dipole_invariant, norm2
from .spatial import unpack_real_modes, DENSITY_FLOOR

THREADS = 64
TILE = 1024
REAL_FIELDS = 26
MODE_SUMS = 52
STRUCT_CHANNELS = 68


@cuda.jit(device=True)
def _local_values(state, walker, x, y, vals):
    re = cuda.local.array((3, 2), float64)
    im = cuda.local.array((3, 2), float64)
    for mu in range(3):
        for a in range(2):
            z = state[walker, x, y, mu, a]
            re[mu, a] = float64(z.real); im[mu, a] = float64(z.imag)
            vals[14+4*mu+2*a] = re[mu, a]
            vals[15+4*mu+2*a] = im[mu, a]
    kappa = 0.
    for k in range(8, 14):
        vals[k] = 0.
    for mu in range(3):
        u, v, s, t = re[mu, 0], im[mu, 0], re[mu, 1], im[mu, 1]
        kappa += 2*(u*t-v*s)
        vals[8] += u*u-v*v; vals[9] += 2*u*v
        vals[10] += s*s-t*t; vals[11] += 2*s*t
        vals[12] += u*s-v*t; vals[13] += u*t+v*s
    vals[0] = kappa; vals[1] = vals[8]-vals[10]; vals[2] = vals[9]-vals[11]
    nx, ny, nz, xy, xz, yz = 0., 0., 0., 0., 0., 0.
    for a in range(2):
        nx += re[0,a]**2+im[0,a]**2; ny += re[1,a]**2+im[1,a]**2
        nz += re[2,a]**2+im[2,a]**2
        xy += re[0,a]*re[1,a]+im[0,a]*im[1,a]
        xz += re[0,a]*re[2,a]+im[0,a]*im[2,a]
        yz += re[1,a]*re[2,a]+im[1,a]*im[2,a]
    vals[3] = (nx-ny)/math.sqrt(2.); vals[4] = (nx+ny-2*nz)/math.sqrt(6.)
    vals[5] = math.sqrt(2.)*xy; vals[6] = math.sqrt(2.)*xz; vals[7] = math.sqrt(2.)*yz


@cuda.jit
def tiled_modes(state, trig, partial):
    walker, mode, tile = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tid = cuda.threadIdx.x; L = state.shape[1]
    vals = cuda.local.array(REAL_FIELDS, float64)
    sums = cuda.local.array(MODE_SUMS, float64)
    shared = cuda.shared.array((MODE_SUMS, THREADS), float64)
    for k in range(2*REAL_FIELDS): sums[k] = 0.
    for site in range(tile*TILE+tid, min((tile+1)*TILE, L*L), THREADS):
        x, y = site//L, site%L
        _local_values(state, walker, x, y, vals)
        co, si = 1., 0.
        if mode == 1: co, si = trig[x,0], trig[x,1]
        elif mode == 2: co, si = trig[y,0], trig[y,1]
        for k in range(REAL_FIELDS):
            sums[2*k] += vals[k]*co; sums[2*k+1] += vals[k]*si
    for k in range(2*REAL_FIELDS): shared[k,tid] = sums[k]
    cuda.syncthreads()
    stride = THREADS//2
    while stride:
        if tid < stride:
            for k in range(2*REAL_FIELDS): shared[k,tid] += shared[k,tid+stride]
        cuda.syncthreads(); stride //= 2
    if tid == 0:
        for k in range(2*REAL_FIELDS): partial[walker,mode,tile,k] = shared[k,0]


@cuda.jit
def finish_modes(partial, mapping, output):
    walker, mode = cuda.blockIdx.x, cuda.blockIdx.y; tid = cuda.threadIdx.x
    shared = cuda.shared.array((MODE_SUMS, THREADS), float64)
    for k in range(2*REAL_FIELDS):
        acc = 0.
        for tile in range(tid, partial.shape[2], THREADS): acc += partial[walker,mode,tile,k]
        shared[k,tid] = acc
    cuda.syncthreads(); stride = THREADS//2
    while stride:
        if tid < stride:
            for k in range(2*REAL_FIELDS): shared[k,tid] += shared[k,tid+stride]
        cuda.syncthreads(); stride //= 2
    if tid == 0:
        for k in range(REAL_FIELDS):
            output[mapping[walker],k,mode,0] = shared[2*k,0]
            output[mapping[walker],k,mode,1] = shared[2*k+1,0]


@cuda.jit(device=True)
def _hist_bin(x):
    if x < 0: return 0
    if x > 1: return 11
    if x == 1: return 10
    # Same stored float64 edges as numpy.linspace(0,1,11), including tails.
    b = 1
    for k in range(1, 10):
        if x >= k*0.1: b = k+1
    return b


@cuda.jit
def tiled_structure(state, coefficients, K, gd, floor, partial):
    walker, tile = cuda.blockIdx.x, cuda.blockIdx.y; tid = cuda.threadIdx.x
    L = state.shape[1]
    sums = cuda.local.array(STRUCT_CHANNELS, float64)
    vals = cuda.local.array(REAL_FIELDS, float64)
    psi = cuda.local.array((3,2), complex64)
    shared = cuda.shared.array((STRUCT_CHANNELS,THREADS), float64)
    for k in range(STRUCT_CHANNELS): sums[k] = 0.
    for site in range(tile*TILE+tid, min((tile+1)*TILE,L*L), THREADS):
        x, y = site//L, site%L
        _local_values(state, walker, x, y, vals)
        n = 0.
        for k in range(14,26): n += vals[k]*vals[k]
        if not math.isfinite(n):
            sums[5] += 1
            continue
        sums[0] += n
        # Existing device Hamiltonian, including its float32 expression order.
        for mu in range(3):
            for a in range(2): psi[mu,a] = state[walker,x,y,mu,a]
        nx, ny = norm2(psi[0,0]), norm2(psi[0,1])
        for mu in range(1,3): nx += norm2(psi[mu,0]); ny += norm2(psi[mu,1])
        dip = dipole_invariant(psi,nx+ny)
        bulk = potential(psi,coefficients,float32(0))
        grad = 0.
        for mu in range(3):
            grad += K*gradient(state[walker,(x+1)%L,y,mu,0]-psi[mu,0],
                               state[walker,(x+1)%L,y,mu,1]-psi[mu,1],
                               state[walker,x,(y+1)%L,mu,0]-psi[mu,0],
                               state[walker,x,(y+1)%L,mu,1]-psi[mu,1])
        sums[13] += bulk; sums[14] += grad; sums[15] += dip
        sums[16] += gd*dip; sums[17] += potential(psi,coefficients,gd)+grad
        if n <= floor:
            sums[2] += n; sums[4] += 1
            continue
        sums[1] += n; sums[3] += 1; sums[19] += abs(vals[0])
        best = -1.; best_n = 0.; best_s = 0.
        for branch in range(2):
            sign = 1 if branch == 0 else -1
            ns, sr, si = 0., 0., 0.
            for mu in range(3):
                u = (vals[14+4*mu]+sign*vals[17+4*mu])/math.sqrt(2.)
                v = (vals[15+4*mu]-sign*vals[16+4*mu])/math.sqrt(2.)
                ns += u*u+v*v; sr += u*u-v*v; si += 2*u*v
            ss = math.hypot(sr,si)
            sums[9+branch] += ns; sums[11+branch] += ss
            if ns+ss > best: best = ns+ss; best_n = ns; best_s = ss
        eps = 1-best/(2*n); orbital = 1-best_n/n; spin = (best_n-best_s)/(2*n)
        sums[6] += n*eps; sums[7] += n*orbital; sums[8] += n*spin
        if eps < -1e-12 or eps > .75+1e-12: sums[18] += 1
        sums[20+_hist_bin(eps)] += 1
        sums[32+_hist_bin(eps)] += n
        sums[44+_hist_bin(orbital)] += 1
        sums[56+_hist_bin(spin)] += 1
    for k in range(STRUCT_CHANNELS): shared[k,tid] = sums[k]
    cuda.syncthreads(); stride = THREADS//2
    while stride:
        if tid < stride:
            for k in range(STRUCT_CHANNELS): shared[k,tid] += shared[k,tid+stride]
        cuda.syncthreads(); stride //= 2
    if tid == 0:
        for k in range(STRUCT_CHANNELS): partial[walker,tile,k] = shared[k,0]


@cuda.jit
def finish_structure(partial,mapping,output):
    walker = cuda.blockIdx.x; k = cuda.threadIdx.x
    if k < STRUCT_CHANNELS:
        acc = 0.
        for tile in range(partial.shape[1]): acc += partial[walker,tile,k]
        output[mapping[walker],k] = acc


class SpatialMeter:
    def __init__(self,L,replicas,density_floor=DENSITY_FLOOR):
        self.L,self.R = L,replicas; self.tiles = (L*L+TILE-1)//TILE
        self.density_floor = float(density_floor)
        angle = 2*np.pi*np.arange(L)/L
        self.trig = cuda.to_device(np.column_stack([np.cos(angle),-np.sin(angle)]))
        self.partial = cuda.device_array((replicas,3,self.tiles,52),np.float64)
        self.output = cuda.device_array((replicas,26,3,2),np.float64)
        self.structure_partial = cuda.device_array((replicas,self.tiles,68),np.float64)
        self.structure_output = cuda.device_array((replicas,68),np.float64)
        self.identity = cuda.to_device(np.arange(replicas,dtype=np.int32))

    def launch(self,state,mapping=None):
        if tuple(state.shape) != (self.R,self.L,self.L,3,2): raise ValueError('meter shape mismatch')
        mapping = self.identity if mapping is None else mapping
        tiled_modes[(self.R,3,self.tiles),THREADS](state,self.trig,self.partial)
        finish_modes[(self.R,3),THREADS](self.partial,mapping,self.output)

    def read(self): return unpack_real_modes(self.output.copy_to_host())

    def measure(self,state,mapping=None):
        self.launch(state,mapping)
        return self.read()

    def structure(self,state,model,mapping=None):
        mapping = self.identity if mapping is None else mapping
        tiled_structure[(self.R,self.tiles),THREADS](state,model.betas,model.K,model.g_d,
                                                  self.density_floor,self.structure_partial)
        finish_structure[self.R,128](self.structure_partial,mapping,self.structure_output)
        result = self.structure_output.copy_to_host()
        if np.any(result[:,5]) or np.any(result[:,18]) or not np.all(np.isfinite(result)):
            raise FloatingPointError('invalid field/manifold diagnostic; no silent repair')
        return result

    @property
    def allocated_bytes(self):
        return sum(x.nbytes for x in (self.trig,self.partial,self.output,self.structure_partial,
                    self.structure_output,self.identity))
