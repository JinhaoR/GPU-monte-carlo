"""Two small GPU reductions; allocations reused; no full-field host transfer.

Eight real local fields: kappa, Re P, Im P, five orthonormal Q components.
First pass: tiles of 1024 sites, one (walker,mode,tile) block, 128 threads.
Second pass: tile sums, scatter by the current walker->temperature map.
Read-only kernels, default stream, same ordering as Runtime/PT.
"""
import math
import numpy as np
from numba import cuda,float64
from observables import unpack_real_modes

THREADS=128
TILE=1024


@cuda.jit
def tiled_modes(state,trig,partial):
    walker,mode,tile=cuda.blockIdx.x,cuda.blockIdx.y,cuda.blockIdx.z
    tid=cuda.threadIdx.x;L=state.shape[1]
    sums=cuda.local.array(16,dtype=float64)
    vals=cuda.local.array(8,dtype=float64)
    re=cuda.local.array((3,2),dtype=float64)
    im=cuda.local.array((3,2),dtype=float64)
    shared=cuda.shared.array((16,THREADS),dtype=float64)
    for k in range(16):sums[k]=0.
    for site in range(tile*TILE+tid,min((tile+1)*TILE,L*L),THREADS):
        x,y=site//L,site%L
        for mu in range(3):
            for a in range(2):
                z=state[walker,x,y,mu,a];re[mu,a]=float64(z.real);im[mu,a]=float64(z.imag)
        kappa,pr,pi=0.,0.,0.
        for mu in range(3):
            kappa+=2*(re[mu,0]*im[mu,1]-im[mu,0]*re[mu,1])
            pr+=re[mu,0]**2-im[mu,0]**2-re[mu,1]**2+im[mu,1]**2
            pi+=2*(re[mu,0]*im[mu,0]-re[mu,1]*im[mu,1])
        nx,ny,nz=0.,0.,0.;xy,xz,yz=0.,0.,0.
        for a in range(2):
            nx+=re[0,a]**2+im[0,a]**2;ny+=re[1,a]**2+im[1,a]**2;nz+=re[2,a]**2+im[2,a]**2
            xy+=re[0,a]*re[1,a]+im[0,a]*im[1,a]
            xz+=re[0,a]*re[2,a]+im[0,a]*im[2,a]
            yz+=re[1,a]*re[2,a]+im[1,a]*im[2,a]
        vals[0]=kappa;vals[1]=pr;vals[2]=pi
        vals[3]=(nx-ny)/math.sqrt(2.)
        vals[4]=(nx+ny-2*nz)/math.sqrt(6.)
        vals[5]=math.sqrt(2.)*xy;vals[6]=math.sqrt(2.)*xz;vals[7]=math.sqrt(2.)*yz
        co,si=1.,0.
        if mode==1:co,si=trig[x,0],trig[x,1]
        elif mode==2:co,si=trig[y,0],trig[y,1]
        for k in range(8):sums[2*k]+=vals[k]*co;sums[2*k+1]+=vals[k]*si
    for k in range(16):shared[k,tid]=sums[k]
    cuda.syncthreads()
    stride=THREADS//2
    while stride:
        if tid<stride:
            for k in range(16):shared[k,tid]+=shared[k,tid+stride]
        cuda.syncthreads();stride//=2
    if tid==0:
        for k in range(16):partial[walker,mode,tile,k]=shared[k,0]


@cuda.jit
def finish_modes(partial,walker_to_temperature,output):
    walker,mode=cuda.blockIdx.x,cuda.blockIdx.y;tid=cuda.threadIdx.x
    shared=cuda.shared.array((16,THREADS),dtype=float64)
    for k in range(16):
        acc=0.
        for tile in range(tid,partial.shape[2],THREADS):acc+=partial[walker,mode,tile,k]
        shared[k,tid]=acc
    cuda.syncthreads();stride=THREADS//2
    while stride:
        if tid<stride:
            for k in range(16):shared[k,tid]+=shared[k,tid+stride]
        cuda.syncthreads();stride//=2
    if tid==0:
        dst=walker_to_temperature[walker]
        for k in range(8):
            output[dst,k,mode,0]=shared[2*k,0];output[dst,k,mode,1]=shared[2*k+1,0]


class SpatialMeter:
    def __init__(self,L,replicas):
        self.L,self.R=L,replicas;self.tiles=(L*L+TILE-1)//TILE
        angle=2*np.pi*np.arange(L)/L
        self.trig=cuda.to_device(np.ascontiguousarray(np.column_stack([np.cos(angle),-np.sin(angle)])))
        self.partial=cuda.device_array((replicas,3,self.tiles,16),dtype=np.float64)
        self.output=cuda.device_array((replicas,8,3,2),dtype=np.float64)
        self.identity=cuda.to_device(np.arange(replicas,dtype=np.int32))

    def launch(self,state,walker_to_temperature=None):
        if tuple(state.shape)!=(self.R,self.L,self.L,3,2):raise ValueError('state shape does not match meter')
        mapping=self.identity if walker_to_temperature is None else walker_to_temperature
        tiled_modes[(self.R,3,self.tiles),THREADS](state,self.trig,self.partial)
        finish_modes[(self.R,3),THREADS](self.partial,mapping,self.output)

    def read(self):return unpack_real_modes(self.output.copy_to_host())

    def measure(self,state,walker_to_temperature=None):
        self.launch(state,walker_to_temperature);return self.read()

    @property
    def allocated_bytes(self):return self.trig.nbytes+self.partial.nbytes+self.output.nbytes+self.identity.nbytes
