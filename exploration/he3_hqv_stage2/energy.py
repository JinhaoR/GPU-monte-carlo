"""CPU evaluation of the existing full 3x2 GL Hamiltonian; no minimizer.

Open boundaries use the same three-invariant forward stencil on complete
interior cells only: x=0..Nx-2,y=0..Ny-2. All site potentials are included.
This explicit open-cell convention is distinct from the periodic production
Hamiltonian and is used consistently for every trial/reference comparison.
"""
import numpy as np

BETAS=np.array([-.5165123343467712,.9420228004455566,.9875290393829346,
                .8790628910064697,-1.152514934539795])


def energy_density(A,betas=BETAS,K=1.,gd=0.,boundary='periodic'):
    A=np.asarray(A,dtype=np.complex128)
    if A.ndim!=4 or A.shape[-2:]!=(3,2):raise ValueError('Expected (Nx,Ny,3,2)')
    x,y=A[...,0],A[...,1]
    nx=np.sum(abs(x)**2,axis=-1);ny=np.sum(abs(y)**2,axis=-1);n=nx+ny
    xx=np.sum(x*x,axis=-1);yy=np.sum(y*y,axis=-1);xy=np.sum(x*y,axis=-1)
    overlap=np.sum(x.conj()*y,axis=-1)
    invariants=np.stack([abs(xx+yy)**2,n*n,nx*nx+ny*ny+2*(overlap**2).real,
        nx*nx+ny*ny+2*abs(overlap)**2,abs(xx)**2+abs(yy)**2+2*abs(xy)**2],axis=-1)
    # Match the production literal float32(2/3), while accumulating in float64.
    dipole=abs(A[...,0,0]+A[...,1,1])**2+abs(A[...,0,0])**2+abs(A[...,1,1])**2+2*(A[...,0,1]*A[...,1,0].conj()).real-float(np.float32(2/3))*n
    onsite=-2*n+np.einsum('...i,i->...',invariants,np.asarray(betas))
    grad=np.zeros(n.shape)
    if boundary=='periodic':
        dx=np.roll(A,-1,axis=0)-A;dy=np.roll(A,-1,axis=1)-A
        a,b,c,d=dx[...,0],dx[...,1],dy[...,0],dy[...,1]
        grad=K*np.sum(3*(abs(a)**2+abs(d)**2)+abs(b)**2+abs(c)**2+2*(a.conj()*d+b.conj()*c).real,axis=-1)
    elif boundary=='open':
        dx=A[1:,:-1]-A[:-1,:-1];dy=A[:-1,1:]-A[:-1,:-1]
        a,b,c,d=dx[...,0],dx[...,1],dy[...,0],dy[...,1]
        grad[:-1,:-1]=K*np.sum(3*(abs(a)**2+abs(d)**2)+abs(b)**2+abs(c)**2+2*(a.conj()*d+b.conj()*c).real,axis=-1)
    else:raise ValueError('boundary must be periodic or open')
    return dict(onsite=onsite,dipole=gd*dipole,gradient=grad,total=onsite+gd*dipole+grad)


def energy(A,**kwargs):
    return {name:float(a.sum()) for name,a in energy_density(A,**kwargs).items()}
