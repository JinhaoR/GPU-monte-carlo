"""Independent NumPy definitions and ensemble estimators; no CUDA dependency.

State axes: (...,x,y,spin,orbital); amplitudes: (...,field,momentum).
Fourier convention sum_r O(r) exp(-2pi i k.r/L), no FFT normalization.
"""
import numpy as np

FIELDS=('chirality','composite','q1','q2','q3','q4','q5')
MOMENTA=((0,0),(1,0),(-1,0),(0,1),(0,-1))
SECTORS=('chirality','composite','nematic')


def local_fields(state):
    a=np.asarray(state,dtype=np.complex128)
    x,y=a[...,0],a[...,1]
    kappa=2*np.sum((x.conj()*y).imag,axis=-1)
    pair=np.sum(x*x-y*y,axis=-1)
    q=np.einsum('...ma,...na->...mn',a,a.conj()).real
    n=np.trace(q,axis1=-2,axis2=-1)
    q=q-n[...,None,None]*np.eye(3)/3
    # Orthonormal basis under Tr(Q^T Q), including off-diagonal factors.
    t=np.stack([(q[...,0,0]-q[...,1,1])/np.sqrt(2),
        (q[...,0,0]+q[...,1,1]-2*q[...,2,2])/np.sqrt(6),
        np.sqrt(2)*q[...,0,1],np.sqrt(2)*q[...,0,2],np.sqrt(2)*q[...,1,2]],axis=-1)
    return kappa,pair,t,q


def numpy_amplitudes(state):
    a=np.asarray(state);single=a.ndim==4
    if single:a=a[None]
    k,p,t,_=local_fields(a)
    fields=np.concatenate([k[...,None],p[...,None],t],axis=-1)
    ft=np.fft.fftn(fields,axes=(1,2))
    out=np.stack([ft[:,kx,ky,:] for kx,ky in MOMENTA],axis=-1)
    return out[0] if single else out


def unpack_real_modes(raw):
    """GPU reduced eight real fields at 0,+kx,+ky; reconstruct BOTH P signs.

    Conjugacy applies to Re(P) and Im(P) individually, not to complex P.
    Output matches numpy_amplitudes: (...,7,5), complex128.
    """
    z=raw[...,0]+1j*raw[...,1]  # (...,8,3)
    out=np.empty((*z.shape[:-2],7,5),dtype=np.complex128)
    for dst,src in [(0,0),(2,3),(3,4),(4,5),(5,6),(6,7)]:
        out[...,dst,0]=z[...,src,0]
        out[...,dst,1]=z[...,src,1];out[...,dst,2]=z[...,src,1].conj()
        out[...,dst,3]=z[...,src,2];out[...,dst,4]=z[...,src,2].conj()
    out[...,1,0]=z[...,1,0]+1j*z[...,2,0]
    for axis,pos,neg in [(1,1,2),(2,3,4)]:
        out[...,1,pos]=z[...,1,axis]+1j*z[...,2,axis]
        out[...,1,neg]=z[...,1,axis].conj()+1j*z[...,2,axis].conj()
    return out


def powers(amplitudes,L):
    p=np.abs(amplitudes)**2/(L*L)
    return np.stack([p[...,0,:],p[...,1,:],p[...,2:,:].sum(axis=-2)],axis=-2)


def correlation_ratio(s0,sk,L):
    """Ensemble powers in, raw ratio + xi/L + status out; never clamp S0<Sk.

    Status: 0 valid finite; 1 S0<Sk; 2 Sk=0,S0>0 (singular);
    3 nonpositive/zero denominator or nonfinite input (undefined).
    """
    s0,sk=np.broadcast_arrays(np.asarray(s0,float),np.asarray(sk,float))
    ratio=np.full(s0.shape,np.nan);xi=np.full(s0.shape,np.nan)
    status=np.full(s0.shape,3,dtype=np.int8)
    # Power of a theoretically absent Fourier mode can consist entirely of
    # cancellation roundoff. This declared relative precision threshold flags
    # that singular case; it never repairs a negative S0/Sk-1.
    floor=256*np.finfo(float).eps**2*np.maximum(abs(s0),abs(sk))
    good=np.isfinite(s0)&np.isfinite(sk)&(sk>floor)&(s0>=0)
    ratio[good]=s0[good]/sk[good];status[good]=1
    valid=good&(ratio>=1);status[valid]=0
    xi[valid]=np.sqrt(ratio[valid]-1)/(2*L*np.sin(np.pi/L))
    singular=np.isfinite(s0)&np.isfinite(sk)&(s0>0)&(sk>=0)&(sk<=floor)
    ratio[singular]=np.inf;status[singular]=2
    return ratio,xi,status


def ensemble(amplitudes,L,axis=0):
    """Average powers FIRST; then form nonlinear directional/combined ratios."""
    s=powers(amplitudes,L).mean(axis=axis)
    out={'spectra':s}
    for label,sk in [('x',(s[...,1]+s[...,2])/2),('y',(s[...,3]+s[...,4])/2),
                     ('xy',s[...,1:].mean(axis=-1))]:
        r,xi,st=correlation_ratio(s[...,0],sk,L)
        out['ratio_'+label]=r;out['xi_over_L_'+label]=xi;out['status_'+label]=st
    mean=amplitudes.mean(axis=axis)
    disconnected=np.stack([abs(mean[...,0,0])**2,abs(mean[...,1,0])**2,
        np.sum(abs(mean[...,2:,0])**2,axis=-1)],axis=-1)/(L*L)
    out['zero_disconnected_estimate']=disconnected
    out['zero_connected_estimate']=s[...,0]-disconnected
    return out
