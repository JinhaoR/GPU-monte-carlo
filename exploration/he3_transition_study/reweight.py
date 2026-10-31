"""Local multiple-histogram estimating equations and nonlinear observables.

Histogram-bin bias is checked separately. Point weights use equal actual sample
counts, not a fictitious independence count. Errors require joint PT blocks.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import root
from prepare import CACHE,NAMES


def load(gd,L,bins=800):
    with np.load(CACHE/f'gd={gd:g}_L={L}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if bins!=len(d['ebins']):
        ratio=len(d['ebins'])//bins
        assert len(d['ebins'])==ratio*bins
        for k in ['H','S','Ssrc','Shold']:
            a=d[k];d[k]=a.reshape(*a.shape[:-1],bins,ratio).sum(axis=-1)
        d['ebins']=d['ebins'].reshape(bins,ratio).mean(axis=1)
        d['edges']=d['edges'][::ratio]
    return d


def solve(H,S,beta,ebins,guess=None):
    """WHAM in a shifted-energy/reference-beta representation, f0=0.

    H: (source,bin) counts; S: (moment,bin) pooled raw moment sums.
    r_i=f_i+log(sum_b h_b exp(-u_ib)/den_b).
    Exact Jacobian delta_ij-sum_b P_i(b) p(source=j|b).
    """
    active=H.sum(axis=1)>0;H=np.asarray(H[active],float);beta=np.asarray(beta)[active]
    occupied=H.sum(axis=0)>0;H=H[:,occupied];S=S[:,occupied];e=ebins[occupied]
    n=H.sum(axis=1);h=H.sum(axis=0);u=(beta-beta[len(beta)//2])[:,None]*e[None,:]
    def equations(x):
        f=np.r_[0.,x];ld=logsumexp(np.log(n)[:,None]+f[:,None]-u,axis=0)
        lg=np.log(h)-ld;lz=logsumexp(lg[None,:]-u,axis=1)
        prob=np.exp(lg[None,:]-u-lz[:,None])
        post=np.exp(np.log(n)[:,None]+f[:,None]-u-ld)
        jac=np.eye(len(n))-prob@post.T
        return (f+lz)[1:],jac[1:,1:]
    x0=np.zeros(len(n)-1) if guess is None else np.asarray(guess)[1:]-np.asarray(guess)[0]
    if len(n)==1:
        x=np.array([]);residual=0.
    else:
        result=root(equations,x0,jac=True,method='hybr',options=dict(xtol=1e-10));x=result.x
        residual=float(np.max(abs(equations(x)[0])))
        if not np.all(np.isfinite(x)) or residual>2e-8:raise RuntimeError(f'WHAM solve failed: {result.message}, residual={residual}')
    f=np.r_[0.,x];ld=logsumexp(np.log(n)[:,None]+f[:,None]-u,axis=0)
    return dict(H=H,S=S,e=e,n=n,h=h,f=f,ld=ld,beta_ref=beta[len(beta)//2],
        active=active,occupied=occupied,residual=residual,source_beta=beta)


def evaluate(fit,T,L,eref=0.,diagnostics=False):
    T=np.atleast_1d(T);u=(1/T-fit['beta_ref'])[:,None]*fit['e'][None,:]
    logw=-u-fit['ld'][None,:];norm=logsumexp(logw+np.log(fit['h'])[None,:],axis=1)
    w=np.exp(logw-norm[:,None])
    a=fit['S']@w.T;m={name:a[k] for k,name in enumerate(NAMES)}
    v=observables(m,T,L,eref)
    if diagnostics:
        v.update(weight_ess=1/np.sum(fit['h'][None,:]*w*w,axis=1),
            max_source_fraction=(fit['H']@w.T).max(axis=0))
    return v


def observables(m,T,L,eref=0.):
    n=L*L
    hx=(m['kx']-(m['ix2']-m['ix']**2)/T)/n
    hy=(m['ky']-(m['iy2']-m['iy']**2)/T)/n
    return dict(energy=(m['dE']+eref)/n,density=m['density'],
        chirality=m['absm'],chirality_raw=m['absc'],phase=np.sqrt(np.maximum(m['p2'],0)),
        chi2=m['m2'],chirality_raw2=m['c2'],phase2=m['p2'],
        binder_chirality=1-m['m4']/(3*m['m2']**2),
        binder_chirality_raw=1-m['c4']/(3*m['c2']**2),
        binder_phase=2-m['p4']/m['p2']**2,helicity=(hx+hy)/2,helicity_x=hx,helicity_y=hy,
        specific_heat=(m['dE2']-m['dE']**2)/(n*T*T),
        chi_response=n*(m['m2']-m['absm']**2)/T,
        chi_raw_response=n*(m['c2']-m['absc']**2)/T,
        phase_response=n*(m['p2']-m['p']**2)/T,
        spin_alignment=m['qzz'])


def direct(d,k,L):
    m={name:np.array([d['Ssrc'][j,k].sum()/d['H'][:,k].sum()]) for j,name in enumerate(NAMES)}
    return {name:float(a[0]) for name,a in observables(m,np.array([1/d['beta'][k]]),L,float(d['eref'])).items()}
