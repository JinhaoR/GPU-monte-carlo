"""Exact full-field constrained classical energy relaxation; no Monte Carlo.

The energy density is reused from Stage 2. G=partial H/partial A* obeys
dH=2 Re sum(conj(G) dA); the packed real gradient is 2(ReG,ImG).
Only the outer frame and two explicit square annuli are removed from variables.
No bulk projection, connecting strip, plane restriction or topology penalty.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'exploration/he3_hqv_stage2'))
sys.path.insert(0,str(ROOT/'exploration/he3_hqv_stage1'))
from energy import energy_density,BETAS
from pair_baseline import trial
from defects import local_geometry,analyze_loop,rectangle_loop,find_candidates

BETA_A=float(BETAS[1]+BETAS[3]+BETAS[4])


def value_gradient(A,gd=0.,K=1.,betas=BETAS):
    """Periodic production stencil and its analytic Wirtinger derivative."""
    n0=(1+gd*float(np.float32(2/3))/2)/float(betas[1]+betas[3]+betas[4])
    e0=-(2+gd*float(np.float32(2/3)))**2/(4*float(betas[1]+betas[3]+betas[4]))
    E=float(np.sum(energy_density(A,betas=betas,K=K,gd=gd)['total']-e0))
    x,y=A[...,0],A[...,1]
    nx=np.sum(abs(x)**2,axis=-1);ny=np.sum(abs(y)**2,axis=-1);n=nx+ny
    xx=np.sum(x*x,axis=-1);yy=np.sum(y*y,axis=-1);xy=np.sum(x*y,axis=-1)
    s=xx+yy;O=np.sum(x.conj()*y,axis=-1)
    b1,b2,b3,b4,b5=betas
    gx=(-2+2*b2*n)[...,None]*x+2*b1*s[...,None]*x.conj()
    gy=(-2+2*b2*n)[...,None]*y+2*b1*s[...,None]*y.conj()
    gx+=2*b3*(nx[...,None]*x+O[...,None]*y)+2*b4*(nx[...,None]*x+O.conj()[...,None]*y)
    gy+=2*b3*(ny[...,None]*y+O.conj()[...,None]*x)+2*b4*(ny[...,None]*y+O[...,None]*x)
    gx+=2*b5*(xx[...,None]*x.conj()+xy[...,None]*y.conj())
    gy+=2*b5*(yy[...,None]*y.conj()+xy[...,None]*x.conj())
    dx=np.roll(A,-1,axis=0)-A;dy=np.roll(A,-1,axis=1)-A
    fx=3*dx[...,0]+dy[...,1];fy=dy[...,0]+dx[...,1]
    gx+=K*(np.roll(fx,1,axis=0)-fx+np.roll(fy,1,axis=1)-fy)
    fx=dx[...,1]+dy[...,0];fy=3*dy[...,1]+dx[...,0]
    gy+=K*(np.roll(fx,1,axis=0)-fx+np.roll(fy,1,axis=1)-fy)
    G=np.stack((gx,gy),axis=-1)
    if gd:
        G-=gd*float(np.float32(2/3))*A
        a,b,c,d=A[...,0,0],A[...,0,1],A[...,1,0],A[...,1,1]
        G[...,0,0]+=gd*(2*a+d);G[...,1,1]+=gd*(a+2*d)
        G[...,0,1]+=gd*c;G[...,1,0]+=gd*b
    return E,G


def problem(L,R,pin=2,gd=0.,initial='noisy',seed=7381,right_rotation=0.):
    """R-independent isolated HQV maps on square annuli (outer radius pin+.5).

    Center positions are half-integer, R even. A square annulus consists of
    perimeter sites at Chebyshev radius pin+.5; the core interior remains free.
    Its amplitude is the homogeneous chiral n0 at this coupling. The left
    isolated template has theta=alpha=(phi-pi)/2, the right=-phi/2.
    Neither annulus includes the other vortex's R-dependent angular field.
    """
    if L%2 or R%2 or R<4*pin+4 or R>=L-4*pin-8:
        raise ValueError('Use even, separated cores clear of the outer frame')
    A,centers,_=trial(L,R,3.,'compact_periodic')
    n0=(1+gd*float(np.float32(2/3))/2)/BETA_A
    A*=np.sqrt(n0/(1/BETA_A))
    vacuum=np.zeros((3,2),complex);vacuum[2]=np.sqrt(n0/2)*np.array([1,1j])
    x,y=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    frame=(x<2)|(y<2)|(x>=L-2)|(y>=L-2)
    pinned=frame.copy();A[frame]=vacuum
    rings=[]
    for index,((cx,cy),q,offset) in enumerate(zip(centers,[.5,-.5],[-np.pi/2,0.])):
        r=np.maximum(abs(x-cx),abs(y-cy))
        ring=np.isclose(r,pin+.5)
        phi=np.arctan2(y-cy,x-cx)
        alpha=q*phi+offset;theta=q*phi+offset
        d=np.stack((np.sin(alpha),np.zeros_like(alpha),np.cos(alpha)),axis=-1)
        if index==1 and right_rotation:
            dx,dy=d[...,0].copy(),d[...,1].copy()
            d[...,0]=np.cos(right_rotation)*dx-np.sin(right_rotation)*dy
            d[...,1]=np.sin(right_rotation)*dx+np.cos(right_rotation)*dy
        target=np.sqrt(n0)*np.exp(1j*theta)[...,None,None]*d[..., :,None]*np.array([1,1j])/np.sqrt(2)
        A[ring]=target[ring];pinned|=ring;rings.append(ring)
    fixed=A.copy()
    if initial=='noisy':
        rng=np.random.default_rng(seed)
        A[~pinned]+=.015*np.sqrt(n0/12)*(rng.normal(size=A[~pinned].shape)+1j*rng.normal(size=A[~pinned].shape))
    elif initial=='nonplanar':
        # Distinct smooth spin path plus complex all-component seed; pins fixed.
        angle=.8*np.sin(np.pi*x/(L-1))*np.sin(2*np.pi*y/(L-1))
        ax,ay=A[...,0,:].copy(),A[...,1,:].copy()
        A[...,0,:]=np.cos(angle)[...,None]*ax-np.sin(angle)[...,None]*ay
        A[...,1,:]=np.sin(angle)[...,None]*ax+np.cos(angle)[...,None]*ay
        rng=np.random.default_rng(seed)
        A[~pinned]+=.01*(rng.normal(size=A[~pinned].shape)+1j*rng.normal(size=A[~pinned].shape))
    elif initial!='planar':raise ValueError(initial)
    A[pinned]=fixed[pinned]
    return A,pinned,dict(L=L,R=R,pin_radius_parameter=pin,pin_chebyshev_radius=pin+.5,
        gd=gd,initial=initial,seed=seed,right_pin_spin_rotation=right_rotation,
        centers=centers,outer_frame_width=2,pinned_sites=int(pinned.sum()),free_real_variables=int((~pinned).sum()*12),
        constraint='exact fixed outer frame and isolated fixed annular maps; all core interiors free')


def topology(A,meta):
    g=local_geometry(A);L=meta['L'];R=meta['R'];p=meta['pin_radius_parameter'];rows=[]
    for core,(center,q) in enumerate(zip(meta['centers'],[.5,-.5])):
        cx,cy=center
        for r in sorted(set([p,p+1,p+3,min(p+5,int(R/2)-2)])):
            if r>=R/2-1:continue
            c=rectangle_loop(int(cx-r-.5),int(cx+r+.5),int(cy-r-.5),int(cy+r+.5))
            result=analyze_loop(A,c,geometry=g,periodic=True)
            rows.append(dict(core=core,radius_parameter=r,expected_charge=q,**result))
    x1=meta['centers'][0][0];x2=meta['centers'][1][0]
    outer=rectangle_loop(max(2,int(x1-9.5)),min(L-3,int(x2+9.5)),max(2,int(L/2-12)),min(L-3,int(L/2+11)))
    rows.append(dict(core='outer',radius_parameter=-1,expected_charge=0,**analyze_loop(A,outer,geometry=g,periodic=True)))
    return rows


def optimize(A,pinned,meta,*,seconds=60.,maxiter=2500,gtol=3e-6,checkpoint=100):
    fixed=A.copy();free=~pinned;nfree=int(free.sum())
    def unpack(v):
        B=fixed.copy();B[free]=v.view(np.complex128).reshape(nfree,3,2);return B
    x0=np.ascontiguousarray(A[free]).view(float).ravel().copy()
    start=perf_counter();history=[];top_hist=[];cache={};last_accepted=x0.copy()
    def fun(v):
        B=unpack(v);e,G=value_gradient(B,gd=meta['gd'])
        grad=np.ascontiguousarray(2*G[free]).view(float).ravel()
        cache.update(x=v.copy(),value=e,grad=grad)
        return e,grad
    e0,g0=fun(x0)
    history.append(dict(iteration=0,energy=e0,gradient_inf=float(max(abs(g0))),gradient_rms=float(np.sqrt(np.mean(g0*g0))),seconds=0.))
    top_hist.append(dict(iteration=0,loops=topology(A,meta)))
    def callback(v):
        nonlocal last_accepted
        last_accepted=v.copy()
        if not np.array_equal(v,cache['x']):fun(v)
        iteration=len(history)
        history.append(dict(iteration=iteration,energy=float(cache['value']),gradient_inf=float(max(abs(cache['grad']))),
            gradient_rms=float(np.sqrt(np.mean(cache['grad']**2))),seconds=perf_counter()-start))
        if iteration%checkpoint==0:top_hist.append(dict(iteration=iteration,loops=topology(unpack(v),meta)))
        if perf_counter()-start>seconds:raise StopIteration
    result=minimize(fun,x0,jac=True,method='L-BFGS-B',callback=callback,
        options=dict(maxiter=maxiter,maxfun=maxiter*3,maxls=40,maxcor=12,gtol=gtol,ftol=1e-15))
    B=unpack(result.x);final,G=value_gradient(B,gd=meta['gd']);freegrad=2*G[free]
    # Projection of excluded variables is exact, not a large penalty limit.
    constraint_error=float(np.max(abs(B[pinned]-fixed[pinned]))) if np.any(pinned) else 0.
    increments=np.diff([r['energy'] for r in history])
    info=dict(**meta,initial_energy=e0,final_energy=final,energy_reduction=e0-final,
        iterations=int(result.nit),function_evaluations=int(result.nfev),scipy_success=bool(result.success),
        scipy_message=str(result.message),gradient_inf=float(np.max(abs(np.ascontiguousarray(freegrad).view(float)))),
        gradient_rms=float(np.sqrt(np.mean(abs(freegrad)**2)/2)),target_gtol=gtol,
        constraint_error=constraint_error,max_accepted_energy_increase=float(max(0.,np.max(increments))) if len(increments) else 0.,
        elapsed_seconds=perf_counter()-start)
    info['gradient_converged']=info['gradient_inf']<=gtol
    top_hist.append(dict(iteration=int(result.nit),loops=topology(B,meta)))
    return B,info,history,top_hist
