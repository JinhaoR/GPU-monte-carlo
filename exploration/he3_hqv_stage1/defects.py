"""Small CPU-only contour diagnostics for a full complex (..., 3, 2) GL field.

Classification is conditional on a resolved loop near ONE chiral manifold.
It is not a projection of the Hamiltonian or a detector valid in every phase.
Coordinates: field[x, y, spin, orbital], positive circulation is CCW in (x,y).
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Thresholds:
    # epsilon is a relative SQUARED Frobenius distance, not its square root.
    max_residual: float = .12
    min_density: float = .01
    min_composite_ratio: float = .5
    min_chirality: float = .7
    min_gap: float = .5
    max_composite_step: float = .75*np.pi
    max_director_step: float = np.pi/3
    max_lifted_phase_step: float = .75*np.pi


def local_geometry(field):
    """Exact best fit A ≈ a d (1,i sigma)/sqrt(2), d real, |d|=1.

    No field is changed. Diagnostics of poor fits are retained and masked only
    when deciding whether a loop has a meaningful chiral-sector interpretation.
    M=Re(z z†), lambda_max=(|z|²+|z·z|)/2. eigh returns an unoriented director.
    """
    A=np.asarray(field)
    if A.shape[-2:]!=(3,2) or not np.iscomplexobj(A):
        raise ValueError('Expected a complex array with final shape (3,2)')
    if not np.all(np.isfinite(A)):
        raise ValueError('Nonfinite GL field')
    x,y=A[...,0],A[...,1]
    n=np.sum(abs(A)**2,axis=(-2,-1))
    zp=(x-1j*y)/np.sqrt(2);zm=(x+1j*y)/np.sqrt(2)
    mp=np.sum(abs(zp)**2,axis=-1);mm=np.sum(abs(zm)**2,axis=-1)
    bp=np.sum(zp*zp,axis=-1);bm=np.sum(zm*zm,axis=-1)
    lp=(mp+abs(bp))/2;lm=(mm+abs(bm))/2
    sigma=np.where(lp>=lm,1,-1)
    z=np.where((sigma==1)[...,None],zp,zm)
    B=np.where(sigma==1,bp,bm)
    m=np.where(sigma==1,mp,mm)
    M=z.real[..., :,None]*z.real[...,None,:]+z.imag[..., :,None]*z.imag[...,None,:]
    eig,d=np.linalg.eigh(M)
    director=d[..., :, -1]
    amplitude=np.sum(director*z,axis=-1)
    safe=np.where(n>0,n,1.)
    eps=np.clip(1-eig[...,-1]/safe,0,1)
    eps=np.where(n>0,eps,1.)
    chi=2*np.imag(np.sum(x.conj()*y,axis=-1))/safe
    p=np.sum(x*x-y*y,axis=-1)
    return dict(density=n,chirality=chi,composite=p,sector_composite=B,
                projected_composite=B,fit_composite=amplitude**2,
                residual=eps,sigma=sigma,director=director,amplitude=amplitude,
                gap=(eig[...,-1]-eig[...,-2])/safe,
                wrong_chirality=(n-m)/safe,nonreal_spin=(m-abs(B))/(2*safe),
                composite_ratio=abs(p)/safe,projected_ratio=abs(B)/safe)


def rectangle_loop(x0,x1,y0,y1):
    """Unit-spaced CCW boundary sites, each corner included once."""
    if not(x0<x1 and y0<y1):
        raise ValueError('Rectangle must have positive width and height')
    return np.array([(x,y0) for x in range(x0,x1)]+
                    [(x1,y) for y in range(y0,y1)]+
                    [(x,y1) for x in range(x1,x0,-1)]+
                    [(x0,y) for y in range(y1,y0,-1)],dtype=int)


def _phase_winding(values):
    steps=np.angle(np.roll(values,-1)*values.conj())
    total=float(steps.sum()/(2*np.pi))
    return int(np.rint(total)),steps,total


def analyze_loop(field,contour,*,geometry=None,thresholds=None,directors=None,periodic=False):
    """Return conservative net defect classification on a closed lattice path.

    The supplied sites omit a repeated final endpoint. With periodic=True,
    paths may use unwrapped integer coordinates, reduced modulo the field shape.
    Noncontractible torus cycles are flagged, since they do not enclose a disk.
    Per-loop minimum density is an absolute GL norm threshold in chosen units.
    A loop classifies NET enclosed charge, never counts neutral enclosed pairs.
    Cached geometry and director overrides are trusted expert inputs; an override
    is intended for sign changes of the extracted unit eigenvectors, not an
    independent arbitrary director field.
    """
    A=np.asarray(field)
    if A.ndim!=4:
        raise ValueError('Expected a single two-dimensional field, shape (Nx,Ny,3,2)')
    c=np.asarray(contour)
    if c.ndim!=2 or c.shape[1]!=2 or len(c)<4 or not np.issubdtype(c.dtype,np.integer):
        raise ValueError('contour must contain at least four integer (x,y) sites')
    if np.array_equal(c[0],c[-1]):c=c[:-1]
    if len({tuple(v) for v in c})!=len(c):
        raise ValueError('Contour contains repeated vertices')
    if not periodic and (np.any(c<0) or np.any(c>=np.array(A.shape[:2]))):
        raise ValueError('Contour leaves an open array; use periodic=True only for periodic fields')
    delta=np.roll(c,-1,axis=0)-c
    # Permit ordinary unit links and periodic closure across a single seam.
    reduced=delta-np.rint(delta/np.array(A.shape[:2]))*np.array(A.shape[:2]) if periodic else delta
    if np.any(np.sum(abs(reduced),axis=1)!=1):
        raise ValueError('Contour must use nearest-neighbor lattice links')
    ix,iy=c[:,0]%A.shape[0],c[:,1]%A.shape[1]
    if len(set(zip(ix.tolist(),iy.tolist())))!=len(c):
        raise ValueError('Contour repeats sites after periodic reduction')
    g=local_geometry(A) if geometry is None else geometry
    th=Thresholds() if thresholds is None else thresholds
    n=g['density'][ix,iy];p=g['composite'][ix,iy];B=g['projected_composite'][ix,iy]
    sig=g['sigma'][ix,iy]
    d=(g['director'] if directors is None else np.asarray(directors))[ix,iy]
    if directors is None:
        amp=g['amplitude'][ix,iy]
    else:
        # Recompute a together with any chosen signs of d; the paired lift is
        # invariant under (d_i,a_i) -> (-d_i,-a_i) independently at each vertex.
        z=(A[ix,iy,:,0]-1j*sig[:,None]*A[ix,iy,:,1])/np.sqrt(2)
        amp=np.sum(d*z,axis=-1)
    overlap=np.sum(d*np.roll(d,-1,axis=0),axis=1)
    signs=np.where(overlap>=0,1.,-1.)
    parity=int(np.prod(signs))
    director_steps=np.arccos(np.clip(abs(overlap),0,1))
    liftsteps=np.angle(np.roll(amp,-1)*amp.conj()*signs)
    qlift=float(liftsteps.sum()/(2*np.pi))
    wp,dp,rawp=_phase_winding(p)
    wb,db,rawb=_phase_winding(B)
    valid_amp=bool(np.all(n>th.min_density) and np.all(abs(p)>0) and np.all(abs(B)>0) and np.all(abs(amp)>0))
    reasons=[]
    if periodic and np.any(np.sum(reduced,axis=0)!=0):reasons.append('noncontractible_torus_cycle')
    if not valid_amp:reasons.append('density_or_complex_amplitude_too_small')
    if np.max(g['residual'][ix,iy])>th.max_residual:reasons.append('outside_chiral_manifold_tolerance')
    if np.any(sig!=sig[0]):reasons.append('mixed_chirality_sector')
    if np.min(abs(g['chirality'][ix,iy]))<th.min_chirality:reasons.append('weak_local_chirality')
    if np.min(g['gap'][ix,iy])<th.min_gap:reasons.append('director_eigenvalue_degeneracy')
    if min(np.min(g['composite_ratio'][ix,iy]),np.min(g['projected_ratio'][ix,iy]))<th.min_composite_ratio:
        reasons.append('weak_charge_two_field')
    if max(np.max(abs(dp)),np.max(abs(db)))>=th.max_composite_step:reasons.append('composite_phase_link_alias_risk')
    if np.max(director_steps)>=th.max_director_step:reasons.append('director_link_ambiguous')
    if np.max(abs(liftsteps))>=th.max_lifted_phase_step:reasons.append('lifted_phase_link_alias_risk')
    if wp!=wb:reasons.append('original_and_sector_composite_winding_disagree')
    if abs(2*qlift-wp)>1e-6:reasons.append('composite_and_lifted_charge_disagree')
    if parity!=(-1 if wp%2 else 1):reasons.append('composite_and_director_parity_disagree')
    if max(abs(rawp-wp),abs(rawb-wb))>1e-6:reasons.append('noninteger_composite_sum')
    accepted=not reasons
    label=('no_net_vortex' if wp==0 else 'hqv' if abs(wp)==1 else 'integer_vortex' if abs(wp)==2 else 'higher_winding') if accepted else 'unresolved'
    return dict(accepted=accepted,label=label,composite_winding=wp if valid_amp else None,
        projected_winding=wb if valid_amp else None,phase_charge=wp/2 if accepted else None,
        director_parity=parity if np.min(g['gap'][ix,iy])>=th.min_gap else None,
        lifted_phase_charge=qlift if valid_amp else None,sigma=int(sig[0]) if np.all(sig==sig[0]) else None,
        max_composite_step=float(max(np.max(abs(dp)),np.max(abs(db)))),
        max_director_step=float(np.max(director_steps)),max_lifted_phase_step=float(np.max(abs(liftsteps))),
        min_density=float(np.min(n)),max_residual=float(np.max(g['residual'][ix,iy])),
        min_gap=float(np.min(g['gap'][ix,iy])),min_abs_chirality=float(np.min(abs(g['chirality'][ix,iy]))),
        min_composite_ratio=float(np.min(g['composite_ratio'][ix,iy])),reasons=reasons,
        contour_vertices=len(c))


def plaquette_map(field,geometry=None,*,periodic=False):
    """Raw elementary candidates, including unresolved composite aliasing.

    A lifted candidate map is needed because an integer vortex can alias in P.
    These maps are proposals for larger-loop checks, NOT accepted populations.
    """
    A=np.asarray(field);g=local_geometry(A) if geometry is None else geometry
    if periodic:
        def corners(a):return (a,np.roll(a,-1,axis=0),np.roll(np.roll(a,-1,axis=0),-1,axis=1),np.roll(a,-1,axis=1))
    else:
        def corners(a):return (a[:-1,:-1],a[1:,:-1],a[1:,1:],a[:-1,1:])
    pc=corners(g['composite']);dc=corners(g['director']);ac=corners(g['amplitude'])
    pw=np.zeros(pc[0].shape);ql=np.zeros_like(pw);h=np.ones_like(pw);min_overlap=np.ones_like(pw)
    for i in range(4):
        j=(i+1)%4
        pw+=np.angle(pc[j]*pc[i].conj())/(2*np.pi)
        dot=np.sum(dc[i]*dc[j],axis=-1);s=np.where(dot>=0,1.,-1.)
        h*=s;min_overlap=np.minimum(min_overlap,abs(dot))
        ql+=np.angle(ac[j]*ac[i].conj()*s)/(2*np.pi)
    return dict(composite_winding=np.rint(pw).astype(int),lifted_phase_charge=ql,
                director_parity=h.astype(int),min_director_overlap=min_overlap,
                candidate=(abs(pw)>.5)|(abs(ql)>.25))


def find_candidates(field,*,periodic=False,thresholds=None,radii=(1,2,4)):
    """Elementary seeds -> connected clusters -> stable surrounding-loop charges.

    A deliberately small proof of concept. Accept at least two surrounding
    contours with the same nonzero charge. Results refer to resolved charged
    clusters, not necessarily individual cores. A neutral pair inside every
    chosen loop, a core within a rejected region, or sublattice structure can
    be missed. Retain rejected seeds and all loop reasons for inspection.
    """
    A=np.asarray(field);g=local_geometry(A);maps=plaquette_map(A,g,periodic=periodic)
    pending=set(map(tuple,np.argwhere(maps['candidate'])))
    shape=np.array(maps['candidate'].shape);results=[]
    while pending:
        seed=min(pending);pending.remove(seed)
        stack=[seed];cluster=[seed]
        while stack:
            x,y=stack.pop()
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                v=np.array([x+dx,y+dy])
                if periodic:v%=shape
                elif np.any(v<0) or np.any(v>=shape):continue
                key=tuple(v)
                if key in pending:
                    pending.remove(key);stack.append(key);cluster.append(key)
        # Use an actual elementary seed as the anchor; this avoids circular
        # averaging conventions when a cluster crosses a periodic seam.
        i,j=seed;loops=[]
        for radius in radii:
            if 2*radius+1>=min(A.shape[:2]):
                loops.append(dict(radius=radius,accepted=False,reasons=['contour_too_large_for_box']))
                continue
            c=rectangle_loop(i-radius,i+1+radius,j-radius,j+1+radius)
            if not periodic and (np.any(c<0) or np.any(c>=np.array(A.shape[:2]))):
                loops.append(dict(radius=radius,accepted=False,reasons=['outer_boundary_prevents_contour']))
                continue
            r=analyze_loop(A,c,geometry=g,thresholds=thresholds,periodic=periodic)
            loops.append(dict(radius=radius,**r))
        good=[r for r in loops if r['accepted'] and r['phase_charge']!=0]
        charges=set(r['phase_charge'] for r in good)
        # All accepted contours, including zero-charge ones, must agree: an
        # enlarging loop which captures a second core is a resolved cluster
        # ambiguity, not a stable single-population count.
        allcharges=set(r['phase_charge'] for r in loops if r['accepted'])
        accepted=len(good)>=2 and len(charges)==1 and len(allcharges)==1
        charge=good[0]['phase_charge'] if accepted else None
        results.append(dict(center_x=i+.5,center_y=j+.5,seed_plaquettes=len(cluster),
            accepted=accepted,phase_charge=charge,label=good[0]['label'] if accepted else 'unresolved_cluster',
            reasons=[] if accepted else ['no_stable_nonzero_charge_on_multiple_contours'],loops=loops))
    return results
