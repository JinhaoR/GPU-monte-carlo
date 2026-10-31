"""Small known matrix fields for HQV detector validation; no GL relaxation.

Fields have shape (Lx, Ly, spin=3, orbital=2); axis 0 is x. Analytic angle
fixtures use open boundaries unless explicitly labelled periodic. They are
kinematic constructions, not energy minima or finite-temperature samples.
"""
import numpy as np


def grid(L=96):
    return np.meshgrid(np.arange(L, dtype=float), np.arange(L, dtype=float), indexing="ij")


def chiral(theta, alpha=0.0, density=1.0, sigma=1, director=None):
    """A=sqrt(n) exp(i theta) d (1,i sigma)/sqrt(2), with real unit d."""
    theta, alpha, density = np.broadcast_arrays(theta, alpha, density)
    if director is None:
        director = np.stack((np.sin(alpha), np.zeros_like(alpha), np.cos(alpha)), axis=-1)
    else:
        director = np.broadcast_to(np.asarray(director, dtype=float), theta.shape+(3,))
        director = director/np.linalg.norm(director, axis=-1, keepdims=True)
    spin = np.sqrt(density)[..., None]*np.exp(1j*theta)[..., None]*director
    orbital = np.array([1, 1j*sigma])/np.sqrt(2)
    return spin[..., :, None]*orbital


def uniform(L=96, sigma=1):
    return chiral(np.zeros((L, L)), sigma=sigma)


def smooth_texture(L=96):
    x, y = grid(L)
    theta=.23*np.sin(2*np.pi*x/L)+.17*np.cos(2*np.pi*y/L)
    alpha=.6*np.sin(2*np.pi*y/L)
    beta=.8*np.sin(2*np.pi*x/L)
    d=np.stack((np.sin(alpha)*np.cos(beta),np.sin(alpha)*np.sin(beta),np.cos(alpha)),axis=-1)
    return chiral(theta,director=d)


def _core_factor(x,y,centers,core):
    amplitude=np.ones_like(x)
    for cx,cy in centers:
        amplitude*=np.tanh(np.hypot(x-cx,y-cy)/core)
    return amplitude


def integer_vortex(L=96, center=None, charge=1, core=2, sigma=1):
    if center is None:center=(L/2-.5,L/2-.5)
    x,y=grid(L);cx,cy=center
    phi=np.arctan2(y-cy,x-cx)
    return chiral(charge*phi,density=_core_factor(x,y,[center],core)**2,sigma=sigma)


def integer_pair(L=96, centers=None, core=2, sigma=1):
    if centers is None:centers=((L/3-.5,L/2-.5),(2*L/3-.5,L/2-.5))
    x,y=grid(L)
    phi=np.arctan2(y-centers[0][1],x-centers[0][0])-np.arctan2(y-centers[1][1],x-centers[1][0])
    return chiral(phi,density=_core_factor(x,y,centers,core)**2,sigma=sigma)


def _hqv_from_unit_scalar(u,amplitude,sigma=1):
    """Square-root-free lift: e^(i alpha)d(alpha), u=e^(2i alpha).

    v=((u-1)/(2i),0,(u+1)/2) has unit norm and v·v=u. It is
    single-valued even across the arbitrary branch cut of arg(u).
    """
    v=np.stack(((u-1)/(2j),np.zeros_like(u),(u+1)/2),axis=-1)
    return amplitude[...,None,None]*v[..., :,None]*np.array([1,1j*sigma])/np.sqrt(2)


def hqv_pair(L=96,centers=None,core=2,sigma=1):
    if centers is None:centers=((L/3-.5,L/2-.5),(2*L/3-.5,L/2-.5))
    x,y=grid(L)
    z1=x-centers[0][0]+1j*(y-centers[0][1])
    z2=x-centers[1][0]+1j*(y-centers[1][1])
    # No individual half-angle or branch cut enters the physical field.
    u=z1*np.conj(z2)
    u=np.divide(u,np.abs(u),out=np.ones_like(u),where=np.abs(u)>0)
    return _hqv_from_unit_scalar(u,_core_factor(x,y,centers,core),sigma)


def same_sign_hqv_pair(L=96,centers=None,core=2,sigma=1):
    """Two +1/2 charges: an outer contour measures net +1, not core count."""
    if centers is None:centers=((L/3-.5,L/2-.5),(2*L/3-.5,L/2-.5))
    x,y=grid(L)
    z1=x-centers[0][0]+1j*(y-centers[0][1])
    z2=x-centers[1][0]+1j*(y-centers[1][1])
    u=z1*z2
    u=np.divide(u,np.abs(u),out=np.ones_like(u),where=np.abs(u)>0)
    return _hqv_from_unit_scalar(u,_core_factor(x,y,centers,core),sigma)


def periodic_hqv_pair(L=96,core=2,sigma=1):
    """A genuinely periodic pair with analytic torus scalar of indices -1,+1.

    f=sin(X)+i(cos(X)+cos(Y)-1) has exactly two zeros, at X=0,
    Y=pi/2 and 3pi/2. A periodic amplitude envelope suppresses their cores.
    This is a validation fixture, not the minimum-energy torus Green field.
    The continuum lift is continuous at a core but need not be differentiable
    there; on this grid the analytic zeros sit between lattice sites.
    """
    x,y=grid(L)
    X=2*np.pi*(x-L/2+.5)/L;Y=2*np.pi*(y+.5)/L
    f=np.sin(X)+1j*(np.cos(X)+np.cos(Y)-1)
    u=f/np.abs(f)
    amplitude=np.tanh(np.abs(f)/(2*np.pi*core/L))
    centers=((L/2-.5,L/4-.5),(L/2-.5,3*L/4-.5))
    return _hqv_from_unit_scalar(u,amplitude,sigma),centers


def chirality_wall(L=96,width=3):
    x,y=grid(L)
    eta=np.pi/4*(1+np.tanh((x-(L/2-.5))/width))
    hp=np.array([1,1j])/np.sqrt(2);hm=np.array([1,-1j])/np.sqrt(2)
    orbital=np.cos(eta)[...,None]*hp+np.sin(eta)[...,None]*hm
    a=np.zeros((L,L,3,2),dtype=complex);a[...,2,:]=orbital
    return a


def composite_zero(L=96,center=None):
    """Odd P winding with finite A norm and no chiral orbital structure."""
    if center is None:center=(L/2-.5,L/2-.5)
    x,y=grid(L);f=.01*((x-center[0])+1j*(y-center[1]))
    a=np.zeros((L,L,3,2),dtype=complex)
    a[...,0,0]=1;a[...,1,0]=1j+f
    a/=np.sqrt(np.sum(np.abs(a)**2,axis=(-2,-1)))[...,None,None]
    return a


def nonunitary_high_chirality(L=96,ellipticity=.65):
    """m_chi=1 but spin vector is complex and not a real director times phase."""
    spin=np.array([1,1j*ellipticity,0])/np.sqrt(1+ellipticity**2)
    a=spin[:,None]*np.array([1,1j])[None,:]/np.sqrt(2)
    return np.broadcast_to(a,(L,L,3,2)).copy()


def density_void(L=96,radius=6):
    x,y=grid(L);a=uniform(L)
    a[np.hypot(x-L/2,y-L/2)<=radius]=0
    return a


def half_phase_without_spin(L=96):
    """Deliberately invalid HQV texture: physical A jumps sign across a cut."""
    x,y=grid(L);phi=np.arctan2(y-(L/2-.5),x-(L/2-.5))
    return chiral(phi/2)


def half_spin_without_phase(L=96):
    """RP2 half-disclination without compensating phase: physical sign seam."""
    x,y=grid(L);phi=np.arctan2(y-(L/2-.5),x-(L/2-.5))
    return chiral(np.zeros_like(phi),alpha=phi/2)


def add_noise(field,strength,seed=7293):
    """Isotropic independent complex component noise; strength is relative RMS."""
    rng=np.random.default_rng(seed)
    scale=np.sqrt(np.mean(np.sum(np.abs(field)**2,axis=(-2,-1)))/12)
    return field+strength*scale*(rng.normal(size=field.shape)+1j*rng.normal(size=field.shape))


def spin_rotate(field,rotation):
    return np.einsum('ab,...bc->...ac',rotation,field)


def make_cases(L=96):
    centers=((L/3-.5,L/2-.5),(2*L/3-.5,L/2-.5))
    periodic,pc=periodic_hqv_pair(L)
    result={}
    def add(name,a,boundary,centers=(),charges=(),description=''):
        result[name]=dict(field=a,boundary=boundary,centers=centers,
                          expected_charges=charges,description=description)
    add('uniform',uniform(L),'periodic',description='Constant positive-chirality state')
    add('smooth_texture',smooth_texture(L),'periodic',description='Smooth spin and phase without defects')
    add('integer_vortex',integer_vortex(L),'open',[(L/2-.5,L/2-.5)],[1],description='One integer phase vortex; use resolved enclosing contours')
    add('integer_pair',integer_pair(L,centers),'open',centers,[1,-1],description='Neutral integer pair in an open analytic-angle field')
    add('hqv_pair',hqv_pair(L,centers),'open',centers,[.5,-.5],description='Branch-independent HQV and anti-HQV')
    add('periodic_hqv_pair',periodic,'periodic',pc,[-.5,.5],description='Torus field constructed from a periodic scalar with two opposite zeros')
    add('chirality_wall',chirality_wall(L),'open',description='Smooth transition between opposite orbital chiralities')
    add('composite_zero',composite_zero(L),'open',[(L/2-.5,L/2-.5)],description='Odd composite winding outside the chiral manifold')
    add('nonunitary_high_chirality',nonunitary_high_chirality(L),'periodic',description='Chirality alone misses complex spin polarization')
    add('density_void',density_void(L),'open',description='Zero-norm disk; loops through it are unresolved')
    add('half_phase_without_spin',half_phase_without_spin(L),'open',description='Uncompensated half phase has an unphysical sign seam')
    add('half_spin_without_phase',half_spin_without_phase(L),'open',description='Uncompensated director half turn has a physical sign seam')
    return result
