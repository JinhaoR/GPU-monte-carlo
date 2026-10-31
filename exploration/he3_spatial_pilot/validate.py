"""Independent FFT/full-tensor/analytic validation on the actual local GPU.

Run python -B exploration/he3_spatial_pilot/validate.py before pilot.py.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0'
from pathlib import Path
import json,sys,warnings,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore',NumbaPerformanceWarning)
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from core.parallel_tempering import ParallelTempering
from observables import numpy_amplitudes,local_fields,powers,ensemble,correlation_ratio,MOMENTA,FIELDS
from gpu_measure import SpatialMeter

OUT=Path(__file__).resolve().parent/'results'


def main():
    if not cuda.is_available():raise RuntimeError('An actual CUDA GPU is required; no CPU simulator fallback')
    if os.environ.get('NUMBA_ENABLE_CUDASIM')=='1':raise RuntimeError('CUDA simulator is not accepted')
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026102901);rows=[];checks=[]
    L=12;N=L*L;n=1.3
    uniform=np.zeros((L,L,3,2),complex);uniform[...,2,0]=np.sqrt(n/2);uniform[...,2,1]=1j*np.sqrt(n/2)
    nonchiral=np.zeros_like(uniform);nonchiral[...,2,0]=np.sqrt(n)
    planar=np.zeros_like(uniform);planar[...,0,0]=np.sqrt(n/2);planar[...,1,1]=np.sqrt(n/2)
    xx,yy=np.meshgrid(np.arange(L),np.arange(L),indexing='ij');phase=np.exp(2j*np.pi*xx/L)
    wave=uniform*(1+(.2+.1j)*phase)[...,None,None]
    sinusoid=uniform*np.sqrt(1+.3*np.cos(2*np.pi*xx/L))[...,None,None]
    spinwave=uniform.copy();spinwave[...,2,:]=0
    spinwave[...,0,:]=uniform[...,2,:]*np.cos(2*np.pi*xx/L)[...,None]
    spinwave[...,2,:]=uniform[...,2,:]*np.sin(2*np.pi*xx/L)[...,None]
    random=(rng.normal(size=uniform.shape)+1j*rng.normal(size=uniform.shape))*.3
    random_nonpower=(rng.normal(size=(10,10,3,2))+1j*rng.normal(size=(10,10,3,2)))*.3
    cases={'zero':np.zeros_like(uniform),'uniform_chiral':uniform,'uniform_nonchiral':nonchiral,
        'uniform_planar':planar,'complex_periodic_wave':wave,'density_sinusoid':sinusoid,
        'spin_texture':spinwave,'random_full_matrix':random,'random_L10':random_nonpower}
    for ls in [34,64,128,256]:
        # Include a partial final tile and multiple complete tiles, beyond the
        # small analytic tests. These are read-only synthetic measurements.
        cases[f'random_L{ls}_tiled']=(rng.normal(size=(ls,ls,3,2))+1j*rng.normal(size=(ls,ls,3,2)))*.3
    for label,field in cases.items():
        a=field.astype(np.complex64);ls=a.shape[0]
        meter=SpatialMeter(ls,1);gpu=meter.measure(cuda.to_device(a[None]))[0];cpu=numpy_amplitudes(a)
        diff=np.max(abs(gpu-cpu));scale=max(1,np.max(abs(cpu)))
        assert diff<3e-12*scale,(label,diff,scale)
        # Independently contract all nine tensor matrix entries in Fourier space.
        k,p,t,q=local_fields(a);qft=np.fft.fftn(q,axes=(0,1))
        reference=np.array([np.sum(abs(qft[i,j])**2)/(ls*ls) for i,j in MOMENTA])
        np.testing.assert_allclose(powers(gpu,ls)[2],reference,rtol=5e-12,atol=2e-22)
        assert np.all(powers(gpu,ls)>=0)
        rows.append(dict(case=label,L=ls,max_absolute_gpu_fft_difference=float(diff),relative_to_largest_amplitude=float(diff/scale)))
    checks.extend([f'GPU versus independent NumPy FFT: {len(cases)} fields/sizes','Q Frobenius contraction: all nine tensor entries','nonnegative structure factors'])
    u=numpy_amplitudes(uniform)
    np.testing.assert_allclose(u[[0,1],0],n*N,atol=1e-10)
    np.testing.assert_allclose(powers(u,L)[2,0],2*n*n*N/3,rtol=1e-12)
    assert np.max(abs(numpy_amplitudes(planar)[:2]))<1e-10
    w=numpy_amplitudes(wave)
    np.testing.assert_allclose(w[1,1],2*n*(.2+.1j)*N,atol=1e-10)
    assert abs(w[1,2])<1e-10 and abs(w[1,1])>10
    ds=numpy_amplitudes(sinusoid)
    np.testing.assert_allclose(ds[0,[0,1,2]],n*N*np.array([1,.15,.15]),atol=1e-10)
    checks.extend(['uniform analytic zero modes and tensor norm','nonchiral composite-cancellation case','complex P positive/negative momenta differ','known sinusoidal Fourier amplitude'])
    # Spin SO(3) and U(1) transformations, including GPU c64 rounding.
    rot,_=np.linalg.qr(rng.normal(size=(3,3)));rot[:,0]*=np.linalg.det(rot)
    aa=np.stack([uniform,random,wave,spinwave]).astype(np.complex64)
    rr=np.ascontiguousarray(np.einsum('mn,rxyna->rxyma',rot,aa,optimize=True),dtype=np.complex64)
    meter=SpatialMeter(L,4);ga=meter.measure(cuda.to_device(aa));gr=meter.measure(cuda.to_device(rr))
    np.testing.assert_allclose(powers(ga,L),powers(gr,L),rtol=2e-6,atol=2e-12)
    np.testing.assert_allclose(gr[:,:2],ga[:,:2],rtol=2e-6,atol=2e-5)
    theta=.71;gp=meter.measure(cuda.to_device((aa*np.exp(1j*theta)).astype(np.complex64)))
    np.testing.assert_allclose(gp[:,1],ga[:,1]*np.exp(2j*theta),rtol=2e-6,atol=2e-5)
    np.testing.assert_allclose(powers(gp,L),powers(ga,L),rtol=2e-6,atol=2e-12)
    checks.extend(['global spin SO(3) invariant scalar spectra','global gauge charge two and invariant spectra'])
    xy=meter.measure(cuda.to_device(aa.transpose(0,2,1,3,4).copy()))
    np.testing.assert_allclose(xy,ga[:,:,[0,3,4,1,2]],rtol=2e-12,atol=2e-11)
    shifted=meter.measure(cuda.to_device(np.roll(aa,2,axis=1)))
    factor=np.exp(-2j*np.pi*np.array([k[0] for k in MOMENTA])*2/L)
    np.testing.assert_allclose(shifted,ga*factor,rtol=2e-12,atol=2e-11)
    mapping=np.array([2,0,3,1],np.int32);permuted=meter.measure(cuda.to_device(aa),cuda.to_device(mapping));expect=np.empty_like(ga);expect[mapping]=ga
    np.testing.assert_allclose(permuted,expect,rtol=2e-12,atol=2e-11)
    checks.extend(['spatial x/y permutation','translation Fourier phase','nontrivial walker->temperature scatter'])
    ratio,xi,status=correlation_ratio(np.array([4,1,2,0]),np.array([1,2,0,0]),L)
    assert list(status)==[0,1,2,3] and np.isnan(xi[1:]).all() and ratio[1]==.5
    np.testing.assert_allclose(xi[0],np.sqrt(3)/(2*L*np.sin(np.pi/L)))
    assert np.all(ensemble(numpy_amplitudes(uniform)[None],L)['status_xy']==2)
    assert np.all(ensemble(numpy_amplitudes(np.zeros_like(uniform))[None],L)['status_xy']==3)
    checks.extend(['negative radicand retains ratio and NaN flag','uniform singular and zero undefined estimators'])
    # Real PT smoke: map changes and cached extensive energies stay unchanged.
    rt=He3GaussianRuntime(He3Model(pressure=20),8,[.37,.38,.39,.41],seed=2026102902)
    pt=ParallelTempering(rt,exchange_interval=5,seed=2026102903)
    sm=SpatialMeter(8,4);cross=[]
    for step in range(1,101):
        pt.sweep()
        if step in [5,50,100]:
            before=rt.state.copy_to_host();mapping=pt.walker_to_temperature.copy_to_host()
            amp=sm.measure(rt.state,pt.walker_to_temperature);sc=pt.measure();after=rt.state.copy_to_host()
            np.testing.assert_array_equal(before,after)
            ref=np.empty_like(amp);ref[mapping]=numpy_amplitudes(before)
            np.testing.assert_allclose(amp,ref,rtol=3e-12,atol=1e-11)
            np.testing.assert_allclose(amp[:,0,0].real,sc['chirality'],rtol=3e-6,atol=2e-5)
            np.testing.assert_allclose(amp[:,1,0],sc['phase_real']+1j*sc['phase_imag'],rtol=3e-6,atol=2e-5)
            qzz=-np.sqrt(2/3)*amp[:,3,0].real
            np.testing.assert_allclose(qzz,(2/3)*sc['density']*sc['spin_alignment'],rtol=3e-6,atol=2e-5)
            cross.append(dict(sweep=step,chirality_max_diff=float(np.max(abs(amp[:,0,0].real-sc['chirality']))),
                composite_max_diff=float(np.max(abs(amp[:,1,0]-sc['phase_real']-1j*sc['phase_imag'])))))
    checks.extend(['real GPU Gaussian/PT smoke','existing zero modes and Qzz normalization','measurement leaves fields unchanged'])
    result=dict(passed=True,check_groups=len(checks),checks=checks,fft_comparisons=rows,pt_cross_checks=cross,
        gpu_name=str(cuda.get_current_device().name),fourier_convention='unnormalized exp(-2pi i k.r/L)',
        fields=FIELDS,momenta=MOMENTA,
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['observables.py','gpu_measure.py']})
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
