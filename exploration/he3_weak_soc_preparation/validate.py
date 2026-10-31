"""Deterministic CPU/GPU checks ONLY. No local updates, PT run or pilot.

The existing-engine integration test is in integration.py; it freezes local updates.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from models.he3 import spatial as sp
OUT=Path(__file__).resolve().parent/'results'
RESULTS={}


def check(name,condition,detail=True):
    if not condition:raise AssertionError(name)
    RESULTS[name]=detail


def eigen_reference(a):
    n=np.sum(abs(a)**2,axis=(-2,-1));best=np.zeros_like(n)
    for sign in (1,-1):
        v=(a[...,0]-1j*sign*a[...,1])/np.sqrt(2)
        matrix=np.einsum('...i,...j->...ij',v.real,v.real)+np.einsum('...i,...j->...ij',v.imag,v.imag)
        best=np.maximum(best,np.linalg.eigvalsh(matrix)[...,-1])
    return 1-best/n


def examples(L=8):
    rng=np.random.default_rng(20261031)
    ideal=np.zeros((L,L,3,2),complex);ideal[...,2,0]=1/np.sqrt(2);ideal[...,2,1]=1j/np.sqrt(2)
    planar=np.zeros_like(ideal);planar[...,0,0]=1/np.sqrt(2);planar[...,1,1]=1/np.sqrt(2)
    complex_spin=np.zeros_like(ideal);d=np.array([1,1j,0])/np.sqrt(2)
    complex_spin[:]=np.outer(d,[1,1j])/np.sqrt(2)
    wave=ideal*(1+(.2+.1j)*np.exp(2j*np.pi*np.arange(L)/L)[:,None,None,None])
    mixed=.8*ideal+.3*ideal.conj()
    random=(rng.normal(size=(L,L,3,2))+1j*rng.normal(size=(L,L,3,2)))*.2
    return dict(ideal=ideal,planar=planar,complex_spin=complex_spin,wave=wave,mixed=mixed,
                random=random,zero=ideal*0,near_zero=ideal*1e-8)


def cpu_checks():
    fields=examples();rng=np.random.default_rng(44);L=8
    a=np.stack(list(fields.values()));local=sp.local_fields(a);amp=sp.numpy_amplitudes(a)
    eps=sp.chiral_distance(a)
    valid=eps['valid']
    ref=eigen_reference(a[:-2])
    check('distance_eigenvalue_reference',np.allclose(eps['epsilon'][:-2],ref,atol=1e-14),float(np.max(abs(eps['epsilon'][:-2]-ref))))
    check('distance_examples',np.allclose(eps['epsilon'][[0,1,2]],np.array([0,.75,.5])[:,None,None],atol=1e-14))
    check('distance_additive_sources',np.allclose(eps['epsilon'][valid],(eps['orbital_loss']+eps['spin_loss'])[valid],atol=1e-14))
    check('zero_near_zero_excluded',not np.any(valid[-2:]) and np.all(np.isnan(eps['epsilon'][-2:])))
    check('planar_P_zero_B_nonzero',np.allclose(local[1,...,1],0) and np.allclose(np.sum(abs(local[1,...,7:10])**2*np.array([1,1,2]),axis=-1),.5))
    check('maximal_chirality_complex_spin_not_ideal',np.allclose(local[2,...,0],1) and np.allclose(eps['epsilon'][2],.5))
    check('complex_fourier_signs',np.allclose(amp[3,1,1],2*(.2+.1j)*L*L) and abs(amp[3,1,2])<1e-12)
    rotation=np.linalg.qr(rng.normal(size=(3,3)))[0];phase=.713
    rotated=np.einsum('ij,...ja->...ia',rotation,a)*np.exp(1j*phase)
    b=sp.numpy_amplitudes(rotated)
    invariant=sp.contracted_components(abs(amp)**2)[...,:5,:]
    check('SO3_U1_spectra',np.allclose(invariant,sp.contracted_components(abs(b)**2)[...,:5,:],atol=1e-10))
    check('gauge_charges',np.allclose(b[:,1],amp[:,1]*np.exp(2j*phase)) and
          np.allclose(b[:,7:10],amp[:,7:10]*np.exp(2j*phase)))
    check('distance_SO3_U1',np.allclose(sp.chiral_distance(rotated)['epsilon'],eps['epsilon'],equal_nan=True,atol=1e-14))
    # Independent full-matrix Frobenius norms, rather than reusing basis weights.
    m=np.einsum('...ma,...na->...mn',a,a.conj()).real
    q=m-np.trace(m,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3
    bfull=np.einsum('...ma,...mb->...ab',a,a)
    for label,tensor,indices in [('Q',q,slice(2,7)),('B',bfull,slice(7,10))]:
        ft=np.fft.fftn(tensor,axes=(1,2));power=np.stack([np.sum(abs(ft[:,x,y])**2,axis=(-2,-1)) for x,y in sp.MOMENTA],-1)
        observed=sp.contracted_components(abs(amp)**2)[:,2 if label=='Q' else 3]
        check(label+'_Frobenius_contraction',np.allclose(power,observed,atol=1e-10))
    err=0.
    for name in ('random','wave','planar'):
        f=sp.local_fields(fields[name]);c=sp.compact_correlations(fields[name])
        for r in range(L):
            x=np.mean(np.roll(f,-r,axis=0)*f.conj(),axis=(0,1))
            y=np.mean(np.roll(f,-r,axis=1)*f.conj(),axis=(0,1))
            err=max(err,float(np.max(abs(x-c['x'][:,r]))),float(np.max(abs(y-c['y'][:,r]))))
        all_pairs=np.empty((L,L,16),complex)
        for dx in range(L):
            for dy in range(L):all_pairs[dx,dy]=np.mean(np.roll(f,(-dx,-dy),(0,1))*f.conj(),axis=(0,1))
        d=np.minimum(np.arange(L),L-np.arange(L));shell=np.floor(np.hypot(d[:,None],d[None,:])).astype(int)
        radial=np.stack([all_pairs[shell==j].mean(0) for j in range(shell.max()+1)],-1)
        err=max(err,float(np.max(abs(radial-c['radial']))))
    check('FFT_direct_directional_radial_pairs',err<1e-12,err)
    r,xi,status=sp.correlation_ratio([2,1,1,0],[1,2,0,0],L)
    check('ratio_flags_no_clipping',np.array_equal(status,[0,1,2,3]) and r[1]==.5 and np.isnan(xi[1]))
    sample=np.stack([amp,amp*np.exp(.1j)])
    check('connected_recomputed_mean',np.allclose(sp.ensemble(sample,L,True)['spectra'],sp.contracted_components(np.mean(abs(sample-sample.mean(0))**2,axis=0))/(L*L)))
    import run as he3
    check('configured_sampling_intervals',he3.sizes==[64,96,128,256]
          and he3.sample_interval==10 and he3.exchange_interval==10)
    t=he3.make_temperature_ladder(**he3.temperature_ladder)
    frozen=np.loadtxt(OUT.parent/'ladders/common_L256.txt')
    assert np.allclose(t,frozen,rtol=0,atol=1e-15)
    check('common_ladder',np.all(np.diff(t)>0) and np.all(np.diff((1/t).astype('f4'))<0))
    from utils.data_io import save_runs,load_runs
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        path=Path(tmp)/'old.npz';old={8:dict(metadata={'L':8},samples={'energy':np.ones((3,2))})}
        save_runs(path,old)
        check('legacy_save_load',np.array_equal(load_runs(path)[8]['samples']['energy'],old[8]['samples']['energy']))


def gpu_checks():
    from numba import cuda
    if not cuda.is_available():RESULTS['gpu']='UNVERIFIED: CUDA unavailable';return
    from models.he3.measure_spatial import SpatialMeter
    from models.he3.model import He3Model
    spec=importlib.util.spec_from_file_location('independent_old_reference',ROOT/'tests/test_he3.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    maxerr=0.;energyerr=0.;distanceerr=0.
    for L in (8,10,34):
        a=np.asarray(list(examples(L).values()),np.complex64);R=len(a)
        state=cuda.to_device(a);mapping=np.roll(np.arange(R,dtype=np.int32),3);inverse=np.argsort(mapping)
        meter=SpatialMeter(L,R);gpu=meter.measure(state,cuda.to_device(mapping))
        cpu=sp.numpy_amplitudes(a)[inverse];maxerr=max(maxerr,float(np.max(abs(cpu-gpu))))
        assert np.allclose(cpu,gpu,rtol=1e-12,atol=1e-10)
        for gd in (0.,.001,.01):
            model=He3Model(pressure=20,g_d=gd)
            s=meter.structure(state,model,cuda.to_device(mapping));old=model.measure(state,temperature_indices=cuda.to_device(mapping))
            e=reference.total_energy(a,model.K,model.betas,g_d=float(model.g_d))[inverse]
            assert np.allclose(s[:,17],old['energy'],rtol=1e-6,atol=1e-6*L*L)
            assert np.allclose(s[:,13]+s[:,14]+s[:,16],old['energy'],rtol=1e-6,atol=1e-6*L*L)
            energyerr=max(energyerr,float(np.max(abs(s[:,13]+s[:,14]+s[:,16]-old['energy']))))
            assert np.allclose(e,old['energy'],rtol=3e-6,atol=3e-6*L*L)
            dip=reference.dipole_invariant_numpy(a).sum((1,2))[inverse]
            assert np.allclose(dip,s[:,15],rtol=2e-6,atol=1e-6*L*L)
            assert np.allclose(s[:,16],float(model.g_d)*s[:,15],rtol=2e-6,atol=1e-7*L*L)
            assert np.allclose(gpu[:,0,0].real,old['chirality'],rtol=1e-6,atol=1e-6*L*L)
            assert np.allclose(gpu[:,1,0],old['phase_real']+1j*old['phase_imag'],rtol=1e-6,atol=1e-6*L*L)
            assert np.allclose(-2*gpu[:,3,0].real/np.sqrt(6),(2/3)*old['density']*old['spin_alignment'],atol=1e-6*L*L)
            h=1e-4
            for axis in (0,1):
                delta=[0.,0.];delta[axis]=h;ep=reference.total_energy(a,model.K,model.betas,twist=delta,g_d=float(model.g_d))[inverse]
                delta[axis]=-h;em=reference.total_energy(a,model.K,model.betas,twist=delta,g_d=float(model.g_d))[inverse]
                assert np.allclose((ep-em)/(2*h),old['current_'+'xy'[axis]],rtol=1e-5,atol=3e-6*L*L)
                assert np.allclose((ep+em-2*e)/h**2,old['curvature_'+'xy'[axis]],rtol=1e-5,atol=3e-6*L*L)
        d=sp.chiral_distance(a)
        expected=np.nansum(d['n']*d['epsilon'],axis=(1,2))[inverse]
        distanceerr=max(distanceerr,float(np.max(abs(expected-s[:,6]))))
        assert np.allclose(expected,s[:,6],atol=1e-10)
        assert np.array_equal(a,state.copy_to_host())
        # Histograms far from floating boundary values must match exactly.
        random_index=list(examples().keys()).index('random');dst=mapping[random_index]
        for offset,key in [(20,'epsilon'),(44,'orbital_loss'),(56,'spin_loss')]:
            assert np.array_equal(s[dst,offset:offset+12],sp.histogram_with_tails(d[key][random_index]))
    check('GPU_FFT_signs_mapping_partial_tiles',True,maxerr)
    check('GPU_energy_parts_old_estimators_twist',True,energyerr)
    check('GPU_manifold_histograms_readonly',True,distanceerr)
    # Real-space-only mode works without a low-mode meter or a full ladder copy.
    model=He3Model(pressure=20,g_d=.001)
    model.realspace_every=2;model.realspace_indices=[0,3]
    for _ in range(4):model.measure(state,cuda.to_device(mapping))
    assert model._spatial_meter is None and not model.spatial_samples
    assert [r[0] for r in model.realspace_samples]==[1,3]
    for record in model.realspace_samples:
        np.testing.assert_array_equal(record[3],inverse[[0,3]])
        for j,w in enumerate(record[3]):
            ref=sp.compact_correlations(a[w])
            np.testing.assert_allclose(record[1][j],np.stack([ref[k][:7,:L//2+1] for k in ('x','y')],axis=1),atol=1e-12)
    check('realspace_only_recording_and_cadence',True)


def realspace_analysis_checks():
    from models.he3.spatial_analysis import summarize_realspace
    from utils.data_io import save_runs,load_runs
    L=8;records=[]
    for i,scale in enumerate([1.,1.3,.7,1.7,.9]):
        c=sp.compact_correlations(examples(L)['ideal']*scale*np.exp(.3j*i))
        profiles=np.stack([c[k][:7,:L//2+1] for k in ('x','y')],axis=1)
        records.append((i,profiles[None],c['means'][None,:7],np.array([1]),np.array([0])))
    run=dict(samples={'energy':np.zeros((5,2))},metadata=dict(L=L,temperatures=np.array([.38,.4]),sample_sweeps=np.arange(5)*10))
    sp.add_spatial_samples(run,[],records)
    assert not records and 'spatial_amplitudes' not in run['samples']
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        path=Path(tmp)/'profiles.npz';save_runs(path,{L:run});loaded=load_runs(path)[L]
        result=summarize_realspace(loaded,block_size=2)
    assert result['dropped_events']==1 and result['blocks']==2
    raw,means=run['samples']['realspace_profiles'][:4],run['samples']['realspace_means'][:4]
    connected=raw.mean(0)-abs(means.mean(0))[...,None,None]**2
    expected=np.stack((connected[:,0],connected[:,1],connected[:,2:].sum(1)),axis=1)
    np.testing.assert_allclose(result['connected'],expected,atol=1e-12)
    assert np.all(result['connected'][0,0].real>0)
    # Every snapshot is uniform: per-snapshot mean subtraction would incorrectly give zero.
    np.testing.assert_allclose(raw-abs(means)[...,None,None]**2,0,atol=1e-12)
    deleted=[]
    for kept in (slice(0,2),slice(2,4)):
        c=raw[kept].mean(0)-abs(means[kept].mean(0))[...,None,None]**2
        deleted.append(np.stack((c[:,0],c[:,1],c[:,2:].sum(1)),axis=1))
    np.testing.assert_allclose(result['connected_se_real'],np.std(np.asarray(deleted).real,axis=0),atol=1e-12)
    check('realspace_ensemble_mean_joint_deletes_and_legacy_NPZ',True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cpu_checks();gpu_checks();realspace_analysis_checks()
    (OUT/'simple_validation.json').write_text(json.dumps(dict(passed=True,checks=RESULTS),indent=2),encoding='utf-8')
    print('Passed',len(RESULTS),'deterministic checks; no MC sweeps')


if __name__=='__main__':main()
