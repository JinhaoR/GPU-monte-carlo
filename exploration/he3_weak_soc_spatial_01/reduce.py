"""Read the eight completed spatial runs; save compact 50k-sweep moment blocks.

No simulation or production writes. Run from any directory with python -B.
The largest live inputs are one run's scalar and spatial arrays, not all runs.
"""
from pathlib import Path
import csv
import gc
import hashlib
import json
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
BLOCK = 50_000
sys.path.insert(0, str(ROOT))
from exploration.he3_soc.summarize import autocorrelation
from models.he3.spatial import FIELDS, STRUCTURE_NAMES


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pt_summary(meta, L, gd):
    pt=meta['parallel_tempering']; h=pt.pop('walker_history')
    sweeps=np.asarray(h['sweeps']); pos=np.asarray(h['temperature_indices'],dtype=np.int16)
    assert np.array_equal(h['walker_ids'],np.arange(pos.shape[1]))
    ids=np.searchsorted(sweeps,meta['spatial_sample_sweeps'])
    assert np.array_equal(sweeps[ids],meta['spatial_sample_sweeps'])
    expected=np.argsort(pos[ids],axis=1).astype(np.int32)
    rid=np.searchsorted(sweeps,meta['realspace_sample_sweeps'])
    expected_r=np.argsort(pos[rid],axis=1)[:,meta['realspace_temperature_indices']].astype(np.int32)
    keep=sweeps>meta['thermalization_sweeps']; ps=pos[keep]; ts=sweeps[keep]
    nt=pos.shape[1]; trips=[]; visited_both=[]; durations=[]; walkers=[]
    up=np.zeros(nt); down=np.zeros(nt)
    for w in range(nt):
        p=ps[:,w]; hit=np.flatnonzero((p==0)|(p==nt-1))
        visited_both.append(bool(np.any(p==0) and np.any(p==nt-1)))
        count=0; origin=None; opposite=False; start=None
        for i in hit:
            endpoint=int(p[i])
            if origin is None:origin=endpoint;start=int(ts[i])
            elif endpoint!=origin:opposite=True
            elif opposite:
                count+=1;durations.append(int(ts[i])-start);start=int(ts[i]);opposite=False
        trips.append(count)
        if len(hit):
            past=np.searchsorted(hit,np.arange(len(p)),side='right')-1
            valid=past>=0; last=p[hit[past[valid]]]
            up+=np.bincount(p[valid][last==0],minlength=nt)
            down+=np.bincount(p[valid][last==nt-1],minlength=nt)
        walkers.append(dict(g_d=gd,L=L,walker=w,production_sampled_loops=count,
                            both_endpoints=visited_both[-1],minimum_index=int(p.min()),maximum_index=int(p.max())))
    rates=np.asarray(pt['acceptance_rates']); T=np.asarray(meta['temperatures'])
    row=dict(g_d=gd,L=L,replicas=nt,T_min=T[0],T_max=T[-1],burn_sweeps=meta['thermalization_sweeps'],
        production_sweeps=meta['measurement_sweeps'],scalar_events=len(meta['sample_sweeps']),
        spatial_events=len(meta['spatial_sample_indices']),profile_events=len(meta['realspace_sample_indices']),
        seed=meta['seed'],exchange_seed=pt['exchange_seed'],exchange_interval=pt['exchange_interval'],
        minimum_swap_acceptance=float(rates.min()),minimum_swap_midpoint=float((T[:-1]+T[1:])[rates.argmin()]/2),
        exact_completed_loops_including_burn=int(sum(pt['walker_round_trips'])),
        exact_mean_completed_loop_sweeps=pt['mean_round_trip_sweeps'],
        production_sampled_completed_loops=sum(trips),production_walkers_with_loop=sum(np.array(trips)>0),
        production_walkers_both_endpoints=sum(visited_both),
        production_sampled_loop_mean=float(np.mean(durations)) if durations else np.nan,
        production_loop_min_per_walker=min(trips),production_loop_median_per_walker=float(np.median(trips)),
        gpu_seconds=meta['timing']['simulation_seconds'],
        nonfinite_rejections=sum(meta['sampling']['nonfinite_rejections']),
        initialization=meta['initialization'],ladder=json.dumps(meta['temperature_ladder']))
    flow=np.divide(up,up+down,out=np.full(nt,np.nan),where=(up+down)>0)
    trace=dict(pt_sweeps=ts[::10],pt_positions=ps[::10,::10],pt_walkers=np.arange(nt)[::10],
               pt_flow=flow,pt_acceptance=rates)
    return row,walkers,expected,expected_r,trace


def reduce_file(path):
    L=int(path.stem.split('=')[1]); gd=float(path.parent.name.split('gd=')[1].split('_')[0]); N=L*L
    tag=f'gd={gd:g}_L={L}'
    print('Reading',tag,flush=True)
    with np.load(path,allow_pickle=False) as z:
        info=json.loads(z['manifest'].item())['runs'][str(L)]; meta=info['metadata']
        T=np.asarray(meta['temperatures']); ss=np.asarray(meta['sample_sweeps'])
        assert np.all(ss>meta['thermalization_sweeps'])
        assert len(ss)*meta['sample_interval']==meta['measurement_sweeps']
        assert tuple(meta['spatial_fields'])==FIELDS and tuple(meta['structure_names'])==STRUCTURE_NAMES
        assert np.array_equal(ss[meta['spatial_sample_indices']],meta['spatial_sample_sweeps'])
        assert np.array_equal(ss[meta['realspace_sample_indices']],meta['realspace_sample_sweeps'])
        inventory,walkers,wm,wr,trace=pt_summary(meta,L,gd)
        def load(name):return z[f'L{L}_{name}']
        checks={}
        checks['spatial_walker_map']=bool(np.array_equal(load('spatial_walkers'),wm))
        checks['profile_walker_map']=bool(np.array_equal(load('realspace_walkers'),wr))
        assert checks['spatial_walker_map'] and checks['profile_walker_map']
        blocks={}; diagnostics=[]; spec={}; ids=np.asarray(meta['spatial_sample_indices'])
        probes=np.asarray(meta['realspace_temperature_indices']); rid=np.asarray(meta['realspace_sample_indices'])
        nscalar=BLOCK//meta['sample_interval']; nb=len(ss)//nscalar
        nsp=BLOCK//meta['spatial_interval']; nr=BLOCK//meta['realspace_interval']
        assert nb*nscalar==len(ss) and len(ids)==nb*nsp and len(rid)==nb*nr
        def block(a,step):return a.reshape((nb,step)+a.shape[1:]).mean(1)
        def add(key,a):
            assert np.isfinite(a).all(),key
            blocks[key]=block(a,nscalar)
        selected=np.unique(np.r_[probes,[np.argmin(abs(T-t)) for t in (.38,.3815,.3825,.384,.385,.387,.39)]])
        def diag(name,a):
            b=a[:,selected]; tau=autocorrelation(b)
            for j,k in enumerate(selected):
                x=b[:,j];mean=x.mean();var=x.var()
                fourth=np.mean((x-mean)**4)
                diagnostics.append(dict(g_d=gd,L=L,T=T[k],observable=name,tau_sweeps=tau[j]*meta['sample_interval'],
                    ess=len(x)/(2*tau[j]),mean=mean,excess_kurtosis=fourth/var**2-3 if var else np.nan))
        g={name:load(name) for name in info['sample_names'] if not name.startswith(('spatial','realspace'))}
        assert all(v.shape==(len(ss),len(T)) for v in g.values())
        e=g['energy']/N;n=g['density']/N;c=g['chirality']/g['density']
        add('e',e);add('e2',e*e);add('n',n);add('n2',n*n);add('chi',c);add('chi_abs',abs(c));add('chi2',c*c);add('chi4',c**4)
        pp=(g['phase_real']**2+g['phase_imag']**2)/N**2
        add('p2',pp);add('p4',pp*pp);add('p_abs',np.sqrt(pp));add('qzz',g['spin_alignment']);add('qzz2',g['spin_alignment']**2)
        for axis in ('x','y'):
            i=g['current_'+axis];k=g['curvature_'+axis]
            add('i'+axis,i);add('i'+axis+'2',i*i);add('k'+axis,k)
            add('i'+axis+'e',i*e);add('i'+axis+'2e',i*i*e);add('k'+axis+'e',k*e)
            diag('current_'+axis,i)
        add('ixiy',g['current_x']*g['current_y'])
        for name,a in [('energy',e),('abs_chirality',abs(c)),('signed_chirality',c),('spin_alignment',g['spin_alignment'])]:diag(name,a)
        for key,name in [('ix','current_x'),('iy','current_y'),('kx','curvature_x'),('ky','curvature_y'),('e','energy'),('n','density')]:
            spec[key]=g[name][ids]/(N if key in ('e','n') else 1)
        for axis in ('x','y'):blocks['sp_i'+axis+'2']=block(spec['i'+axis]**2,nsp)
        for key,v in spec.items():blocks['sp_'+key]=block(v,nsp)
        checks['spatial_energy']=float(np.max(abs(load('spatial_energy')-g['energy'][ids])))
        checks['profile_energy']=float(np.max(abs(load('realspace_energy')-g['energy'][rid][:,probes])))
        a=load('spatial_amplitudes');s=load('spatial_structure')
        assert np.isfinite(a).all() and np.isfinite(s).all()
        checks['kappa_zero_mode_per_site']=float(np.max(abs(a[...,0,0]-g['chirality'][ids]))/N)
        checks['P_zero_mode_per_site']=float(np.max(abs(a[...,1,0]-g['phase_real'][ids]-1j*g['phase_imag'][ids]))/N)
        checks['Qzz_zero_mode_per_site']=float(np.max(abs(-2*a[...,3,0]/np.sqrt(6)-(2/3)*g['density'][ids]*g['spin_alignment'][ids]))/N)
        checks['energy_parts_per_site']=float(np.max(abs(s[...,13]+s[...,14]+s[...,16]-g['energy'][ids]))/N)
        checks['energy_direct_per_site']=float(np.max(abs(s[...,17]-g['energy'][ids]))/N)
        checks['soc_identity_per_site']=float(np.max(abs(s[...,16]-gd*s[...,15]))/N)
        checks['density_per_site']=float(np.max(abs(s[...,0]-g['density'][ids]))/N)
        checks['residual_additivity_per_site']=float(np.max(abs(s[...,6]-s[...,7]-s[...,8]))/N)
        checks['chiral_weights_per_site']=float(np.max(abs(s[...,9]+s[...,10]-s[...,0]))/N)
        checks['chiral_weight_difference_per_site']=float(np.max(abs(s[...,9]-s[...,10]-a[...,0,0]))/N)
        checks['excluded_sites_max']=float(s[...,4].max());checks['nonfinite_sites_max']=float(s[...,5].max())
        for hi in (0,2,3):checks[f'hist_count_{hi}']=float(np.max(abs(s[...,20+12*hi:32+12*hi].sum(-1)-s[...,3])))
        checks['hist_density']=float(np.max(abs(s[...,32:44].sum(-1)-s[...,1]))/N)
        assert max(v for k,v in checks.items() if 'per_site' in k)<5e-6
        assert checks['spatial_energy']==0 and checks['profile_energy']==0
        assert checks['excluded_sites_max']==0 and checks['nonfinite_sites_max']==0
        blocks['amplitude']=block(a,nsp);blocks['power']=block(abs(a)**2,nsp);blocks['structure']=block(s,nsp)
        b=np.stack(((a[...,7,:]+a[...,8,:])/np.sqrt(2),(a[...,7,:]-a[...,8,:])/np.sqrt(2),np.sqrt(2)*a[...,9,:]),axis=-2)
        blocks['b_power']=block(abs(b)**2,nsp)
        blocks['b_cross']=block(np.stack([b[...,i,:]*b[...,j,:].conj() for i,j in [(0,1),(0,2),(1,2)]],axis=-2),nsp)
        checks['B_difference']=float(np.max(abs(b[...,1,:]-a[...,1,:]/np.sqrt(2)))/N)
        del b,spec,e,n,c,pp,s
        r=load('realspace_profiles');m=load('realspace_means')
        assert np.isfinite(r).all() and np.isfinite(m).all()
        assert np.array_equal(np.asarray(meta['realspace_sample_sweeps']),np.asarray(meta['spatial_sample_sweeps'])[np.searchsorted(ids,rid)])
        matched_a=a[np.searchsorted(ids,rid)][:,probes][:,:,:7,0]/N
        checks['profile_mean_zero_modes']=float(np.max(abs(m-matched_a)))
        checks['profile_r0_directions']=float(np.max(abs(r[:,:,:,0,0]-r[:,:,:,1,0])))
        checks['profile_r0_imaginary']=float(np.max(abs(r[...,0].imag)))
        checks['profile_negative_r0']=float(min(0,r[...,0].real.min()))
        blocks['profiles']=block(r,nr);blocks['profile_means']=block(m,nr)
        # Gaussian comparisons and contact tests use synchronized probe temperatures.
        meta['parallel_tempering'].pop('temperature_betas',None)
        compact={k:meta[k] for k in ('L','temperatures','seed','model_parameters','thermalization_sweeps','measurement_sweeps',
            'sample_interval','spatial_interval','realspace_interval','realspace_temperatures','realspace_temperature_indices',
            'initialization','temperature_ladder','sampler_parameters','campaign','timing')}
        compact['parallel_tempering']={k:meta['parallel_tempering'][k] for k in ('exchange_interval','exchange_seed','mean_round_trip_sweeps')}
        compact.update(block_sweeps=BLOCK,blocks=nb,checks=checks)
        del a,r,m,g,meta,info
    with zipfile.ZipFile(path) as z:crcs={x.filename:f'{x.CRC:08x}' for x in z.infolist()}
    compact['source']=dict(path=str(path.relative_to(ROOT)),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,member_crc32=crcs)
    np.savez_compressed(OUT/(tag+'.npz'),T=T,profile_T=np.asarray(compact['realspace_temperatures']),
                        profile_indices=probes,**blocks,**trace)
    (OUT/(tag+'.json')).write_text(json.dumps(compact,indent=2),encoding='utf-8')
    csv_write(OUT/(tag+'_acf.csv'),diagnostics);csv_write(OUT/(tag+'_walkers.csv'),walkers)
    print('Finished',tag,'RT',inventory['exact_mean_completed_loop_sweeps'],'swap min',inventory['minimum_swap_acceptance'],flush=True)
    gc.collect()
    return inventory


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in (ROOT/'data/weak_soc_spatial_simple_01').rglob('L=*.npz')
                 if int(p.stem[2:]) in (64,96,128,160))
    assert len(paths)==8
    audit=dict(branch=__import__('subprocess').check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if HERE not in p.parents and '.git' not in p.parts},
        datasets={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths})
    audit_path=HERE/'preservation_before.json'
    if not audit_path.exists():audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    inventory=[]
    for p in paths:inventory.append(reduce_file(p))
    csv_write(OUT/'inventory.csv',inventory)


if __name__=='__main__':main()
