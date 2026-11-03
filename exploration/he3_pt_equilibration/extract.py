"""Read manifests and raw histories, one completed archive at a time."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,csv,zipfile,hashlib,subprocess
sys.dont_write_bytecode=True  # Reusing utilities must not refresh tracked bytecode.
import numpy as np
from scipy.special import erfc
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from exploration.he3_soc.summarize import autocorrelation

def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def paths():
    for campaign,folder in [('old','data/he3'),('new','data/weak_soc_spatial_simple_01')]:
        for gd in (0,.01):
            for p in sorted((ROOT/folder/f'P=20_gd={gd:g}_gaussian').glob('L=*.npz')):
                if int(p.stem[2:])<=256:yield campaign,gd,p

def main():
    inputs=list(paths());before=HERE/'preservation_before.json'
    if not before.exists():
        before.write_text(json.dumps(dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if HERE not in p.parents and '.git' not in p.parts},
            data={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for _,_,p in inputs},
            branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    inventory=[];edges=[];acfs=[]
    for campaign,gd,path in inputs:
        L=int(path.stem[2:]);N=L*L;tag=f'{campaign}_gd={gd:g}_L={L}'
        with np.load(path) as z:
            meta=json.loads(z['manifest'].item())['runs'][str(L)]['metadata'];pt=meta['parallel_tempering'];hist=pt['walker_history']
            T=np.array(meta['temperatures']);R=len(T);sweeps=np.array(hist['sweeps']);x=np.array(hist['temperature_indices'],dtype=np.int16)
            assert np.array_equal(hist['walker_ids'],np.arange(R)) and np.all(np.sort(x,axis=1)==np.arange(R))
            ss=np.array(meta['sample_sweeps']);burn=meta['thermalization_sweeps'];interval=meta['sample_interval']
            phys_sweeps=ss[(ss%500)==0];si=np.searchsorted(ss,phys_sweeps);hi=np.searchsorted(sweeps,phys_sweeps)
            assert np.array_equal(sweeps[hi],phys_sweeps)
            pos=x[hi];nb=len(ss)//1000;blocks={};walk={};dat=dict(T=T,sweeps=sweeps,positions=x,physical_sweeps=phys_sweeps,physical_positions=pos)
            def load(name):return z[f'L{L}_{name}']
            def add(name,a):blocks[name]=a.reshape(nb,1000,R).mean(1)
            def walker(a):return np.take_along_axis(a[si],pos,axis=1).astype(np.float32)
            E=load('energy');meanE=E.mean(0);stdE=E.std(0,ddof=1);cv=stdE**2/(N*T*T)
            tc=T[np.argmax(cv)];selected=np.unique(np.r_[0,R-1,[np.argmin(abs(T-t)) for t in (T[0]+.002,tc-.0015,tc-.0005,tc-.0002,tc,tc+.0002,tc+.0005,tc+.0015,.40,.41)]])
            def diag(name,a):
                v=a[:,selected];tau=autocorrelation(v)
                for j,k in enumerate(selected):acfs.append(dict(campaign=campaign,g_d=gd,L=L,T=T[k],observable=name,tau_sweeps=tau[j]*interval,ess=len(a)/(2*tau[j])))
            diag('energy',E);add('e',E/N);add('e2',(E/N)**2);walk['energy']=walker(E/N)
            walk['energy_residual']=walker((E-meanE)/stdE)
            centers=[];counts=[]
            for j in range(R):
                hh,ee=np.histogram(E[:,j],bins=160);counts.append(hh);centers.append((ee[1:]+ee[:-1])/2)
            for j in range(R-1):
                ee=np.linspace(min(E[:,j].min(),E[:,j+1].min()),max(E[:,j].max(),E[:,j+1].max()),161)
                a=np.histogram(E[:,j],ee)[0]/len(E);b=np.histogram(E[:,j+1],ee)[0]/len(E)
                db=1/T[j]-1/T[j+1]
                prodacc=np.mean([np.exp(np.minimum(0,db*(E[:,j]-np.roll(E[:,j+1],shift)))).mean() for shift in (10003,20011,30007)])
                edges.append(dict(campaign=campaign,g_d=gd,L=L,edge=j,T_left=T[j],T_right=T[j+1],acceptance=pt['acceptance_rates'][j],product_hist_acceptance=prodacc,
                    simultaneous_sample_acceptance=np.exp(np.minimum(0,db*(E[:,j]-E[:,j+1]))).mean(),gaussian_acceptance=erfc(db*np.sqrt((stdE[j]**2+stdE[j+1]**2)/2)/2),overlap=np.minimum(a,b).sum()))
            dat.update(energy_mean=meanE,energy_sigma=stdE,Cv=cv,energy_bin_centers=np.array(centers),energy_hist=np.array(counts),acceptance=np.array(pt['acceptance_rates']))
            del E
            density=load('density');c=load('chirality')/density
            add('n',density/N);add('m',c);add('abs_m',abs(c));add('m2',c*c);add('m4',c**4)
            for name,a in [('signed_chirality',c),('abs_chirality',abs(c)),('m2',c*c)]:diag(name,a)
            M2=(c*c).mean(0);M4=(c**4).mean(0)
            diag('Binder_influence',-(c**4)/(3*M2**2)+2*M4*c*c/(3*M2**3))
            walk['chirality']=walker(c);walk['abs_m_residual']=walker((abs(c)-abs(c).mean(0))/abs(c).std(0))
            del c,density
            pr=load('phase_real')/N;pi=load('phase_imag')/N;p2=pr*pr+pi*pi
            for key,a in [('p_real',pr),('p_imag',pi),('p2',p2)]:add(key,a)
            diag('P_real',pr);diag('P2',p2);walk['P_real']=walker(pr);walk['P_imag']=walker(pi);walk['P2_residual']=walker((p2-p2.mean(0))/p2.std(0))
            del pr,pi,p2
            q=load('spin_alignment');add('qzz',q);add('qzz2',q*q);diag('spin_alignment',q);walk['qzz']=walker(q);walk['qzz_residual']=walker((q-q.mean(0))/q.std(0));del q
            influence=np.zeros((len(ss),len(selected)))
            for axis in ('x','y'):
                i=load('current_'+axis);k=load('curvature_'+axis)
                add('i'+axis,i);add('i'+axis+'2',i*i);add('k'+axis,k)
                influence+=(k[:,selected]-i[:,selected]**2/T[selected]+2*i.mean(0)[selected]*i[:,selected]/T[selected])/(2*N)
                del i,k
            tau=autocorrelation(influence)
            for j,k in enumerate(selected):acfs.append(dict(campaign=campaign,g_d=gd,L=L,T=T[k],observable='helicity_influence',tau_sweeps=tau[j]*interval,ess=len(ss)/(2*tau[j])))
            if campaign=='new':
                amp=load('spatial_amplitudes');sp_s=np.array(meta['spatial_sample_sweeps']);spi=np.searchsorted(sweeps,sp_s);sp_pos=x[spi]
                assert np.array_equal(sp_s,sweeps[spi]) and np.array_equal(np.argsort(sp_pos,axis=1),load('spatial_walkers'))
                qv=amp[...,2:7,0].real/N
                dat['Q_sweeps']=sp_s;dat['Q_walkers']=np.take_along_axis(qv,sp_pos[...,None],axis=1).astype(np.float32)
                # Keep low Fourier powers for independent spin/composite checks.
                dat['spatial_power_blocks']=(abs(amp[:,:,:7,:])**2).reshape(nb,50,R,7,5).mean(1)
                del amp,qv
            dat.update({'block_'+k:v for k,v in blocks.items()});dat.update({'walker_'+k:v for k,v in walk.items()})
            np.savez_compressed(OUT/(tag+'.npz'),**dat)
            sampling=meta['sampling'];ar=np.array(sampling['acceptance_rates'])
            with zipfile.ZipFile(path) as zz:rawbytes=sum(m.file_size for m in zz.infolist());crcs={m.filename:f'{m.CRC:08x}' for m in zz.infolist()}
            row=dict(tag=tag,campaign=campaign,campaign_id=meta.get('campaign',{}).get('id',''),g_d=gd,L=L,R=R,T_min=T[0],T_max=T[-1],burn=burn,production=meta['measurement_sweeps'],sample_interval=interval,
                history_interval=int(np.median(np.diff(sweeps))),exchange_interval=pt['exchange_interval'],gpu_seconds=meta['timing']['simulation_seconds'],file_bytes=path.stat().st_size,raw_bytes=rawbytes,
                acceptance_min=float(dat['acceptance'].min()),acceptance_mean=float(dat['acceptance'].mean()),acceptance_T=float((T[:-1]+T[1:])[np.argmin(dat['acceptance'])]/2),
                exact_trips=int(sum(pt['walker_round_trips'])),exact_walkers_with_trips=int(np.count_nonzero(pt['walker_round_trips'])),completed_trip_mean=pt['mean_round_trip_sweeps'],
                gaussian_acceptance_mean=float(ar.mean()),gaussian_acceptance_min=float(ar.min()),gaussian_acceptance_max=float(ar.max()),rho=sampling['rho'],mass=sampling['proposal_mass'],Cv_peak=float(cv.max()),Cv_peak_T=tc,
                source=str(path.relative_to(ROOT)),initialization=meta.get('initialization','not recorded'))
            inventory.append(row)
            compact=dict(row=row,model=meta['model_parameters'],ladder=meta['temperature_ladder'],seed=meta['seed'],exchange_seed=pt['exchange_seed'],member_crc32=crcs,
                exact_trips=pt['walker_round_trips'],exact_trip_sums=pt['walker_round_trip_total_sweeps'],gaussian_acceptance_by_physical_walker=ar.tolist())
            (OUT/(tag+'.json')).write_text(json.dumps(compact,indent=2))
            print(tag,'R',R,'trip',round(row['completed_trip_mean']),'Amin',round(row['acceptance_min'],3),flush=True)
    write('inventory.csv',inventory);write('energy_edges.csv',edges);write('fixed_temperature_acf.csv',acfs)

if __name__=='__main__':main()
