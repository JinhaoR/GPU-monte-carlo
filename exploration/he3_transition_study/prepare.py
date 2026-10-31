"""Read-only, streaming preparation for local energy-histogram reweighting.

python -B exploration/he3_transition_study/prepare.py
One original file/channel is decompressed at a time. 30 joint PT time blocks,
15 neighboring temperatures and full extensive energies are retained as binned
counts and sums of observable moments. No trajectory averaging before binning.
"""
import csv
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results';CACHE=OUT/'cache'
COUPLINGS=(0,.001,.01);SIZES=(64,96,128,160);BLOCK=2000;BINS=1600
NAMES=('dE','dE2','density','m','absm','m2','m4','c','absc','c2','c4',
       'p','p2','p4','ix','ix2','kx','iy','iy2','ky','qzz')


def crc(path):
    with zipfile.ZipFile(path) as z:return ';'.join(f'{i.filename}:{i.CRC:08x}' for i in z.infolist())


def main():
    CACHE.mkdir(parents=True,exist_ok=True);inventory=[]
    for gd in COUPLINGS:
        for L in SIZES:
            source=ROOT/f'data/he3/P=20_gd={gd:g}_gaussian/L={L}.npz'
            cached=CACHE/f'gd={gd:g}_L={L}.npz';metapath=CACHE/f'gd={gd:g}_L={L}.json'
            fingerprint=crc(source)
            if cached.exists() and metapath.exists():
                old=json.loads(metapath.read_text())
                if old['source_crc32']==fingerprint:inventory.append(old);print('Cached',gd,L,flush=True);continue
            print('Preparing',gd,L,flush=True)
            with np.load(source,allow_pickle=False) as z:
                info=json.loads(z['manifest'].item())['runs'][str(L)];meta=info['metadata']
                T=np.array(meta['temperatures']);center=int(np.argmin(abs(T-(.3854363 if gd==.01 else .3822107))))
                ids=np.arange(center-7,center+8);K=len(ids);hold=np.array([6,7,8]);nt=len(meta['sample_sweeps']);nb=nt//BLOCK
                assert nt==60000 and nb==30
                def get(name):return z[f'L{L}_{name}'][:,ids].copy()
                E=get('energy');eref=float(E.mean());lo=float(E.min());hi=float(E.max())
                edges=np.linspace(lo-1e-7,hi+1e-7,BINS+1);ebins=(edges[:-1]+edges[1:])/2-eref
                bindex=np.clip(np.searchsorted(edges,E,side='right')-1,0,BINS-1)
                H=np.zeros((nb,K,BINS),dtype=np.int32)
                for b in range(nb):
                    for k in range(K):H[b,k]=np.bincount(bindex[b*BLOCK:(b+1)*BLOCK,k],minlength=BINS)
                S=np.zeros((len(NAMES),nb,BINS));Ssrc=np.zeros((len(NAMES),K,BINS))
                Shold=np.zeros((len(hold),len(NAMES),nb,BINS))
                def add(name,values):
                    m=NAMES.index(name)
                    assert values.shape==(nt,K) and np.all(np.isfinite(values))
                    for b in range(nb):
                        sl=slice(b*BLOCK,(b+1)*BLOCK)
                        S[m,b]=np.bincount(bindex[sl].ravel(),weights=values[sl].ravel(),minlength=BINS)
                        for h,k in enumerate(hold):Shold[h,m,b]=np.bincount(bindex[sl,k],weights=values[sl,k],minlength=BINS)
                    for k in range(K):Ssrc[m,k]=np.bincount(bindex[:,k],weights=values[:,k],minlength=BINS)
                add('dE',E-eref);add('dE2',(E-eref)**2);del E
                D=get('density');add('density',D/L**2)
                C=get('chirality');m=C/D;c=C/L**2
                for name,a in [('m',m),('absm',abs(m)),('m2',m*m),('m4',m**4),('c',c),('absc',abs(c)),('c2',c*c),('c4',c**4)]:add(name,a)
                del D,C,m,c
                pr=get('phase_real')/L**2;pi=get('phase_imag')/L**2;p2=pr*pr+pi*pi;del pr,pi
                add('p',np.sqrt(p2));add('p2',p2);add('p4',p2*p2);del p2
                for axis in ['x','y']:
                    I=get('current_'+axis);add('i'+axis,I);add('i'+axis+'2',I*I);del I
                    a=get('curvature_'+axis);add('k'+axis,a);del a
                a=get('spin_alignment');add('qzz',a);del a
                beta=np.array(meta['parallel_tempering']['temperature_betas'])[ids]
                np.savez_compressed(cached,T=T[ids],beta=beta,eref=eref,ebins=ebins,edges=edges,
                    H=H,S=S,Ssrc=Ssrc,Shold=Shold,hold=hold,names=np.array(NAMES))
                pt=meta['parallel_tempering']
                record=dict(g_d=gd,L=L,source=str(source.relative_to(ROOT)),source_crc32=fingerprint,
                    source_bytes=source.stat().st_size,selected_temperature_indices=ids.tolist(),
                    selected_temperatures=T[ids].tolist(),measurements=nt,block_measurements=BLOCK,bins=BINS,
                    thermalization_sweeps=meta['thermalization_sweeps'],measurement_sweeps=meta['measurement_sweeps'],
                    sample_interval=meta['sample_interval'],exchange_interval=pt['exchange_interval'],
                    seed=meta['seed'],exchange_seed=pt['exchange_seed'],model_parameters=meta['model_parameters'],
                    exact_round_trips=sum(pt['walker_round_trips']),acceptance=sum(pt['accepted'])/sum(pt['attempts']))
                metapath.write_text(json.dumps(record,indent=2),encoding='utf-8');inventory.append(record)
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('Prepared',len(inventory),'read-only datasets',flush=True)


if __name__=='__main__':main()
