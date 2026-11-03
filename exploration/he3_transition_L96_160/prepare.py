"""Read only six latest archives; retain 50k-sweep blocks and local histograms."""
from pathlib import Path
import json,sys,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';OUT.mkdir(exist_ok=True)
SIZES=(96,128,160)
KEYS=('chi2','chi4','ix','ix2','kx','iy','iy2','ky')


def main():
    inventory=[]
    for gd in (0,.01):
        for L in SIZES:
            tag=f'gd={gd:g}_L={L}';path=ROOT/f'data/weak_soc_spatial_simple_01/P=20_gd={gd:g}_gaussian/L={L}.npz'
            prior=ROOT/f'exploration/he3_weak_soc_spatial_01/results/{tag}.json'
            meta=json.loads(prior.read_text())
            with zipfile.ZipFile(path) as z:crc={x.filename:f'{x.CRC:08x}' for x in z.infolist()}
            assert crc==meta['source']['member_crc32']
            T=np.array(meta['temperatures']);ns=60000;nb=60;step=1000
            windows={name:np.arange(np.argmin(abs(T-c))-half,np.argmin(abs(T-c))+half+1)
                for name,c,half in [('critical',.3818 if gd==0 else .3856,6),('integer',.3836 if gd==0 else .3894,5)]}
            data={'T':T};stored={}
            with np.load(path) as z:
                E=z[f'L{L}_energy']
                for name,ids in windows.items():
                    e=E[:,ids];ref=e.mean();edges=np.linspace(e.min()-1e-6,e.max()+1e-6,801)
                    bins=np.searchsorted(edges,e,side='right')-1
                    H=np.zeros((nb,len(ids),800));S=np.zeros((nb,8,800));src=np.zeros((len(ids),8,800))
                    for b in range(nb):
                        for j in range(len(ids)):H[b,j]=np.bincount(bins[b*step:(b+1)*step,j],minlength=800)
                    stored[name]=(ids,bins,H,S,src)
                    data[name+'_energy']=((edges[1:]+edges[:-1])/2-ref);data[name+'_beta']=(1/T[ids]).astype(np.float32).astype(float)
                    data[name+'_T']=T[ids]
                del E
                c=z[f'L{L}_chirality']/z[f'L{L}_density']
                for k,key in enumerate(KEYS):
                    if key=='chi2':a=c*c
                    elif key=='chi4':a=c**4
                    else:
                        raw={'ix':'current_x','ix2':'current_x','kx':'curvature_x','iy':'current_y','iy2':'current_y','ky':'curvature_y'}[key]
                        a=z[f'L{L}_{raw}']
                        if key.endswith('2'):a=a*a
                    assert a.shape==(ns,len(T)) and np.isfinite(a).all()
                    data[key]=a.reshape(nb,step,len(T)).mean(1)
                    for name,(ids,bins,H,S,src) in stored.items():
                        v=a[:,ids]
                        for b in range(nb):
                            sl=slice(b*step,(b+1)*step);S[b,k]=np.bincount(bins[sl].ravel(),weights=v[sl].ravel(),minlength=800)
                        for j in range(len(ids)):src[j,k]=np.bincount(bins[:,j],weights=v[:,j],minlength=800)
                    del a
            with np.load(ROOT/f'exploration/he3_weak_soc_spatial_01/results/{tag}.npz') as old:
                differences={k:float(np.max(abs(data[k]-old[k]))) for k in KEYS}
            assert max(differences.values())<1e-8
            for name,(ids,bins,H,S,src) in stored.items():data[name+'_H']=H;data[name+'_S']=S;data[name+'_src']=src
            np.savez_compressed(OUT/(tag+'.npz'),**data)
            inventory.append(dict(g_d=gd,L=L,source=str(path.relative_to(ROOT)),size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,crc_verified=True,raw_cache_max_errors=differences,
                production_sweeps=meta['measurement_sweeps'],burn=meta['thermalization_sweeps'],sample_interval=meta['sample_interval']))
            print('Prepared',tag,flush=True)
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')


if __name__=='__main__':main()
