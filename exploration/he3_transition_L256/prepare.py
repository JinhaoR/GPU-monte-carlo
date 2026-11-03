"""Reduce only the new L256 run, reusing validated spatial/PT reduction."""
from pathlib import Path
import json,sys,zipfile,hashlib,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from exploration.he3_weak_soc_spatial_01 import reduce as spatial_reduce
L=256
KEYS=('chi2','chi4','ix','ix2','kx','iy','iy2','ky')

def main():
    path=ROOT/'data/weak_soc_spatial_simple_01/P=20_gd=0_gaussian/L=256.npz'
    audit=HERE/'preservation_before.json'
    if not audit.exists():
        audit.write_text(json.dumps(dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if HERE not in p.parents and '.git' not in p.parts},branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    spatial_reduce.OUT=OUT/'spatial'
    if (spatial_reduce.OUT/'gd=0_L=256.npz').exists():
        prior=json.loads((spatial_reduce.OUT/'gd=0_L=256.json').read_text())
        with zipfile.ZipFile(path) as z:assert {x.filename:f'{x.CRC:08x}' for x in z.infolist()}==prior['source']['member_crc32']
        with np.load(path) as z:md=json.loads(z['manifest'].item())['runs']['256']['metadata']
        pt=spatial_reduce.pt_summary(md,256,0)[0]
    else:pt=spatial_reduce.reduce_file(path)
    (OUT/'pt_summary.json').write_text(json.dumps(pt,indent=2,default=lambda x:x.item()))
    with np.load(path) as z:
        meta=json.loads(z['manifest'].item())['runs']['256']['metadata'];T=np.array(meta['temperatures'])
        old=json.loads((ROOT/'exploration/he3_weak_soc_spatial_01/results/gd=0_L=160.json').read_text())
        assert meta['model_parameters']==old['model_parameters'] and np.array_equal(T,old['temperatures'])
        assert meta['sample_interval']==50 and meta['measurement_sweeps']==3_000_000
        with np.load(OUT/'spatial/gd=0_L=256.npz') as b:data={k:b[k] for k in KEYS}
        data['T']=T; nb=60; step=1000; stored={}
        E=z['L256_energy']
        for name,c,half in [('critical',.3818,6),('integer',.3836,5)]:
            ids=np.arange(np.argmin(abs(T-c))-half,np.argmin(abs(T-c))+half+1)
            e=E[:,ids];ref=e.mean();edges=np.linspace(e.min()-1e-6,e.max()+1e-6,801)
            bins=np.searchsorted(edges,e,side='right')-1; H=np.zeros((nb,len(ids),800));S=np.zeros((nb,8,800));src=np.zeros((len(ids),8,800))
            for b in range(nb):
                for j in range(len(ids)):H[b,j]=np.bincount(bins[b*step:(b+1)*step,j],minlength=800)
            stored[name]=(ids,bins,H,S,src)
            data[name+'_energy']=(edges[1:]+edges[:-1])/2-ref
            data[name+'_beta']=np.array(meta['parallel_tempering']['temperature_betas'])[ids]
            data[name+'_T']=T[ids]
        del E
        c=z['L256_chirality']/z['L256_density']
        for k,key in enumerate(KEYS):
            if key=='chi2':a=c*c
            elif key=='chi4':a=c**4
            else:
                raw={'ix':'current_x','ix2':'current_x','kx':'curvature_x','iy':'current_y','iy2':'current_y','ky':'curvature_y'}[key]
                a=z['L256_'+raw]
                if key.endswith('2'):a=a*a
            np.testing.assert_allclose(a.reshape(nb,step,len(T)).mean(1),data[key],rtol=0,atol=1e-8)
            for name,(ids,bins,H,S,src) in stored.items():
                v=a[:,ids]
                for b in range(nb):
                    sl=slice(b*step,(b+1)*step);S[b,k]=np.bincount(bins[sl].ravel(),weights=v[sl].ravel(),minlength=800)
                for j in range(len(ids)):src[j,k]=np.bincount(bins[:,j],weights=v[:,j],minlength=800)
            del a
    for name,(ids,bins,H,S,src) in stored.items():data[name+'_H']=H;data[name+'_S']=S;data[name+'_src']=src
    np.savez_compressed(OUT/'gd=0_L=256.npz',**data)
    print('Histogram preparation complete',flush=True)

if __name__=='__main__':main()
