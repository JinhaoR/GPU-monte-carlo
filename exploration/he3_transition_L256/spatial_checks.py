"""Existing spatial estimators plus critical-region ACF and time-block checks."""
from pathlib import Path
import sys,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'exploration/he3_weak_soc_spatial_01'))
from exploration.he3_weak_soc_spatial_01 import analyze as spatial
from exploration.he3_soc.summarize import autocorrelation

def main():
    with np.load(OUT/'spatial/gd=0_L=256.npz') as z:
        T=z['T'];pt=z['profile_T'];blocks={k:z[k] for k in z.files if k not in ('T','profile_T','profile_indices') and not k.startswith('pt_')}
    v,e,jk=spatial.estimate(blocks,T,256)
    np.savez_compressed(OUT/'spatial_summary.npz',T=T,profile_T=pt,**v,**{k+'_se':x for k,x in e.items()})
    keys=['helicity','chirality','binder_chirality','epsilon','orbital_loss','spin_loss','energy_bulk','energy_gradient','D','S','density','phase']
    rows=[]
    for j,t in enumerate(T):
        row={'T':t}
        for k in keys:row[k]=v[k][j];row[k+'_se']=e[k][j]
        for c,name in enumerate(spatial.NAMES):
            row['xi_'+name]=v['full_xi_xy'][j,c];row['xi_'+name+'_se']=e['full_xi_xy'][j,c]
        rows.append(row)
    spatial.csv_write(OUT/'spatial_curves.csv',rows)
    path=ROOT/'data/weak_soc_spatial_simple_01/P=20_gd=0_gaussian/L=256.npz'
    selected=np.where((T>.3813)&(T<.3823))[0];acf=[]
    with np.load(path) as z:
        c=z['L256_chirality'][:,selected]/z['L256_density'][:,selected]
        arrays={'signed_chirality':c,'abs_chirality':abs(c),'m2':c*c,'m4':c**4,'energy':z['L256_energy'][:,selected]}
        for name,a in arrays.items():
            tau=autocorrelation(a)
            for j,k in enumerate(selected):acf.append(dict(observable=name,T=T[k],tau_sweeps=tau[j]*50,ess=len(a)/(2*tau[j])))
    spatial.csv_write(OUT/'critical_acf.csv',acf)
    # Scalar block trends use the original 50k-sweep sufficient statistics.
    trend=[]
    for b in range(60):
        for k in selected:
            m={name:x[b,k] for name,x in blocks.items() if x.shape==(60,len(T))}
            trend.append(dict(sweep=(b+1)*50000,T=T[k],energy=m['e'],m2=m['chi2'],m4=m['chi4'],signed_chirality=m['chi'],abs_chirality=m['chi_abs'],Y=(m['kx']+m['ky']-((m['ix2']-m['ix']**2)+(m['iy2']-m['iy']**2))/T[k])/(2*256**2)))
    spatial.csv_write(OUT/'time_blocks.csv',trend)
    print('Spatial summary and critical ACF complete')

if __name__=='__main__':main()
