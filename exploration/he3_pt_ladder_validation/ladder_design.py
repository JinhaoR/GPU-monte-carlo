"""Freeze three comparison ladders and predict overlap; no GPU work."""
import sys,os,json,csv,hashlib,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'exploration/he3_pt_equilibration';OUT=HERE/'results'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(OLD))
from exploration.he3_pt_equilibration.design import density_grid,predicted_acceptance
from exploration.he3_pt_equilibration.transport import null_transport
from utils.temperature_ladder import make_temperature_ladder

REFERENCE=dict(T_min=.37,T_max=.42,n_T=200,Tc=.384,tc_window=.008,tc_fraction=.25)

def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    audit=HERE/'preservation_before.json'
    if not audit.exists():
        inv=list(csv.DictReader((OLD/'results/inventory.csv').open()))
        audit.write_text(json.dumps(dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if HERE not in p.parents and '.git' not in p.parts},
            datasets={m['source']:dict(size=(ROOT/m['source']).stat().st_size,mtime_ns=(ROOT/m['source']).stat().st_mtime_ns) for m in inv},
            branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    with np.load(OLD/'results/design_profiles.npz') as z:profile=dict(z)
    grid=profile['T'];peaks=list(csv.DictReader((OLD/'results/refined_energy_peaks.csv').open()))
    # Test the same common density proposed for L320, rather than changing
    # placement separately for every pilot size. Peak-location allowance and
    # energy uncertainty are already in the previous envelope.
    cv={}
    for g,p in zip((0,.01),peaks):
        tc=float(p['reweighted_peak_T']);growth=np.exp(-((grid-tc)/.0015)**4)
        cv[g]=profile[f'envelope_{g:g}']*(1+((320/256)**2-1)*growth)
    density=np.maximum.reduce([np.sqrt(c)/grid for c in cv.values()])
    ladders={'A':make_temperature_ladder(**REFERENCE),
             'B':density_grid(grid,density,200,.375,.405),
             'C':density_grid(grid,density,160,.375,.405)}
    for name,T in ladders.items():
        assert np.all(np.diff(T)>0) and np.all(np.diff((1/T).astype(np.float32))<0)
        np.savetxt(HERE/f'ladder_{name}.txt',T,fmt='%.12f')
    np.testing.assert_allclose(ladders['C'],np.loadtxt(OLD/'results/narrow_energy_R160_candidate.txt'),atol=6e-13,rtol=0)
    rows=[]
    for L in (96,128,160,256,320,384):
        for g in (0,.01):
            if L<=256:
                with np.load(OLD/f'results/new_gd={g:g}_L={L}.npz') as z:C=np.interp(grid,z['T'],z['Cv'])
                scenarios={'measured_Cv':C}
            else:
                base=profile[f'envelope_{g:g}'];tc=float(peaks[0 if g==0 else 1]['reweighted_peak_T'])
                scenarios={'fixed_Cv':base,'growing_Cv':base*(1+((L/256)**2-1)*np.exp(-((grid-tc)/.0015)**4))}
            for scenario,C in scenarios.items():
                for name,T in ladders.items():
                    a=predicted_acceptance(T,grid,C,L);_,rt,_=null_transport(a,5)
                    rows.append(dict(L=L,g_d=g,scenario=scenario,ladder=name,R=len(T),T_min=T[0],T_max=T[-1],min_spacing=np.diff(T).min(),min_acceptance=a.min(),null_RT=rt))
    write('offline_predictions.csv',rows)
    (OUT/'design.json').write_text(json.dumps(dict(reference=REFERENCE,method='Common L320 stress-envelope energy density max_g sqrt(Cv)/T; no flow term in GPU comparisons',
        distinction='A vs B changes range and placement at fixed R; B vs C isolates R at fixed density/range',
        ladders={k:dict(R=len(t),minimum_spacing=float(np.diff(t).min()),range=[t[0],t[-1]]) for k,t in ladders.items()}),indent=2))
    print('A/B/C ladders frozen; offline predictions and preservation baseline saved.')

if __name__=='__main__':main()
