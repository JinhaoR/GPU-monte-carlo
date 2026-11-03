"""Check split-run sensitivity using primitive moments, and audit preserved inputs.

No GPU calls. The small saved moment blocks also make the split checks reproducible
without repeatedly decompressing the original archives.
"""
import json
import csv
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from analyze import ROOT, HERE, OUT, DATA, PILOT, moments, interpolate, estimate


def main():
    rows=[];burns=[]
    inventory=list(csv.DictReader((OUT/'summary.csv').open()))
    for case in inventory:
        tag=case['case'];g=float(case['g_d']);T=np.loadtxt(PILOT/f"ladder_{case['ladder']}.txt")
        tc=.38210530380170976 if g==0 else .3856721419201371
        probes=np.array([.375,tc-.0003,tc,tc+.0003,.402,.405])
        path=OUT/(tag+'_moments.npz')
        with np.load(DATA/(tag+'.npz')) as z:
            a={k:z['L96_'+k] for k in ('energy','density','chirality','phase_real','phase_imag',
               'spin_alignment','current_x','current_y','curvature_x','curvature_y')}
        m=moments(a,96)
        b={k:interpolate(v.reshape(300,100,-1).mean(1),T,probes) for k,v in m.items()}
        del a,m
        np.savez_compressed(path,T=probes,**b)
        prod={k:v[100:] for k,v in b.items()}
        base=dict(case=tag,g_d=g,ladder=case['ladder'],initialization=case['initialization'])
        for bs in (5000,25000,50000,100000,250000):
            v,e=estimate({k:x[:100] for k,x in prod.items()},probes,96,bs//5000)
            w,f=estimate({k:x[100:] for k,x in prod.items()},probes,96,bs//5000)
            for key in v:
                for j,t in enumerate(probes):
                    err=np.hypot(e[key][j],f[key][j]);diff=w[key][j]-v[key][j]
                    rows.append(dict(**base,T=t,observable=key,block_sweeps=bs,blocks_per_half=500000//bs,
                        first=v[key][j],second=w[key][j],difference=diff,se=err,z=diff/err))
        for burn in (250000,500000,750000,1000000):
            v,e=estimate({k:x[burn//5000:] for k,x in b.items()},probes,96,10)
            for key in v:
                for j,t in enumerate(probes):burns.append(dict(**base,burn=burn,T=t,observable=key,value=v[key][j],se=e[key][j]))
        print(tag,'split-run sensitivity checked',flush=True)
    for name,data in [('split_sensitivity',rows),('burn_sensitivity',burns)]:
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Independent reduction agrees with the 100k checks in the main analysis.
    prior=list(csv.DictReader((OUT/'half_contrasts.csv').open()))
    for r in rows:
        if r['block_sweeps']!=100000:continue
        q=next(q for q in prior if q['case']==r['case'] and float(q['T'])==r['T'] and q['observable']==r['observable'])
        np.testing.assert_allclose([r['difference'],r['se']],[float(q['difference']),float(q['se'])],rtol=1e-8,atol=1e-11)
    checks=list(csv.DictReader((OUT/'validation.csv').open()))
    assert len(checks)==12
    for c in checks:
        path=DATA/(c['case']+'.npz')
        assert path.stat().st_size==int(c['source_bytes']) and path.stat().st_mtime_ns==int(c['source_mtime_ns'])
        assert c['passed']=='True'
    old=json.loads((PILOT/'preservation_before.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in old['sources'].items())
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
    record=dict(cases=12,all_raw_validation_passed=True,independent_moment_reduction_agrees=True,
                inputs_size_mtime_unchanged=True,previous_python_files_unchanged=len(old['sources']),tracked_diff_empty=True,
                no_GPU_or_simulation_calls=True)
    (OUT/'audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
