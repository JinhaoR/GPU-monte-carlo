"""Compact decomposition of raw thermal manifold failure; no extra sampling."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'exploration/he3_hqv_stage1'))
from defects import local_geometry,analyze_loop,rectangle_loop,Thresholds


def main():
    rows=[];wide=[]
    for run in OUT.glob('pilot_gd=*/run.json'):
        rm=json.loads(run.read_text())
        for p in sorted(run.parent.glob('snapshot_*.npz')):
            with np.load(p,allow_pickle=False) as z:
                meta=json.loads(z['metadata'].item());fields=z['fields']
            for t,A in zip(meta['temperatures'],fields):
                g=local_geometry(A)
                np.testing.assert_allclose(g['residual'],g['wrong_chirality']+g['nonreal_spin'],atol=3e-7)
                rows.append(dict(g_d=rm['g_d'],temperature=t,sweep=meta['sweep'],
                    residual_mean=float(g['residual'].mean()),
                    wrong_orbital_weight_mean=float(g['wrong_chirality'].mean()),
                    nonreal_spin_penalty_mean=float(g['nonreal_spin'].mean()),
                    minimum_density=float(g['density'].min()),
                    density_floor_site_fraction=float(np.mean(g['density']<=.01)),
                    chi_mean=float(g['chirality'].mean())))
    # Wider loops on the representative smoothed candidate regions explicitly
    # test whether the default seed-neighborhood size alone caused rejection.
    from thermal_analysis import smooth
    detail=json.loads((OUT/'representative_candidates.json').read_text())
    path=OUT/'pilot_gd=0_L=64/snapshot_090000.npz'
    with np.load(path) as z:
        meta=json.loads(z['metadata'].item());fields=z['fields']
    for target in [.375,.3815]:
        j=meta['temperatures'].index(target);A=smooth(fields[j],8);g=local_geometry(A)
        record=next(r for r in detail if r['T']==target and r['setting']=='smooth8_proxy')
        for index,candidate in enumerate(record['candidates']):
            i=int(np.floor(candidate['center_x']));k=int(np.floor(candidate['center_y']))
            for radius in [1,2,4,8,12]:
                res=analyze_loop(A,rectangle_loop(i-radius,i+radius+1,k-radius,k+radius+1),geometry=g,periodic=True)
                wide.append(dict(temperature=target,candidate=index,x=candidate['center_x'],y=candidate['center_y'],
                    radius=radius,accepted=res['accepted'],charge=res['phase_charge'],
                    composite_winding=res['composite_winding'],director_parity=res['director_parity'],
                    max_residual=res['max_residual'],reasons=';'.join(res['reasons'])))
    for name,rr in [('thermal_geometry.csv',rows),('wider_candidate_loops.csv',wide)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    print('Decomposition checks passed;',len(wide),'candidate/loop sizes, accepted',sum(r['accepted'] for r in wide))


if __name__=='__main__':main()
