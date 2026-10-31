"""Apply unchanged Stage-1 geometry/loops to pilot fields; no MC is launched.

Run: python -B exploration/he3_hqv_stage2/thermal_analysis.py
Primary counts use raw fields. Full-grid heat-kernel smoothing is a separately
labelled exploratory proxy, never a replacement for saved raw configurations.
"""
from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'exploration/he3_hqv_stage1'))
from defects import local_geometry,find_candidates,Thresholds,plaquette_map,analyze_loop,rectangle_loop
from synthetic import make_cases,hqv_pair,add_noise


def smooth(A,passes):
    """Periodic full-grid linear heat kernel; per-axis variance passes/4."""
    A=np.asarray(A,dtype=np.complex128).copy()
    for _ in range(passes):A=.5*A+.125*(np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1))
    return A


def settings():
    return [('raw_strict',0,Thresholds()),
        ('raw_eps020',0,replace(Thresholds(),max_residual=.2,min_chirality=.6,min_gap=.6,min_composite_ratio=.6)),
        ('raw_eps030',0,replace(Thresholds(),max_residual=.3,min_chirality=.4,min_gap=.4,min_composite_ratio=.4)),
        ('smooth4_proxy',4,Thresholds()),('smooth8_proxy',8,Thresholds())]


def valid_sites(g,th):
    return ((g['density']>th.min_density)&(g['residual']<=th.max_residual)&
        (abs(g['chirality'])>=th.min_chirality)&(g['gap']>=th.min_gap)&
        (g['composite_ratio']>=th.min_composite_ratio)&(g['projected_ratio']>=th.min_composite_ratio))


def savecsv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def calibration():
    rows=[]
    for name in ['uniform','integer_pair','hqv_pair','periodic_hqv_pair','composite_zero','nonunitary_high_chirality']:
        case=make_cases(64)[name]
        for passes in [0,4,8]:
            # Only periodic input is rolled in the thermal analysis. Open plane
            # calibration charges are checked away from the boundary seam.
            A=smooth(case['field'],passes)
            found=find_candidates(A,periodic=case['boundary']=='periodic')
            accepted=[r['phase_charge'] for r in found if r['accepted']]
            expected=sorted(case['expected_charges'])
            assert sorted(accepted)==expected,(name,passes,accepted,expected)
            rows.append(dict(case=name,passes=passes,accepted_charges=str(sorted(accepted)),expected=str(expected),passed=True))
    # Demonstrate smoothing's loss of spatial information, not guaranteed recovery.
    for passes in [0,4,8]:
        A=smooth(hqv_pair(64,((30.5,31.5),(32.5,31.5))),passes)
        found=find_candidates(A,periodic=False)
        rows.append(dict(case='close_pair_separation2',passes=passes,
            accepted_charges=str([r['phase_charge'] for r in found if r['accepted']]),
            expected='two underlying synthetic cores; counting may fail',passed=True))
    savecsv('smoothing_calibration.csv',rows)
    return rows


def main():
    started=perf_counter();calibration_rows=calibration()
    rows=[];rejections=[];candidates=[];details=[];saved_fields={}
    for run in sorted(OUT.glob('pilot_gd=*/run.json')):
        runmeta=json.loads(run.read_text());gd=runmeta['g_d']
        for path in sorted(run.parent.glob('snapshot_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                fields=z['fields'];meta=json.loads(z['metadata'].item())
            for index,(T,A) in enumerate(zip(meta['temperatures'],fields)):
                for label,passes,th in settings():
                    B=A if passes==0 else smooth(A,passes)
                    g=local_geometry(B);valid=valid_sites(g,th);maps=plaquette_map(B,g,periodic=True)
                    found=find_candidates(B,periodic=True,thresholds=th)
                    good=[r for r in found if r['accepted']]
                    # Count each loop reason, retaining overlap among failure types.
                    reason=Counter(v for r in found for loop in r['loops'] for v in loop.get('reasons',[]))
                    attempted=sum(len(r['loops']) for r in found)
                    invalid=sum(not q['accepted'] for r in found for q in r['loops'])
                    row=dict(g_d=gd,temperature=T,sweep=meta['sweep'],production_sweep=meta['production_relative_sweep'],
                        walker=meta['physical_walkers'][index],setting=label,smoothing_passes=passes,
                        snapshot=str(path.relative_to(ROOT)),saved_index=index,
                        hqv_positive=sum(r['phase_charge']==.5 for r in good),
                        hqv_negative=sum(r['phase_charge']==-.5 for r in good),
                        integer_net=sum(abs(r['phase_charge'])==1 for r in good),
                        higher_net=sum(abs(r['phase_charge'])>1 for r in good),
                        accepted_clusters=len(good),unresolved_clusters=len(found)-len(good),
                        raw_seed_plaquettes=int(maps['candidate'].sum()),
                        invalid_site_fraction=float(1-valid.mean()),bad_residual_fraction=float(np.mean(g['residual']>th.max_residual)),
                        wrong_sector_site_fraction=float(min(np.mean(g['sigma']==1),np.mean(g['sigma']==-1))),
                        chi_mean=float(np.mean(g['chirality'])),density_mean=float(np.mean(g['density'])),
                        residual_median=float(np.median(g['residual'])),residual_p90=float(np.quantile(g['residual'],.9)),
                        contour_attempts=attempted,invalid_contour_fraction=invalid/attempted if attempted else None,
                        all_plaquette_composite_winding_sum=int(maps['composite_winding'].sum()))
                    rows.append(row)
                    rejections.extend(dict(g_d=gd,temperature=T,sweep=meta['sweep'],setting=label,
                        reason=k,loop_occurrences=v,contour_attempts=attempted) for k,v in reason.items())
                    for j,r in enumerate(found):
                        candidates.append(dict(g_d=gd,temperature=T,sweep=meta['sweep'],setting=label,
                            candidate=j,center_x=r['center_x'],center_y=r['center_y'],accepted=r['accepted'],
                            label=r['label'],phase_charge=r['phase_charge'],seed_plaquettes=r['seed_plaquettes'],
                            loop_acceptance=';'.join(f"r{q['radius']}:{q.get('phase_charge')}:{q['accepted']}" for q in r['loops'])))
                    # A compact JSON record gives every reason on representative
                    # last snapshots; CSV keeps all candidate/contour outcomes.
                    if meta['sweep']==runmeta['actual_sweeps']:
                        details.append(dict(g_d=gd,T=T,setting=label,candidates=found))
                        saved_fields[gd,T,label]=(B,g,valid,found)
        print('Analyzed',run.parent.name,flush=True)
    savecsv('thermal_snapshots.csv',rows);savecsv('thermal_rejections.csv',rejections);savecsv('thermal_candidates.csv',candidates)
    aggregate=[]
    for gd in sorted(set(r['g_d'] for r in rows)):
        for T in sorted(set(r['temperature'] for r in rows if r['g_d']==gd)):
            for label,_,_ in settings():
                rr=[r for r in rows if r['g_d']==gd and r['temperature']==T and r['setting']==label]
                rec=dict(g_d=gd,temperature=T,setting=label,snapshots=len(rr))
                for key in ['hqv_positive','hqv_negative','integer_net','unresolved_clusters','invalid_site_fraction',
                            'invalid_contour_fraction','residual_median','chi_mean','density_mean']:
                    a=np.asarray([r[key] for r in rr],dtype=float);finite=a[np.isfinite(a)]
                    rec[key+'_mean']=float(finite.mean()) if len(finite) else np.nan
                    rec[key+'_min']=float(finite.min()) if len(finite) else np.nan
                    rec[key+'_max']=float(finite.max()) if len(finite) else np.nan
                aggregate.append(rec)
    savecsv('thermal_summary.csv',aggregate)
    (OUT/'representative_candidates.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    # Last recorded gD=0 fields, with raw quality failures and proxy candidates.
    temps=[.35,.3815,.41]
    fig,axs=plt.subplots(5,3,figsize=(12,15))
    for col,T in enumerate(temps):
        A,g,valid,found=saved_fields[0.,T,'raw_strict']
        vminvmax=[(-np.pi,np.pi),(-1,1),(0,1.8),(0,.65),(0,1)]
        for row,(arr,cmap,title) in enumerate([(np.angle(g['composite']),'twilight','arg(P)'),
                (g['chirality'],'coolwarm','local chirality'),(g['density'],'viridis','density'),
                (g['residual'],'magma','squared manifold residual'),(valid.astype(float),'gray','locally valid sites')]):
            im=axs[row,col].imshow(arr.T,origin='lower',cmap=cmap,vmin=vminvmax[row][0],vmax=vminvmax[row][1]);fig.colorbar(im,ax=axs[row,col],fraction=.046)
            if row==0:
                for r in found:axs[row,col].plot(r['center_x'],r['center_y'],'x',ms=2,color='white',alpha=.5)
            axs[row,col].set_title(f'T={T:g}: {title}');axs[row,col].set(xlabel='x',ylabel='y')
    fig.suptitle('Actual last pilot snapshots: raw fields, strict Stage-1 quality\nWhite crosses are unresolved seed clusters, not confirmed vortices')
    fig.tight_layout();fig.savefig(OUT/'thermal_raw_fields.png',dpi=145);plt.close(fig)
    # Director LINE glyphs: segments, not arrows, with fixed spin-coordinate projection.
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for col,T in enumerate(temps):
        for row,label in enumerate(['raw_strict','smooth8_proxy']):
            A,g,valid,found=saved_fields[0.,T,label]
            im=axs[row,col].imshow(np.angle(g['composite']).T,origin='lower',cmap='twilight',vmin=-np.pi,vmax=np.pi)
            for x in range(0,64,3):
                for y in range(0,64,3):
                    if valid[x,y]:
                        dx,dy=.95*g['director'][x,y,[0,2]]
                        axs[row,col].plot([x-dx,x+dx],[y-dy,y+dy],color='white',lw=.55)
            for r in found:
                if r['accepted']:
                    color='red' if r['phase_charge']>0 else 'cyan';axs[row,col].plot(r['center_x'],r['center_y'],'o',mfc='none',mec=color,ms=8)
                else:axs[row,col].plot(r['center_x'],r['center_y'],'x',color='gray',ms=2)
            axs[row,col].set_title(f'{label}, T={T:g}\naccepted clusters={sum(r["accepted"] for r in found)}')
            axs[row,col].set(xlabel='x',ylabel='y')
    fig.suptitle('Director-line diagnostic: glyph components are (spin dx, spin dz), not physical flow\nRaw fields above; full-grid smoothed proxies below; no coordinate downsampling of saved fields')
    fig.tight_layout();fig.savefig(OUT/'thermal_director_proxies.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for label,_,_ in settings():
        rr=[r for r in aggregate if r['g_d']==0 and r['setting']==label]
        x=[r['temperature'] for r in rr]
        axs[0].plot(x,[r['invalid_site_fraction_mean'] for r in rr],'.-',label=label)
        axs[1].plot(x,[r['hqv_positive_mean']+r['hqv_negative_mean'] for r in rr],'.-')
        axs[2].plot(x,[r['unresolved_clusters_mean'] for r in rr],'.-')
    for ax in axs:ax.set_xlabel('Temperature');ax.grid(alpha=.2)
    axs[0].set(ylabel='Fraction locally invalid',title='Quality, not absence of defects');axs[0].legend(fontsize=7)
    axs[1].set(ylabel='Mean accepted HQV clusters',title='Smoothed counts are proxies')
    axs[2].set(ylabel='Mean unresolved clusters',title='One pilot history; no equilibrium errors')
    fig.tight_layout();fig.savefig(OUT/'thermal_quality_summary.png',dpi=165);plt.close(fig)
    (OUT/'thermal_analysis.json').write_text(json.dumps(dict(settings=[r[0] for r in settings()],
        analyzed_field_settings=len(rows),raw_fields=sum(r['setting']=='raw_strict' for r in rows),
        elapsed_seconds=perf_counter()-started,smoothing_calibration_passed=True,
        smoothing_variance_per_axis='passes/4; snapshots stay full resolution',
        interpretations='Raw accepted counts conditional on quality; smoothed proxies do not establish raw GL HQVs'),indent=2),encoding='utf-8')
    print('Thermal analysis seconds',perf_counter()-started,flush=True)


if __name__=='__main__':main()
