"""Scientific summaries, extended topology checks and selected figures.

python -B exploration/he3_hqv_stage3/analyze.py
Reads completed relaxation outputs only. Does not launch an optimizer.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relax import ROOT,problem,value_gradient,BETA_A,local_geometry,analyze_loop,rectangle_loop,find_candidates,trial,energy_density

OUT=Path(__file__).resolve().parent/'results'


def save(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def extended_loops(A,m,phase):
    L,R,p=m['L'],m['R'],m['pin_radius_parameter'];g=local_geometry(A);rows=[]
    for core,(cx,cy) in enumerate(m['centers']):
        radii=sorted(set([p,p+1,p+3,min(p+5,int(R/2)-2),int(R/3),int(R/2)-3]))
        for r in radii:
            if r<p or r>=R/2-1:continue
            c=rectangle_loop(int(cx-r-.5),int(cx+r+.5),int(cy-r-.5),int(cy+r+.5))
            res=analyze_loop(A,c,geometry=g,periodic=True)
            rows.append(dict(case=m['case'],phase=phase,core=str(core),radius=r,expected=[.5,-.5][core],
                accepted=res['accepted'],charge=res['phase_charge'],wP=res['composite_winding'],
                director_parity=res['director_parity'],max_residual=res['max_residual'],min_density=res['min_density'],
                max_phase_step=res['max_composite_step'],reasons=';'.join(res['reasons'])))
    x1,x2=m['centers'][0][0],m['centers'][1][0]
    c=rectangle_loop(max(2,int(x1-10.5)),min(L-3,int(x2+10.5)),max(2,L//2-14),min(L-3,L//2+13))
    res=analyze_loop(A,c,geometry=g,periodic=True)
    rows.append(dict(case=m['case'],phase=phase,core='outer',radius=-1,expected=0,accepted=res['accepted'],
        charge=res['phase_charge'],wP=res['composite_winding'],director_parity=res['director_parity'],
        max_residual=res['max_residual'],min_density=res['min_density'],max_phase_step=res['max_composite_step'],reasons=';'.join(res['reasons'])))
    return rows


def main():
    start=perf_counter();infos={};fields={};rows=[];tops=[];defects=[];parts=[]
    for p in sorted(OUT.glob('*/info.json')):
        m=json.loads(p.read_text());infos[m['case']]=m
        if m.get('exclude_from_comparison'):continue
        with np.load(p.parent/'fields.npz') as z:A=z['initial'];B=z['final'];mask=z['pinned']
        fields[m['case']]=(A,B,mask)
        for label,F in [('initial',A),('final',B)]:tops.extend(extended_loops(F,m,label))
        good=find_candidates(B,periodic=True)
        for r in good:
            defects.append(dict(case=m['case'],x=r['center_x'],y=r['center_y'],accepted=r['accepted'],
                charge=r['phase_charge'],label=r['label'],seed_plaquettes=r['seed_plaquettes']))
        g=local_geometry(B);ed=energy_density(B,gd=m['gd'])
        n0=(1+m['gd']*float(np.float32(2/3))/2)/BETA_A
        onsite0=-2*n0+BETA_A*n0*n0;dipole0=-m['gd']*float(np.float32(2/3))*n0
        # Stage2 trial without overwriting annuli or adding initialization noise.
        original,_,_=trial(m['L'],m['R'],3.,'compact_periodic');original*=np.sqrt(n0*BETA_A)
        trial_E=value_gradient(original,gd=m['gd'])[0]
        if not m.get('released_from'):
            quiet,_,_=problem(m['L'],m['R'],pin=m['pin_radius_parameter'],initial='planar',
                right_rotation=m['right_pin_spin_rotation'],gd=m['gd'])
            quiet_E=value_gradient(quiet,gd=m['gd'])[0]
        else:quiet_E=np.nan
        data=dict(case=m['case'],L=m['L'],R=m['R'],gd=m['gd'],pin=m['pin_radius_parameter'],
            initial=m['initial'],right_pin_rotation=m['right_pin_spin_rotation'],released=bool(m.get('released_from')),
            stage2_unpatched_trial_energy=trial_E,quiet_same_constraints_energy=quiet_E,
            noisy_initial_energy=m['initial_energy'],relaxed_energy=m['final_energy'],
            gradient_inf=m['gradient_inf'],iterations=m['iterations'],seconds=m['elapsed_seconds'],
            gradient_converged=m['gradient_converged'],constraint_error=m['constraint_error'],
            max_energy_increase=m['max_accepted_energy_increase'],minimum_density=float(g['density'].min()),
            maximum_residual=float(g['residual'].max()),minimum_chirality=float(g['chirality'].min()),
            spin_y_norm_fraction=float(np.sum(abs(B[...,1,:])**2)/np.sum(abs(B)**2)),
            wrong_chirality_weight_max=float(g['wrong_chirality'].max()),nonreal_spin_penalty_max=float(g['nonreal_spin'].max()),
            accepted_hqv_positive=sum(r['accepted'] and r['phase_charge']==.5 for r in good),
            accepted_hqv_negative=sum(r['accepted'] and r['phase_charge']==-.5 for r in good),
            accepted_other=sum(r['accepted'] and abs(r['phase_charge'])!=.5 for r in good),
            unresolved_clusters=sum(not r['accepted'] for r in good))
        rows.append(data)
        parts.append(dict(case=m['case'],L=m['L'],R=m['R'],gd=m['gd'],
            onsite_excess=float(np.sum(ed['onsite']-onsite0)),dipole_excess=float(np.sum(ed['dipole']-dipole0)),
            gradient=float(ed['gradient'].sum()),total=float(sum(np.sum(ed[k]-v) for k,v in [('onsite',onsite0),('dipole',dipole0),('gradient',0.)]))))
    save('case_summary.csv',rows);save('topology_checks.csv',tops);save('detected_clusters.csv',defects);save('energy_parts.csv',parts)
    baseline=[r for r in rows if r['case'].endswith('_p2') and r['gd']==0]
    differences=[];slopes=[]
    for L in sorted(set(r['L'] for r in baseline)):
        rr=sorted([r for r in baseline if r['L']==L],key=lambda r:r['R']);ref=rr[0]
        for r in rr:differences.append(dict(L=L,R=r['R'],reference_R=ref['R'],
            relaxed_difference=r['relaxed_energy']-ref['relaxed_energy'],
            stage2_trial_difference=r['stage2_unpatched_trial_energy']-ref['stage2_unpatched_trial_energy'],
            quiet_same_constraints_difference=r['quiet_same_constraints_energy']-ref['quiet_same_constraints_energy'],
            bare_log_difference=4*np.pi/BETA_A*np.log(r['R']/ref['R'])))
        for a,b in zip(rr[:-1],rr[1:]):
            slopes.append(dict(L=L,R_low=a['R'],R_high=b['R'],
                effective_log_slope=(b['relaxed_energy']-a['relaxed_energy'])/np.log(b['R']/a['R']),
                bare=4*np.pi/BETA_A))
    save('energy_differences.csv',differences);save('effective_slopes.csv',slopes)
    # Purely diagnostic finite-size extrapolation, deliberately no critical fit.
    extrap=[]
    for R in [24,32,48]:
        rr=[r for r in differences if r['R']==R]
        if len(rr)<3:continue
        for subset in ['all_sizes','L_at_least96']:
            use=[r for r in rr if subset=='all_sizes' or r['L']>=96]
            if len(use)<2:continue
            x=np.array([1/r['L']**2 for r in use]);y=np.array([r['relaxed_difference'] for r in use])
            slope,intercept=np.polyfit(x,y,1)
            extrap.append(dict(R=R,subset=subset,points=len(use),infinite_box_difference=intercept,
                resulting_log_coefficient=intercept/np.log(R/16),max_fit_residual=float(max(abs(y-intercept-slope*x))),
                interpretation='Exploratory 1/L^2 correction only; not a controlled limit'))
    save('finite_size_diagnostic.csv',extrap)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    colors={64:'#4477aa',96:'#ee7733',128:'#228833',192:'#aa3377'}
    for L in sorted(colors):
        rr=[r for r in differences if r['L']==L]
        if not rr:continue
        x=np.array([r['R'] for r in rr]);y=np.array([r['relaxed_difference'] for r in rr])
        axs[0].plot(x,y,'o-',color=colors[L],label=f'L={L}')
        axs[1].plot(np.log(x/16),y,'o-',color=colors[L],label=f'relaxed L={L}')
        if L in [64,192]:axs[1].plot(np.log(x/16),[r['stage2_trial_difference'] for r in rr],':',color=colors[L],label=f'unpatched trial L={L}')
        ss=[r for r in slopes if r['L']==L]
        axs[2].plot([np.sqrt(r['R_low']*r['R_high']) for r in ss],[r['effective_log_slope'] for r in ss],'o-',color=colors[L])
    xx=np.linspace(0,np.log(3),100);axs[1].plot(xx,4*np.pi/BETA_A*xx,'k--',label='bare trial coefficient')
    axs[2].axhline(4*np.pi/BETA_A,color='k',ls='--')
    axs[0].set(xlabel='R',ylabel='E(R)-E(16)',title='Constrained relaxed energies')
    axs[1].set(xlabel='ln(R/16)',ylabel='E(R)-E(16)',title='A log-looking curve is not a fit proof')
    axs[2].set(xlabel='Geometric-mean R',ylabel='Delta E / Delta ln R',title='Finite box and finite-R corrections')
    for ax in axs:ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);axs[1].legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'interaction_energies.png',dpi=170);plt.close(fig)
    # The final representative field, with a view of ALL tensor diagnostics.
    name='L128_R32_p2';m=infos[name];A,B,pin=fields[name];g=local_geometry(B);ed=energy_density(B)
    fig,axs=plt.subplots(2,3,figsize=(14,9))
    center=m['L']/2-.5
    content=[(g['density'],'viridis','Density'),(g['chirality'],'coolwarm','Local chirality'),
        (np.angle(g['composite']),'twilight','arg(P)'),(g['residual'],'magma','Squared manifold residual'),
        (ed['total']+1/BETA_A,'inferno','Full excess energy density'),(np.angle(g['composite']),'twilight','Director LINE glyphs: spin (dx,dz)')]
    for ax,(arr,cmap,title) in zip(axs.flat,content):
        im=ax.imshow(arr.T,origin='lower',cmap=cmap);fig.colorbar(im,ax=ax,fraction=.046)
        ax.contour(pin.T,levels=[.5],colors='cyan',linewidths=.6)
        ax.set(title=title,xlabel='x',ylabel='y',xlim=(center-31,center+31),ylim=(center-22,center+22))
        for (cx,cy),q in zip(m['centers'],[.5,-.5]):ax.text(cx+3,cy+3,f'{q:+g}',color='white')
    ax=axs[1,2]
    for x in range(int(center-30),int(center+31),3):
        for y in range(int(center-21),int(center+22),3):
            dx,dz=g['director'][x,y,[0,2]]*.95
            ax.plot([x-dx,x+dx],[y-dz,y+dz],color='white',lw=.7)
    fig.suptitle('L128 R32: converged full-GL field, cyan annuli fixed; core interiors free\nNo connecting strip or bulk manifold constraint')
    fig.tight_layout();fig.savefig(OUT/'relaxed_field_maps.png',dpi=160);plt.close(fig)
    # Core reconstruction and convergence; optimizer iteration is not real time.
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for name,color in [('L64_R24_p2','#4477aa'),('L64_R24_nonplanar','#228833'),('L64_R24_released','#aa3377')]:
        h=list(csv.DictReader((OUT/name/'history.csv').open()));it=np.array([int(r['iteration']) for r in h]);E=np.array([float(r['energy']) for r in h]);gg=[float(r['gradient_inf']) for r in h]
        axs[0,0].semilogy(it,np.maximum(E-E[-1],1e-12),color=color,label=name)
        axs[0,1].semilogy(it,gg,color=color,label=name)
    axs[0,1].axhline(3e-6,color='k',ls='--')
    A,B,_=fields['L64_R24_p2'];gi=local_geometry(A);gf=local_geometry(B);cx,cy=infos['L64_R24_p2']['centers'][0]
    x=np.arange(int(cx)-8,int(cx)+10);j=int(cy)
    axs[1,0].plot(x-cx,gi['density'][x,j],'o-',label='initial with pins + noise');axs[1,0].plot(x-cx,gf['density'][x,j],'o-',label='relaxed')
    axs[1,1].plot(x-cx,gf['wrong_chirality'][x,j],label='wrong orbital weight')
    axs[1,1].plot(x-cx,gf['nonreal_spin'][x,j],label='nonreal-spin penalty')
    axs[1,1].plot(x-cx,gf['residual'][x,j],'--',label='total squared residual')
    axs[0,0].set(xlabel='Optimizer iteration',ylabel='E-E(final)',title='Accepted objective decrease')
    axs[0,1].set(xlabel='Optimizer iteration',ylabel='Free real-gradient infinity norm')
    axs[1,0].set(xlabel='x-cx (row y=floor(cy))',ylabel='Density',title='Density need not vanish at lattice core sites')
    axs[1,1].set(xlabel='x-cx',ylabel='Residual contributions',title='Complex spin structure in the core')
    for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'convergence_and_core.png',dpi=165);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    groups=[('p2',['L64_R16_p2','L64_R24_p2','L64_R32_p2']),('p3',['L64_R16_p3','L64_R32_p3']),
        ('right pin spin plane rotated',['L64_R16_rotated_pin','L64_R24_rotated_pin','L64_R32_rotated_pin'])]
    for label,names in groups:
        mm=[infos[n] for n in names];ref=mm[0]['final_energy'];axs[0].plot([v['R'] for v in mm],[v['final_energy']-ref for v in mm],'o-',label=label)
    weak=[]
    for R in [16,32]:
        zero=infos[f'L128_R{R}_p2'];soc=infos[f'L128_R{R}_gd0001']
        B0=fields[f'L128_R{R}_p2'][1]
        # Uniform amplitude rescaling meets the gd-specific vacuum and pin
        # amplitudes before comparing frozen vs fully relaxed SOC energies.
        scaled=B0*np.sqrt(1+.001*float(np.float32(2/3))/2)
        frozen=value_gradient(scaled,gd=.001)[0]
        weak.append(dict(R=R,E0=zero['final_energy'],Esoc=soc['final_energy'],extra=soc['final_energy']-zero['final_energy'],
            SOC_energy_of_scaled_zero_SOC_minimum=frozen,SOC_relaxation_gain=frozen-soc['final_energy']))
    save('weak_soc_comparison.csv',weak)
    axs[1].plot([r['R'] for r in weak],[r['extra'] for r in weak],'o-')
    axs[0].set(xlabel='R',ylabel='E(R)-E(16)',title='Unnecessary pinned spin texture affects the curve');axs[0].legend(fontsize=7)
    axs[1].set(xlabel='R',ylabel='E(gD=.001)-E(gD=0)',title='Two sub-dipole-length comparisons, not confinement')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'pinning_and_weak_soc.png',dpi=170);plt.close(fig)
    # Exact SOC density, rather than a director-sign seam or restricted proxy.
    name='L128_R32_gd0001';m=infos[name];_,B,pin=fields[name];g=local_geometry(B);e=energy_density(B,gd=.001)
    n0=(1+.001*float(np.float32(2/3))/2)/BETA_A;D=e['dipole']+.001*float(np.float32(2/3))*n0
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,arr,title,cmap in zip(axs,[D,np.sum(abs(B[...,2,:])**2,axis=-1)/g['density'],g['residual']],
            ['Actual SOC excess density','Spin-z norm fraction','Squared manifold residual'],['inferno','viridis','magma']):
        im=ax.imshow(arr.T,origin='lower',cmap=cmap);fig.colorbar(im,ax=ax,fraction=.046)
        ax.set(title=title,xlabel='x',ylabel='y');ax.contour(pin.T,levels=[.5],colors='cyan',linewidths=.5)
    fig.suptitle('Weak SOC: the computed texture is not an imposed director-sign string')
    fig.tight_layout();fig.savefig(OUT/'weak_soc_field.png',dpi=160);plt.close(fig)
    (OUT/'analysis_validation.json').write_text(json.dumps(dict(cases=len(rows),excluded_repeat_cases=sum(m.get('exclude_from_comparison',False) for m in infos.values()),
        all_gradient_converged=all(r['gradient_converged'] for r in rows),all_constraints_exact=all(r['constraint_error']==0 for r in rows),
        total_recorded_solver_seconds=sum(m['elapsed_seconds'] for m in infos.values()),analysis_seconds=perf_counter()-start,
        checks='Before/after extended loops, full-field candidate search, energy parts and differences retained'),indent=2),encoding='utf-8')
    print('Analyzed',len(rows),'cases; recorded solver seconds',sum(m['elapsed_seconds'] for m in infos.values()),flush=True)


if __name__=='__main__':main()
