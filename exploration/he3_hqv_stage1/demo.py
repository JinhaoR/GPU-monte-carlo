"""Generate a few Stage-1 figures and explicitly synthetic matrix fixtures.

Run after test_stage1.py: python -B exploration/he3_hqv_stage1/demo.py
No fields are relaxed or sampled. No production files or original NPZs are used.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from defects import local_geometry,analyze_loop,rectangle_loop,plaquette_map,find_candidates
from synthetic import make_cases,integer_vortex

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
plt.rcParams.update({'font.size':9,'axes.grid':False})


def around(center,radius=6):
    x,y=center
    return rectangle_loop(int(np.floor(x-radius)),int(np.ceil(x+radius)),
                          int(np.floor(y-radius)),int(np.ceil(y+radius)))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cases=make_cases(96)
    automatic=[]
    for name,case in cases.items():
        found=find_candidates(case['field'],periodic=case['boundary']=='periodic')
        if not found:
            automatic.append(dict(case=name,candidate_index='',center_x='',center_y='',accepted=False,
                label='no_elementary_seed',phase_charge='',seed_plaquettes=0,reason='No seed; not a proof of absence in arbitrary fields'))
        for j,r in enumerate(found):
            automatic.append(dict(case=name,candidate_index=j,center_x=r['center_x'],center_y=r['center_y'],
                accepted=r['accepted'],label=r['label'],phase_charge=r['phase_charge'],seed_plaquettes=r['seed_plaquettes'],reason='; '.join(r['reasons'])))
    with (OUT/'automatic_detection.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(automatic[0]));w.writeheader();w.writerows(automatic)
    names=['uniform','integer_vortex','hqv_pair','periodic_hqv_pair','chirality_wall','composite_zero']
    fig,axs=plt.subplots(3,6,figsize=(18,9),sharex=True,sharey=True)
    summaries=[]
    for col,name in enumerate(names):
        case=cases[name];A=case['field'];g=local_geometry(A)
        loop=around(case['centers'][0] if case['centers'] else (47.5,47.5),12 if name=='chirality_wall' else 6)
        r=analyze_loop(A,loop,geometry=g,periodic=case['boundary']=='periodic')
        summaries.append(dict(case=name,boundary=case['boundary'],**r))
        for row,(arr,cmap,lo,hi) in enumerate([(g['density'],'viridis',0,1.05),
                (np.angle(g['composite']),'twilight',-np.pi,np.pi),
                (g['residual'],'magma',0,1)]):
            im=axs[row,col].imshow(arr.T,origin='lower',extent=(-.5,95.5,-.5,95.5),cmap=cmap,vmin=lo,vmax=hi)
            closed=np.vstack((loop,loop[0]))
            axs[row,col].plot(closed[:,0],closed[:,1],color='cyan' if row==2 else 'white',lw=.7)
            for (x,y),q in zip(case['centers'],case['expected_charges']):
                axs[row,col].plot(x,y,'o',mfc='none',mec='black',ms=7)
                axs[row,col].text(x+3,y+3,f'{q:+g}',color='black',fontsize=8)
            if row==2:axs[row,col].set_xlabel('x (lattice sites)')
            if col==0:axs[row,col].set_ylabel(['density n','arg(P), y','squared residual, y'][row])
            if col==5:fig.colorbar(im,ax=axs[row,col],fraction=.046,pad=.04)
        axs[0,col].set_title(name.replace('_',' '))
        axs[2,col].text(.04,.04,('accepted: '+r['label']) if r['accepted'] else 'unresolved',
            transform=axs[2,col].transAxes,color='white',fontsize=8,
            bbox=dict(facecolor='black',alpha=.5,edgecolor='none'))
    fig.suptitle('Known matrix textures: density, charge-two phase and distance from the chiral manifold\nWhite/cyan contours are tested; open analytic fields are not periodic equilibrium configurations')
    fig.tight_layout();fig.savefig(OUT/'synthetic_textures.png',dpi=155);plt.close(fig)

    # Show quantities transported around exactly the same lattice contour.
    selected=['hqv_pair','integer_vortex','composite_zero','half_phase_without_spin']
    fig,axs=plt.subplots(3,4,figsize=(14,9),sharex='col')
    for col,name in enumerate(selected):
        case=cases[name];A=case['field'];g=local_geometry(A)
        c=around(case['centers'][0] if case['centers'] else (47.5,47.5))
        x,y=c.T;d=g['director'][x,y].copy();amp=g['amplitude'][x,y]
        p=g['composite'][x,y];s=np.where(np.sum(d*np.roll(d,-1,axis=0),axis=1)>=0,1.,-1.)
        psteps=np.angle(np.roll(p,-1)*p.conj())
        lsteps=np.angle(np.roll(amp,-1)*amp.conj()*s)
        t=np.arange(len(c)+1)/len(c)
        axs[0,col].plot(t,np.r_[0,np.cumsum(psteps)]/(4*np.pi),label='composite winding / 2')
        axs[0,col].plot(t,np.r_[0,np.cumsum(lsteps)]/(2*np.pi),'--',label='lifted phase charge')
        lifted=np.empty((len(d)+1,3));lifted[0]=d[0]
        sign=1.
        for j in range(1,len(d)):
            sign*=s[j-1];lifted[j]=sign*d[j]
        lifted[-1]=sign*s[-1]*d[0]
        for k,label in enumerate(['d_x','d_y','d_z']):axs[1,col].plot(t,lifted[:,k],label=label)
        axs[2,col].plot(t[:-1],g['residual'][x,y],label='squared residual')
        axs[2,col].plot(t[:-1],abs(g['chirality'][x,y]),'--',label='absolute chirality')
        axs[0,col].set_title(name.replace('_',' '))
        axs[1,col].set_ylim(-1.1,1.1);axs[2,col].set_ylim(-.03,1.05)
        axs[2,col].set_xlabel('Fraction of CCW contour traversal')
        r=analyze_loop(A,c,geometry=g)
        axs[2,col].text(.04,.08,'accepted' if r['accepted'] else 'unresolved',transform=axs[2,col].transAxes)
        for row in range(3):axs[row,col].grid(alpha=.2)
    for row in range(3):axs[row,0].legend(fontsize=7,loc='best')
    axs[0,0].set_ylabel('Cumulative circulation / 2 pi')
    axs[1,0].set_ylabel('Continuously lifted real director')
    axs[2,0].set_ylabel('Local validity diagnostics')
    fig.suptitle('A valid HQV combines half phase circulation, director sign reversal and a good manifold fit\nOdd P winding alone fails both the nonchiral-zero and uncompensated half-phase controls')
    fig.tight_layout();fig.savefig(OUT/'contour_signatures.png',dpi=165);plt.close(fig)

    rows=list(csv.DictReader((OUT/'noise_sensitivity.csv').open()))
    strengths=sorted(set(float(r['noise_relative_rms']) for r in rows))
    limits=sorted(set(float(r['max_squared_residual']) for r in rows))
    heat=np.array([[next(int(r['correct'])/int(r['loops']) for r in rows if float(r['noise_relative_rms'])==v and float(r['max_squared_residual'])==lim) for v in strengths] for lim in limits])
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    im=axs[0].imshow(heat,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
    axs[0].set_xticks(range(len(strengths)),labels=[f'{v:g}' for v in strengths])
    axs[0].set_yticks(range(len(limits)),labels=[f'{v:g}' for v in limits])
    axs[0].set(xlabel='Synthetic component-noise relative RMS',ylabel='Maximum squared manifold residual',title='Fraction accepted with correct charge (20 loops/cell)')
    for i in range(len(limits)):
        for j in range(len(strengths)):axs[0].text(j,i,f'{heat[i,j]:.2g}',ha='center',va='center',color='white' if heat[i,j]<.5 else 'black')
    fig.colorbar(im,ax=axs[0],fraction=.05)
    A=integer_vortex(96);multi=[]
    for radius in [0,1,2,3,4,6,8,12]:
        c=rectangle_loop(47,48,47,48) if radius==0 else around((47.5,47.5),radius)
        r=analyze_loop(A,c);multi.append(dict(radius=radius,**r))
    axs[1].plot([r['radius'] for r in multi],[r['lifted_phase_charge'] for r in multi],'o-',label='joint lifted charge')
    axs[1].plot([r['radius'] for r in multi],[r['composite_winding']/2 for r in multi],'s--',label='raw composite winding / 2')
    for r in multi:
        if not r['accepted']:axs[1].axvspan(r['radius']-.25,r['radius']+.25,color='red',alpha=.2)
    axs[1].set(xlabel='Square-loop radius (0 = elementary plaquette)',ylabel='Apparent phase charge',ylim=(-.1,1.6),title='Integer vortex: composite alias on the smallest contour')
    axs[1].legend();axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'resolution_and_noise.png',dpi=165);plt.close(fig)
    (OUT/'demo_loops.json').write_text(json.dumps(dict(known_loops=summaries,integer_resolution=multi),indent=2),encoding='utf-8')
    # Explicit Stage-1 fixtures only, not an implementation of MC snapshot saving.
    keep=['uniform','integer_vortex','hqv_pair','periodic_hqv_pair']
    meta={name:{k:v for k,v in cases[name].items() if k!='field'} for name in keep}
    meta['origin']='Analytic synthetic fixtures; no Monte Carlo or relaxation'
    np.savez_compressed(OUT/'synthetic_fields.npz',**{name:cases[name]['field'] for name in keep},metadata=np.array(json.dumps(meta)))
    print('Saved three figures, contour diagnostics and explicitly synthetic fixtures.')


if __name__=='__main__':main()
