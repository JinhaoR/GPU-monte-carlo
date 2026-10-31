"""Full-GL energies of UNRELAXED neutral HQV pair trial fields, gD=0.

Run: python -B exploration/he3_hqv_stage2/pair_baseline.py
Two explicit boundary/texture families: plane-angle open cells, and a smooth
compact periodic field whose boundary is exactly the homogeneous vacuum.
Core amplitude vanishes quadratically (C1 at the continuum core), width>=3.
No constraints, minimizer, pair dynamics or finite-T sampling is implemented.
"""
import ast
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'exploration/he3_hqv_stage1'))
from synthetic import hqv_pair,chiral,grid,_hqv_from_unit_scalar
from defects import analyze_loop,rectangle_loop,local_geometry
from energy import energy,energy_density,BETAS

OUT=HERE/'results'
BETA_A=float(BETAS[1]+BETAS[3]+BETAS[4])
N0=1/BETA_A
EXPECTED=4*np.pi*N0


def trial(L,R,core=3.,family='open_plane'):
    centers=((L/2-.5-R/2,L/2-.5),(L/2-.5+R/2,L/2-.5))
    x,y=grid(L)
    distances=[np.hypot(x-cx,y-cy) for cx,cy in centers]
    amplitude=np.sqrt(N0)*np.prod([np.tanh((r/core)**2) for r in distances],axis=0)
    if family=='open_plane':
        A=hqv_pair(L,centers,core=core)
        old=np.prod([np.tanh(r/core) for r in distances],axis=0)
        A*=np.divide(amplitude,old,out=np.zeros_like(amplitude),where=old>0)[...,None,None]
        boundary='open'
    elif family=='compact_periodic':
        phi=np.arctan2(y-centers[0][1],x-centers[0][0])-np.arctan2(y-centers[1][1],x-centers[1][0])
        r=np.hypot(x-(L/2-.5),y-(L/2-.5))
        t=np.clip((r-.30*L)/(.16*L),0,1)
        cutoff=1-(6*t**5-15*t**4+10*t**3)
        # The only angle branch seam lies between cores, where cutoff=1.
        # Multiplying an angle by a general cutoff without this property fails.
        A=_hqv_from_unit_scalar(np.exp(1j*cutoff*phi),amplitude)
        boundary='periodic'
        np.testing.assert_allclose(A[0],A[-1],atol=1e-13)
        np.testing.assert_allclose(A[:,0],A[:,-1],atol=1e-13)
    else:raise ValueError(family)
    return A,centers,boundary


def around(center,r):
    x,y=center
    return rectangle_loop(int(np.floor(x-r)),int(np.ceil(x+r)),int(np.floor(y-r)),int(np.ceil(y+r)))


def validate_energy():
    # Independent source scalar loops: CUDA decorators removed only in memory.
    source=ast.parse((ROOT/'models/he3/update.py').read_text(encoding='utf-8'))
    defs=[]
    for node in source.body:
        if isinstance(node,ast.FunctionDef) and node.name in ['norm2','dipole_invariant','potential','gradient']:
            node.decorator_list=[];defs.append(node)
    ns={'float32':np.float32}
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'source_polynomials','exec'),ns)
    rng=np.random.default_rng(82913);checks=[]
    for bc in ['periodic','open']:
        for gd in [0.,.001]:
            A=.3*(rng.normal(size=(8,9,3,2))+1j*rng.normal(size=(8,9,3,2)))
            onsite=sum(ns['potential'](A[i,j],BETAS,gd) for i in range(8) for j in range(9))
            gradient=0.
            for i in range(8 if bc=='periodic' else 7):
                for j in range(9 if bc=='periodic' else 8):
                    dx=A[(i+1)%8,j]-A[i,j];dy=A[i,(j+1)%9]-A[i,j]
                    for mu in range(3):gradient+=ns['gradient'](dx[mu,0],dx[mu,1],dy[mu,0],dy[mu,1])
            total=energy(A,gd=gd,boundary=bc)['total']
            # Same production float32 literal coefficients; float64 sums.
            np.testing.assert_allclose(total,onsite+gradient,rtol=1e-10,atol=1e-7)
            checks.append(dict(boundary=bc,gd=gd,cpu=total,source_scalar=float(onsite+gradient),difference=float(total-onsite-gradient)))
    A=chiral(np.zeros((16,16)),density=N0)
    e=energy(A)
    np.testing.assert_allclose(e['total'],-16**2/BETA_A,rtol=1e-13)
    assert e['gradient']==0
    checks.append(dict(homogeneous_energy=e['total'],analytic=-16**2/BETA_A,density=N0))
    # Match production GPU energy on one of the saved actual thermal fields.
    for path in sorted(OUT.glob('pilot_gd=*/snapshot_030000.npz')):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(z['metadata'].item());params=meta['model_parameters']
            totals=[energy(a,betas=params['betas'],K=params['K'],gd=params['g_d'])['total'] for a in z['fields']]
            gpu=z['scalar_energy']
        np.testing.assert_allclose(totals,gpu,rtol=3e-6,atol=.003)
        checks.append(dict(dataset=str(path.relative_to(ROOT)),max_cpu_gpu_energy_difference=float(max(abs(np.array(totals)-gpu))),gpu_energies=gpu.tolist()))
    return checks


def main():
    started=perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    checks=validate_energy();rows=[];loops=[]
    for family in ['open_plane','compact_periodic']:
        for L in [128,256,512]:
            for R in [8,12,16,24,32,48,64]:
                A,centers,bc=trial(L,R,3.,family)
                parts=energy(A,boundary=bc)
                ref=-L**2/BETA_A
                g=local_geometry(A)
                qualities=[]
                # Small, resolved distinct contours even at shortest R=8.
                for center,q in zip(centers,[.5,-.5]):
                    for rad in [1.5,2.5]:
                        res=analyze_loop(A,around(center,rad),geometry=g,periodic=bc=='periodic')
                        assert res['accepted'] and res['phase_charge']==q,(family,L,R,rad,res)
                        qualities.append(res['max_residual'])
                        loops.append(dict(family=family,L=L,R=R,core=3.,contour=f'core_q={q:g}_radius={rad:g}',
                            charge=res['phase_charge'],accepted=res['accepted'],max_residual=res['max_residual'],min_density=res['min_density']))
                outer=rectangle_loop(int(centers[0][0]-5.5),int(centers[1][0]+5.5),int(L/2-8),int(L/2+7))
                outer_result=analyze_loop(A,outer,geometry=g,periodic=bc=='periodic')
                assert outer_result['accepted'] and outer_result['phase_charge']==0,outer_result
                loops.append(dict(family=family,L=L,R=R,core=3.,contour='outer_both',charge=outer_result['phase_charge'],
                    accepted=outer_result['accepted'],max_residual=outer_result['max_residual'],min_density=outer_result['min_density']))
                rows.append(dict(family=family,boundary=bc,L=L,R=R,core=3.,reference_energy=ref,
                    total_energy=parts['total'],excess_energy=parts['total']-ref,
                    gradient=parts['gradient'],onsite_excess=parts['onsite']-ref,
                    expected_log_coefficient=EXPECTED,max_contour_residual=max(qualities),
                    seam_bond_max=float(max(np.max(abs(A[0]-A[-1])),np.max(abs(A[:,0]-A[:,-1]))))))
    # Core prescription sensitivity, at the largest box only, five separations.
    for core in [4.,6.]:
        for R in [16,24,32,48,64]:
            A,centers,bc=trial(512,R,core,'open_plane');parts=energy(A,boundary=bc);ref=-512**2/BETA_A
            for center,q in zip(centers,[.5,-.5]):
                for radius in [3.,5.]:
                    r=analyze_loop(A,around(center,radius),periodic=False)
                    assert r['accepted'] and r['phase_charge']==q,r
                    loops.append(dict(family='open_plane',L=512,R=R,core=core,contour=f'core_q={q:g}_radius={radius:g}',
                        charge=r['phase_charge'],accepted=r['accepted'],max_residual=r['max_residual'],min_density=r['min_density']))
            outer=rectangle_loop(int(centers[0][0]-5.5),int(centers[1][0]+5.5),248,263)
            outer_result=analyze_loop(A,outer,periodic=False)
            assert outer_result['accepted'] and outer_result['phase_charge']==0,outer_result
            loops.append(dict(family='open_plane',L=512,R=R,core=core,contour='outer_both',charge=outer_result['phase_charge'],
                accepted=outer_result['accepted'],max_residual=outer_result['max_residual'],min_density=outer_result['min_density']))
            rows.append(dict(family='open_plane',boundary=bc,L=512,R=R,core=core,reference_energy=ref,
                total_energy=parts['total'],excess_energy=parts['total']-ref,gradient=parts['gradient'],
                onsite_excess=parts['onsite']-ref,expected_log_coefficient=EXPECTED,
                max_contour_residual=r['max_residual'],
                seam_bond_max=float(max(np.max(abs(A[0]-A[-1])),np.max(abs(A[:,0]-A[:,-1]))))))
    with (OUT/'pair_energies.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'pair_topology_checks.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(loops[0]));w.writeheader();w.writerows(loops)
    fits=[]
    for family in ['open_plane','compact_periodic']:
        for L in [128,256,512]:
            rr=[r for r in rows if r['family']==family and r['L']==L and r['core']==3]
            for lo,hi in [(12,32),(16,64),(24,64)]:
                selected=[r for r in rr if lo<=r['R']<=hi]
                x=np.log([r['R'] for r in selected]);y=np.array([r['excess_energy'] for r in selected])
                slope,intercept=np.polyfit(x,y,1)
                fits.append(dict(family=family,L=L,core=3.,R_min=lo,R_max=hi,points=len(selected),
                    slope=float(slope),expected=EXPECTED,relative_difference=float(slope/EXPECTED-1),
                    max_residual=float(max(abs(y-intercept-slope*x)))))
    for core in [4.,6.]:
        rr=[r for r in rows if r['core']==core];x=np.log([r['R'] for r in rr]);y=np.array([r['excess_energy'] for r in rr])
        slope,intercept=np.polyfit(x,y,1)
        fits.append(dict(family='open_plane',L=512,core=core,R_min=16,R_max=64,points=len(rr),slope=float(slope),expected=EXPECTED,
            relative_difference=float(slope/EXPECTED-1),max_residual=float(max(abs(y-intercept-slope*x)))))
    with (OUT/'pair_log_fits.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(fits[0]));w.writeheader();w.writerows(fits)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    colors={128:'#4477aa',256:'#ee7733',512:'#228833'}
    for family,style in [('open_plane','-'),('compact_periodic','--')]:
        for L in [128,256,512]:
            rr=[r for r in rows if r['family']==family and r['L']==L and r['core']==3]
            R=np.array([r['R'] for r in rr]);E=np.array([r['excess_energy'] for r in rr]);E0=E[R==16][0]
            axs[0].plot(R,E,style+'o',color=colors[L],ms=3,label=f'{family}, L={L}')
            axs[1].plot(np.log(R/16),E-E0,style+'o',color=colors[L],ms=3)
            axs[2].plot(np.sqrt(R[:-1]*R[1:]),np.diff(E)/np.diff(np.log(R)),style+'o',color=colors[L],ms=3)
    xs=np.linspace(np.log(.5),np.log(4),100);axs[1].plot(xs,EXPECTED*xs,'k:',label='bare phase + spin benchmark')
    axs[2].axhline(EXPECTED,color='k',ls=':',label=f'4 pi K n0 = {EXPECTED:.4f}')
    axs[0].set(xlabel='Separation R',ylabel='Full lattice excess energy',title='Unrelaxed full-matrix trial energies')
    axs[1].set(xlabel='ln(R/16)',ylabel='E(R)-E(16)',title='Reference coefficient includes both sectors')
    axs[2].set(xlabel='Geometric-mean separation',ylabel='Delta E / Delta ln R',title='Finite box and core-overlap effects')
    axs[0].legend(fontsize=6);axs[1].legend(fontsize=7);axs[2].legend(fontsize=7)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'pair_energy_baseline.png',dpi=170);plt.close(fig)
    # Explicit visual check of compact periodic texture and reference subtraction.
    A,centers,bc=trial(128,32,3,'compact_periodic');g=local_geometry(A);maps=energy_density(A)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,a,title in zip(axs,[np.angle(g['composite']),g['density'],maps['total']+1/BETA_A],
            ['arg(P): compact periodic trial','Full order-parameter density','Local excess energy (no imposed wall)']):
        im=ax.imshow(a.T,origin='lower',cmap='twilight' if title.startswith('arg') else 'viridis');fig.colorbar(im,ax=ax,fraction=.046)
        for (x,y),q in zip(centers,[.5,-.5]):ax.plot(x,y,'wo',mfc='none');ax.text(x+2,y+3,f'{q:+g}',color='white')
        ax.set(title=title,xlabel='x');ax.set_ylabel('y')
    fig.tight_layout();fig.savefig(OUT/'pair_trial_texture.png',dpi=155);plt.close(fig)
    (OUT/'energy_validation.json').write_text(json.dumps(dict(passed=True,checks=checks,
        n0=N0,beta_chiral=BETA_A,expected_neutral_pair_log_coefficient=EXPECTED,
        configurations=len(rows),elapsed_seconds=perf_counter()-started,
        meaning='Unrelaxed configuration energies, not a minimized potential or free energy'),indent=2),encoding='utf-8')
    print('Validated',len(rows),'unrelaxed pair energies; expected coefficient',EXPECTED,'elapsed',perf_counter()-started,flush=True)


if __name__=='__main__':main()
