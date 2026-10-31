"""Read-only checks of recorded mappings, scalar agreement and pilot drift.

Run after pilot.py: python -B exploration/he3_hqv_stage2/pilot_checks.py
Small within-run block errors are descriptive, never equilibration guarantees.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from energy import energy

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'


def main():
    rows=[];checks=[]
    for run in OUT.glob('pilot_gd=*/run.json'):
        meta=json.loads(run.read_text());temps=np.array(meta['temperatures'])
        source_match={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in meta['source_sha256'].items()}
        assert all(source_match.values()),source_match
        for path in sorted(run.parent.glob('snapshot_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                m=json.loads(z['metadata'].item());A=z['fields'];tw=z['temperature_to_walker'];wt=z['walker_to_temperature']
                np.testing.assert_array_equal(wt[tw],np.arange(len(temps)))
                np.testing.assert_array_equal(tw[m['temperature_indices']],m['physical_walkers'])
                np.testing.assert_allclose(temps[m['temperature_indices']],m['temperatures'],rtol=0,atol=1e-14)
                assert A.shape==(5,64,64,3,2) and A.dtype==np.complex64
                den=np.sum(abs(A.astype(np.complex128))**2,axis=(1,2,3,4))
                chi=2*np.imag(np.sum(A[...,0].astype(np.complex128).conj()*A[...,1],axis=(1,2,3)))
                np.testing.assert_allclose(den,z['scalar_density'],rtol=2e-6,atol=1e-4)
                np.testing.assert_allclose(chi,z['scalar_chirality'],rtol=2e-6,atol=1e-4)
                checks.append(dict(path=str(path.relative_to(ROOT)),field_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                    density_max_absolute_difference=float(max(abs(den-z['scalar_density']))),
                    chirality_max_absolute_difference=float(max(abs(chi-z['scalar_chirality'])))))
        with np.load(run.parent/'scalar_history.npz',allow_pickle=False) as z:
            sweeps=z['sweeps'];T=z['temperatures'];production=sweeps>30000
            fig,axs=plt.subplots(1,3,figsize=(14,4))
            for t in [.35,.375,.3815,.389,.41]:
                j=np.argmin(abs(T-t))
                values=dict(energy=z['energy'][:,j]/64**2,density=z['density'][:,j]/64**2,
                    abs_chirality=abs(z['chirality'][:,j]/z['density'][:,j]),spin_alignment=z['spin_alignment'][:,j])
                for name,a in values.items():
                    a=a[production];half=len(a)//2;first=a[:half];second=a[half:]
                    f=first.reshape(-1,5).mean(axis=1);s=second.reshape(-1,5).mean(axis=1)
                    se=np.sqrt(f.var(ddof=1)/len(f)+s.var(ddof=1)/len(s))
                    rows.append(dict(g_d=meta['g_d'],temperature=t,observable=name,saved_scalar_spacing_sweeps=250,
                        block_sweeps=1250,blocks_per_half=len(f),first_half=first.mean(),second_half=second.mean(),
                        difference=second.mean()-first.mean(),descriptive_difference_error=se,
                        descriptive_ratio=(second.mean()-first.mean())/se if se else np.nan))
                axs[0].plot(sweeps/1000,values['energy'],lw=.65,label=f'T={t:g}')
                axs[1].plot(sweeps/1000,values['spin_alignment'],lw=.65)
            pt=meta['pt'];Tm=(T[:-1]+T[1:])/2
            axs[2].plot(Tm,pt['acceptance_rates'],'.-')
            for ax in axs[:2]:ax.axvline(30,color='black',ls='--');ax.set_xlabel('Thousand sweeps');ax.grid(alpha=.2)
            axs[0].set_ylabel('Energy/site');axs[0].legend(fontsize=7);axs[1].set_ylabel('Qzz')
            axs[2].set(xlabel='Adjacent-temperature midpoint',ylabel='PT swap acceptance');axs[2].grid(alpha=.2)
            fig.suptitle('Short ordered-start pilot: traces and cumulative PT acceptance; zero completed loops')
            fig.tight_layout();fig.savefig(OUT/'pilot_sampling_diagnostics.png',dpi=160);plt.close(fig)
    with (OUT/'pilot_half_checks.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'snapshot_validation.json').write_text(json.dumps(dict(passed=True,snapshots=len(checks),checks=checks,
        source_hashes_match=True,meaning='Mappings and arithmetic validated; no equilibrium assertion'),indent=2),encoding='utf-8')
    print('Validated',len(checks),'full-resolution snapshots and source hashes.')


if __name__=='__main__':main()
