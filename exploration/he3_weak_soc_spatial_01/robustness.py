"""Challenge the orbital interpretation with time halves and phase-mean checks."""
import json
import numpy as np
from findings import load,OUT
from orbital_channels import estimate as orbital_estimate, validate
from reduce import csv_write


def main():
    stability=[];means=[];consistency=[]
    identities={};quality={}
    for gd in (0,.01):
        runs={};orbits={}
        for L in (64,96,128,160):
            z=load(gd,L);o=load(gd,L,'orbital');runs[L]=z;orbits[L]=o
            with np.load(OUT/f'gd={gd:g}_L={L}.npz') as raw:
                a=raw['amplitude'].mean(0)
                structure=raw['structure']
                quality[f'gd={gd:g}_L={L}']=dict(
                    residual_outside_roundoff_max_block=float(structure[...,18].max()),
                    histogram_under_overflow_max_block=float(structure[...,[20,31,32,43,44,55,56,67]].max()))
            s=z['full_spectra']
            residuals=dict(Q_partition=s[:,2]-s[:,5:8].sum(1),
                B_partition=s[:,3]-s[:,8:11].sum(1),P_Bdifference=s[:,1]-2*s[:,9],
                A_partition=s[:,4]-o['A_plus_S']-o['A_minus_S'],
                B_rotated_partition=s[:,8]+s[:,10]-o['B_plus_S']-o['B_minus_S'])
            for name,value in residuals.items():
                identities[name]=max(identities.get(name,0.),float(np.max(abs(value))))
            T=z['T'];N=L*L
            am=(a[:,14,0]+a[:,15,0])/np.sqrt(2)
            bm=(a[:,7,0]+a[:,8,0]+2*a[:,9,0])/2
            for j,t in enumerate(T):means.append(dict(g_d=gd,L=L,T=t,
                A_plus_z_disconnected_fraction=abs(am[j])**2/(N*o['A_plus_z_S'][j,0]),
                B_plus_disconnected_fraction=abs(bm[j])**2/(N*o['B_plus_S'][j,0])))
            blocks={k[6:]:v for k,v in o.items() if k.startswith('block_')}
            for subset,portion in [('first',slice(0,30)),('second',slice(30,None))]:
                b={k:v[portion].reshape((15,2)+v.shape[1:]).mean(1) for k,v in blocks.items()}
                m={k:v.mean(0) for k,v in b.items()};v=orbital_estimate(m,L)
                jk=orbital_estimate({k:(15*m[k]-x)/14 for k,x in b.items()},L)
                for t in z['profile_T'][1:4]:
                    j=np.argmin(abs(T-t))
                    for name in ('A_plus_z','A_minus','B_plus'):
                        for metric in ('xi','S'):
                            value=v[name+'_'+metric][j] if metric=='xi' else v[name+'_'+metric][j,0]
                            deleted=jk[name+'_'+metric][:,j] if metric=='xi' else jk[name+'_'+metric][:,j,0]
                            stability.append(dict(g_d=gd,L=L,T=T[j],subset=subset,observable=name+'_'+metric,
                                value=value,se=np.sqrt(14*np.var(deleted))))
        for l1,l2 in ((64,96),(96,128),(128,160)):
            T=runs[l1]['T'];lr=np.log(l2/l1);o1=orbits[l1];o2=orbits[l2]
            a1=o1['A_plus_z_S'][:,0];a2=o2['A_plus_z_S'][:,0]
            b1=o1['B_plus_S'][:,0];b2=o2['B_plus_S'][:,0]
            ea=2-np.log(a2/a1)/lr;eb=2-np.log(b2/b1)/lr
            f1=4*np.log(o1['jk_A_plus_z_S'][:,:,0])-np.log(o1['jk_B_plus_S'][:,:,0])
            f2=4*np.log(o2['jk_A_plus_z_S'][:,:,0])-np.log(o2['jk_B_plus_S'][:,:,0])
            se=np.sqrt(29*(np.var(f1,axis=0)+np.var(f2,axis=0)))/lr
            for j,t in enumerate(T):consistency.append(dict(g_d=gd,L1=l1,L2=l2,T=t,eta_A_plus_z=ea[j],eta_B_plus=eb[j],
                difference=eb[j]-4*ea[j],difference_se=se[j],eta_B_scalar_Y=2*t/(np.pi*runs[l2]['helicity'][j])))
    csv_write(OUT/'orbital_stability.csv',stability);csv_write(OUT/'charged_mean_check.csv',means)
    csv_write(OUT/'charge_consistency.csv',consistency)
    assert max(identities.values())<1e-9
    assert all(v==0 for row in quality.values() for v in row.values())
    (OUT/'summary_identity_checks.json').write_text(json.dumps(identities,indent=2),encoding='utf-8')
    (OUT/'structure_quality_checks.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    validation=validate()
    controls=[]
    for h in (0.,.2,1.,-1.):
        u=1.;phase=.37;d=np.array([0.,0.,1.]);ap=u*np.exp(1j*phase)*d;am=1j*h*np.exp(1j*phase)*d
        ax=(ap+am)/np.sqrt(2);ay=(ap-am)/np.sqrt(2)
        kappa=2*np.vdot(ax,ay).imag;P=np.sum(ax*ax-ay*ay);Bpp=np.sum(ap*ap)
        assert abs(P+1j*kappa*np.exp(2j*phase))<1e-14
        controls.append(dict(h=h,kappa=kappa,P_abs=abs(P),Bpp_abs=abs(Bpp)))
    validation['restricted_chirality_phase_controls']=controls
    (OUT/'interpretation_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')


if __name__=='__main__':main()
