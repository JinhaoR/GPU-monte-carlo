"""Endpoint-labelled structural memory, and raw-grid transition stability."""
import csv,numpy as np
from extract import OUT,write
from transport import events
from physical import evaluate
from exploration.he3_transition_L96_160.analyze import crossing,interval

def main():
    inv=list(csv.DictReader((OUT/'inventory.csv').open()));rows=[];within=[]
    for m in inv:
        tag=m['tag'];L=int(m['L']);T=None;base=dict(campaign=m['campaign'],g_d=float(m['g_d']),L=L)
        with np.load(OUT/(tag+'.npz')) as z:d=dict(z)
        T=d['T'];R=len(T);tc=float(m['Cv_peak_T']);selected=np.where(abs(T-tc)<.0015)[0]
        labels=events(d['positions'],d['sweeps'])[-1];ids=np.searchsorted(d['sweeps'],d['physical_sweeps']);p=d['physical_positions'];order=np.argsort(p,axis=1)
        labels=np.take_along_axis(labels[ids],order,axis=1)
        values=dict(energy=d['walker_energy'],m2=d['walker_chirality']**2,P2=d['walker_P_real']**2+d['walker_P_imag']**2,spin=d['walker_qzz'])
        for name,a in values.items():
            v=np.take_along_axis(a,order,axis=1).astype(float);scale=v.std(0)
            counts=[];sums=[]
            for endpoint in (0,1):
                mask=labels==endpoint;counts.append(mask.reshape(30,200,R).sum(1));sums.append((v*mask).reshape(30,200,R).sum(1))
            counts=np.array(counts);sums=np.array(sums)
            for k in selected:
                n=counts[:,:,k].sum(1);s=sums[:,:,k].sum(1)
                if n.min()<100:continue
                means=s/n;jk=(s[:,None]-sums[:,:,k])/(n[:,None]-counts[:,:,k]);diff=jk[0]-jk[1]
                rows.append(dict(**base,T=T[k],observable=name,cold_count=int(n[0]),hot_count=int(n[1]),cold_label_mean=means[0],hot_label_mean=means[1],difference=means[0]-means[1],difference_se=np.sqrt(29*np.var(diff)),difference_in_sigma=(means[0]-means[1])/scale[k]))
        # Staying in a band is a selection condition, not a canonical ACF estimate.
        temp=T[p];bad=np.vstack((np.zeros((1,R),int),np.cumsum(abs(temp-tc)>.0005,axis=0)))
        for lag in (1,2,10):
            keep=(bad[lag+1:]-bad[:-lag-1])==0
            for key in ('energy_residual','abs_m_residual','P2_residual','qzz_residual'):
                a=d['walker_'+key];x=a[:-lag][keep];y=a[lag:][keep]
                within.append(dict(**base,observable=key,lag=lag*500,pairs=len(x),correlation=np.corrcoef(x,y)[0,1] if len(x)>20 else np.nan))
    write('endpoint_conditioned_structure.csv',rows);write('critical_residence_memory.csv',within)
    # Four-size transition comparisons remain L>=96; L64 is only used elsewhere
    # for transport/scaling as explicitly requested in this investigation.
    transitions=[];contrasts=[];rng=np.random.default_rng(20261102)
    for gd in (0,.01):
        datasets=[]
        for L in (96,128,160,256):
            with np.load(OUT/f'new_gd={gd:g}_L={L}.npz') as z:datasets.append({k[6:]:z[k] for k in z.files if k.startswith('block_')});T=z['T']
        center=.38183 if gd==0 else .38558
        for scenario,factor,sl in [('main',2,slice(None)),('500k',10,slice(None)),('first',2,slice(0,30)),('second',2,slice(30,None))]:
            central=[];boot=[];B=600 if scenario=='main' else 200
            for L,d in zip((96,128,160,256),datasets):
                b={k:v[sl].reshape((-1,factor,len(T))).mean(1) for k,v in d.items()};n=len(b['e']);w=rng.multinomial(n,np.full(n,1/n),size=B)/n
                central.append(evaluate({k:v.mean(0) for k,v in b.items()},T,L));boot.append(evaluate({k:w@v for k,v in b.items()},T,L))
            cb={}
            for a,b in ((0,1),(1,2),(0,2),(0,3),(1,3),(2,3)):
                t,nr=crossing(T,central[a]['Binder']-central[b]['Binder'],center-.00065,center+.00065,center)
                bb=np.array([crossing(T,x-y,center-.00065,center+.00065,t)[0] for x,y in zip(boot[a]['Binder'],boot[b]['Binder'])]);cb[a,b]=(t,bb)
                transitions.append(dict(g_d=gd,scenario=scenario,quantity='Binder',sizes=f'{(96,128,160,256)[a]}_{(96,128,160,256)[b]}',T=t,**interval(bb)))
            for q in (.5,1.):
                jump=2*T/(np.pi*q*q);t,nr=crossing(T,central[3]['Y']-jump,T[0],T[-1]);bb=np.array([crossing(T,v-jump,T[0],T[-1],t)[0] for v in boot[3]['Y']])
                transitions.append(dict(g_d=gd,scenario=scenario,quantity=f'direct_q{q:g}',sizes='256',T=t,**interval(bb)))
                contrasts.append(dict(g_d=gd,scenario=scenario,q=q,difference=cb[2,3][0]-t,**interval(cb[2,3][1]-bb)))
    write('raw_crossing_stability.csv',transitions);write('raw_crossing_contrasts.csv',contrasts)
    print('Endpoint-conditioned and transition checks complete',flush=True)

if __name__=='__main__':main()
