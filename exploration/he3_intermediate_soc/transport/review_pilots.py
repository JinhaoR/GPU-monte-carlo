"""Independent preparation/half comparisons and production-only walker checks.

Reads completed local pilots; never runs MC or rewrites their analysis outputs.
"""
from pathlib import Path
import csv,json,sys
from itertools import combinations
import numpy as np
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from analyze_pilots import moments,evaluate

PROBES=(0,6,10,12,23)
OBS=('e','chi','q','Y','eps','planar','Azplus_S0','Aplanar_S0','Yxy')


def write(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def persistent(flags,hold,step):
    forward=reverse=changed=0
    for f in flags.T:
        cuts=np.r_[0,1+np.flatnonzero(f[1:]!=f[:-1]),len(f)]
        accepted=[f[lo] for lo,hi in zip(cuts[:-1],cuts[1:]) if f[lo] and (hi-lo-1)*step>=hold]
        pairs=[(a,b) for a,b in zip(accepted[:-1],accepted[1:]) if a!=b]
        forward+=sum(a==1 for a,b in pairs);reverse+=sum(a==-1 for a,b in pairs);changed+=bool(pairs)
    return int(forward),int(reverse),int(changed)


def main():
    curves=list(csv.DictReader((PARENT/'pilot_results/curves.csv').open()))
    by={(r['label'],r['subset'],float(r['T'])):r for r in curves}
    analyzed={r['label'] for r in curves}
    prep=[];halves=[];walk=[];block_rows=[];cdf=[];loaded={};metas=[]
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for path in sorted((PARENT/'pilots').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['label'] not in analyzed:continue
        metas.append(meta)
        label=meta['label'];L=meta['L'];g=meta['g_d'];burn=meta['analysis_burn'];end=meta['total_sweeps'];N=L*L
        with np.load(path) as z:raw=dict(z)
        loaded[label]=raw
        T=raw['T'];ids=raw['sweeps']>burn;v=raw['values'][ids];mapping=raw['temperature_to_walker'][ids]
        hist_ids=np.searchsorted(raw['walker_history_sweeps'],raw['sweeps'][ids])
        pos=raw['walker_history_indices'][hist_ids]
        assert np.array_equal(np.argsort(mapping,axis=1),pos)
        w=np.empty_like(v);w[np.arange(len(v))[:,None],mapping]=v
        q=w[:,:,9];chi=abs(w[:,:,2]/w[:,:,1])
        flags=np.where((q>.25)&(chi>.25),1,np.where((q<-.1)&(chi<.15),-1,0))
        high_inventory=(flags==1).sum(1);low_inventory=(flags==-1).sum(1)
        if L==16:
            ax=axs.flat[(.04,.06,.08,.1).index(g)];factor=25
            ax.plot((raw['sweeps'][ids].reshape(-1,factor).mean(1)-burn)/1000,
                high_inventory.reshape(-1,factor).mean(1),
                label=f"{meta['preparation']}; loops={sum(meta['pt']['walker_round_trips'])}")
            ax.set(title=f'g_D={g:g}, L=16',xlabel='production sweeps / 1000',ylabel='chiral-like inventory, 500-sweep mean')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        both=loops=0
        for p in pos.T:
            hit=p[(p==0)|(p==len(T)-1)]
            both+=bool(np.any(hit==0) and np.any(hit==len(T)-1))
            if len(hit):
                origin=hit[0];other=False
                for h in hit:
                    if h!=origin:other=True
                    elif other:loops+=1;other=False
        for hold in (100,200,400):
            forward,reverse,changed=persistent(flags,hold,meta['scalar_interval'])
            walk.append(dict(label=label,L=L,g_d=g,preparation=meta['preparation'],scope='production',hold_sweeps=hold,
                chiral_like_to_low_q=forward,low_q_to_chiral_like=reverse,walkers_with_conversion=changed,
                sampled_production_loops=loops,production_walkers_both_endpoints=both,
                high_inventory_first_half=float(high_inventory[:len(v)//2].mean()),high_inventory_second_half=float(high_inventory[len(v)//2:].mean()),
                low_inventory_first_half=float(low_inventory[:len(v)//2].mean()),low_inventory_second_half=float(low_inventory[len(v)//2:].mean()),
                exact_loops_including_burn=sum(meta['pt']['walker_round_trips'])))
        b=moments(raw,L,burn)
        for factor in (1,2,3):
            grouped={k:a.reshape((12//factor,factor)+a.shape[1:]).mean(1) for k,a in b.items()}
            B=12//factor;m={k:a.mean(0) for k,a in grouped.items()};value=evaluate(m,T,L)
            jk=evaluate({k:(B*m[k]-a)/(B-1) for k,a in grouped.items()},T,L)
            for j in PROBES:
                for key in OBS:
                    block_rows.append(dict(label=label,L=L,g_d=g,preparation=meta['preparation'],T=T[j],observable=key,
                        blocks=B,block_sweeps=(end-burn)//B,value=float(value[key][j]),se=float(np.sqrt((B-1)*np.var(jk[key][:,j])))))
        for j in PROBES:
            first=by[(label,'first',T[j])];second=by[(label,'second',T[j])]
            for key in OBS:
                a,bv=float(first[key]),float(second[key]);ae,be=float(first[key+'_se']),float(second[key+'_se'])
                halves.append(dict(label=label,L=L,g_d=g,preparation=meta['preparation'],T=T[j],observable=key,
                    first=a,first_se=ae,second=bv,second_se=be,difference=bv-a,conditional_z=(bv-a)/np.hypot(ae,be)))
    for L,g in sorted(set((m['L'],m['g_d']) for m in metas)):
        cases=[m for m in metas if m['L']==L and m['g_d']==g]
        for a,b in combinations(cases,2):
            T=loaded[a['label']]['T'];assert np.array_equal(T,loaded[b['label']]['T'])
            for j in PROBES:
                x=by[(a['label'],'full',T[j])];y=by[(b['label'],'full',T[j])]
                for key in OBS:
                    xv,yv=float(x[key]),float(y[key]);xe,ye=float(x[key+'_se']),float(y[key+'_se'])
                    prep.append(dict(L=L,g_d=g,T=T[j],first=a['preparation'],second=b['preparation'],observable=key,
                        first_value=xv,first_se=xe,second_value=yv,second_se=ye,difference=yv-xv,conditional_z=(yv-xv)/np.hypot(xe,ye)))
                for key,column in [('q',9),('e',0),('chi',2)]:
                    arrays=[]
                    for meta in (a,b):
                        z=loaded[meta['label']];v=z['values'][z['sweeps']>meta['analysis_burn'],j]
                        arrays.append(v[:,column] if key=='q' else v[:,column]/L**2 if key=='e' else abs(v[:,2]/v[:,1]))
                    # Empirical CDF distance only: no iid p-value from correlated MC.
                    cdf.append(dict(L=L,g_d=g,T=T[j],first=a['preparation'],second=b['preparation'],observable=key,
                        empirical_CDF_distance=float(ks_2samp(*arrays).statistic)))
    write('pilot_preparation_comparisons.csv',prep);write('pilot_half_comparisons.csv',halves)
    write('pilot_production_walkers.csv',walk);write('pilot_block_sensitivity.csv',block_rows);write('pilot_distribution_comparisons.csv',cdf)
    fig.savefig(HERE/'pilot_inventories.png',dpi=180);plt.close(fig)
    print('Reviewed',len(metas),'completed pilots')


if __name__=='__main__':main()
