"""Comparison tables and plots from the reduced cluster data; CPU only."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
FIG=HERE/'figures'
COLORS=dict(A='C0',B='C1',C='C2')


def read(path):
    rows=list(csv.DictReader(path.open(encoding='utf8')))
    for row in rows:
        for k,v in row.items():
            try:row[k]=float(v)
            except ValueError:pass
    return rows


def select(rows,**kw):
    return [r for r in rows if all(r[k]==v for k,v in kw.items())]


def write(name,rows):
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    st=read(OUT/'statistics.csv');s=read(OUT/'summary.csv');native=read(OUT/'native_statistics.csv')
    reference=read(ROOT/'exploration/he3_pt_ladder_validation/results/long_reference.csv')
    comparisons=[];refs=[];halves=[];pooled=[];spatial_contrasts=[]
    for a in st:
        if a['initialization']=='ordered':
            b=select(st,g_d=a['g_d'],ladder=a['ladder'],initialization='random',T=a['T'],observable=a['observable'],block_sweeps=a['block_sweeps'])[0]
            err=np.hypot(a['se'],b['se'])
            comparisons.append(dict(g_d=a['g_d'],ladder=a['ladder'],T=a['T'],observable=a['observable'],block_sweeps=a['block_sweeps'],
                ordered=a['value'],random=b['value'],difference=a['value']-b['value'],se=err,z=(a['value']-b['value'])/err))
        if a['block_sweeps']==100000:
            r=select(reference,g_d=a['g_d'],T=a['T'],observable=a['observable'])
            if r:
                err=np.hypot(a['se'],r[0]['se']);d=a['value']-r[0]['value']
                refs.append(dict(case=a['case'],g_d=a['g_d'],ladder=a['ladder'],initialization=a['initialization'],T=a['T'],observable=a['observable'],difference=d,se=err,z=d/err))
            halves.append(dict(case=a['case'],g_d=a['g_d'],ladder=a['ladder'],initialization=a['initialization'],T=a['T'],observable=a['observable'],
                difference=a['half_shift'],se=a['half_combined_se'],z=a['half_shift']/a['half_combined_se']))
    for g in (0,.01):
        for name in ('A','B','C'):
            a=select(s,g_d=g,ladder=name);cost=sum(x['simulation_seconds'] for x in a)
            pooled.append(dict(g_d=g,ladder=name,R=a[0]['R'],simulation_hours=cost/3600,
                seconds_per_passage=cost/sum(x['production_passages'] for x in a),
                seconds_per_common_passage=cost/sum(x['common_passages'] for x in a),
                flux_time=4*a[0]['R']*1e6/sum(x['production_passages'] for x in a),
                passages_per_walker_M=sum(x['production_passages'] for x in a)/(2*a[0]['R']),
                loops=sum(x['production_loops'] for x in a),walkers_with_loop=sum(x['walkers_loop'] for x in a)))
    spatial=read(OUT/'spatial.csv')
    for g in (0,.01):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for name in ('A','B','C'):
            rows=select(spatial,g_d=g,ladder=name,initialization='ordered')
            for label,target in [('cold',.375),('critical',tc),('hot',.402)]:
                temp=min({x['T'] for x in rows},key=lambda t:abs(t-target))
                for a in select(rows,T=temp):
                    b=select(spatial,g_d=g,ladder=name,initialization='random',T=temp,observable=a['observable'])[0]
                    err=np.hypot(a['se'],b['se']);d=a['value']-b['value']
                    spatial_contrasts.append(dict(g_d=g,ladder=name,probe=label,T=temp,observable=a['observable'],ordered=a['value'],random=b['value'],difference=d,se=err,z=d/err if err>0 else np.nan))
    for name,rows in [('initialization_contrasts',comparisons),('reference_contrasts',refs),('half_contrasts',halves),('pooled_transport',pooled),('spatial_contrasts',spatial_contrasts)]:write(name,rows)

    gains=[]
    for g in (0,.01):
        for key in ('Binder','Y'):
            for bs in (50000,100000,200000):
                costs={name:np.mean([r['variance_times_seconds'] for r in select(native,g_d=g,ladder=name,observable=key,block_sweeps=bs)]) for name in ('A','B','C')}
                gains.append(dict(g_d=g,observable=key,block_sweeps=bs,B_gain=costs['A']/costs['B'],C_gain=costs['A']/costs['C']))
    write('precision_gains',gains)
    rates={name:np.mean([r['simulation_seconds']/1500000 for r in select(s,ladder=name)]) for name in ('A','B','C')}
    costs=[dict(experiment='L96 C, four fresh runs, 1M burn + 3M comparison',hours_low=4*4e6*rates['C']/3600,hours_high=4*4e6*rates['C']/3600)]
    costs.append(dict(experiment='L128 A+C, eight runs, 2M burn + 3M comparison',
        hours_low=4*5e6*(rates['A']+rates['C'])*(128/96)**2/3600,
        hours_high=4*5e6*(rates['A']+rates['C'])*(128/96)**2.501519718614108/3600))
    for r in costs:r['hours_with_20percent_allowance']=1.2*r['hours_high']
    write('next_test_costs',costs)

    # Temporal deletes keep correlated walkers/events in the same block.
    returns=[]
    for table in ('excursions','Q_excursions'):
        raw=read(OUT/(table+'.csv'))
        for case in s:
            a=select(raw,case=case['case'])
            for first_only in (False,True):
                if first_only:
                    first={}
                    for r in sorted(a,key=lambda r:r['end']):first.setdefault(r['walker'],r)
                    a=list(first.values())
                def evaluate(rr):
                    x=np.array([r['sign_before'] for r in rr]);y=np.array([r['sign_after'] for r in rr])
                    return np.mean(x!=y),x.mean()*(1-y.mean())+(1-x.mean())*y.mean()
                obs,ind=evaluate(a);deleted=[]
                for b in range(10):
                    rr=[r for r in a if not 500000+b*100000<r['end']<=500000+(b+1)*100000]
                    deleted.append(evaluate(rr))
                deleted=np.asarray(deleted)
                row=dict(case=case['case'],g_d=case['g_d'],ladder=case['ladder'],initialization=case['initialization'],
                    cadence=50 if table=='excursions' else 1000,first_per_walker=first_only,events=len(a),walkers=len(set(r['walker'] for r in a)),
                    opposite_fraction=obs,opposite_se=np.sqrt(9*np.var(deleted[:,0])),independent_marginal_opposite=ind,
                    difference_from_independent=obs-ind,difference_se=np.sqrt(9*np.var(deleted[:,0]-deleted[:,1])),
                    Q_dot_mean=np.mean([r['Q_dot'] for r in a]) if table=='Q_excursions' else np.nan)
                returns.append(row)
    write('excursion_summary',returns)

    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'axes.formatter.useoffset':False})
    labels=[f'{name}\n{start}' for name in ('A','B','C') for start in ('ordered','random')]
    cases=[(name,start) for name in ('A','B','C') for start in ('ordered','random')]
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,(file,key,label) in enumerate([('acceptance','acceptance','Adjacent swap acceptance'),('flow','flow','Last-cold endpoint fraction')]):
            rows=read(OUT/(file+'.csv'))
            for name,start in cases:
                d=select(rows,g_d=g,ladder=name,initialization=start)
                ax[row,col].plot([r['T'] for r in d],[r[key] for r in d],'-' if start=='ordered' else '--',color=COLORS[name],label=f'{name}, {start}')
            ax[row,col].set(xlabel='Temperature',ylabel=label,title=f'g_D={g:g}');ax[row,col].legend(fontsize=8,ncol=2)
    save(fig,'01_acceptance_flow')

    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for i,(name,start) in enumerate(cases):
            r=select(s,g_d=g,ladder=name,initialization=start)[0]
            ax[0,col].bar(i,r['flux_time']/1e6,color=COLORS[name],alpha=1 if start=='ordered' else .5)
            ax[0,col].errorbar(i,r['flux_time']/1e6,yerr=r['flux_se']/1e6,color='k',capsize=2)
            ax[0,col].plot(i,r['null_RT']/1e6,'kx',label='Acceptance-only prediction' if i==0 else None)
            ax[1,col].bar(i,r['seconds_per_passage'],color=COLORS[name],alpha=1 if start=='ordered' else .5)
        for row in range(2):ax[row,col].set_xticks(range(6),labels,fontsize=8);ax[row,col].set_title(f'g_D={g:g}')
        ax[0,col].legend(fontsize=8)
    ax[0,0].set_ylabel('Endpoint traffic scale (M sweeps)');ax[1,0].set_ylabel('Total simulation seconds / production passage')
    save(fig,'02_transport_cost')

    fig,ax=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for col,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for row,key in enumerate(('energy','m2','Binder','Y')):
            rr=select(reference,g_d=g,T=tc,observable=key)[0];a=ax[row,col]
            a.axhline(rr['value'],color='k');a.axhspan(rr['value']-rr['se'],rr['value']+rr['se'],color='.7',alpha=.4)
            for i,(name,start) in enumerate(cases):
                r=select(st,g_d=g,ladder=name,initialization=start,T=tc,observable=key,block_sweeps=100000)[0]
                a.errorbar(i,r['value'],yerr=r['se'],fmt='o' if start=='ordered' else 's',color=COLORS[name],capsize=3)
            a.set_xticks(range(6),labels,fontsize=8);a.set_ylabel(key)
        ax[0,col].set_title(f'Critical probe, g_D={g:g}; 100k-block errors')
    save(fig,'03_critical_means')

    time=read(OUT/'time_blocks.csv')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for col,name in enumerate(('A','B','C')):
            for start in ('ordered','random'):
                d=select(time,g_d=g,ladder=name,initialization=start,T=tc)
                ax[row,col].plot(np.array([r['sweep'] for r in d])/1e6,[r['m2'] for r in d],'-' if start=='ordered' else '--',color=COLORS[name],label=start)
            ax[row,col].axvline(.5,color='.5',ls=':');ax[row,col].set(xlabel='Sweep (millions)',ylabel='Squared chirality',title=f'{name}, g_D={g:g}');ax[row,col].legend(fontsize=8)
    save(fig,'04_relaxation')

    fig,ax=plt.subplots(4,2,figsize=(12,13),layout='constrained')
    for col,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for row,key in enumerate(('Binder','Y','spin','chirality')):
            for name,start in cases:
                d=select(st,g_d=g,ladder=name,initialization=start,T=.375 if key in ('spin','chirality') else tc,observable=key)
                ax[row,col].plot(np.array([r['block_sweeps'] for r in d])/1000,[r['se'] for r in d],'-o' if start=='ordered' else '--s',color=COLORS[name],ms=3,label=f'{name}, {start}')
            ax[row,col].set(xscale='log',yscale='log',xlabel='Block length (k sweeps)',ylabel=f'{key} standard error',title=f'g_D={g:g}');ax[row,col].legend(fontsize=7,ncol=2)
    save(fig,'05_blocking')

    wm=read(OUT/'walker_memory.csv')
    fig,ax=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,key in enumerate(('energy','spin','chirality')):
            for name,start in cases:
                d=select(wm,g_d=g,ladder=name,initialization=start,observable=key)
                ax[row,col].plot([r['lag'] for r in d],[r['correlation'] for r in d],'-o' if start=='ordered' else '--s',color=COLORS[name],ms=3,label=f'{name}, {start}')
            ax[row,col].set(xscale='log',xlabel='Lag (sweeps)',ylabel=f'Same-walker {key} residual correlation',title=f'g_D={g:g}');ax[row,col].legend(fontsize=7,ncol=2)
    save(fig,'06_walker_memory')

    surv=read(OUT/'survival.csv')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,kind in enumerate(('first_passage','first_loop')):
            for name,start in cases:
                d=select(surv,g_d=g,ladder=name,initialization=start,kind=kind)
                ax[row,col].step(np.array([r['time'] for r in d])/1e6,[r['survival'] for r in d],where='post',linestyle='-' if start=='ordered' else '--',color=COLORS[name],label=f'{name}, {start}')
            ax[row,col].set(xlabel='Time since first endpoint (M sweeps)',ylabel=kind+' survival',ylim=(-.02,1.02),title=f'g_D={g:g}');ax[row,col].legend(fontsize=7,ncol=2)
    save(fig,'07_censoring')

    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,key in enumerate(('Binder','Y')):
            for i,(name,start) in enumerate(cases):
                d=select(native,g_d=g,ladder=name,initialization=start,observable=key)
                mid=select(d,block_sweeps=100000)[0]['ESS_per_total_hour']
                ax[row,col].plot([i,i],[min(x['ESS_per_total_hour'] for x in d),max(x['ESS_per_total_hour'] for x in d)],color=COLORS[name])
                ax[row,col].plot(i,mid,'o' if start=='ordered' else 's',color=COLORS[name])
            ax[row,col].set(yscale='log',ylabel=f'{key}: conditional ESS / total GPU-hour',title=f'g_D={g:g}');ax[row,col].set_xticks(range(6),labels,fontsize=8)
    save(fig,'08_precision_cost')

    hot=read(OUT/'hot_memory.csv')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,key in enumerate(('energy','Q_tensor')):
            for name,start in cases:
                d=[r for r in select(hot,g_d=g,ladder=name,initialization=start,observable=key) if r['lag']<=5000]
                ax[row,col].plot([r['lag'] for r in d],[r['correlation'] for r in d],'-o' if start=='ordered' else '--s',color=COLORS[name],ms=3,label=f'{name}, {start}')
            ax[row,col].set(xscale='log',xlabel='Lag (sweeps)',ylabel=f'Continuous hot-stay {key} correlation',title=f'g_D={g:g}');ax[row,col].legend(fontsize=7,ncol=2)
    save(fig,'09_hot_memory')
    print('Comparison tables and nine PNG/PDF figure pairs saved.')


if __name__=='__main__':main()
