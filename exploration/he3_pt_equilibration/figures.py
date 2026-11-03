"""Make the report figures from the saved tables; no simulation imports."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT, HERE

FIG = HERE/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2,
                     'savefig.dpi': 170, 'axes.formatter.useoffset': False})

def read(name):
    rows=list(csv.DictReader((OUT/(name+'.csv')).open())); cols={}
    for key in rows[0]:
        if key=='sizes':
            cols[key]=np.array([row[key] for row in rows]);continue
        try: cols[key]=np.array([float(row[key]) for row in rows])
        except ValueError: cols[key]=np.array([row[key] for row in rows])
    return cols

def subset(d, **filters):
    mask=np.ones(len(next(iter(d.values()))),bool)
    for k,v in filters.items():mask &= d[k]==v
    return {k:v[mask] for k,v in d.items()}

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    inv=read('inventory');tr=read('transport_summary');edges=read('energy_edges');flow=read('flow')
    colors=dict(zip((64,96,128,160,256),plt.cm.viridis(np.linspace(.08,.9,5))))
    fig,ax=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for col,g in enumerate((0,.01)):
        for L,c in colors.items():
            d=subset(edges,campaign='new',g_d=g,L=L);f=subset(flow,campaign='new',g_d=g,L=L)
            ax[0,col].plot((d['T_left']+d['T_right'])/2,d['acceptance'],color=c,label=f'L={L}')
            ax[1,col].plot(f['T'],f['flow'],color=c)
            with np.load(OUT/f'new_gd={g:g}_L={L}.npz') as z:ax[2,col].plot(z['T'],z['Cv'],color=c)
        old=subset(edges,campaign='old',g_d=g,L=160)
        ax[0,col].plot((old['T_left']+old['T_right'])/2,old['acceptance'],'k--',lw=1,label='old L=160')
        ax[0,col].set_title(f'g_D={g:g}')
        for a in ax[:,col]:a.set_xlim(.379 if g==0 else .3825,.386 if g==0 else .3895);a.set_xlabel('Temperature')
        ax[0,col].set_ylim(0,1.02);ax[1,col].set_ylim(-.02,1.02)
    ax[0,0].set_ylabel('Measured neighbor acceptance');ax[1,0].set_ylabel('Fraction last visiting cold end');ax[2,0].set_ylabel('Specific heat per site')
    ax[0,0].legend(fontsize=8,ncol=2);save(fig,'01_acceptance_flow_heat')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    surv=read('survival')
    for col,g in enumerate((0,.01)):
        for campaign,ls in [('old','--'),('new','-')]:
            d=subset(tr,campaign=campaign,g_d=g);m=subset(inv,campaign=campaign,g_d=g)
            ix=np.argsort(d['L']);im=np.argsort(m['L'])
            ax[0,col].plot(m['L'][im],m['completed_trip_mean'][im]/1e6,'o'+ls,label=f'{campaign}: completed means',color='C0')
            ax[0,col].errorbar(d['L'][ix],d['passage_flux_time'][ix]/1e6,yerr=d['passage_flux_time_block_se'][ix]/1e6,fmt='s'+ls,label=f'{campaign}: inverse passage flux',color='C3')
            if campaign=='new':
                ax[0,col].plot(d['L'][ix],d['null_alternating_RT'][ix]/1e6,':',color='C2',label='alternating acceptance-only model')
                ax[0,col].plot(d['L'][ix],d['coarse500_markov_RT'][ix]/1e6,'-.',color='C4',label='500-sweep Markov model')
        for L in (128,160,256):
            d=subset(surv,campaign='new',g_d=g,L=L,kind='first_loop')
            ax[1,col].step(d['time']/1e6,d['survival'],where='post',color=colors[L],label=f'L={L}')
        ax[0,col].set(xscale='log',yscale='log',xlabel='L',ylabel='Transport scale (million sweeps)',title=f'g_D={g:g}')
        ax[1,col].axhline(.5,color='.5',lw=.8);ax[1,col].set(xlabel='Time since first production endpoint (M sweeps)',ylabel='First-loop survival (descriptive KM)',ylim=(0,1.02),xlim=(0,3))
        ax[1,col].legend();ax[0,col].legend(fontsize=7)
    save(fig,'02_transport_censoring')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');diff=read('diffusion');cond=read('endpoint_conditioned_structure')
    for col,g in enumerate((0,.01)):
        for lag,c in [(500,'C0'),(10000,'C1')]:
            d=subset(diff,campaign='new',g_d=g,L=256,lag=lag)
            ax[0,col].plot(d['T'],d['D_index'],color=c,label=f'measured, lag {lag}')
            ax[0,col].plot(d['T'],d['D_null'],'--',color=c,alpha=.7,label=f'acceptance-only, lag {lag}')
        d=subset(cond,campaign='new',g_d=g,L=256,observable='m2')
        for name,c in [('cold','C0'),('hot','C3')]:ax[1,col].plot(d['T'],d[name+'_label_mean'],'o-',ms=3,color=c,label=f'Last endpoint: {name}')
        ax[0,col].set(yscale='log',xlabel='Temperature',ylabel='Var(index displacement) / (2 lag)',title=f'L=256, g_D={g:g}')
        ax[1,col].set(xlabel='Temperature',ylabel='Mean squared chirality, same temperature')
        ax[0,col].legend(fontsize=7);ax[1,col].legend(fontsize=8)
        ax[1,col].set_xlim((.3809,.383) if g==0 else (.3845,.3865))
    save(fig,'03_diffusion_structural_memory')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');acf=read('fixed_temperature_acf');block=read('blocking_and_drift')
    for col,g in enumerate((0,.01)):
        for key in ('energy','Binder_influence','P2','spin_alignment','helicity_influence'):
            xx=[];yy=[]
            for L in colors:
                tc=subset(inv,campaign='new',g_d=g,L=L)['Cv_peak_T'][0];d=subset(acf,campaign='new',g_d=g,L=L,observable=key)
                j=np.argmin(abs(d['T']-tc));xx.append(L);yy.append(d['tau_sweeps'][j])
            ax[0,col].plot(xx,yy,'o-',label=key)
        tc=subset(inv,campaign='new',g_d=g,L=256)['Cv_peak_T'][0]
        for key in ('energy','Binder','Y','P2'):
            d=subset(block,campaign='new',g_d=g,L=256,observable=key,T=tc)
            ax[1,col].plot(d['block_sweeps']/1000,d['se']/d['se'][0],'o-',label=key)
        ax[0,col].set(yscale='log',xlabel='L',ylabel='Fixed-temperature descriptive tau (sweeps)',title=f'g_D={g:g}')
        ax[1,col].axhline(1,color='.5',lw=.8);ax[1,col].set(xlabel='Block length (thousand sweeps)',ylabel='L=256 standard error / 100k-block error')
        ax[0,col].legend(fontsize=8);ax[1,col].legend(fontsize=8)
    save(fig,'04_acf_blocking')

    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');cr=read('raw_crossing_stability')
    for col,g in enumerate((0,.01)):
        for key,sz,offset,c in [('Binder','160_256',-.08,'C0'),('direct_q0.5','256',.08,'C3')]:
            ds=subset(cr,g_d=g,quantity=key,sizes=sz)
            for i,scenario in enumerate(('main','500k','first','second')):
                d=subset(ds,scenario=scenario);ax[col].errorbar(i+offset,d['T'],yerr=d['se'],fmt='o',color=c,label=key if i==0 else None)
        ax[col].set_xticks(range(4),['100k blocks','500k blocks','first half','second half'],rotation=15)
        ax[col].set(title=f'g_D={g:g}',ylabel='Finite-size crossing temperature');ax[col].legend(fontsize=8)
    save(fig,'05_transition_stability')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');ld=read('ladder_predictions')
    names=('current','simple_best','energy','blend25','blend50','feedback','narrow_energy');labels=('current 200','simple 200','energy 200','25% flow 200','50% flow 200','flow only 200','narrow 160')
    for col,g in enumerate((0,.01)):
        d=subset(ld,L=320,scenario='area_Cv_critical',g_d=g)
        for i,(name,label) in enumerate(zip(names,labels)):
            a=subset(d,design=name,R=160 if name=='narrow_energy' else 200)
            ax[0,col].bar(i,a['min_predicted_acceptance'][0],color=plt.cm.tab10(i))
            ax[1,col].plot(i,a['R_times_null_RT'][0]/1e6,'o',color=plt.cm.tab10(i))
        ax[0,col].set(title=f'L=320 stress scenario, g_D={g:g}',ylabel='Minimum predicted neighbor acceptance',ylim=(0,1))
        ax[1,col].set_ylabel('R x acceptance-only round trip (M replica sweeps)')
        for a in ax[:,col]:a.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
    save(fig,'06_ladder_tradeoffs')

    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');cost=read('resource_scenarios')
    for R,c in [(160,'C0'),(200,'C1'),(256,'C3')]:
        d=subset(cost,R=R,total_sweeps=16000000);ix=np.argsort(d['L'])
        ax[0].plot(d['L'][ix],d['hours_recent'][ix],'o-',color=c,label=f'R={R}, recent cost trend')
        ax[0].fill_between(d['L'][ix],d['hours_area'][ix],d['hours_recent'][ix],alpha=.12,color=c)
        ax[1].plot(d['L'][ix],d['GPU_core_GiB'][ix],'o-',color=c,label=f'R={R}')
    ax[0].axhline(72,color='k',ls='--',label='72 h wall limit');ax[0].axhline(57.6,color='.4',ls=':',label='20% wall-time reserve')
    ax[0].set(xlabel='L',ylabel='Hours for 8M + 8M sweeps',title='Conditional on comparable hardware')
    ax[1].set(xlabel='L',ylabel='Core field + RNG + counters (GiB)',title='Workspace, context and history additional')
    for a in ax:a.legend(fontsize=8)
    save(fig,'07_resources')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');res=read('residence')
    for col,g in enumerate((0,.01)):
        for L in (96,128,160,256):
            d=subset(surv,campaign='new',g_d=g,L=L,kind='passage')
            ax[0,col].step(d['time']/1e6,d['survival'],where='post',color=colors[L],label=f'L={L}')
        for region,name,c in [(0,'below critical band','C0'),(1,'inside critical band','C2'),(2,'above critical band','C3')]:
            d=subset(res,campaign='new',g_d=g,L=256,region=region,left_censored='False',right_censored='False')
            x=np.sort(d['duration']);ax[1,col].step(x,1-np.arange(len(x))/len(x),where='post',color=c,label=name)
        ax[0,col].set(xlabel='Endpoint-to-opposite-end passage time (M sweeps)',ylabel='Passage survival (descriptive KM)',title=f'g_D={g:g}',xlim=(0,3),ylim=(0,1.02))
        ax[1,col].set(xlabel='Completed sampled residence (sweeps)',ylabel='Fraction longer than duration',xscale='log',yscale='log',title='L=256; boundary-censored episodes excluded')
        for a in ax[:,col]:a.legend(fontsize=8)
    save(fig,'08_passages_residence')
    print('Eight PNG/PDF report figures written.')

if __name__=='__main__':main()
