"""Figures from pilot tables; no GPU access."""
import csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ladder_design import HERE,OUT
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160,'axes.formatter.useoffset':False})
COLORS={'A':'C0','B':'C1','C':'C2'}

def read(name):
    rows=list(csv.DictReader((OUT/(name+'.csv')).open()));d={}
    for k in rows[0]:
        try:d[k]=np.array([float(r[k]) for r in rows])
        except ValueError:d[k]=np.array([r[k] for r in rows])
    return d

def select(d,**filters):
    keep=np.ones(len(next(iter(d.values()))),bool)
    for k,v in filters.items():keep &= d[k]==v
    return {k:v[keep] for k,v in d.items()}

def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    s=read('summary');st=read('statistics');ref=read('long_reference')
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ('A','B','C'):
        T=np.loadtxt(HERE/f'ladder_{name}.txt')
        ax[0].plot(T,np.arange(len(T)),color=COLORS[name],label=f'{name}: R={len(T)}')
        ax[1].plot((T[1:]+T[:-1])/2,np.diff(T),color=COLORS[name],label=name)
    ax[0].set(xlabel='Temperature',ylabel='Temperature index');ax[1].set(xlabel='Temperature',ylabel='Neighbor temperature spacing',yscale='log')
    for a in ax:a.legend()
    save(fig,'01_ladders')
    labels=[f'{a}\n{b}' for a in ('A','B','C') for b in ('ordered','random')]
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for i,(ladder,start) in enumerate((a,b) for a in ('A','B','C') for b in ('ordered','random')):
            d=select(s,g_d=g,ladder=ladder,initialization=start)
            if not len(d['L']):continue
            for row,key in enumerate(('passages_per_walker_M','common_passages_per_walker_M')):
                ax[row,col].bar(i,d[key][0],color=COLORS[ladder],alpha=1 if start=='ordered' else .5)
                if row==0:ax[row,col].errorbar(i,d[key][0],yerr=d['passage_rate_block_se'][0],color='k',capsize=2)
        for row in range(2):ax[row,col].set_xticks(range(6),labels,fontsize=8);ax[row,col].set_title(f'g_D={g:g}')
    ax[0,0].set_ylabel('Endpoint passages / walker / M sweeps');ax[1,0].set_ylabel('Common-surface passages / walker / M sweeps')
    save(fig,'02_passages')

    fig,ax=plt.subplots(4,2,figsize=(11,12),layout='constrained')
    for col,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for row,key in enumerate(('energy','m2','Binder','Y')):
            r=select(ref,g_d=g,T=tc,observable=key);a=ax[row,col]
            a.axhline(r['value'][0],color='k',label='long saved L96 reference')
            a.axhspan(r['value'][0]-r['se'][0],r['value'][0]+r['se'][0],color='.5',alpha=.2)
            for i,(ladder,start) in enumerate((a,b) for a in ('A','B','C') for b in ('ordered','random')):
                d=select(st,g_d=g,T=tc,observable=key,block_sweeps=5000,ladder=ladder,initialization=start)
                if len(d['value']):a.errorbar(i,d['value'][0],yerr=d['se'][0],fmt='o' if start=='ordered' else 's',color=COLORS[ladder],capsize=3)
            a.set_xticks(range(6),labels,fontsize=8);a.set_ylabel(key)
        ax[0,col].set_title(f'Critical-region probe, g_D={g:g}')
    save(fig,'03_initialization_reference')

    blocks=read('time_blocks');fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for col,name in enumerate(('A','B','C')):
            for start,style in [('ordered','-'),('random','--')]:
                d=select(blocks,g_d=g,ladder=name,initialization=start,target_T=tc)
                if len(d['sweep']):
                    ax[row,col].plot(d['sweep']/1000,d['m2'],style,lw=.7,alpha=.3,color=COLORS[name])
                    ax[row,col].plot(d['sweep'].reshape(-1,5).mean(1)/1000,d['m2'].reshape(-1,5).mean(1),style,label=start,color=COLORS[name])
            r=select(ref,g_d=g,T=tc,observable='m2');ax[row,col].axhline(r['value'][0],color='k',lw=.7)
            ax[row,col].axvline(20,color='.5',ls=':');ax[row,col].set(title=f'{name}, g_D={g:g}',xlabel='Sweep (thousands)',ylabel='Squared chirality near heat peak');ax[row,col].legend(fontsize=8)
    save(fig,'04_relaxation')

    flow=read('flow');fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for name in ('A','B','C'):
            for start,style in [('ordered','-'),('random','--')]:
                d=select(flow,g_d=g,ladder=name,initialization=start)
                for row,key in enumerate(('flow','labelled_fraction')):ax[row,col].plot(d['T'],d[key],style,color=COLORS[name],label=f'{name}, {start}')
        for a in ax[:,col]:a.set(xlabel='Temperature',ylim=(-.02,1.02));a.legend(fontsize=7,ncol=2)
        ax[0,col].set_title(f'g_D={g:g}')
    ax[0,0].set_ylabel('Fraction last visiting cold endpoint');ax[1,0].set_ylabel('Fraction with a known previous endpoint')
    save(fig,'05_flow_coverage')

    surv=read('survival');fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,kind in enumerate(('passage','first_loop')):
            for name in ('A','B','C'):
                for start,style in [('ordered','-'),('random','--')]:
                    d=select(surv,g_d=g,ladder=name,initialization=start,kind=kind)
                    ax[row,col].step(d['time']/1000,d['survival'],where='post',linestyle=style,color=COLORS[name],label=f'{name}, {start}')
            ax[row,col].set(xlabel='Time since starting endpoint (k sweeps)',ylabel=kind+' survival',ylim=(-.02,1.02),xlim=(0,40));ax[row,col].legend(fontsize=7,ncol=2)
        ax[0,col].set_title(f'g_D={g:g}')
    save(fig,'06_censoring')

    mem=read('memory');fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        for row,observable in enumerate(('energy','spin')):
            for name in ('A','B','C'):
                for start,style in [('ordered','-'),('random','--')]:
                    d=select(mem,g_d=g,ladder=name,initialization=start,band='hot',observable=observable)
                    keep=d['lag_sweeps']<=1000
                    ax[row,col].plot(d['lag_sweeps'][keep],d['correlation'][keep],style+'o',color=COLORS[name],label=f'{name}, {start}',ms=3)
            ax[row,col].set(xlabel='Lag (sweeps)',ylabel=f'Conditional {observable} residual correlation',title=f'Continuous hot stay, g_D={g:g}',xticks=[500,1000]);ax[row,col].legend(fontsize=7,ncol=2)
    save(fig,'07_physical_memory')

    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,g in enumerate((0,.01)):
        tc=.38210530380170976 if g==0 else .3856721419201371
        for row,key in enumerate(('Binder','Y')):
            for i,(name,start) in enumerate((a,b) for a in ('A','B','C') for b in ('ordered','random')):
                d=select(st,g_d=g,T=tc,observable=key,ladder=name,initialization=start)
                if len(d['block_ESS_per_total_hour']):
                    mid=d['block_ESS_per_total_hour'][d['block_sweeps']==5000][0]
                    ax[row,col].plot([i,i],[d['block_ESS_per_total_hour'].min(),d['block_ESS_per_total_hour'].max()],color=COLORS[name])
                    ax[row,col].plot(i,mid,'o' if start=='ordered' else 's',color=COLORS[name])
            ax[row,col].set(yscale='log',ylabel=f'{key}: conditional ESS / total simulated hour',title=f'g_D={g:g}');ax[row,col].set_xticks(range(6),labels,fontsize=8)
    save(fig,'08_conditional_efficiency')
    print('Eight PNG/PDF figure pairs written; all pilot errors are conditional.')

if __name__=='__main__':main()
