"""Joint blocked, preparation-resolved analysis of bounded local experiments."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'pilot_results';OUT.mkdir(exist_ok=True)


def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def moments(z,L,burn,stop=None):
    ids=(z['sweeps']>burn)&(z['sweeps']<=(stop or z['sweeps'][-1]))
    sid=ids[4::5];v=z['values'][ids];sv=z['sparse_values'][sid]
    a=z['amplitudes'][sid];s=z['structure'][sid];x=z['extra'][sid];N=L*L
    n=v[:,:,1];chi=v[:,:,2]/n
    m=dict(e=v[:,:,0]/N,n=n/N,chi=abs(chi),signed=chi,q=v[:,:,9],
        D=(v[:,:,6]+v[:,:,8])/(2*N),ix=v[:,:,5],iy=v[:,:,7],ix2=v[:,:,5]**2,iy2=v[:,:,7]**2,
        kx=v[:,:,6]/N,ky=v[:,:,8]/N,ixiy=v[:,:,5]*v[:,:,7],
        skx=sv[:,:,6]/N,sky=sv[:,:,8]/N,skxy=x[:,:,0]/N,six=sv[:,:,5],siy=sv[:,:,7],
        six2=sv[:,:,5]**2,siy2=sv[:,:,7]**2,sixiy=sv[:,:,5]*sv[:,:,7],
        sn=s[:,:,0],eps=s[:,:,6],orb=s[:,:,7],spin=s[:,:,8],planar=x[:,:,1],
        Q1=abs(a[:,:,5:7,:])**2,Q1mean=a[:,:,5:7,:])
    fields={'P':a[:,:,1,:], 'Bpp':(a[:,:,7,:]+a[:,:,8,:]+2*a[:,:,9,:])/2,
        'Bmm':(a[:,:,7,:]+a[:,:,8,:]-2*a[:,:,9,:])/2,'Btrace':(a[:,:,7,:]+a[:,:,8,:])/np.sqrt(2),
        'Azplus':(a[:,:,14,:]+a[:,:,15,:])/np.sqrt(2),'Azminus':(a[:,:,14,:]-a[:,:,15,:])/np.sqrt(2),
        'Aplanar':(a[:,:,11,:]-a[:,:,12,:])/np.sqrt(2)}
    for name,f in fields.items():m[name]=abs(f)**2;m[name+'mean']=f
    f=a[:,:,10:,:].reshape(len(a),len(z['T']),3,2,5)
    for name,sgn in [('Aplus',1),('Aminus',-1)]:m[name]=(abs((f[:,:,:,0,:]+sgn*f[:,:,:,1,:])/np.sqrt(2))**2).sum(2)
    # Twelve joint time blocks; both cadences divide their physical durations.
    return {k:r.reshape((12,len(r)//12)+r.shape[1:]).mean(1) for k,r in m.items()}


def evaluate(m,T,L):
    N=L*L;out={k:m[k] for k in ('e','n','chi','signed','q','D')}
    out['Y']=m['D']-(m['ix2']-m['ix']**2+m['iy2']-m['iy']**2)/(2*N*T)
    out['Yxx']=m['kx']-(m['ix2']-m['ix']**2)/(N*T)
    out['Yyy']=m['ky']-(m['iy2']-m['iy']**2)/(N*T)
    out['Yxy']=m['skxy']-(m['ixiy']-m['ix']*m['iy'])/(N*T)
    # Retain the synchronized sparse-only comparison; the main estimates use
    # all available currents and equal-time-window joint blocks at both cadences.
    out['Yxx_sparse']=m['skx']-(m['six2']-m['six']**2)/(N*T)
    out['Yyy_sparse']=m['sky']-(m['siy2']-m['siy']**2)/(N*T)
    out['Yxy_sparse']=m['skxy']-(m['sixiy']-m['six']*m['siy'])/(N*T)
    d=(out['Yxx']+out['Yyy'])/2;split=np.hypot((out['Yxx']-out['Yyy'])/2,out['Yxy'])
    out['lambda_min']=d-split;out['lambda_max']=d+split
    det=out['Yxx']*out['Yyy']-out['Yxy']**2
    out['sqrt_det']=np.sqrt(np.where((out['lambda_min']>0),det,np.nan))
    for key in ('eps','orb','spin','planar'):out[key]=m[key]/m['sn']
    for key in ('P','Bpp','Bmm','Btrace','Azplus','Azminus','Aplanar','Aplus','Aminus'):
        p=m[key]/N;out[key+'_S0']=p[...,0]
        rr=p[...,0]/p[...,1:].mean(-1)-1
        out[key+'_xi']=np.sqrt(np.where(rr>=0,rr,np.nan))/(2*L*np.sin(np.pi/L))
        if key+'mean' in m:out[key+'_disconnected']=abs(m[key+'mean'][...,0])**2/(N*p[...,0])
    qp=(m['Q1']-abs(m['Q1mean'])**2).sum(-2)/N
    rr=qp[...,0]/qp[...,1:].mean(-1)-1
    out['tilt_xi']=np.sqrt(np.where(rr>=0,rr,np.nan))/(2*L*np.sin(np.pi/L))
    out['Aplus_fraction']=out['Aplus_S0']/(out['Aplus_S0']+out['Aminus_S0'])
    out['Bpp_fraction']=out['Bpp_S0']/(out['Bpp_S0']+out['Bmm_S0'])
    return out


def conversions(z,meta):
    v=z['values'];mapping=z['temperature_to_walker'];w=np.empty_like(v)
    w[np.arange(len(v))[:,None],mapping]=v
    q=w[:,:,9];chi=abs(w[:,:,2]/w[:,:,1]);rows=[]
    # Threshold visits are structural descriptors, not phase assignments.
    for hold in (100,200,400):
        flags=np.where((q>.25)&(chi>.25),1,np.where((q<-.1)&(chi<.15),-1,0))
        forward=reverse=walkers=0
        for j in range(w.shape[1]):
            f=flags[:,j];cuts=np.r_[0,1+np.flatnonzero(f[1:]!=f[:-1]),len(f)]
            accepted=[]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if f[lo] and (hi-lo-1)*20>=hold:accepted.append((int(f[lo]),int(z['sweeps'][lo])))
            changes=[(a,b) for a,b in zip(accepted[:-1],accepted[1:]) if a[0]!=b[0]]
            forward+=sum(a[0]==1 for a,b in changes);reverse+=sum(a[0]==-1 for a,b in changes)
            walkers+=bool(changes)
        rows.append(dict(label=meta['label'],L=meta['L'],g_d=meta['g_d'],preparation=meta['preparation'],
            hold_sweeps=hold,chiral_like_to_low_q=forward,low_q_to_chiral_like=reverse,walkers_changing=walkers))
    return rows


def main():
    rows=[];transport=[];convert=[];saved={};joint=[]
    for path in sorted((HERE/'pilots').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());L=meta['L'];gd=meta['g_d']
        with np.load(path) as zz:z=dict(zz)
        T=z['T'];n=meta['total_sweeps'];burn=meta['analysis_burn']
        for label,lo,hi in [('full',burn,n),('first',burn,(burn+n)//2),('second',(burn+n)//2,n)]:
            b=moments(z,L,lo,hi);m={k:x.mean(0) for k,x in b.items()};e=evaluate(m,T,L)
            jk=evaluate({k:(12*m[k]-x)/11 for k,x in b.items()},T,L)
            for j,t in enumerate(T):
                row=dict(label=meta['label'],L=L,g_d=gd,preparation=meta['preparation'],subset=label,T=t)
                for k,x in e.items():row[k]=float(x[j]);row[k+'_se']=float(np.sqrt(11*np.var(jk[k][:,j])))
                rows.append(row)
            if label=='full':saved[meta['label']]=(T,e,{k:np.sqrt(11*np.var(jk[k],axis=0)) for k in e},meta,z)
        pt=meta['pt'];transport.append(dict(label=meta['label'],L=L,g_d=gd,preparation=meta['preparation'],
            total_sweeps=n,loop_seconds=meta['loop_seconds'],min_acceptance=min(pt['acceptance_rates']),
            exact_round_trips=sum(pt['walker_round_trips']),walkers_with_trip=sum(x>0 for x in pt['walker_round_trips']),
            mean_completed_trip_sweeps=pt['mean_round_trip_sweeps']))
        convert.extend(conversions(z,meta))
        sid=z['sweeps'][4::5]>burn
        # Retain selected event-level joint distributions, without mixing preparations.
        for j in (0,6,10,12,23):
            sv=z['sparse_values'][sid,j];st=z['structure'][sid,j];a=z['amplitudes'][sid,j];ex=z['extra'][sid,j]
            for i in range(len(sv)):
                bp=(a[i,7,0]+a[i,8,0]+2*a[i,9,0])/2;bm=(a[i,7,0]+a[i,8,0]-2*a[i,9,0])/2
                joint.append(dict(label=meta['label'],L=L,g_d=gd,preparation=meta['preparation'],T=T[j],
                    sweep=int(z['sweeps'][4::5][sid][i]),e=sv[i,0]/L**2,q=sv[i,9],chi=sv[i,2]/sv[i,1],
                    epsilon=st[i,6]/st[i,0],planar_weight=ex[i,1]/st[i,0],
                    Bpp_power=abs(bp)**2/L**4,Bmm_power=abs(bm)**2/L**4))
    write('curves.csv',rows);write('transport.csv',transport);write('conversions.csv',convert);write('joint_samples.csv',joint)
    for L,gd in sorted(set((m[3]['L'],m[3]['g_d']) for m in saved.values())):
        fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        selected=[v for v in saved.values() if v[3]['L']==L and v[3]['g_d']==gd]
        for T,e,se,meta,z in selected:
            for ax,key in zip(axs.flat,('chi','q','Y','eps','Aplus_fraction','Yxy')):
                ax.errorbar(T,e[key],se[key],label=meta['label'],lw=1,alpha=.8)
                ax.set(xlabel='MC temperature',ylabel=key);ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7);fig.suptitle(f'L={L}, g_D={gd:g}: separate initial preparations, conditional block errors')
        fig.savefig(OUT/f'comparison_L{L}_gd{gd:g}.png',dpi=160);plt.close(fig)
    print('Analyzed',len(saved),'pilots;',len(rows),'curve rows')


if __name__=='__main__':main()
