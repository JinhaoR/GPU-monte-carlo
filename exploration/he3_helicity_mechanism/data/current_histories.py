"""Selective raw-history diagnostics for old soc_scan_02 only; no simulations.

One source NPZ and channel is read at a time. Only selected fixed-temperature
columns are kept in memory. Derived CSVs and histograms are written beside this
script. Fixed-temperature PT sequences are NOT real-time trajectories.
Run: python -B exploration/he3_helicity_mechanism/data/current_histories.py
"""
from pathlib import Path
import csv
import json
import zipfile
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BLOCK=2000


def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def block_means(a):
    return np.asarray(a).reshape(-1,BLOCK).mean(1)


def jack(m):
    n=len(next(iter(m.values())))
    return {k: (n*v.mean(0)-v)/(n-1) for k,v in m.items()}


def se(v):
    return np.sqrt((len(v)-1)*np.var(v,axis=0))


def evaluate(m,T,N):
    u=m['j']; variance=m['j2']-u*u
    third=m['j3']-3*u*m['j2']+2*u**3
    fourth=m['j4']-4*u*m['j3']+6*u*u*m['j2']-3*u**4
    dD=N*(m['de']-m['d']*m['e'])/T**2
    dS=N*(m['j2e']-m['j2']*m['e']-2*u*(m['je']-u*m['e']))/T**2-variance/T
    r=dict(j_mean=u,j_variance=variance,D=m['d'],Y=m['d']-variance,
           skewness=third/variance**1.5,excess_kurtosis=fourth/variance**2-3,
           mean_correction=u*u,dD_dT=dD,dS_dT=dS,dY_dT=dD-dS,
           derivative_curvature_share=dD/(dD-dS))
    for name in ('density','abs_chi','chi','qzz'):
        cov_j=m['j_'+name]-u*m[name]
        cov_j2=m['j2_'+name]-m['j2']*m[name]-2*u*cov_j
        var_other=m[name+'2']-m[name]**2
        r['corr_j_'+name]=cov_j/np.sqrt(variance*var_other)
        r['corr_centered_j2_'+name]=cov_j2/np.sqrt((fourth-variance**2)*var_other)
    for fraction in ('001','01','05'):
        # Tail membership fixed using the full-history centered |j| quantile.
        contribution=m['tail_j2_'+fraction]-2*u*m['tail_j_'+fraction]+u*u*m['tail_p_'+fraction]
        r['tail_variance_share_'+fraction]=contribution/variance
    r['fraction_gt3sigma']=m['gt3']
    return r


def conditional_metrics(m):
    p=m['count']; mu=m['j']/p; second=m['j2']/p
    overall=np.sum(m['j'],axis=-1)
    total=np.sum(m['j2'],axis=-1)-overall*overall
    between=np.sum(p*(mu-overall[...,None])**2,axis=-1)
    within=np.sum(p*(second-mu*mu),axis=-1)
    return dict(p=p,mean_j=mu,variance=second-mu*mu,D=m['d']/p,
                moment_difference=m['d']/p-(second-mu*mu),
                density=m['density']/p,abs_chi=m['abs_chi']/p,
                between_share=between/total,within_share=within/total,
                decomposition_residual=within+between-total)


def xy_metrics(m):
    cov=m['xy']-m['x']*m['y']
    return dict(current_cross_covariance_over_TN=cov,
                corr_ix_iy=cov/np.sqrt((m['x2']-m['x']**2)*(m['y2']-m['y']**2)))


def main():
    stats,conds,mixes,derivatives,provenance,check=[],[],[],[],[],[]
    hist={}; joint={}; edges=np.linspace(-6,6,121)
    for gd in (0,.001,.01):
        for L in (64,96,128,160):
            source=ROOT/f'data/he3/P=20_gd={gd:g}_gaussian/L={L}.npz'
            cached=np.load(ROOT/f'exploration/he3_soc/results/blocks/gd={gd:g}_L={L}.npz')
            T=cached['T']; center=71 if gd<.01 else 76
            ids=np.array(sorted(set([int(np.argmin(abs(T-.35))),center-5,center-3,center-1,
                                     center,center+1,center+3,center+5,int(np.argmin(abs(T-.40))),
                                     int(np.argmin(abs(T-.45)))])))
            print(f'Reading old g_D={gd:g}, L={L}, {len(ids)} selected temperatures',flush=True)
            with np.load(source,allow_pickle=False) as archive:
                meta=json.loads(archive['manifest'].item())['runs'][str(L)]['metadata']
                assert meta['campaign']['id']=='soc_scan_02'
                assert len(meta['sample_sweeps'])==60000
                assert meta['model_parameters']['K']==1
                raw={}
                for name in ('current_x','current_y','curvature_x','curvature_y','density','chirality','spin_alignment','energy'):
                    raw[name]=archive[f'L{L}_{name}'][:,ids].copy()
            with zipfile.ZipFile(source) as archive:
                crc=';'.join(f'{x.filename}:{x.CRC:08x}' for x in archive.infolist())
            inventory=json.loads((ROOT/f'exploration/he3_soc/results/inventory_gd={gd:g}_L={L}.json').read_text())
            assert inventory['zip_crc32']==crc and inventory['bytes']==source.stat().st_size
            provenance.append(dict(g_d=gd,L=L,source=str(source.relative_to(ROOT)),campaign='soc_scan_02',
                                   selected_indices=ids.tolist(),source_bytes=source.stat().st_size,
                                   crc32=crc,block_records=BLOCK))
            for k,index in enumerate(ids):
                t=T[index]; N=L*L
                density=raw['density'][:,k]/N; chi=raw['chirality'][:,k]/raw['density'][:,k]
                base={'density':density,'abs_chi':abs(chi),'chi':chi,'qzz':raw['spin_alignment'][:,k]}
                energy=raw['energy'][:,k]/N
                groups={}
                for name in ('density','abs_chi','qzz'):
                    bounds=np.quantile(base[name],[.25,.5,.75])
                    groups[name]=(np.searchsorted(bounds,base[name]),bounds)
                groups['chi_sign']=((chi>=0).astype(int),np.array([0.]))
                x=raw['current_x'][:,k]/np.sqrt(t*N); y=raw['current_y'][:,k]/np.sqrt(t*N)
                xyb={name:block_means(a) for name,a in [('x',x),('y',y),('xy',x*y),('x2',x*x),('y2',y*y)]}
                xyv=xy_metrics({name:a.mean() for name,a in xyb.items()}); xyj=xy_metrics(jack(xyb))
                all_direction=[]
                for axis in ('x','y'):
                    j=raw['current_'+axis][:,k]/np.sqrt(t*N)
                    d=raw['curvature_'+axis][:,k]/N
                    a={'j':j,'j2':j*j,'j3':j**3,'j4':j**4,'d':d,'e':energy,
                       'je':j*energy,'j2e':j*j*energy,'de':d*energy}
                    for name,x in base.items():
                        a[name]=x; a[name+'2']=x*x; a['j_'+name]=j*x; a['j2_'+name]=j*j*x
                    standardized=(j-j.mean())/j.std()
                    for label,f in [('001',.001),('01',.01),('05',.05)]:
                        mask=abs(standardized)>=np.quantile(abs(standardized),1-f)
                        a['tail_p_'+label]=mask.astype(float)
                        a['tail_j_'+label]=j*mask; a['tail_j2_'+label]=j*j*mask
                    a['gt3']=(abs(standardized)>3).astype(float)
                    bm={name:block_means(x) for name,x in a.items()}
                    m={name:x.mean() for name,x in bm.items()}; jm=jack(bm)
                    v=evaluate(m,t,N); jv=evaluate(jm,t,N)
                    row=dict(g_d=gd,L=L,T=t,index=index,axis=axis)
                    for name,value in v.items(): row[name],row[name+'_se']=value,se(jv[name])
                    row['q99_abs_z']=np.quantile(abs(standardized),.99)
                    row['q999_abs_z']=np.quantile(abs(standardized),.999)
                    row['max_abs_z']=np.max(abs(standardized))
                    row['tail01_max_block_share']=np.max(bm['tail_p_01'])/np.sum(bm['tail_p_01'])
                    for name,value in xyv.items(): row[name],row[name+'_se']=value,se(xyj[name])
                    row['tail01_first_half_fraction']=a['tail_p_01'][:30000].mean()
                    row['tail01_second_half_fraction']=a['tail_p_01'][30000:].mean()
                    for length in (100,500,2000,5000):
                        means=((j-j.mean())**2).reshape(-1,length).mean(1)
                        row[f'variance_se_block{length}']=means.std(ddof=1)/np.sqrt(len(means))
                    stats.append(row); all_direction.append((v,jv))
                    hist[f'g{gd:g}_L{L}_i{index}_{axis}']=np.histogram(standardized,bins=edges)[0]
                    # Verify selected full raw moments against pre-existing cache.
                    for rawkey,cachekey in [('current_'+axis,'i'+axis),('curvature_'+axis,'k'+axis)]:
                        diff=raw[rawkey][:,k].mean()-cached[cachekey][:,index].mean()
                        check.append(dict(g_d=gd,L=L,T=t,observable=rawkey,raw_minus_cache=diff))
                    for grouping,(labels,bounds) in groups.items():
                        ng=len(bounds)+1
                        cm={name:[] for name in ('count','j','j2','d','density','abs_chi')}
                        for label in range(ng):
                            mask=(labels==label).astype(float)
                            for name,x in [('count',np.ones_like(j)),('j',j),('j2',j*j),('d',d),('density',density),('abs_chi',abs(chi))]:
                                cm[name].append(block_means(mask*x))
                        cm={name:np.array(x).T for name,x in cm.items()}
                        cv=conditional_metrics({name:x.mean(0) for name,x in cm.items()})
                        cj=conditional_metrics(jack(cm))
                        mixes.append(dict(g_d=gd,L=L,T=t,index=index,axis=axis,grouping=grouping,
                                          between_share=cv['between_share'],between_share_se=se(cj['between_share']),
                                          within_share=cv['within_share'],within_share_se=se(cj['within_share']),
                                          decomposition_residual=cv['decomposition_residual']))
                        for label in range(ng):
                            cr=dict(g_d=gd,L=L,T=t,index=index,axis=axis,grouping=grouping,group=label,
                                    lower=-np.inf if label==0 else bounds[label-1],
                                    upper=np.inf if label==ng-1 else bounds[label])
                            for name in ('p','mean_j','variance','D','moment_difference','density','abs_chi'):
                                cr[name],cr[name+'_se']=cv[name][label],se(cj[name])[label]
                            conds.append(cr)
                    if L==160 and index in (center-3,center,center+3):
                        joint[f'g{gd:g}_i{index}_{axis}']=np.histogram2d(abs(chi),standardized,bins=(np.linspace(0,1,81),edges))[0]
                # Average derivatives with the x/y covariance retained.
                v0,j0=all_direction[0]; v1,j1=all_direction[1]
                dr=dict(g_d=gd,L=L,T=t,index=index)
                for name in ('dD_dT','dS_dT','dY_dT'):
                    dr[name]=.5*(v0[name]+v1[name]); dr[name+'_se']=se(.5*(j0[name]+j1[name]))
                dr['curvature_share']=dr['dD_dT']/dr['dY_dT']
                dr['curvature_share_se']=se((j0['dD_dT']+j1['dD_dT'])/(j0['dY_dT']+j1['dY_dT']))
                dr['slope_resolved_3se']=abs(dr['dY_dT'])>3*dr['dY_dT_se']
                derivatives.append(dr)
            del raw
    write('current_statistics.csv',stats); write('conditional_moments.csv',conds)
    write('total_variance.csv',mixes); write('canonical_derivatives.csv',derivatives)
    write('raw_cache_checks.csv',check)
    (OUT/'raw_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'current_histograms.npz',edges=edges,**hist)
    np.savez_compressed(OUT/'joint_histograms.npz',z_edges=edges,chi_edges=np.linspace(0,1,81),**joint)
    # Standardized current shape across the representative transition interval.
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    mid=.5*(edges[1:]+edges[:-1]); dx=edges[1]-edges[0]
    for ax,gd in zip(axes,(0,.001,.01)):
        center=71 if gd<.01 else 76
        for index in (center-3,center,center+3):
            h=hist[f'g{gd:g}_L160_i{index}_x']+hist[f'g{gd:g}_L160_i{index}_y']
            t=next(r['T'] for r in stats if r['g_d']==gd and r['L']==160 and r['index']==index)
            ax.plot(mid,h/h.sum()/dx,label=f'T={t:.5f}')
        ax.plot(mid,norm.pdf(mid),'k--',label='Normal'); ax.set_yscale('log'); ax.set_ylim(1e-5,1)
        ax.set_title(f'g_D={gd:g}, L=160'); ax.set_xlabel('(I - mean I) / std I'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Probability density'); fig.tight_layout(); fig.savefig(OUT/'current_distributions.png',dpi=180); plt.close(fig)
    normal={}
    for label,f in [('001',.001),('01',.01),('05',.05)]:
        cutoff=norm.ppf(1-f/2); normal[label]=dict(fraction=f,cutoff=cutoff,variance_share=f+2*cutoff*norm.pdf(cutoff))
    (OUT/'normal_tail_reference.json').write_text(json.dumps(normal,indent=2),encoding='utf-8')
    print('Finished selected old raw histories; maximum cache difference',max(abs(r['raw_minus_cache']) for r in check),flush=True)


if __name__=='__main__': main()
