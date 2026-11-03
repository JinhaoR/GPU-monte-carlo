"""Exact helicity decomposition from old soc_scan_02 block moments; no simulation.

Run: python -B exploration/he3_helicity_mechanism/data/decompose.py
The 600 original cache rows each average 100 saved records (5000 sweeps).
Joint time blocks preserve covariance of all observables and temperatures.
Errors are conditional history errors, not guarantees of equilibrium.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CACHE = ROOT / 'exploration/he3_soc/results/blocks'
GD = (0, .001, .01)
SIZES = (64, 96, 128, 160)
KEYS = ('kx', 'ky', 'ix', 'ix2', 'iy', 'iy2', 'n', 'chi', 'chi_abs', 'qzz')


def write(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def eval_m(m, T, L):
    r = {'density': m['n'], 'abs_chirality': m['chi_abs'],
         'signed_chirality': m['chi'], 'qzz': m['qzz']}
    for a in ('x', 'y'):
        r['D'+a] = m['k'+a] / L**2
        r['S'+a] = (m['i'+a+'2'] - m['i'+a]**2) / (T * L**2)
        r['M'+a] = m['i'+a]**2 / (T * L**2)
        r['Y'+a] = r['D'+a] - r['S'+a]
        r['I'+a] = m['i'+a]
    for name in ('D', 'S', 'M', 'Y'):
        r[name] = .5 * (r[name+'x'] + r[name+'y'])
    r['Yx_minus_Yy'] = r['Yx'] - r['Yy']
    r['D_over_4n'] = r['D'] / (4*r['density'])
    r['screen_fraction'] = r['S'] / r['D']
    r['cancellation_condition'] = (abs(r['D'])+abs(r['S'])) / abs(r['Y'])
    return r


def estimate(b, T, L, group=20):
    count = len(b['n']) // group
    c = {k: v[:count*group].reshape(count, group, -1).mean(1) for k,v in b.items()}
    m = {k: v.mean(0) for k,v in c.items()}
    v = eval_m(m, T, L)
    j = eval_m({k: (count*m[k]-x)/(count-1) for k,x in c.items()}, T, L)
    e = {k: np.sqrt((count-1)*np.var(x,axis=0)) for k,x in j.items()}
    return v, j, e


def window(v, T, lo, hi):
    r = {}
    for name in ('D','S','Y','density','D_over_4n'):
        r[name+'_lo'] = v[name][...,lo]
        r[name+'_hi'] = v[name][...,hi]
        r[name+'_change'] = v[name][...,hi]-v[name][...,lo]
        x = T[lo:hi+1] - T[lo:hi+1].mean()
        r[name+'_OLS_slope'] = np.sum(v[name][...,lo:hi+1]*x,axis=-1) / np.sum(x*x)
    loss = -r['Y_change']
    r['curvature_loss_share'] = -r['D_change']/loss
    r['screening_loss_share'] = r['S_change']/loss
    r['curvature_slope_share'] = r['D_OLS_slope']/r['Y_OLS_slope']
    # D = 4 n R, with R=D/(4n). Symmetric product difference is exact.
    r['density_product_loss'] = -2*(r['D_over_4n_hi']+r['D_over_4n_lo'])*r['density_change']
    r['ratio_product_loss'] = -2*(r['density_hi']+r['density_lo'])*r['D_over_4n_change']
    r['density_share_of_curvature_loss'] = r['density_product_loss']/(-r['D_change'])
    r['ratio_share_of_curvature_loss'] = r['ratio_product_loss']/(-r['D_change'])
    r['density_share_of_helicity_loss'] = r['density_product_loss']/loss
    r['ratio_share_of_helicity_loss'] = r['ratio_product_loss']/loss
    return r


def main():
    curves, windows, halves, sensitivity, shoulders = [], [], [], [], []
    figures, axes = plt.subplots(3, 3, figsize=(13, 11), sharex='col')
    for col,gd in enumerate(GD):
        for L in SIZES:
            z = np.load(CACHE/f'gd={gd:g}_L={L}.npz')
            T = z['T']; b = {k:z[k] for k in KEYS}; center = 71 if gd<.01 else 76
            v,j,e = estimate(b,T,L)
            for t in range(len(T)):
                row = dict(g_d=gd,L=L,T=T[t],index=t,block_records=2000)
                for k in v: row[k],row[k+'_se'] = v[k][t],e[k][t]
                curves.append(row)
            for width in (1,2,3,5,8):
                lo,hi = center-width,center+width
                w,wj = window(v,T,lo,hi),window(j,T,lo,hi)
                row = dict(g_d=gd,L=L,halfwidth_indices=width,T_lo=T[lo],T_hi=T[hi],block_records=2000)
                for k in w:
                    row[k] = w[k]
                    row[k+'_se'] = np.sqrt((len(wj[k])-1)*np.var(wj[k]))
                windows.append(row)
            for offset_lo,offset_hi in ((3,6),(3,9),(3,12),(-3,12)):
                lo,hi=center+offset_lo,center+offset_hi
                w,wj=window(v,T,lo,hi),window(j,T,lo,hi)
                row=dict(g_d=gd,L=L,offset_lo=offset_lo,offset_hi=offset_hi,T_lo=T[lo],T_hi=T[hi])
                for k in w:row[k]=w[k];row[k+'_se']=np.sqrt((len(wj[k])-1)*np.var(wj[k]))
                shoulders.append(row)
            for group in (1,5,20,50):
                vg,jg,eg = estimate(b,T,L,group)
                wj = window(jg,T,center-3,center+3)
                for t in (center-3,center,center+3):
                    row = dict(g_d=gd,L=L,T=T[t],index=t,block_records=group*100,
                               Y=vg['Y'][t],Y_se=eg['Y'][t],D_se=eg['D'][t],S_se=eg['S'][t],
                               M=vg['M'][t],M_se=eg['M'][t])
                    row['curvature_share_se'] = np.sqrt((len(wj['curvature_loss_share'])-1)*np.var(wj['curvature_loss_share']))
                    sensitivity.append(row)
            # 300 original rows per half, 15 independent jackknife groups per half.
            v1,_,e1 = estimate({k:a[:300] for k,a in b.items()},T,L)
            v2,_,e2 = estimate({k:a[300:] for k,a in b.items()},T,L)
            for t in [int(np.argmin(abs(T-.35))), center-3,center,center+3,int(np.argmin(abs(T-.40)))]:
                for name in ('D','S','Y','M','density','abs_chirality','signed_chirality','qzz'):
                    se = np.hypot(e1[name][t],e2[name][t])
                    halves.append(dict(g_d=gd,L=L,T=T[t],observable=name,first=v1[name][t],second=v2[name][t],
                                       change=v2[name][t]-v1[name][t],change_se=se,z=(v2[name][t]-v1[name][t])/se))
            sl = slice(center-12,center+16)
            for ax,name in zip(axes[:,col],('D','S','Y')):
                ax.errorbar(T[sl],v[name][sl],yerr=e[name][sl],label=f'L={L}',lw=1)
                ax.axvspan(T[center-3],T[center+3],alpha=.07,color='black')
                ax.set_ylabel(name); ax.grid(alpha=.2); ax.xaxis.set_major_locator(MaxNLocator(4))
            axes[0,col].set_title(f'g_D={gd:g}'); axes[2,col].set_xlabel('T')
        axes[0,col].legend(fontsize=8)
    figures.tight_layout(); figures.savefig(OUT/'decomposition.png',dpi=180); plt.close(figures)
    write('curves.csv',curves); write('transition_windows.csv',windows)
    write('time_halves.csv',halves); write('block_sensitivity.csv',sensitivity)
    write('hotter_shoulders.csv',shoulders)
    # Spatial normalization check: log Var(I) = 2 log L + log(T S).
    scaling=[]
    for gd in GD:
        for index in ((65,68,71,74,79,94) if gd<.01 else (70,73,76,79,84,94)):
            selected=[r for r in curves if r['g_d']==gd and r['index']==index]
            for axis in ('x','y'):
                x=np.log([r['L'] for r in selected]); y=np.log([r['S'+axis]*r['T']*r['L']**2 for r in selected])
                slope=np.polyfit(x,y,1)[0]
                scaling.append(dict(g_d=gd,T=selected[0]['T'],index=index,axis=axis,varI_logL_slope=slope,
                                    S_L64=selected[0]['S'+axis],S_L160=selected[-1]['S'+axis]))
    write('current_size_scaling.csv',scaling)
    metadata=dict(source='soc_scan_02',K=1,N='L^2',base_cache_records=100,saved_record_sweeps=50,
                  default_block_records=2000,production_records=60000,
                  center_index={'0':71,'0.001':71,'0.01':76},primary_halfwidth=3,
                  uncertainty='paired delete-one joint PT time-block jackknife, conditional on sampled histories',
                  identities=['Y=D-S','D=<Kx+Ky>/(2N)','S=(Var(Ix)+Var(Iy))/(2TN)',
                              'M=(<Ix>^2+<Iy>^2)/(2TN)','D=4<n> R, R=D/(4<n>)',
                              'Delta D=4 R_mid Delta<n> + 4 n_mid Delta R'])
    (OUT/'method.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for r in windows:
        if r['L']==160 and r['halfwidth_indices']==3:
            print({k:round(float(r[k]),7) for k in ('g_d','T_lo','T_hi','curvature_loss_share','curvature_loss_share_se',
                   'D_over_4n_lo','D_over_4n_hi','density_share_of_curvature_loss','density_share_of_curvature_loss_se')})


if __name__=='__main__': main()
