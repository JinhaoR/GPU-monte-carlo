"""Check density confounding of the observed current-power/chirality association.

Old soc_scan_02, L=160 only, five selected T columns, no simulation. Partial
correlations are descriptive linear adjustments, not causal estimands.
"""
from pathlib import Path
import csv
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def corr(m,a,b):
    return (m[''.join(sorted(a+b))]-m[a]*m[b])/np.sqrt((m[a+a]-m[a]**2)*(m[b+b]-m[b]**2))


def evaluate(m):
    # Recompute both current means in every deleted-block sample.
    mx,my=m['x'],m['y']; power_mean=m['q']-.5*(mx*mx+my*my)
    power_var=(m['qq']-m['q']**2+mx*mx*(m['xx']-mx*mx)+my*my*(m['yy']-my*my)
               +2*mx*my*(m['xy']-mx*my)-2*mx*(m['qx']-m['q']*mx)-2*my*(m['qy']-m['q']*my))
    power_m={'y':power_mean,'yy':power_var+power_mean**2,'n':m['n'],'nn':m['nn'],
             'c':m['c'],'cc':m['cc'],'cn':m['cn']}
    for label in ('c','n'):
        covariance=(m['q'+label]-m['q']*m[label]-mx*(m['x'+label]-mx*m[label])
                    -my*(m['y'+label]-my*m[label]))
        power_m[''.join(sorted('y'+label))]=covariance+power_mean*m[label]
    yc=corr(power_m,'y','c'); yn=corr(power_m,'y','n'); cn=corr(power_m,'c','n')
    return dict(corr_power_abs_chi=yc,corr_power_density=yn,corr_density_abs_chi=cn,
                partial_power_chi_given_density=(yc-yn*cn)/np.sqrt((1-yn*yn)*(1-cn*cn)),
                partial_power_density_given_chi=(yn-yc*cn)/np.sqrt((1-yc*yc)*(1-cn*cn)))


def main():
    rows=[]
    for gd in (0,.001,.01):
        L=160; center=71 if gd<.01 else 76
        ids=np.array([center-3,center-1,center,center+1,center+3])
        T=np.load(ROOT/f'exploration/he3_soc/results/blocks/gd={gd:g}_L={L}.npz')['T']
        with np.load(ROOT/f'data/he3/P=20_gd={gd:g}_gaussian/L={L}.npz') as z:
            x=z[f'L{L}_current_x'][:,ids].copy(); y=z[f'L{L}_current_y'][:,ids].copy()
            n=z[f'L{L}_density'][:,ids].copy(); c=abs(z[f'L{L}_chirality'][:,ids].copy()/n); n/=L*L
        x/=np.sqrt(T[ids]*L*L); y/=np.sqrt(T[ids]*L*L)
        power=.5*(x*x+y*y)
        for j,index in enumerate(ids):
            data=dict(x=x[:,j],y=y[:,j],q=power[:,j],c=c[:,j],n=n[:,j]); moments=data.copy()
            for a,b in [('q','q'),('x','x'),('y','y'),('x','y'),('q','x'),('q','y'),('c','c'),('n','n'),('c','n'),
                        ('q','c'),('q','n'),('x','c'),('x','n'),('y','c'),('y','n')]:moments[a+b]=data[a]*data[b]
            blocks={k:v.reshape(30,2000).mean(1) for k,v in moments.items()}
            m={k:v.mean() for k,v in blocks.items()}
            jk={k:(30*m[k]-v)/29 for k,v in blocks.items()}
            r,jr=evaluate(m),evaluate(jk)
            row=dict(g_d=gd,L=L,T=T[index],index=index)
            for k,v in r.items():row[k]=v;row[k+'_se']=np.sqrt(29*np.var(jr[k]))
            rows.append(row)
    with (OUT/'partial_associations.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:
        if r['index']==(71 if r['g_d']<.01 else 76):print(r)


if __name__=='__main__':main()
