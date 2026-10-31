"""Test interpolation and ladder resolution of fitted WM diagnostic temperatures."""
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import least_squares
from compare import load
from summarize import OUT, SIZES, COUPLINGS, estimate, write_csv
from bkt_checks import fit_arrays


def main():
    with (OUT/'wm_sensitivity.csv').open() as f:
        central=[r for r in csv.DictReader(f) if r['block_size']=='2000' and r['subset']=='full' and r['window']=='all' and r['sizes']=='64:96:128:160']
    rows=[]
    for gd in COUPLINGS:
        T,_=load(gd,64)
        d=[estimate(load(gd,L)[1],T,L,2000) for L in SIZES]
        rho=np.array([v['helicity'] for v,e in d]);err=np.array([e['helicity'] for v,e in d])
        for q in [.5,1.]:
            ref=next(r for r in central if float(r['g_d'])==gd and float(r['charge'])==q)
            tc=float(ref['temperature']);a0=1/(2*np.log(64)+float(ref['C']))
            for name in ['linear','pchip','cubic','even_temperatures','odd_temperatures']:
                if name in ['linear','even_temperatures','odd_temperatures']:
                    ids=np.arange(len(T)) if name=='linear' else np.arange(0 if name.startswith('even') else 1,len(T),2)
                    f=fit_arrays(T[ids],SIZES,rho[:,ids],err[:,ids],q)
                else:
                    factory=PchipInterpolator if name=='pchip' else CubicSpline
                    rfun=factory(T,rho,axis=1);sfun=factory(T,err,axis=1)
                    logs=2*np.log(np.array(SIZES)/64)
                    def residual(par):
                        t,a=par
                        return (q*q*rfun(t)-2*t/np.pi*(1+a/(1+a*logs)))/(q*q*sfun(t))
                    span=.0015 if gd<.1 else .005
                    ff=[least_squares(residual,[tt,max(a0,1e-5)],bounds=([tc-span,0],[tc+span,np.inf])) for tt in [tc-.0003,tc,tc+.0003]]
                    best=min(ff,key=lambda x:sum(x.fun**2));t,a=best.x
                    f=dict(temperature=t,C=1/a-2*np.log(64),chi2=sum(best.fun**2),reason='local interpolation test')
                rows.append(dict(g_d=gd,charge=q,interpolation=name,temperature=f['temperature'],
                    difference_from_linear=f['temperature']-tc if f['temperature'] is not None else '',
                    C=f['C'],reduced_chi2=f['chi2']/2,reason=f['reason']))
    write_csv(OUT/'wm_interpolation.csv',rows)
    print('Saved interpolation checks')


if __name__=='__main__':main()
