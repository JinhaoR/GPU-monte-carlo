"""Read the twelve cluster comparisons; no simulation or GPU access.

Run from the repository root: python -B analysis/pt_ladder_test/analyze.py
Optional positional arguments select individual NPZ files for reanalysis.
Previous pilot utilities supply the moment estimators and PT event definitions.
"""
import os
import sys
import json
import csv
import gc
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / 'exploration/he3_pt_ladder_validation'
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(PILOT))
from exploration.he3_pt_ladder_validation.analyze import moments, interpolate, blocked
from exploration.he3_pt_equilibration.transport import events, km, acceptance_chain, null_transport
from exploration.he3_soc.summarize import autocorrelation
from models.he3.spatial_analysis import check_spatial, summarize_spatial

OUT = HERE / 'results'
DATA = ROOT / 'data/pt_ladder_test'
BLOCKS = (5_000, 25_000, 50_000, 100_000, 200_000)


def estimate(b, T, L, block):
    value, error, means = blocked(b, T, L, block)
    value['chirality'] = means['m'].mean(0)
    error['chirality'] = np.sqrt(means['m'].var(0) / (len(means['m']) - 1))
    return value, error


def influences(b, T, L):
    avg = {k: v.mean(0) for k, v in b.items()}
    return dict(energy=b['e'], m2=b['m2'], P2=b['p2'], spin=b['qzz'],
        chirality=b['m'], abs_chirality=b['abs_m'],
        Binder=-b['m4']/(3*avg['m2']**2)+2*avg['m4']*b['m2']/(3*avg['m2']**3),
        Y=(b['kx']+b['ky']-(b['ix2']+b['iy2']-2*avg['ix']*b['ix']-2*avg['iy']*b['iy'])/T)/(2*L*L))


def cold_hot_returns(temp, chi, sweeps, cold, hot, Q=None):
    """Last sampled cold state before hot visit, then first sampled cold return."""
    rows = []
    for w in range(temp.shape[1]):
        last_cold = before = None
        reached = False
        for i, t in enumerate(temp[:, w]):
            if t <= cold:
                if reached:
                    if min(abs(chi[before, w]), abs(chi[i, w])) > .1:
                        row = dict(walker=w, start=int(sweeps[before]), end=int(sweeps[i]),
                                   sign_before=int(chi[before, w] > 0), sign_after=int(chi[i, w] > 0))
                        if Q is not None:
                            x, y = Q[before, w], Q[i, w]
                            row['Q_dot'] = float(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)))
                        rows.append(row)
                    reached = False
                last_cold = i
            if t >= hot and last_cold is not None and not reached:
                before = last_cold
                reached = True
    return rows


def analyze(path):
    start = perf_counter()
    tables = {}
    def add(name, **row):
        tables.setdefault(name, []).append(dict(**base, **row))

    with np.load(path) as z:
        m = json.loads(z['manifest'].item())['runs']['96']['metadata']
        pt = m['parallel_tempering']
        h = pt.pop('walker_history')
        hs = np.asarray(h['sweeps'])
        pos = np.asarray(h['temperature_indices'], dtype=np.int16)
        assert h['stride'] == 1
        del h
        a = {k: z['L96_'+k] for k in ('energy', 'density', 'chirality', 'phase_real', 'phase_imag',
             'spin_alignment', 'current_x', 'current_y', 'curvature_x', 'curvature_y')}
        amp = z['L96_spatial_amplitudes']
        structure = z['L96_spatial_structure']
        sw = z['L96_spatial_walkers']
        se = z['L96_spatial_energy']
    L = int(m['L']); T = np.asarray(m['temperatures']); R = len(T)
    g = round(m['model_parameters']['g_d'], 2)
    base = dict(case=path.stem, L=L, g_d=g, ladder=m['ladder'], initialization=m['initialization'], R=R)
    ts = np.asarray(m['sample_sweeps']); sp = np.asarray(m['spatial_sample_sweeps'])
    burn = m['diagnostic_burn_sweeps']; duration = m['diagnostic_production_sweeps']; end = burn+duration
    assert (L, burn, duration, m['sample_interval'], pt['exchange_interval']) == (96, 500000, 1000000, 50, 5)
    assert m['sampler_parameters'] == dict(rho=-.9, proposal_mass=0.)
    np.testing.assert_allclose(T, np.loadtxt(PILOT/f"ladder_{m['ladder']}.txt"), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(ts, np.arange(50, end+1, 50))
    np.testing.assert_array_equal(hs, np.arange(0, end+1, 5))
    np.testing.assert_array_equal(sp, np.arange(1000, end+1, 1000))
    for lo in range(0, len(pos), 5000):
        assert np.all(np.sort(pos[lo:lo+5000], axis=1) == np.arange(R))
    assert all(np.isfinite(x).all() for x in (*a.values(), amp, structure, se))
    assert pt['sweeps'] == end and pt['exchange_rounds'] == end//5 and pt['next_parity'] == 0
    assert np.all(np.asarray(pt['attempts']) == end//10)
    assert np.all(np.asarray(m['sampling']['attempts']) == end*L*L)
    assert not np.any(m['sampling']['nonfinite_rejections'])
    np.testing.assert_array_equal(pt['temperature_betas'], (1/T).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(pt['walker_to_temperature'])[pt['temperature_to_walker']], np.arange(R))
    np.testing.assert_array_equal(pos[-1], pt['walker_to_temperature'])
    np.testing.assert_array_equal(sw, np.argsort(pos[np.searchsorted(hs, sp)], axis=1))
    raw = dict(a, spatial_amplitudes=amp, spatial_structure=structure, spatial_walkers=sw, spatial_energy=se)
    spatial_check = check_spatial(dict(samples=raw, metadata=m))
    keep = ts > burn
    b = moments(a, L); bp = {k: v[keep] for k, v in b.items()}
    tc = .38210530380170976 if g == 0 else .3856721419201371
    probes = np.array([.375, tc-.0003, tc, tc+.0003, .402, .405])

    # Joint moments, preserving same-time correlations between neighboring slots.
    q = {k: interpolate(v, T, probes) for k, v in bp.items()}
    inf = influences(q, probes, L)
    taus = {k: autocorrelation(v)*50 for k, v in inf.items()}
    half = len(q['e'])//2
    fv, fe = estimate({k:v[:half] for k,v in q.items()}, probes, L, 2000)
    sv, ee = estimate({k:v[half:] for k,v in q.items()}, probes, L, 2000)
    for bs in BLOCKS:
        v, e = estimate(q, probes, L, bs//50)
        for key in v:
            for j, t in enumerate(probes):
                ess = inf[key][:,j].var(ddof=1)/e[key][j]**2
                add('statistics', T=t, observable=key, block_sweeps=bs, blocks=duration//bs,
                    value=v[key][j], se=e[key][j], tau_sweeps=taus[key][j],
                    acf_ESS=duration/(2*taus[key][j]), block_ESS=ess,
                    ESS_per_total_hour=ess*3600/m['timing']['simulation_seconds'],
                    variance_times_seconds=e[key][j]**2*m['timing']['simulation_seconds'],
                    first_half=fv[key][j], second_half=sv[key][j], half_shift=sv[key][j]-fv[key][j],
                    half_combined_se=np.hypot(fe[key][j],ee[key][j]))
    v, e = estimate(bp, T, L, 2000)
    for j,t in enumerate(T):
        add('curves', T=t, **{k:float(x[j]) for k,x in v.items()}, **{k+'_se':float(x[j]) for k,x in e.items()})
    j = np.argmin(abs(T-tc)); nb = {k:x[:,j:j+1] for k,x in bp.items()}; ni = influences(nb, T[j:j+1], L)
    for bs in BLOCKS:
        nv, ne = estimate(nb, T[j:j+1], L, bs//50)
        for key,x in ni.items():
            ess = x.var(ddof=1)/ne[key][0]**2
            add('native_statistics', T=T[j], observable=key, block_sweeps=bs,
                value=nv[key][0], se=ne[key][0], tau_sweeps=autocorrelation(x)[0]*50,
                block_ESS=ess, ESS_per_total_hour=ess*3600/m['timing']['simulation_seconds'],
                variance_times_seconds=ne[key][0]**2*m['timing']['simulation_seconds'])
    for hi in range(500, len(ts)+1, 500):
        bb = {k:interpolate(x[hi-500:hi],T,probes) for k,x in b.items()}
        vv,_ = estimate(bb,probes,L,100)
        for j,t in enumerate(probes):add('time_blocks', sweep=ts[hi-1], T=t, **{k:float(x[j]) for k,x in vv.items()})

    # Exact five-sweep paths, independently checked against GPU trip counters.
    fl, fp, _, fw, labels = events(pos, hs)
    np.testing.assert_array_equal([w['loops'] for w in fw], pt['walker_round_trips'])
    sel = hs >= burn
    loops, passages, first, walkers, _ = events(pos[sel], hs[sel])
    first_pass = {}
    for r in passages:first_pass.setdefault(r[2],r)
    for kind, records in [('first_loop',first),('first_passage',list(first_pass.values()))]:
        for t,s,risk,nevent in km(records)[0]:add('survival',kind=kind,time=t,survival=s,risk=risk,events=nevent)
    for kind, records in [('loop',loops),('passage',passages)]:
        for d,event,w,origin,lo,hi in records:add('events',kind=kind,duration=d,completed=event,walker=w,origin=origin,start=lo,end=hi)
    carry = [r[5] for r in fp if r[1] and r[5] > burn]
    edges = np.arange(burn,end+1,100000)
    counts = np.histogram(carry,edges)[0]
    npass = len(carry); flux=2*R*duration/npass
    flux_jk = 2*R*(duration-100000)/(npass-counts)
    cold = .379 if g==0 else .383
    cold_index = np.searchsorted(T,cold,side='right')-1
    hot_index = np.searchsorted(T,.4)
    surfaces = np.where(pos<=cold_index,0,np.where(pos>=hot_index,R-1,1)).astype(np.int16)
    _,cp,_,_,_ = events(surfaces,hs)
    common = [r[5] for r in cp if r[1] and r[5] > burn]
    common_counts = np.histogram(common,edges)[0]
    for i in range(10):add('traffic_blocks',end=edges[i+1],passages=int(counts[i]),common_passages=int(common_counts[i]))
    atT = np.take_along_axis(labels[sel],np.argsort(pos[sel],axis=1),axis=1)
    h0=(atT==0).sum(0);h1=(atT==1).sum(0);flow=h0/(h0+h1)
    accept=np.asarray(pt['acceptance_rates']); _,null,_=null_transport(accept,5)
    for j,t in enumerate(T):add('flow',T=t,flow=flow[j],labelled_fraction=(h0[j]+h1[j])/len(atT))
    for j,v in enumerate(accept):add('acceptance',T=(T[j]+T[j+1])/2,acceptance=v)
    elapsed=m['timing']['simulation_seconds']
    add('summary',burn=burn,production=duration,gpu=m['gpu'],seed=m['seed'],exchange_seed=pt['exchange_seed'],
        simulation_seconds=elapsed,ms_per_sweep=elapsed/end*1000,min_acceptance=accept.min(),mean_acceptance=accept.mean(),
        full_loops=sum(pt['walker_round_trips']),production_loops=sum(r[1] for r in loops),
        completed_loop_mean=np.mean([r[0] for r in loops if r[1]]),
        walkers_loop=sum(w['loops']>0 for w in walkers),walkers_both=sum(w['both'] for w in walkers),
        walkers_no_endpoint=sum(not w['started'] for w in walkers),first_loop_censored=sum(not r[1] for r in first),
        first_loop_KM_median=km(first)[1],first_passage_KM_median=km(list(first_pass.values()))[1],
        production_passages=npass,flux_time=flux,flux_se=np.sqrt(9*np.var(flux_jk)),null_RT=null,flux_to_null=flux/null,
        common_passages=len(common),seconds_per_passage=elapsed/npass,seconds_per_common_passage=elapsed/len(common),
        min_labelled_fraction=np.min((h0+h1)/len(atT)))
    del atT,surfaces,fl,fp,fw,loops,passages,first

    # Temperature-slot statistics above do not follow the same physical field.
    si=np.searchsorted(hs,ts[keep]); p=pos[si]; temps=T[p]
    lab=np.take_along_axis(labels[si],np.argsort(p,axis=1),axis=1)
    j=np.argmin(abs(T-tc))
    for key,ch in [('energy','e'),('m2','m2'),('P2','p2'),('spin','qzz')]:
        ns=[];ss=[]
        for side in (0,1):
            mask=lab[:,j]==side
            ns.append(mask.reshape(10,-1).sum(1));ss.append((bp[ch][:,j]*mask).reshape(10,-1).sum(1))
        ns=np.array(ns);ss=np.array(ss);tot=ns.sum(1);sums=ss.sum(1)
        deleted=(sums[:,None]-ss)/(tot[:,None]-ns);dd=deleted[0]-deleted[1]
        add('conditioned',T=T[j],observable=key,cold_n=int(tot[0]),hot_n=int(tot[1]),
            difference=sums[0]/tot[0]-sums[1]/tot[1],se=np.sqrt(9*np.var(dd)))
    residual={key:np.take_along_axis((bp[ch]-bp[ch].mean(0))/bp[ch].std(0),p,axis=1)
              for key,ch in [('energy','e'),('abs_chirality','abs_m'),('chirality','m'),('P2','p2'),('spin','qzz')]}
    for key,x in residual.items():
        x=x[::10];x=x-x.mean();var=np.mean(x*x)
        for lag in (500,1000,5000,10000,50000,100000,250000,500000):
            d=lag//500
            add('walker_memory',observable=key,lag=lag,correlation=np.mean(x[:-d]*x[d:])/var,
                walker_mean_variance_fraction=np.var(x.mean(0))/var)
    # Check every intervening exchange for continuous hot residence.
    bad=np.vstack((np.zeros((1,R),dtype=np.int32),np.cumsum(pos<hot_index,axis=0,dtype=np.int32)))
    for lag in (500,1000,5000,10000):
        d=lag//50;valid=(bad[si[d:]+1]-bad[si[:-d]])==0
        for key,x in residual.items():
            xx=x[:-d][valid];yy=x[d:][valid]
            add('hot_memory',observable=key,lag=lag,pairs=len(xx),correlation=np.corrcoef(xx,yy)[0,1] if len(xx)>20 else np.nan)
    chi=np.take_along_axis(bp['m'],p,axis=1)
    exc=cold_hot_returns(temps,chi,ts[keep],cold,.4)
    for row in exc:add('excursions',cadence=50,**row)

    # Full-tensor hot memory and cold-hot-cold orientation checks.
    use=sp>burn;spi=np.searchsorted(hs,sp[use]);qp=pos[spi]
    Q=amp[use,:,2:7,0].real/L**2
    Qw=np.take_along_axis(Q,qp[:,:,None],axis=1)
    Qres=np.take_along_axis(Q-Q.mean(0),qp[:,:,None],axis=1)
    for lag in (1000,5000,10000):
        d=lag//1000;valid=(bad[spi[d:]+1]-bad[spi[:-d]])==0
        x=Qres[:-d][valid];y=Qres[d:][valid];corr=np.nan
        if len(x)>20:
            x=x-x.mean(0);y=y-y.mean(0);corr=np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))
        add('hot_memory',observable='Q_tensor',lag=lag,pairs=len(x),correlation=corr)
    csp=np.take_along_axis((a['chirality']/a['density'])[np.searchsorted(ts,sp[use])],qp,axis=1)
    for row in cold_hot_returns(T[qp],csp,sp[use],cold,.4,Qw):add('Q_excursions',cadence=1000,**row)
    delta=np.diff(p.astype(float),axis=0)
    both=(delta[:-1]!=0)&(delta[1:]!=0)&(abs(temps[1:-1]-tc)<.0005)
    M=(acceptance_chain(accept)**10).toarray()[:R,:R]
    ib=np.triu(M,1).sum(0);ia=np.tril(M,-1).sum(0);ob=np.tril(M,-1).sum(1);oa=np.triu(M,1).sum(1);near=abs(T-tc)<.0005
    tables['summary'][0].update(reversal_fraction=np.mean((delta[:-1]*delta[1:]<0)[both]),
        null_reversal_fraction=(ib*ob+ia*oa)[near].sum()/((ib+ia)*(ob+oa))[near].sum())
    del bad,labels,residual,bp,b,delta,chi,temps,lab,pos

    # Reuse the validated spatial summary, retaining the original scalar indices.
    prod_meta=dict(m,spatial_sample_indices=np.asarray(m['spatial_sample_indices'])[use],spatial_sample_sweeps=sp[use])
    prod_samples=dict(a,spatial_amplitudes=amp[use],spatial_structure=structure[use],spatial_walkers=sw[use],spatial_energy=se[use])
    spatial=summarize_spatial(dict(samples=prod_samples,metadata=prod_meta),block_size=100)
    for j,t in enumerate(T):
        for k,sector in enumerate(spatial['sectors']):
            for label in ('full','connected'):
                add('spatial',T=t,observable=f'{sector}_{label}_S0_per_site',
                    value=spatial[label+'_spectra'][j,k,0]/L**2,se=spatial[label+'_spectra_se'][j,k,0]/L**2)
        for k,name in enumerate(spatial['local_ratio_names']):
            add('spatial',T=t,observable=str(name),value=spatial['local_ratios'][j,k],se=spatial['local_ratios_se'][j,k])
        for k in (13,14,16):
            add('spatial',T=t,observable=m['structure_names'][k]+'_per_site',value=spatial['structure'][j,k]/L**2,se=spatial['structure_se'][j,k]/L**2)
    add('validation',passed=True,finite_channels=True,exact_trip_counters=True,
        scalar_events=len(ts),spatial_events=len(sp),history_rows=len(hs),
        source_bytes=path.stat().st_size,source_mtime_ns=path.stat().st_mtime_ns,
        max_spatial_residual=max(spatial_check['maximum_absolute_residuals'].values()))
    print(path.name, 'passages',npass,'loops',tables['summary'][0]['production_loops'],
          'flux',round(flux),'analyzed in',round(perf_counter()-start,1),'s',flush=True)
    return tables


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(x) for x in sys.argv[1:]] if len(sys.argv)>1 else sorted(DATA.glob('*.npz'))
    for path in paths:
        tables=analyze(path)
        with (OUT/(path.stem+'.json')).open('w',encoding='utf8') as f:
            json.dump(tables,f,default=lambda x:x.item() if isinstance(x,np.generic) else x,allow_nan=True)
        del tables
        gc.collect()
    combined={}
    for path in sorted(OUT.glob('L=96_*.json')):
        for name,rows in json.loads(path.read_text(encoding='utf8')).items():combined.setdefault(name,[]).extend(rows)
    for name,rows in combined.items():
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Tables saved in',OUT,flush=True)


if __name__=='__main__':
    main()
