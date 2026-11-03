"""Bounded numerical checks, long-block ESS diagnostics, and preservation audit."""
import csv,json,hashlib,subprocess,ast
import numpy as np
from extract import ROOT,HERE,OUT,write
from transport import acceptance_chain,null_transport,events,km
from physical import block_estimate

def main():
    checks={}
    for R in (4,8,20):
        P,rt,symmetric=null_transport(np.ones(R-1),5)
        np.testing.assert_allclose(P.sum(0),1);np.testing.assert_allclose(P.sum(1),1)
        assert abs(rt-2*R*5)<1e-8
        assert abs(symmetric-10*R*(R-1))<1e-8
        P=acceptance_chain(np.linspace(.2,.8,R-1))
        np.testing.assert_allclose(P.sum(0),1);np.testing.assert_allclose(P.sum(1),1)
    checks['alternating_chain_stationarity_and_ballistic_limit']=True
    pos=np.array([[0,1,2,3],[1,0,3,2],[3,2,1,0],[2,3,0,1],[0,1,2,3]])
    loops,passages,first,w,labels=events(pos,np.arange(5)*100)
    assert w[0]['loops']==1 and w[0]['both'] and first[0][0]==400
    _,_,first_short,wshort,_=events(pos[:3],np.arange(3)*100)
    assert not any(r[1] for r in first_short) and wshort[0]['both']
    assert np.isnan(km(first_short)[1])
    # A temperature-ordered measurement must recover the original walker value.
    walker_values=100*np.arange(5)[:,None]+np.arange(4)[None,:]
    atT=np.take_along_axis(walker_values,np.argsort(pos,axis=1),axis=1)
    np.testing.assert_array_equal(np.take_along_axis(atT,pos,axis=1),walker_values)
    checks['endpoint_censoring_and_permutation_toys']=True
    rows=[];width=[]
    inv=list(csv.DictReader((OUT/'inventory.csv').open()))
    for m in inv:
        if m['campaign']!='new':continue
        L=int(m['L']);gd=float(m['g_d']);N=L*L
        with np.load(OUT/(m['tag']+'.npz')) as z:d=dict(z)
        T=d['T'];j=np.argmax(d['Cv']);cv=d['Cv'];half=cv[j]/2
        il=np.flatnonzero(cv[:j]<half);ir=np.flatnonzero(cv[j+1:]<half)+j+1
        lo=np.interp(half,cv[il[-1]:il[-1]+2],T[il[-1]:il[-1]+2]) if len(il) else np.nan
        hi=np.interp(half,cv[ir[0]-1:ir[0]+1][::-1],T[ir[0]-1:ir[0]+1][::-1]) if len(ir) else np.nan
        width.append(dict(g_d=gd,L=L,Cv_peak=cv[j],T_peak=T[j],raw_half_height_width=hi-lo,sigma_H_peak=d['energy_sigma'][j]))
        if L not in (160,256):continue
        with np.load(ROOT/m['source']) as z:
            def channel(name):return z[f'L{L}_{name}'][:,j]
            e=channel('energy')/N;c=channel('chirality')/channel('density')
            M2=np.mean(c*c);M4=np.mean(c**4)
            influence=-(c**4)/(3*M2**2)+2*M4*c*c/(3*M2**3)
            y=np.zeros(len(e))
            for axis in ('x','y'):
                i=channel('current_'+axis);k=channel('curvature_'+axis)
                y+=(k-i*i/T[j]+2*i.mean()*i/T[j])/(2*N)
        np.testing.assert_allclose(e.mean(),d['block_e'][:,j].mean(),rtol=1e-12)
        np.testing.assert_allclose(M2,d['block_m2'][:,j].mean(),rtol=1e-12)
        blocks={k[6:]:v for k,v in d.items() if k.startswith('block_')}
        for factor in (2,5,10):
            _,se=block_estimate(blocks,T,L,factor)
            for name,a in [('energy',e),('m2',c*c),('Binder',influence),('Y',y)]:
                variance=a.var(ddof=1);ess=variance/se[name][j]**2
                rows.append(dict(g_d=gd,L=L,T=T[j],observable=name,block_sweeps=factor*50000,instant_variance=variance,block_se=se[name][j],naive_se=np.sqrt(variance/len(a)),block_ESS=ess,block_tau_sweeps=3e6/(2*ess)))
    write('long_block_ess.csv',rows);write('peak_widths.csv',width)
    checks['raw_moment_vs_cached_blocks']=True
    before=json.loads((HERE/'preservation_before.json').read_text())
    source_bad=[p for p,h in before['sources'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    data_bad=[p for p,s in before['data'].items() if (ROOT/p).stat().st_size!=s['size'] or (ROOT/p).stat().st_mtime_ns!=s['mtime_ns']]
    assert not source_bad and not data_bad
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==before['branch']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==before['head']
    assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    checks.update(existing_sources_unchanged=len(before['sources']),datasets_size_and_mtime_unchanged=len(before['data']),branch_and_head_unchanged=True,tracked_diff_empty=True,analysis_syntax_valid=True)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
