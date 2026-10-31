"""Independent raw-channel checks and a focused weak-SOC symmetry-sector check.

Run after summarize.py: python exploration/he3_soc/validate.py
Only two raw NPZ files are read, one channel at a time. All generated material
is written inside exploration/he3_soc/results.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from exploration.he3_soc.summarize import estimate
from utils.Analysis.statistics import blocked_jackknife

OUT=Path(__file__).resolve().parent/"results"


def independent_estimator(m,T,L):
    n=L*L
    hx=(m["kx"]-(m["ix2"]-m["ix"]*m["ix"])/T)/n
    hy=(m["ky"]-(m["iy2"]-m["iy"]*m["iy"])/T)/n
    return dict(energy=m["e"],density=m["n"],chirality=m["chi_abs"],
                signed_chirality=m["chi"],phase=np.sqrt(m["p2"]),
                binder_chirality=1-m["chi4"]/(3*m["chi2"]*m["chi2"]),
                binder_phase=2-m["p4"]/(m["p2"]*m["p2"]),
                helicity_x=hx,helicity_y=hy,helicity=(hx+hy)/2,
                spin_alignment=m["qzz"],
                specific_heat=n*(m["e2"]-m["e"]*m["e"])/(T*T))


def main():
    results=[]
    for gd,L,targets in [(0,64,[.33,.3815,.4]),(.1,160,[.33,.37099,.44335,.46595])]:
        cachepath=OUT/"blocks"/f"gd={gd:g}_L={L}.npz"
        with np.load(cachepath,allow_pickle=False) as z:
            T=z["T"];cache={k:z[k] for k in z.files if k!="T"}
        allvalues,_=estimate(cache,T,L,100)
        ids=sorted(set([int(np.argmin(abs(T-t))) for t in targets]+
                       [int(np.argmin(allvalues["binder_chirality"]))]))
        tt=T[ids]
        path=ROOT/"data/he3"/f"P=20_gd={gd:g}_gaussian"/f"L={L}.npz"
        with np.load(path,allow_pickle=False) as z:
            # Indexing a compressed member decompresses that member, so retain
            # only the selected columns before reading the next channel.
            def channel(name):
                return z[f"L{L}_{name}"][:,ids].copy()
            raw={}
            e=channel("energy")/(L*L)
            raw.update(e=e,e2=e*e)
            density=channel("density")
            chi=channel("chirality")/density
            raw.update(n=density/(L*L),chi=chi,chi_abs=abs(chi),chi2=chi*chi,chi4=chi**4)
            pr=channel("phase_real");pi=channel("phase_imag")
            p2=(pr*pr+pi*pi)/(L**4)
            raw.update(p2=p2,p4=p2*p2,p_abs=np.sqrt(p2))
            for suffix in ("x","y"):
                current=channel("current_"+suffix)
                raw["i"+suffix]=current;raw["i"+suffix+"2"]=current*current
                raw["k"+suffix]=channel("curvature_"+suffix)
            raw["qzz"]=channel("spin_alignment")
            raw["qzz2"]=raw["qzz"]**2
        cache_errors={}
        for key,a in raw.items():
            direct=a.reshape(600,100,len(ids)).mean(axis=1)
            saved=cache[key][:,ids]
            np.testing.assert_allclose(direct,saved,rtol=2e-12,atol=1e-10)
            cache_errors[key]=float(np.max(abs(direct-saved)))
        names=list(independent_estimator({k:a.mean(axis=0) for k,a in raw.items()},tt,L))
        def packed(m):
            value=independent_estimator(m,tt,L)
            return np.stack([value[k] for k in names])
        for bs in (100,2000):
            direct_value,direct_error=blocked_jackknife(raw,packed,bs)
            value,error=estimate(cache,T,L,bs)
            checks={}
            for k,v,e in zip(names,direct_value,direct_error):
                np.testing.assert_allclose(v,value[k][ids],rtol=5e-10,atol=2e-9)
                np.testing.assert_allclose(e,error[k][ids],rtol=2e-6,atol=2e-9)
                checks[k]=dict(max_value_absolute_difference=float(np.max(abs(v-value[k][ids]))),
                               max_error_absolute_difference=float(np.max(abs(e-error[k][ids]))))
            results.append(dict(g_d=gd,L=L,dataset=str(path.relative_to(ROOT)),temperatures=tt.tolist(),
                                block_size=bs,cache_moment_max_absolute_differences=cache_errors,
                                jackknife_comparison=checks))
        print(f"Validated raw channels, cache moments, values and jackknife errors: gd={gd:g}, L={L}",flush=True)
    trend=[]
    fig,axs=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey="row")
    for col,gd in enumerate((0,.001,.01)):
        L=160
        with np.load(OUT/"blocks"/f"gd={gd:g}_L={L}.npz") as z:
            T=z["T"];j=int(np.argmin(abs(T-.3504517922107261)))
            rawblocks={k:z[k][:,j] for k in ("chi","chi_abs","n","e","qzz")}
        fine_time=(2000000+(np.arange(600)+.5)*5000)/1e6
        time=(2000000+(np.arange(12)+.5)*250000)/1e6
        for row,key in enumerate(("chi","chi_abs")):
            a=rawblocks[key]
            axs[row,col].plot(fine_time,a,color=".65",lw=.4,alpha=.55,label="100-measurement blocks")
            axs[row,col].plot(time,a.reshape(12,50).mean(axis=1),"o-",color="#4477aa",lw=1.6,ms=4,label="5,000-measurement blocks")
            axs[row,col].axvline(3.5,color="black",ls="--",lw=.7)
            axs[row,col].set(xlabel="Million local sweeps" if row else "")
            axs[row,col].grid(alpha=.2)
        axs[0,col].set_title(f"$g_D={gd:g}$, L=160, T={T[j]:.6f}")
        for key,a in rawblocks.items():
            for fac in (1,5,20,50,100):
                f=a[:300].reshape(-1,fac).mean(axis=1)
                s=a[300:].reshape(-1,fac).mean(axis=1)
                se=np.sqrt(f.var(ddof=1)/len(f)+s.var(ddof=1)/len(s))
                trend.append(dict(g_d=gd,L=L,temperature=T[j],observable=key,block_measurements=fac*100,
                                  blocks_per_half=len(f),first_half=f.mean(),second_half=s.mean(),
                                  half_difference=s.mean()-f.mean(),half_difference_se=se,
                                  descriptive_half_z=(s.mean()-f.mean())/se))
    axs[0,0].set_ylabel(r"Signed $m_\chi$")
    axs[1,0].set_ylabel(r"Absolute $|m_\chi|$")
    axs[0,0].legend(fontsize=7,loc="lower left")
    fig.suptitle("Weak-SOC symmetry-sector drift need not imply drift of even observables\nBlock traces are descriptive; temperatures share walkers and are not independent chains")
    fig.tight_layout();fig.savefig(OUT/"weak_chirality_sector_drift.png",dpi=170);plt.close(fig)
    with (OUT/"weak_sector_drift.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(trend[0]));writer.writeheader();writer.writerows(trend)
    output=dict(passed=True,checks=results,
                notes=["Raw formulas and original blocked_jackknife match cache estimates at selected temperatures.",
                       "Validation checks arithmetic, not statistical convergence or equilibration.",
                       "Root tau uses initial-monotone paired ACF sums, biased FFT ACF, and clamp tau>=0.5.",
                       "Geyer reversible-chain justification is not a convergence guarantee for alternating nonreversible PT.",
                       "Same L seeds recur across g_D: cross-coupling covariance is not measured; independence is an assumption.",
                       "Half-z scores are descriptive, from few blocks and many correlated comparisons."])
    (OUT/"validation.json").write_text(json.dumps(output,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
