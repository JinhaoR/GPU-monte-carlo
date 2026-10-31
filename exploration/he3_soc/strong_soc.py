"""Read-only investigation of strong-SOC branches and rare-event estimators.

Run from project root: python exploration/he3_soc/strong_soc.py
Only one source channel is decompressed at a time, apart from Q, n and chi.
Outputs describe this finite trajectory; none establish equilibrium coexistence.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration/he3_soc/results"
SIZES = (64, 96, 128, 160)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binder(x):
    return 1 - np.mean(x**4, axis=0) / (3 * np.mean(x*x, axis=0)**2)


def mean_branch(x, mask):
    count = mask.sum(axis=0)
    total = np.einsum("ij,ij->j", x, mask)
    return np.divide(total, count, out=np.full(x.shape[1], np.nan), where=count > 0)


def helicity(data, T, N):
    return .5 * sum((np.mean(data[f"curvature_{d}"], axis=0)
                    - np.var(data[f"current_{d}"], axis=0) / T) / N
                   for d in ("x", "y"))


def estimate_selected(data, T, N):
    return np.array([np.mean(data["high"]), np.mean(data["q"]),
                     np.mean(data["density"]), np.mean(data["energy"]),
                     np.mean(abs(data["chi"])), binder(data["chi"]),
                     helicity(data, T, N)])


def jackknife_selected(data, T, N, block):
    # Joint deletion of contiguous blocks preserves cross-channel dependence.
    n = len(data["q"])
    B = n // block
    trimmed = {k: v[:B * block] for k, v in data.items()}
    estimate = estimate_selected(trimmed, T, N)
    # Only small selected-temperature arrays are copied; no full dataset copies.
    deleted = np.array([
        estimate_selected({k: np.r_[v[:b*block], v[(b+1)*block:]]
                           for k, v in trimmed.items()}, T, N)
        for b in range(B)
    ])
    return estimate, np.sqrt((B-1) * np.var(deleted, axis=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, anomalies, events, uncertainties, correlations, classifications, rare_events = [], [], [], [], [], [], []
    inventory_plot, joint_plot, rare_plot, response_plot = [], [], [], []
    for L in SIZES:
        N = L*L
        path = ROOT / f"data/he3/P=20_gd=0.1_gaussian/L={L}.npz"
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(z["manifest"].item())["runs"][str(L)]["metadata"]
            T = np.asarray(meta["temperatures"])
            sweeps = np.asarray(meta["sample_sweeps"])
            q = z[f"L{L}_spin_alignment"]
            n = z[f"L{L}_density"]
            chi = z[f"L{L}_chirality"] / n
            n /= N
            high = q > .3
            p = high.mean(axis=0)
            U = binder(chi)
            qmean = q.mean(axis=0)
            jrare = int(np.argmin(U))
            jstep = int(np.argmax(abs(np.diff(qmean))))
            jmix = int(np.argmin(abs(p-.5)))
            indices = sorted(set([0, jrare, jstep, jstep+1, jmix,
                                  int(np.argmin(abs(T-.388))),
                                  int(np.argmin(abs(T-.443)))]))
            selected = {j: {"q": q[:, j].copy(), "chi": chi[:, j].copy(),
                            "density": n[:, j].copy(), "high": high[:, j].copy()}
                        for j in indices}
            stats = {"Qzz": qmean, "density": n.mean(axis=0),
                     "abs_chi": abs(chi).mean(axis=0),
                     "chi2": (chi*chi).mean(axis=0), "chi4": (chi**4).mean(axis=0),
                     "binder": U, "high_fraction": p,
                     "high_fraction_first": high[:30000].mean(axis=0),
                     "high_fraction_second": high[30000:].mean(axis=0),
                     "Qzz_high": mean_branch(q, high), "Qzz_low": mean_branch(q, ~high),
                     "density_high": mean_branch(n, high), "density_low": mean_branch(n, ~high),
                     "abs_chi_high": mean_branch(abs(chi), high),
                     "abs_chi_low": mean_branch(abs(chi), ~high)}
            inv = high.sum(axis=1)
            for cutoff in (.2, .3, .4):
                count = (q > cutoff).sum(axis=1)
                classifications.append(dict(L=L, classifier="Qzz", threshold=cutoff,
                    initial=int(count[0]), final=int(count[-1]),
                    positive_inventory_changes=int(np.sum(np.diff(count)>0)),
                    negative_inventory_changes=int(np.sum(np.diff(count)<0)),
                    disagreements_with_Q03=int(np.sum((q>cutoff)!=high))))
            count = (abs(chi) > .1).sum(axis=1)
            classifications.append(dict(L=L, classifier="abs_chi", threshold=.1,
                initial=int(count[0]), final=int(count[-1]),
                positive_inventory_changes=int(np.sum(np.diff(count)>0)),
                negative_inventory_changes=int(np.sum(np.diff(count)<0)),
                disagreements_with_Q03=int(np.sum((abs(chi)>.1)!=high))))
            # Metadata positions and measurements share exact sweep coordinates.
            hist = meta["parallel_tempering"]["walker_history"]
            hs = np.asarray(hist["sweeps"])
            keep = hs > meta["thermalization_sweeps"]
            hs = hs[keep]
            pos = np.asarray(hist["temperature_indices"])[keep]
            ix = np.searchsorted(sweeps, hs)
            assert np.array_equal(sweeps[ix], hs)
            ids = np.asarray(hist["walker_ids"])
            wq = np.take_along_axis(q[ix], pos, axis=1)
            whigh = wq > .3
            change_t, change_w = np.where(whigh[1:] != whigh[:-1])
            for t, w in zip(change_t, change_w):
                events.append(dict(L=L, walker=int(ids[w]),
                    preceding_history_sweep=int(hs[t]), following_history_sweep=int(hs[t+1]),
                    previous_T=float(T[pos[t,w]]), following_T=float(T[pos[t+1,w]]),
                    previous_Qzz=float(wq[t,w]), following_Qzz=float(wq[t+1,w]),
                    direction="high_to_low" if whigh[t,w] else "low_to_high"))
            # Fourth-moment concentration and deletion challenges.
            cr = chi[:, jrare]
            weight = cr**4
            rare = abs(cr) > .1
            order = np.argsort(abs(cr))[::-1]
            challenge = {}
            for k in (1, 2, 5, 10, 20):
                challenge[f"binder_remove_top{k}"] = float(binder(cr[np.argsort(abs(cr))[:-k]]))
            anomalies.append(dict(L=L, source=str(path.relative_to(ROOT)),
                Q_jump_T_lo=float(T[jstep]), Q_jump_T_hi=float(T[jstep+1]),
                mix_T=float(T[jmix]), high_fraction=float(p[jmix]),
                high_fraction_first=float(stats["high_fraction_first"][jmix]),
                high_fraction_second=float(stats["high_fraction_second"][jmix]),
                inventory_initial=int(inv[0]), inventory_final=int(inv[-1]),
                physical_high_to_low=int(sum(whigh[t,w] for t,w in zip(change_t,change_w))),
                physical_low_to_high=int(sum(not whigh[t,w] for t,w in zip(change_t,change_w))),
                binder_min_T=float(T[jrare]), binder_min=float(U[jrare]),
                chi2=float(stats["chi2"][jrare]), chi4=float(stats["chi4"][jrare]),
                tail_count=int(rare.sum()), tail_fraction=float(rare.mean()),
                tail_episodes=int(np.sum(rare & np.r_[True, ~rare[:-1]])),
                high_branch_negative_chirality_count=int(np.sum(high & (chi < 0))),
                tail_chi4_fraction=float(weight[rare].sum()/weight.sum()),
                top1_chi4_fraction=float(weight[order[0]]/weight.sum()),
                binder_first=float(binder(cr[:30000])), binder_second=float(binder(cr[30000:])),
                binder_without_abs_chi_gt01=float(binder(cr[~rare])), **challenge))
            inventory_plot.append((L, sweeps, inv, hs, whigh, ids))
            rare_plot.append((L, T, U.copy(), cr.copy(), q[:,jrare].copy(), sweeps, order,
                              float(T[jrare])))
            # Free large chi/n arrays before decompressing current channels.
            del chi, n
            for name in ("energy", "current_x", "curvature_x", "current_y", "curvature_y"):
                arr = z[f"L{L}_{name}"]
                if name == "energy":
                    arr /= N
                for j in indices:
                    selected[j][name] = arr[:,j].copy()
                stats[name] = arr.mean(axis=0)
                stats[name+"_high"] = mean_branch(arr, high)
                stats[name+"_low"] = mean_branch(arr, ~high)
                if name.startswith("current"):
                    stats[name+"2"] = np.einsum("ij,ij->j", arr, arr)/len(arr)
                    stats[name+"2_high"] = mean_branch(arr*arr, high)
                    stats[name+"2_low"] = mean_branch(arr*arr, ~high)
                del arr
            for d in ("x", "y"):
                for suffix in ("", "_high", "_low"):
                    stats[f"helicity_{d}{suffix}"] = (stats[f"curvature_{d}{suffix}"]
                        -(stats[f"current_{d}2{suffix}"]-stats[f"current_{d}{suffix}"]**2)/T)/N
                stats[f"mean_current_correction_{d}"] = stats[f"current_{d}"]**2/T/N
            for suffix in ("", "_high", "_low"):
                stats["helicity"+suffix] = .5*(stats["helicity_x"+suffix]+stats["helicity_y"+suffix])
            for i in np.flatnonzero(rare):
                d = selected[jrare]
                rare_events.append(dict(L=L,T=float(T[jrare]),sweep=int(sweeps[i]),
                    chi=float(d["chi"][i]),Qzz=float(d["q"][i]),
                    energy=float(d["energy"][i]),density=float(d["density"][i]),
                    current_x=float(d["current_x"][i]),current_y=float(d["current_y"][i])))
            for j in range(len(T)):
                all_rows.append(dict(L=L, T=float(T[j]), **{k:float(v[j]) for k,v in stats.items()}))
            # Only 7 selected temperatures are used for expensive joint deletion.
            for j in indices:
                data = selected[j]
                for block in (200, 1000, 5000, 10000):
                    estimate, error = jackknife_selected(data, T[j], N, block)
                    for name,value,se in zip(("high_fraction","Qzz","density","energy","abs_chi","binder","helicity"),estimate,error):
                        uncertainties.append(dict(L=L,T=float(T[j]),index=j,
                            role="binder_min" if j==jrare else "mix" if j==jmix else "selected",
                            block_samples=block,block_sweeps=block*50,
                            observable=name,estimate=float(value),jackknife_se=float(se)))
                # Correlations pooled and conditional; branch mixture can dominate.
                channels={"energy":data["energy"], "density":data["density"],
                          "Qzz":data["q"], "abs_chi":abs(data["chi"]),
                          "current_x":data["current_x"],"current_y":data["current_y"]}
                names=list(channels)
                for scope, mask in (("pooled",np.ones(60000,bool)),("high",data["high"]),("low",~data["high"])):
                    for ia,a in enumerate(names):
                        for b in names[ia+1:]:
                            if mask.sum()>10:
                                correlations.append(dict(L=L,T=float(T[j]),scope=scope,n=int(mask.sum()),
                                    x=a,y=b,r=float(np.corrcoef(channels[a][mask],channels[b][mask])[0,1])))
            joint_plot.append((L,float(T[jmix]),selected[jmix]))
            response_plot.append((L,T,{k:v for k,v in stats.items() if "helicity" in k or "mean_current" in k},
                                  selected[jmix],float(T[jmix])))
            del q, high, selected
            print(f"L={L}: branch inventory {inv[0]}->{inv[-1]}, Umin={U[jrare]:.3f}, "
                  f"rare count={int(rare.sum())}",flush=True)
    write_csv("strong_temperature.csv",all_rows)
    write_csv("strong_anomalies.csv",anomalies)
    write_csv("strong_branch_events.csv",events)
    write_csv("strong_uncertainties.csv",uncertainties)
    write_csv("strong_correlations.csv",correlations)
    write_csv("strong_classification.csv",classifications)
    write_csv("strong_rare_events.csv",rare_events)

    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,(L,sweeps,inv,hs,whigh,ids) in zip(axes.flat,inventory_plot):
        ax.step((sweeps[::20]-2e6)/1e6,inv[::20],where="post",color="C0")
        ax.set(title=f"L={L}: high-Q branch inventory",ylabel="Number of 160 walkers",xlabel="Production sweeps (millions)")
        ax.set_yticks(np.arange(inv.min(),inv.max()+1))
        ax.grid(alpha=.25)
    fig.suptitle(r"$g_D=0.1$: $Q_{zz}>0.3$ inventory only decreases; temperature swaps conserve this count")
    fig.tight_layout(); fig.savefig(OUT/"strong_branch_inventory.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for col,(L,t,d) in enumerate(joint_plot):
        for row,(key,label) in enumerate((("energy","Energy/site"),("chi","Signed normalized chirality"))):
            h=axes[row,col].hist2d(d["q"],d[key],bins=60,norm=matplotlib.colors.LogNorm(),cmap="magma")
            axes[row,col].set(xlabel=r"$Q_{zz}$",ylabel=label,title=f"L={L}, T={t:.6f}")
    fig.suptitle("Strong SOC: separated high-Q chiral and low-Q nearly nonchiral populations\nHistograms pool a nonstationary PT trajectory")
    fig.tight_layout();fig.savefig(OUT/"strong_joint_branches.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(15,10))
    for col,(L,T,U,c,q,sweeps,order,t) in enumerate(rare_plot):
        axes[0,col].plot(T,U,color="C3");axes[0,col].set(xlim=(.35,.4),title=f"L={L}",xlabel="T",ylabel="Chirality Binder")
        axes[1,col].scatter((sweeps-2e6)/1e6,c,s=1,alpha=.5);axes[1,col].set(title=f"Binder minimum T={t:.6f}",xlabel="Production sweeps (millions)",ylabel=r"$m_\chi$")
        weights=c[order]**4
        axes[2,col].plot(np.arange(1,101),np.cumsum(weights[:100])/weights.sum());axes[2,col].set(xlabel="Largest |chirality| samples retained",ylabel="Fraction of fourth moment",ylim=(0,1.03),xscale="log");axes[2,col].grid(alpha=.25)
    fig.suptitle("Giant negative Binder values are dominated by 2–13 chiral intrusions among 60,000 samples")
    fig.tight_layout();fig.savefig(OUT/"strong_binder_rare_events.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for col,(L,T,s,d,t) in enumerate(response_plot):
        ax=axes[0,col]
        p=np.array([r["high_fraction"] for r in all_rows if r["L"]==L])
        high_h=np.where(p>=.005,s["helicity_high"],np.nan)
        low_h=np.where((1-p)>=.005,s["helicity_low"],np.nan)
        ax.plot(T,s["helicity"],label="Pooled");ax.plot(T,high_h,"--",label="High-Q conditional");ax.plot(T,low_h,"--",label="Low-Q conditional")
        ax.plot(T,8*T/np.pi,":",c="black",label="q=1/2 jump")
        ax.set(title=f"L={L}",xlabel="T",ylabel="Helicity",ylim=(-.5,3.2));ax.grid(alpha=.25)
        ax=axes[1,col]
        ax.plot(T,s["mean_current_correction_x"],label="x");ax.plot(T,s["mean_current_correction_y"],label="y")
        ax.set(xlabel="T",ylabel=r"$\langle I_\mu\rangle^2/(TN)$",title="Contribution from sampled mean current")
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);axes[1,0].legend()
    fig.suptitle("Strong-SOC helicity: conditional diagnostics require ≥0.5% branch occupancy\nConditional samples do not define separate equilibrium ensembles")
    fig.tight_layout();fig.savefig(OUT/"strong_helicity_components.png",dpi=170);plt.close(fig)


if __name__ == "__main__":
    main()
