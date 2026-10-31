"""Saved-walker diagnostics; run `python exploration/he3_soc/pt_diagnostics.py`.

Only NPZ manifests are read.  Original data and production code stay untouched.
Passages and boundary crossings reconstructed from 500-sweep snapshots are lower
bounds; exact stored round-trip counters include thermalization.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
GDS = (0, 0.001, 0.01, 0.1)
SIZES = (64, 96, 128, 160)
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_history(x, sweeps, ids):
    """Sampled last-endpoint labels and alternating-endpoint passage events."""
    labels = np.full(x.shape[1], -1, dtype=int)
    history = np.empty(x.shape, dtype=np.int16)
    events = []
    for k, row in enumerate(x):
        at = np.flatnonzero((row == 0) | (row == x.shape[1] - 1))
        changed = at[(labels[at] >= 0) & (labels[at] != row[at])]
        for walker in changed:
            events.append((int(sweeps[k]), int(ids[walker]),
                           "cold_to_hot" if row[walker] else "hot_to_cold"))
        labels[at] = row[at]
        history[k] = labels
    return history, events


def crossing_counts(x):
    """Counts of observed side changes at each edge, not exact swap counts."""
    edges = np.arange(x.shape[1] - 1)
    counts = np.zeros(len(edges), dtype=np.int64)
    distinct = np.zeros(len(edges), dtype=np.int64)
    for edge in edges:
        side = x > edge
        changes = side[1:] != side[:-1]
        counts[edge] = changes.sum()
        distinct[edge] = np.any(changes, axis=0).sum()
    return counts, distinct


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, edges, walkers, events_out, windows = [], [], [], [], []
    runs = {}
    for gd in GDS:
        for L in SIZES:
            path = ROOT / "data/he3" / f"P=20_gd={gd:g}_gaussian" / f"L={L}.npz"
            with np.load(path, allow_pickle=False) as file:
                manifest = json.loads(file["manifest"].item())
            d = manifest["runs"][str(L)]["metadata"]
            pt = d["parallel_tempering"]
            h = pt["walker_history"]
            ids = np.asarray(h["walker_ids"], dtype=int)
            order = np.argsort(ids)
            ids = ids[order]
            x = np.asarray(h["temperature_indices"], dtype=np.int16)[:, order]
            sweeps = np.asarray(h["sweeps"], dtype=int)
            T = np.asarray(d["temperatures"])
            assert x.shape == (len(sweeps), len(T))
            assert np.array_equal(ids, np.arange(len(T)))
            assert np.all(np.sort(x, axis=1) == np.arange(len(T)))
            assert np.all(np.diff(sweeps) > 0)
            prod = sweeps >= d["thermalization_sweeps"]
            xp = x[prod]
            attempted = np.asarray(pt["attempts"])
            accepted = np.asarray(pt["accepted"])
            acc = accepted / attempted
            labels, events = endpoint_history(x, sweeps, ids)
            events_prod = [e for e in events if e[0] > d["thermalization_sweeps"]]
            # A strict production-only passage requires both endpoint visits in
            # production, unlike a full-history passage whose completion is late.
            _, strict_events = endpoint_history(xp, sweeps[prod], ids)
            low, high = xp.min(axis=0), xp.max(axis=0)
            span = high - low
            observed_crossings, crossing_walkers = crossing_counts(xp)
            occ = np.zeros((len(T), len(T)), dtype=int)
            for w in range(len(T)):
                occ[w] = np.bincount(xp[:, w], minlength=len(T))
            probability = occ / len(xp)
            entropy = -(probability * np.log(np.maximum(probability, 1e-300))).sum(axis=1) / np.log(len(T))
            # Invert each saved permutation to get last-endpoint label at T.
            label_at_T = np.take_along_axis(labels, np.argsort(x, axis=1), axis=1)
            labelled_cold = (label_at_T[prod] == 0).sum(axis=0)
            labelled_hot = (label_at_T[prod] == len(T) - 1).sum(axis=0)
            f = np.divide(labelled_cold, labelled_cold + labelled_hot,
                          out=np.full(len(T), np.nan), where=(labelled_cold + labelled_hot) > 0)
            steep = int(np.nanargmax(f[:-1] - f[1:]))
            pair_min = int(np.argmin(acc))
            n_way = lambda seq, direction: sum(e[2] == direction for e in seq)
            summary = dict(g_d=gd, L=L, dataset=str(path.relative_to(ROOT)).replace("\\", "/"),
                           exchange_interval=pt["exchange_interval"],
                           history_spacing_sweeps=int(np.median(np.diff(sweeps))),
                           diagnostics_scope=pt["diagnostics_scope"],
                           exact_round_trips=int(np.sum(pt["walker_round_trips"])),
                           walkers_with_exact_trips=int(np.count_nonzero(pt["walker_round_trips"])),
                           mean_completed_trip_sweeps=pt["mean_round_trip_sweeps"],
                           aggregate_acceptance=float(accepted.sum() / attempted.sum()),
                           min_acceptance=float(acc[pair_min]), min_acceptance_T=float((T[pair_min] + T[pair_min+1])/2),
                           sampled_cold_to_hot=n_way(events, "cold_to_hot"),
                           sampled_hot_to_cold=n_way(events, "hot_to_cold"),
                           sampled_production_completion_cold_to_hot=n_way(events_prod, "cold_to_hot"),
                           sampled_production_completion_hot_to_cold=n_way(events_prod, "hot_to_cold"),
                           sampled_strict_production_cold_to_hot=n_way(strict_events, "cold_to_hot"),
                           sampled_strict_production_hot_to_cold=n_way(strict_events, "hot_to_cold"),
                           sampled_walkers_both_endpoints_full=int(np.sum((x.min(axis=0)==0)&(x.max(axis=0)==len(T)-1))),
                           sampled_walkers_both_endpoints_production=int(np.sum((low==0)&(high==len(T)-1))),
                           median_production_index_span=float(np.median(span)),
                           median_production_visited_temperatures=float(np.median((occ>0).sum(axis=1))),
                           mean_production_walker_entropy=float(entropy.mean()),
                           min_production_distinct_walkers_per_T=int((occ>0).sum(axis=0).min()),
                           max_endpoint_flow_drop=float(f[steep]-f[steep+1]),
                           max_endpoint_flow_drop_T=float((T[steep]+T[steep+1])/2))
            summaries.append(summary)
            for e in events:
                events_out.append(dict(g_d=gd,L=L,sweep=e[0],walker=e[1],direction=e[2],
                                       completion_in_production=e[0] > d["thermalization_sweeps"]))
            for j in range(len(T)-1):
                edges.append(dict(g_d=gd,L=L,edge=j,T_left=T[j],T_right=T[j+1],
                                  attempts=attempted[j],accepted=accepted[j],acceptance=acc[j],
                                  sampled_production_side_changes=observed_crossings[j],
                                  sampled_production_distinct_crossing_walkers=crossing_walkers[j],
                                  production_cold_last_endpoint_fraction_left=f[j],
                                  production_cold_last_endpoint_fraction_right=f[j+1]))
            for w in ids:
                walkers.append(dict(g_d=gd,L=L,walker=int(w),
                                    exact_round_trips=pt["walker_round_trips"][w],
                                    min_production_temperature_index=low[w],max_production_temperature_index=high[w],
                                    production_distinct_temperatures=int((occ[w]>0).sum()),
                                    normalized_production_occupancy_entropy=entropy[w]))
            for start in np.arange(0,5000000,500000):
                keep = (sweeps>=start)&(sweeps<=start+500000)
                xx=x[keep]
                ww=events if start==0 else [e for e in events if e[0]>start]
                ww=[e for e in ww if e[0]<=start+500000]
                _, wc=crossing_counts(xx)
                windows.append(dict(g_d=gd,L=L,start_sweep=int(start),end_sweep=int(start+500000),
                                    sampled_cold_to_hot=n_way(ww,"cold_to_hot"),
                                    sampled_hot_to_cold=n_way(ww,"hot_to_cold"),
                                    median_walker_index_span=float(np.median(np.ptp(xx,axis=0))),
                                    min_distinct_crossing_walkers=int(wc.min()),
                                    min_turnover_T=float((T[np.argmin(wc)]+T[np.argmin(wc)+1])/2)))
            runs[gd,L] = dict(T=T,x=x,sweeps=sweeps,f=f,acc=acc,occ=occ,
                             thermalization=d["thermalization_sweeps"],summary=summary)
            print(f"g_D={gd:g} L={L}: {summary['exact_round_trips']} exact trips; "
                  f"sampled passages {summary['sampled_cold_to_hot']} cold-to-hot, "
                  f"{summary['sampled_hot_to_cold']} hot-to-cold", flush=True)
    write_csv("pt_summary.csv",summaries)
    write_csv("pt_edges.csv",edges)
    write_csv("pt_walkers.csv",walkers)
    write_csv("pt_endpoint_events.csv",events_out)
    write_csv("pt_time_windows.csv",windows)
    fig, axs = plt.subplots(2,4,figsize=(15,7),sharex="col")
    for col,gd in enumerate(GDS):
        for color,L in zip(COLORS,SIZES):
            r=runs[gd,L]; T=r["T"]
            axs[0,col].plot((T[:-1]+T[1:])/2,r["acc"],color=color,label=f"L={L}")
            axs[1,col].plot(T,r["f"],color=color)
        axs[0,col].set(title=f"$g_D={gd:g}$",ylim=(0,1))
        axs[1,col].set(xlabel="Temperature",ylim=(-.02,1.02))
        axs[0,col].grid(alpha=.2);axs[1,col].grid(alpha=.2)
    axs[0,0].set_ylabel("Cumulative adjacent swap acceptance")
    axs[1,0].set_ylabel("Production cold last-endpoint fraction")
    axs[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Similar average acceptance can hide very different endpoint transport\nFlow uses sampled endpoint labels; counters include thermalization")
    fig.tight_layout();fig.savefig(OUT/"pt_acceptance_flow.png",dpi=170);plt.close(fig)
    fig, axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,gd in zip(axs.flat,GDS):
        r=runs[gd,160]
        mesh=ax.imshow(r["T"][r["x"]].T,aspect="auto",origin="lower",cmap="coolwarm",
                       extent=(0,5,0,160),vmin=.33,vmax=.47,rasterized=True)
        ax.axvline(2,color="black",ls="--",lw=.8)
        ax.set(title=f"$g_D={gd:g}$, L=160; exact trips={r['summary']['exact_round_trips']}",
               xlabel="Million local sweeps",ylabel="Physical walker ID")
    fig.suptitle("Physical walkers retain their identity; dashed line begins production\n500-sweep snapshots can miss brief excursions")
    fig.subplots_adjust(top=.9,right=.86,hspace=.3,wspace=.2)
    fig.colorbar(mesh,cax=fig.add_axes([.9,.17,.02,.64]),label="Assigned temperature")
    fig.savefig(OUT/"pt_walker_histories_L160.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9),sharex=True)
    for ax,L in zip(axs.flat,SIZES):
        r=runs[.1,L]
        occ=r["occ"] / r["occ"].sum(axis=1,keepdims=True)
        # The ladder is nonuniform, so use a pcolormesh in actual temperatures.
        tedge=np.r_[r["T"][0],(r["T"][:-1]+r["T"][1:])/2,r["T"][-1]]
        mesh=ax.pcolormesh(tedge,np.arange(161),np.log10(np.maximum(occ,1e-5)),
                           cmap="viridis",vmin=-4,vmax=-1,shading="flat",rasterized=True)
        ax.set(title=f"$g_D=0.1$, L={L}",xlabel="Temperature",ylabel="Physical walker ID")
    fig.suptitle("Strong SOC: production walkers explore restricted temperature sectors\nOccupancy zeros are snapshot-level absence, not proof of an uncrossed edge")
    fig.subplots_adjust(top=.9,right=.86,hspace=.3,wspace=.2)
    fig.colorbar(mesh,cax=fig.add_axes([.9,.17,.02,.64]),label=r"$\log_{10}$ sampled production occupation")
    fig.savefig(OUT/"pt_strong_occupation.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    main()
