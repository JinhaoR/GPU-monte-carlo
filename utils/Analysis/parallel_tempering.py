"""Analysis of saved parallel-tempering counters."""

import matplotlib.pyplot as plt
import numpy as np


class PTAnalysis:
    def __init__(self, runs):
        self.results = {}
        self._histories = {}
        self._thermalization = {}
        for L, run in sorted(runs.items()):
            if "parallel_tempering" not in run["metadata"]:
                continue
            pt = run["metadata"]["parallel_tempering"]
            attempted = np.asarray(pt["attempts"])
            accepted = np.asarray(pt["accepted"])
            trips = np.asarray(pt["walker_round_trips"])
            times = np.asarray(pt["walker_round_trip_total_sweeps"])
            T = np.asarray(run["metadata"]["temperatures"], dtype=float)
            history = pt.get("walker_history")
            if history is not None:
                ids = np.asarray(history["walker_ids"], dtype=np.int64)
                sweeps = np.asarray(history["sweeps"], dtype=np.int64)
                positions = np.asarray(history["temperature_indices"])
                if (ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids)
                        or np.any(ids < 0) or np.any(ids >= len(T))
                        or sweeps.ndim != 1 or len(sweeps) == 0
                        or np.any(np.diff(sweeps) < 0)
                        or positions.shape != (len(sweeps), len(ids))
                        or not np.issubdtype(positions.dtype, np.integer)
                        or np.any(positions < 0) or np.any(positions >= len(T))):
                    raise ValueError(f"L={L}: invalid walker history")
                self._histories[L] = (ids, sweeps, positions)
            self._thermalization[L] = run["metadata"].get("thermalization_sweeps")
            self.results[L] = {
                "temperatures": T.tolist(),
                "attempts": attempted.tolist(),
                "accepted": accepted.tolist(),
                "acceptance_rates": [float(a / n) if n else None for a, n in zip(accepted, attempted)],
                "aggregate_acceptance": float(accepted.sum() / attempted.sum()) if attempted.sum() else None,
                "walker_round_trips": trips.tolist(),
                "walker_mean_round_trip_sweeps": [float(t / n) if n else None for t, n in zip(times, trips)],
                "mean_round_trip_sweeps": float(times.sum() / trips.sum()) if trips.sum() else None,
                "diagnostics_scope": pt["diagnostics_scope"],
                "history_available": history is not None,
            }

    def summary(self):
        return self.results

    def format_summary(self):
        lines = ["Parallel tempering diagnostics (including thermalization)",
                 "Trip means exclude unfinished trips; zero trips does not imply zero one-way crossings."]
        for L, result in self.results.items():
            lines.append(f"L={L}: {sum(result['accepted'])}/{sum(result['attempts'])} accepted, "
                         f"{sum(result['walker_round_trips'])} trips, "
                         f"mean {result['mean_round_trip_sweeps']} sweeps/trip")
            lines.append("  walker paths recorded" if result["history_available"]
                         else "  walker paths were not recorded; final mappings cannot reconstruct trajectories")
        return "\n".join(lines)

    def plot(self, show=True):
        if not self.results:
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for L, result in self.results.items():
            T = np.asarray(result["temperatures"])
            ax1.plot((T[:-1] + T[1:]) / 2, result["acceptance_rates"], ".-", label=f"L={L}")
            ax2.plot(result["walker_round_trips"], ".-", label=f"L={L}")
        ax1.set(xlabel="Temperature midpoint", ylabel="Swap acceptance", ylim=(0, 1.05))
        ax2.set(xlabel="Walker", ylabel="Completed round trips")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_walkers(self, L=None, show=True):
        """Plot every recorded walker's sampled temperature history as a heatmap.

        Downsampled paths can miss short excursions. Exact trip counters remain
        authoritative. Runs without saved histories produce no substitute plot.
        """
        sizes = sorted(self._histories) if L is None else [L] if L in self._histories else []
        if not sizes:
            return None
        fig, axes = plt.subplots(len(sizes), 1, figsize=(11, 3.5 * len(sizes)), squeeze=False)
        for size, ax in zip(sizes, axes[:, 0]):
            T = np.asarray(self.results[size]["temperatures"])
            ids, sweeps, positions = self._histories[size]
            order = np.argsort(ids)
            ids, positions = ids[order], positions[:, order]
            # Manual exchanges may share a sweep coordinate. Display the last
            # recorded assignment at each time, without inventing elapsed time.
            keep = np.r_[np.diff(sweeps) > 0, True]
            times = sweeps[keep].astype(float)
            temperatures = T[positions[keep]].T
            if len(times) == 1:
                time_edges = [times[0] - .5, times[0] + .5]
            else:
                time_edges = np.r_[times[0], (times[:-1] + times[1:]) / 2, times[-1]]
            mesh = ax.pcolormesh(time_edges, np.arange(len(ids) + 1) - .5,
                                 temperatures, cmap="coolwarm", vmin=T[0], vmax=T[-1],
                                 shading="flat", rasterized=True)
            ticks = np.unique(np.linspace(0, len(ids) - 1, min(7, len(ids))).astype(int))
            ax.set_yticks(ticks, labels=ids[ticks])
            coverage = "all walkers" if len(ids) == len(T) else f"{len(ids)} recorded walkers"
            ax.set(title=f"L={size}: temperature history of {coverage} (sampled)",
                   xlabel="Local sweeps since PT construction", ylabel="Walker ID")
            thermalization = self._thermalization[size]
            if thermalization is not None and times[0] <= thermalization <= times[-1]:
                ax.axvline(thermalization, color="black", linestyle="--", linewidth=.8)
            fig.colorbar(mesh, ax=ax, label="Assigned temperature")
        fig.tight_layout()
        if show:
            plt.show()
        return fig
