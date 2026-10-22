"""Analysis of saved parallel-tempering counters."""

import matplotlib.pyplot as plt
import numpy as np


class PTAnalysis:
    def __init__(self, runs):
        self.results = {}
        for L, run in sorted(runs.items()):
            if "parallel_tempering" not in run["metadata"]:
                continue
            pt = run["metadata"]["parallel_tempering"]
            attempted = np.asarray(pt["attempts"])
            accepted = np.asarray(pt["accepted"])
            trips = np.asarray(pt["walker_round_trips"])
            times = np.asarray(pt["walker_round_trip_total_sweeps"])
            self.results[L] = {
                "temperatures": np.asarray(run["metadata"]["temperatures"]).tolist(),
                "attempts": attempted.tolist(),
                "accepted": accepted.tolist(),
                "acceptance_rates": [float(a / n) if n else None for a, n in zip(accepted, attempted)],
                "aggregate_acceptance": float(accepted.sum() / attempted.sum()) if attempted.sum() else None,
                "walker_round_trips": trips.tolist(),
                "walker_mean_round_trip_sweeps": [float(t / n) if n else None for t, n in zip(times, trips)],
                "mean_round_trip_sweeps": float(times.sum() / trips.sum()) if trips.sum() else None,
                "diagnostics_scope": pt["diagnostics_scope"],
            }

    def summary(self):
        return self.results

    def format_summary(self):
        lines = ["Parallel tempering diagnostics (includes thermalization)"]
        for L, result in self.results.items():
            lines.append(f"L={L}: {sum(result['accepted'])}/{sum(result['attempts'])} accepted, "
                         f"{sum(result['walker_round_trips'])} trips, "
                         f"mean {result['mean_round_trip_sweeps']} sweeps/trip")
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
