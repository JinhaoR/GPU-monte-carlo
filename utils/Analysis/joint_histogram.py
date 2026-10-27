import numpy as np
import matplotlib.pyplot as plt

def joint_histogram(e, m, bins=40, ranges=None):

    e = np.asarray(e, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    if e.ndim != 1 or m.shape != e.shape:
        raise ValueError("e and m must be paired 1D arrays")

    counts, e_edges, m_edges = np.histogram2d(
        e, m,
        bins=bins,
        range=ranges
    )

    de = np.diff(e_edges)
    dm = np.diff(m_edges)

    bin_areas = de[:, None] * dm[None, :]

    P = counts / (counts.sum() * bin_areas)

    F_over_T = np.full_like(P, np.nan)

    occupied = P > 0
    F_over_T[occupied] = -np.log(P[occupied])

    F_over_T -= np.nanmin(F_over_T)

    return {
        "counts": counts,
        "density": P,
        "F_over_T": F_over_T,
        "e_edges": e_edges,
        "m_edges": m_edges,
    }

def joint_histogram_from_run(
    run,
    temperature,
    x_name,
    y_name,
    bins=40,
    x_transform=None,
    y_transform=None,
):
    T = np.asarray(run["metadata"]["temperatures"])
    index = np.argmin(np.abs(T - temperature))
    actual_T = float(T[index])

    if temperature < T.min() or temperature > T.max():
        print(
            f"Warning: T={temperature} outside simulated range. "
            f"Using T={actual_T:.5f}"
        )

    samples = run["samples"]

    x = np.asarray(samples[x_name])
    y = np.asarray(samples[y_name])

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            "Observables must have matching "
            "(n_samples, n_temperatures) shapes"
        )

    # Select simultaneous measurements at one temperature
    x = x[:, index]
    y = y[:, index]

    # Optional transformations (normalization, absolute value...)
    if x_transform is not None:
        x = x_transform(x)

    if y_transform is not None:
        y = y_transform(y)

    hist = joint_histogram(x, y, bins=bins)

    return actual_T, hist

def plot_joint_histogram(
    hist,
    x_label="Observable A",
    y_label="Observable B",
    title="Joint distribution",
):
    figures = {}

    # 2D heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    mesh = ax.pcolormesh(
        hist["e_edges"],
        hist["m_edges"],
        hist["density"].T,
        shading="auto",
        cmap="viridis",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="P(A,B)")

    figures["joint_heatmap"] = fig

    # 3D surface
    x_edges = hist["e_edges"]
    y_edges = hist["m_edges"]

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    X, Y = np.meshgrid(
        x_centers, y_centers, indexing="ij"
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X, Y, hist["density"],
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("P(A,B)")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)

    fig.colorbar(surface, ax=ax, shrink=0.6)

    figures["joint_surface"] = fig

    return figures