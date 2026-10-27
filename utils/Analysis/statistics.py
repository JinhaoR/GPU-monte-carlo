"""Time blocking, jackknife, and joint block bootstrap."""

import numpy as np


def block_means(data, block_size):
    n = len(data) // block_size
    return np.asarray(data[:n * block_size]).reshape(
        n, block_size, *data.shape[1:]
    ).mean(axis=1)


def blocked_mean_error(data, block_size):
    blocks = block_means(data, block_size)
    return (
        blocks.mean(axis=0),
        blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    )


def jackknife_from_blocks(blocks, estimator):
    """Recompute a possibly nonlinear estimator after omitting each time block."""
    B = len(next(iter(blocks.values())))
    totals = {name: x.sum(axis=0) for name, x in blocks.items()}
    estimate = estimator({name: total / B for name, total in totals.items()})
    deleted = np.array([
        estimator({name: (totals[name] - x[b]) / (B - 1) for name, x in blocks.items()})
        for b in range(B)
    ])
    error = np.sqrt((B - 1) * np.var(deleted, axis=0, ddof=0))
    return estimate, error


def blocked_jackknife(data, estimator, block_size):
    return jackknife_from_blocks(
        {name: block_means(x, block_size) for name, x in data.items()}, estimator
    )


def bootstrap_block_means(blocks, n_resamples=400, rng=None):
    """Use the same resampled block indices for every channel and temperature."""
    B = len(next(iter(blocks.values())))
    counts = np.random.default_rng(rng).multinomial(
        B, [1 / B] * B, size=n_resamples
    )
    return {
        name: np.tensordot(counts / B, x, axes=(1, 0))
        for name, x in blocks.items()
    }

def bootstrap_error(estimates):
    values = np.asarray(estimates)
    return float(values.std(ddof=1)) if len(values) > 1 and np.all(np.isfinite(values)) else None
