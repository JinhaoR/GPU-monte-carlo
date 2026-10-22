
import json
from pathlib import Path

import numpy as np

def _json_ready(value):
    """Convert NumPy metadata into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _json_ready(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]

    return value


def save_runs(path, runs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    manifest = {
        "version": 1,
        "runs": {},
    }

    for L, run in runs.items():
        samples = run["samples"]

        manifest["runs"][str(L)] = {
            "metadata": _json_ready(run["metadata"]),
            "sample_names": list(samples),
        }

        for name, values in samples.items():
            arrays[f"L{L}_{name}"] = np.asarray(values)

    arrays["manifest"] = np.array(json.dumps(manifest))

    np.savez_compressed(path, **arrays)


def load_runs(path):
    runs = {}

    with np.load(path, allow_pickle=False) as file:
        manifest = json.loads(file["manifest"].item())

        if manifest["version"] != 1:
            raise ValueError("Unsupported data format version")

        for L_str, info in manifest["runs"].items():
            L = int(L_str)

            samples = {
                name: file[f"L{L}_{name}"].copy()
                for name in info["sample_names"]
            }

            runs[L] = {
                "samples": samples,
                "metadata": info["metadata"],
            }

    return runs
