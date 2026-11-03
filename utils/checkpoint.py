"""Small, pickle-free NPZ state files; arrays stay binary, including metadata.

Checkpoints are written beside their destination and atomically replaced only
after the complete temporary archive has been read back successfully.
"""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def checkpoint_stack():
    import platform
    from importlib.metadata import version, PackageNotFoundError
    from numba import cuda, __version__ as numba_version
    from numba.cuda.cudadrv.driver import driver
    try:
        numba_cuda = version('numba-cuda')
    except PackageNotFoundError:
        numba_cuda = 'bundled'
    return dict(gpu=str(cuda.get_current_device().name), python=platform.python_version(),
                numba=numba_version, numba_cuda=numba_cuda, numpy=np.__version__,
                cuda_driver=list(driver.get_version()), cuda_runtime=list(cuda.runtime.get_version()))


def pack_tree(value, arrays):
    if isinstance(value, np.ndarray):
        name = f'array_{len(arrays)}'
        arrays[name] = value
        return {'__array__': name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): pack_tree(v, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack_tree(v, arrays) for v in value]
    return value


def unpack_tree(value, archive):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return archive[value['__array__']].copy()
        return {k: unpack_tree(v, archive) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_tree(v, archive) for v in value]
    return value


def read_checkpoint(path):
    with open(path, 'rb') as handle, np.load(handle, allow_pickle=False) as archive:
        manifest = json.loads(archive['manifest'].item())
        if manifest['checkpoint_version'] != 1:
            raise ValueError('Unsupported checkpoint version')
        return unpack_tree(manifest['state'], archive)


def write_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    manifest = dict(checkpoint_version=1, state=pack_tree(state, arrays))
    arrays['manifest'] = np.array(json.dumps(manifest))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)  # no expensive compression of random lattice fields
            handle.flush()
            os.fsync(handle.fileno())
        # Read one array at a time: CRC/truncation check without a second full state copy.
        with np.load(temporary, allow_pickle=False) as archive:
            for name in archive.files:
                if archive[name].shape != arrays[name].shape:
                    raise ValueError('Checkpoint write verification failed')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def check_array(value, target, name):
    if not isinstance(value, np.ndarray) or value.shape != target.shape or value.dtype != target.dtype:
        raise ValueError(f'Checkpoint shape/dtype mismatch: {name}')


def same_settings(a, b):
    """Compare JSON-normalized settings (including numpy scalars and tuples)."""
    from .data_io import _json_ready
    return _json_ready(a) == _json_ready(b)
