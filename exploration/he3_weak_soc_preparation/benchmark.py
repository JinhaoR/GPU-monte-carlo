"""Static-field measurement timings for the enabled sizes; no MC or PT updates."""
from pathlib import Path
from time import perf_counter
import json
import sys
import numpy as np
from numba import cuda
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import run as he3
from models.he3.model import He3Model
from models.he3.measure_spatial import SpatialMeter
OUT=Path(__file__).resolve().parent/'results'


def milliseconds(fn):
    fn();cuda.synchronize()
    times=[]
    for _ in range(3):
        start=perf_counter()
        for _ in range(8):fn()
        cuda.synchronize();times.append(1000*(perf_counter()-start)/8)
    return float(np.median(times))


def main():
    if not cuda.is_available():
        print('CUDA unavailable: benchmark unverified');return
    R=len(he3.make_temperature_ladder(**he3.temperature_ladder))
    rows=[]
    for L in he3.sizes:
        needed=R*L*L*48+10*1024**2
        if needed>cuda.current_context().get_memory_info().free/4:
            raise MemoryError('static benchmark allocation exceeds a quarter of free VRAM')
        rng=np.random.default_rng(L)
        a=(.2*rng.normal(size=(R,L,L,3,2))+.2j*rng.normal(size=(R,L,L,3,2))).astype(np.complex64)
        state=cuda.to_device(a);mapping=cuda.to_device(np.arange(R,dtype=np.int32))
        model=He3Model(pressure=20,g_d=.01);meter=SpatialMeter(L,R)
        energy=cuda.device_array(R,np.float64);model.measure_energy(state,energy)
        old=milliseconds(lambda:model.measure(state,mapping,energy))
        modes=milliseconds(lambda:meter.measure(state,mapping))
        structure=milliseconds(lambda:meter.structure(state,model,mapping))
        model.realspace_every=1
        model.realspace_indices=np.linspace(0,R-1,5,dtype=int)
        def profile_event():
            model.measure(state,mapping,energy)
            model.realspace_samples.clear()
        # Full event includes scalar measurement; subtract its separately timed cost.
        profile_total=milliseconds(profile_event)
        profile_extra=profile_total-old
        ns=he3.measurement_sweeps//he3.sample_interval
        extra=he3.measurement_sweeps//he3.spatial_interval
        raw=ns*R*10*8+extra*R*(16*5*16+68*8+4+8)
        profile_bytes=(he3.measurement_sweeps//he3.realspace_interval)*5*(7*2*(L//2+1)*16+7*16+4+8)
        rows.append(dict(L=L,R=R,old_scalar_ms=old,low_modes_ms=modes,structure_energy_ms=structure,
            extra_microseconds_per_measurement_sweep=1000*(modes+structure)/he3.spatial_interval,
            meter_workspace_bytes=meter.allocated_bytes,raw_arrays_MB=raw/1e6,
            realspace_probes=5,realspace_full_event_ms=profile_total,realspace_added_ms=profile_extra,
            realspace_extra_microseconds_per_sweep=1000*profile_extra/he3.realspace_interval,
            realspace_raw_arrays_MB=profile_bytes/1e6))
        print(rows[-1],flush=True)
        del state,meter,energy,a
    (OUT/'prelaunch_benchmark.json').write_text(json.dumps(dict(
        device=str(cuda.get_current_device().name),static_only=True,repetitions_per_component=24,rows=rows),indent=2))


if __name__=='__main__':main()
