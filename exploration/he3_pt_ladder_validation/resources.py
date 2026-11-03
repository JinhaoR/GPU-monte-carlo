"""Conditional Kraken budgets; pilot speedups are not extrapolated to large L."""
import csv,json
import numpy as np
from ladder_design import OUT,OLD,write

def main():
    pred=list(csv.DictReader((OUT/'offline_predictions.csv').open()))
    previous=list(csv.DictReader((OLD/'results/transport_summary.csv').open()))
    costs=list(csv.DictReader((OLD/'results/resource_scenarios.csv').open()))
    inventory=list(csv.DictReader((OLD/'results/inventory.csv').open()))
    rows=[]
    for L,burn,production in ((320,2_000_000,4_000_000),(384,4_000_000,8_000_000)):
        for g in (0,.01):
            baseline=next(r for r in previous if r['campaign']=='new' and float(r['g_d'])==g and int(r['L'])==256)
            for name,R in [('A',200),('B',200),('C',160)]:
                d=[r for r in pred if int(r['L'])==L and float(r['g_d'])==g and r['ladder']==name]
                basecost=next(r for r in costs if int(r['L'])==L and int(r['R'])==R)
                lo=float(basecost['seconds_per_sweep_area']);hi=float(basecost['seconds_per_sweep_recent'])
                f=[float(baseline['passage_flux_time'])*float(r['null_RT'])/float(baseline['null_alternating_RT']) for r in d]
                low=min(f);high=max(f)*(L/256)
                scalar=10*8*R*production/50;spatial=(16*5*16+68*8+4+8)*R*production/1000
                profiles=(7*2*(L//2+1)*16+7*16+4+8)*5*production/10000
                history5=4*R*(burn+production)/5;history100=history5/20
                rows.append(dict(L=L,g_d=g,ladder=name,R=R,T_min=float(d[0]['T_min']),T_max=float(d[0]['T_max']),
                    min_acceptance_stress=min(float(r['min_acceptance']) for r in d),min_acceptance_fixed=max(float(r['min_acceptance']) for r in d),
                    flux_scenario_low=low,flux_scenario_high=high,diagnostic_burn=burn,diagnostic_production=production,
                    hours_area=(burn+production)*lo/3600,hours_recent=(burn+production)*hi/3600,
                    GPU_core_GiB=88*R*L*L/2**30,GPU_history_5_GiB=history5/2**30,GPU_history_100_GiB=history100/2**30,
                    measurement_payload_GiB=(scalar+spatial+profiles)/2**30,
                    host_array_peak_history5_GiB=(2*(scalar+spatial+profiles)+4*history5)/2**30,
                    provisional_total_5flux_low=5*low,provisional_total_5flux_high=5*high,
                    provisional_hours_5flux_low=5*low*hi/3600,provisional_hours_5flux_high=5*high*hi/3600))
    write('larger_size_scenarios.csv',rows)
    experiments=[]
    # L128 rows are alternatives: compare A+B or A+C after the L96 screen.
    for L,ladders,burn,production in [(96,['A','B','C'],500000,1000000),(128,['A','B','C'],2000000,3000000)]:
        ref=next(r for r in inventory if r['campaign']=='new' and int(r['L'])==L and float(r['g_d'])==0)
        seconds=float(ref['gpu_seconds'])/6e6
        for name in ladders:
            R=160 if name=='C' else 200
            experiments.append(dict(L=L,ladder=name,R=R,cases=4,burn=burn,production=production,
                seconds_per_case=(burn+production)*seconds*R/200,
                combined_hours=4*(burn+production)*seconds*R/200/3600))
    write('next_experiment_costs.csv',experiments)
    print('L320/L384 planning scenarios and costed follow-up experiments saved.')

if __name__=='__main__':main()
