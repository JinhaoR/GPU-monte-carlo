"""Write a proposed PT ladder; this file cannot launch Monte Carlo jobs.

Spacing uses the existing Cv curve and a conservative logarithmic increment
when L>160. It is a planning estimate, to be recalibrated from a future pilot.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
from prepare import ROOT,OUT
from scaling import read
from study import KEYS,csvwrite


def main():
    a=read(0,160)
    with np.load(ROOT/'exploration/he3_soc/results/blocks/gd=0_L=160.npz') as z:
        T=z['T'];cv=(z['e2'].mean(0)-z['e'].mean(0)**2)*160**2/T**2
    def cvguess(t,L):
        v=np.interp(t,a['T'],a['point'][KEYS.index('specific_heat')]) if a['T'][0]<=t<=a['T'][-1] else np.interp(t,T,cv)
        return max(1,v+44.61*max(0,np.log(L/160)))
    ladders={};rows=[]
    for L in (64,128,160,192,256,384):
        tt=[.37]
        while tt[-1]<.4-1e-12:
            t=tt[-1];step=.7*t/(L*np.sqrt(cvguess(t,L)))
            # Cap the full decision window even for small lattices.
            if .3795<=t<=.384:step=min(step,.0001*min(1,256/L))
            tt.append(min(.4,t+step))
        tt=np.array(tt)
        # Check predicted edge separation after constructing the nonuniform grid.
        midpoint=(tt[:-1]+tt[1:])/2
        metric=(1/tt[:-1]-1/tt[1:])*np.array([t*L*np.sqrt(cvguess(t,L)) for t in midpoint])
        ladders[str(L)]=tt.tolist();rows.append(dict(L=L,temperatures=len(tt),Tmin=tt[0],Tmax=tt[-1],
            min_spacing=np.diff(tt).min(),max_spacing=np.diff(tt).max(),predicted_max_delta_beta_sigmaE=metric.max(),
            peak_spacing_rule=.7*.382/(L*np.sqrt(cvguess(.382,L)))))
    result=dict(status='PROPOSAL ONLY: no simulations launched',coupling=0,pressure=20,
        coefficient_reduced_temperature=1,sampler='unchanged full-Hamiltonian conditional Gaussian MH',
        exchange_interval_sweeps=5,global_scalar_interval_sweeps=50,
        kmin_fourier_interval_sweeps=250,spatial_correlation_interval_sweeps=5000,
        kmin_added_real_scalars=8,kmin_recommended_with_negative_complex_modes=12,
        optional_full_field_interval_sweeps=200000,optional_full_fields_cap_per_selected_temperature=20,
        checkpoint_interval_sweeps=100000,round_trip_history_interval_sweeps=100,
        dense_decision_window=[.3795,.384],spacing_criterion='delta_beta * sigma(E) about 0.7 or smaller; pilot-adjusted',
        cv_extrapolation='Cv160(T)+44.61*max(0,log(L/160)); planning assumption, not a scaling conclusion',
        first_gate_sizes=[64,128,160,256],seeds_by_size={'64':2,'128':4,'160':4,'256':4},
        initializations=['ordered + chirality','ordered - chirality','hot random','hot random, distinct seed'],
        provisional_burn_sweeps={'64':2000000,'128':2000000,'160':2000000,'256':4000000},
        provisional_production_sweeps={'64':4000000,'128':4000000,'160':4000000,'256':8000000},
        batch_sweeps=1000000,target_effective_samples_per_seed=1000,
        followup_sizes_if_needed=[192,384],weak_SOC_comparison={'g_D':.001,'sizes':[128,256],'independent_seeds':4,'temperature_shift_starting_guess':.00045},
        proposed_temperature_ladders=ladders)
    (OUT/'simulation_proposal.json').write_text(json.dumps(result,indent=2),encoding='utf-8');csvwrite('proposed_ladder_summary.csv',rows)
    print(rows,flush=True)


if __name__=='__main__':main()
