"""Pilot ladder from existing L64 heat-capacity/energy-overlap information."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results'
TARGETS=np.array([.37,.375,.38,.3815,.382,.3825,.385,.395,.405,.415])


def make_ladder():
    with np.load(ROOT/'exploration/he3_soc/results/blocks/gd=0_L=64.npz') as z:
        T=z['T'];cv=(z['e2'].mean(0)-z['e'].mean(0)**2)*64**2/T**2
    with np.load(ROOT/'exploration/he3_transition_study/results/curves_gd=0_L=64.npz') as z:
        fineT=z['T'];finecv=z['point'][list(z['keys']).index('specific_heat')]
    def c(t):return np.interp(t,fineT,finecv) if fineT[0]<=t<=fineT[-1] else np.interp(t,T,cv)
    ts=[.37]
    while ts[-1]<.415-1e-10:
        t=ts[-1];step=.65*t/(64*np.sqrt(c(t)))
        for _ in range(3):mid=t+step/2;step=.65*mid/(64*np.sqrt(c(mid)))
        ts.append(min(.415,t+step))
    # Exact controls take priority if an adaptive point lies almost on one.
    ts=[t for t in ts if np.min(abs(TARGETS-t))>5e-5]
    temperatures=np.unique(np.round(np.r_[ts,TARGETS],9))
    mid=(temperatures[:-1]+temperatures[1:])/2
    metric=(1/temperatures[:-1]-1/temperatures[1:])*np.array([t*64*np.sqrt(c(t)) for t in mid])
    record=dict(temperatures=temperatures.tolist(),R=len(temperatures),L=64,g_d=0,
        design='Delta beta * sigma_H about 0.65, with sigma_H=T L sqrt(Cv), existing L64 data',
        control_temperatures=TARGETS.tolist(),predicted_edge_delta_beta_sigma_H=metric.tolist(),
        predicted_max_delta_beta_sigma_H=float(metric.max()),
        source_files=['exploration/he3_soc/results/blocks/gd=0_L=64.npz',
                      'exploration/he3_transition_study/results/curves_gd=0_L=64.npz'])
    return temperatures,record


if __name__=='__main__':
    T,record=make_ladder();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'ladder.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('R',len(T),'range',T[[0,-1]],'max predicted Delta beta sigma',record['predicted_max_delta_beta_sigma_H'])
