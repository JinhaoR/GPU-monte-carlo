"""Bounded read-only check for resolved energy coexistence at the heat peaks.

Mode counts are descriptive smoothing checks, not a first-order transition test.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from findings import OUT,ROOT,load
from reduce import csv_write


def main():
    rows=[];hist=[]
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row,g in enumerate((0,.01)):
        for col,L in enumerate((64,96,128,160)):
            summary=load(g,L);j=np.argmax(summary['specific_heat']);T=summary['T'][j]
            path=ROOT/f'data/weak_soc_spatial_simple_01/P=20_gd={g:g}_gaussian/L={L}.npz'
            with np.load(path) as z:x=z[f'L{L}_energy'][:,j]/L**2
            mu=x.mean();sd=x.std();lo=x.min();hi=x.max()
            for bins in (100,150,200):
                counts,edges=np.histogram(x,bins=bins,range=(lo,hi));centers=(edges[1:]+edges[:-1])/2
                for width in (.08,.12,.18):
                    smooth=gaussian_filter1d(counts.astype(float),width*sd/(edges[1]-edges[0]))
                    peaks,_=find_peaks(smooth,prominence=.1*smooth.max(),distance=max(1,int(.4*sd/(edges[1]-edges[0]))))
                    rows.append(dict(g_d=g,L=L,T=T,bins=bins,smoothing_in_sd=width,prominence_fraction=.1,
                        prominent_modes=len(peaks),peak_energies=';'.join(map(str,centers[peaks]))))
                if bins==150:
                    ax=axs[row,col];ax.plot(centers,counts/counts.sum()/(edges[1]-edges[0]),label='full')
                    for name,part in [('first half',x[:len(x)//2]),('second half',x[len(x)//2:])]:
                        hh,_=np.histogram(part,edges);ax.plot(centers,hh/hh.sum()/(edges[1]-edges[0]),alpha=.6,label=name)
                    for i,c in enumerate(centers):hist.append(dict(g_d=g,L=L,T=T,energy_per_site=c,probability_density=counts[i]/counts.sum()/(edges[1]-edges[0])))
                    ax.set(title=f'g_D={g:g}, L={L}\nT={T:.6f}',xlabel='energy / site',ylabel='density');ax.legend(fontsize=7)
    fig.savefig(OUT/'figures/energy_histograms.png',dpi=180,bbox_inches='tight');plt.close(fig)
    csv_write(OUT/'energy_modes.csv',rows);csv_write(OUT/'energy_histograms.csv',hist)


if __name__=='__main__':main()
