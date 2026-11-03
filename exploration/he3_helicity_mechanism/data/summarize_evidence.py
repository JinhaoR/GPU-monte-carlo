"""Small arithmetic checks and a figure from this investigation's saved CSVs."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent


def read(name):
    rows=[]
    for row in csv.DictReader((OUT/name).open()):
        for k,v in row.items():
            try: row[k]=float(v)
            except ValueError: pass
        rows.append(row)
    return rows


def main():
    curves=read('curves.csv'); windows=read('transition_windows.csv')
    raw=read('current_statistics.csv'); mix=read('total_variance.csv')
    checks=dict(max_Y_identity_error=max(abs(r['Y']-r['D']+r['S']) for r in curves),
                max_share_sum_error=max(abs(r['curvature_loss_share']+r['screening_loss_share']-1) for r in windows),
                max_product_difference_error=max(abs(r['density_product_loss']+r['ratio_product_loss']+r['D_change']) for r in windows),
                max_raw_cache_mean_error=max(abs(r['raw_minus_cache']) for r in read('raw_cache_checks.csv')),
                max_total_variance_residual=max(abs(r['decomposition_residual']) for r in mix),
                max_mean_current_correction=max(r['M'] for r in curves),
                max_mean_current_correction_over_Y_se=max(r['M']/r['Y_se'] for r in curves),
                largest_conditional_mean_between_share=max(r['between_share'] for r in mix),
                current_kurtosis_range=[min(r['excess_kurtosis'] for r in raw),max(r['excess_kurtosis'] for r in raw)],
                top_one_percent_variance_share_range=[min(r['tail_variance_share_01'] for r in raw),max(r['tail_variance_share_01'] for r in raw)],
                largest_one_percent_tail_block_fraction=max(r['tail01_max_block_share'] for r in raw),
                curve_points=len(curves),raw_directional_points=len(raw),conditional_partitions=len(mix))
    assert checks['max_Y_identity_error']<1e-13
    assert checks['max_share_sum_error']<1e-12
    assert checks['max_product_difference_error']<1e-12
    assert checks['max_raw_cache_mean_error']<2e-10
    assert checks['max_total_variance_residual']<1e-13
    (OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    rows=read('conditional_moments.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for col,gd in enumerate((0,.001,.01)):
        center=71 if gd<.01 else 76
        for row,grouping in enumerate(('abs_chi','density')):
            ax=axes[row,col]
            selected=[r for r in rows if r['g_d']==gd and r['L']==160 and r['index']==center and r['axis']=='x' and r['grouping']==grouping]
            for name,label in [('D','Conditional curvature'),('variance','Conditional current variance'),('moment_difference','Difference of conditional moments')]:
                ax.errorbar([r['group']+1 for r in selected],[r[name] for r in selected],
                            yerr=[r[name+'_se'] for r in selected],marker='o',label=label)
            ax.set_title(f'g_D={gd:g}; T={selected[0]["T"]:.5f}')
            ax.set_xlabel(('Absolute chirality' if grouping=='abs_chi' else 'Density')+' quartile')
            ax.set_xticks([1,2,3,4]);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('x direction, normalized by N and T')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'conditional_moments.png',dpi=180);plt.close(fig)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
