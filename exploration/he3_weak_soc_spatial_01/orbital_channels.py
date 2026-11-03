"""Inspect a fixed diagonal orbital basis using existing Fourier samples.

A_+=(A_x+A_y)/sqrt(2), A_-=(A_x-A_y)/sqrt(2).
P=2 A_+ dot A_- exactly; B_++ and B_-- are their charge-two squares.
The basis follows the observed b0/bxy coherence and the stencil's x/y symmetry.
No fitted eigenvectors and no new Monte Carlo.
"""
import ast
import math
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from models.he3.spatial import local_fields,numpy_amplitudes,unpack_real_modes,correlation_ratio
from reduce import csv_write


def validate():
    # Independently exercise the actual local GPU measurement polynomial on CPU.
    source=ROOT/'models/he3/measure_spatial.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_local_values')
    fn.decorator_list=[]
    ns=dict(np=np,math=math,float64=np.float64,cuda=SimpleNamespace(local=SimpleNamespace(array=lambda shape,dtype:np.empty(shape,dtype))))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    rng=np.random.default_rng(2061);a=(rng.normal(size=(6,6,3,2))+1j*rng.normal(size=(6,6,3,2))).astype(np.complex64)
    raw=np.empty((6,6,26))
    for x in range(6):
        for y in range(6):ns['_local_values'](a[None],0,x,y,raw[x,y])
    ft=np.fft.fftn(raw,axes=(0,1));modes=np.stack((ft[0,0],ft[1,0],ft[0,1]),axis=-1)
    unpacked=unpack_real_modes(np.stack((modes.real,modes.imag),axis=-1))
    expected=numpy_amplitudes(a)
    err=float(np.max(abs(unpacked-expected)))
    f=local_fields(a);plus=(a[...,0]+a[...,1])/np.sqrt(2);minus=(a[...,0]-a[...,1])/np.sqrt(2)
    # Promote before multiplying, matching production observable precision.
    aa=a.astype(complex);plus=(aa[...,0]+aa[...,1])/np.sqrt(2);minus=(aa[...,0]-aa[...,1])/np.sqrt(2)
    perr=float(np.max(abs(f[...,1]-2*np.sum(plus*minus,axis=-1))))
    bpp=(f[...,7]+f[...,8]+2*f[...,9])/2
    berr=float(np.max(abs(bpp-np.sum(plus*plus,axis=-1))))
    assert max(err,perr,berr)<1e-11
    return dict(local_polynomial_and_fourier_max_error=err,P_rotated_identity_error=perr,Bpp_rotated_identity_error=berr)


def estimate(m,L):
    out={}
    for name,p in [('A_plus',m['ap'][...,0,:]),('A_minus',m['ap'][...,1,:]),
        ('A_plus_z',m['aspin'][...,2,0,:]),('A_minus_z',m['aspin'][...,2,1,:]),
        ('A_plus_transverse',m['aspin'][...,:2,0,:].sum(-2)),
        ('B_plus',(m['bp'][...,0,:]+m['bp'][...,2,:]+2*m['bc'][...,1,:].real)/2),
        ('B_minus',(m['bp'][...,0,:]+m['bp'][...,2,:]-2*m['bc'][...,1,:].real)/2)]:
        out[name+'_S']=p/(L*L);out[name+'_xi']=correlation_ratio(p[...,0],p[...,1:].mean(-1),L)[1]
    out['A_plus_fraction']=m['ap'][...,0,0]/m['ap'][...,0:2,0].sum(-1)
    out['B_plus_fraction']=out['B_plus_S'][...,0]/(out['B_plus_S'][...,0]+out['B_minus_S'][...,0])
    return out


def main():
    validation=validate();rows=[]
    for p in sorted((ROOT/'data/weak_soc_spatial_simple_01').rglob('L=*.npz')):
        L=int(p.stem[2:]);gd=float(p.parent.name.split('gd=')[1].split('_')[0]);tag=f'gd={gd:g}_L={L}'
        if L not in (64,96,128,160):continue
        print('Orbital channels',tag,flush=True)
        with np.load(p) as z:a=z[f'L{L}_spatial_amplitudes'][...,10:,:].reshape(3000,200,3,2,5)
        aspin=np.stack((abs((a[...,0,:]+a[...,1,:])/np.sqrt(2))**2,
                        abs((a[...,0,:]-a[...,1,:])/np.sqrt(2))**2),axis=-2)
        ap=aspin.sum(-3)
        blocks={'ap':ap.reshape(60,50,200,2,5).mean(1),
                'aspin':aspin.reshape(60,50,200,3,2,5).mean(1)}
        del ap,aspin,a
        with np.load(OUT/(tag+'.npz')) as z:
            T=z['T'];blocks['bp']=z['b_power'];blocks['bc']=z['b_cross']
        grouped={k:x.reshape((30,2)+x.shape[1:]).mean(1) for k,x in blocks.items()}
        m={k:x.mean(0) for k,x in grouped.items()};v=estimate(m,L)
        jk=estimate({k:(30*m[k]-x)/29 for k,x in grouped.items()},L)
        se={k:np.sqrt(29*np.var(x,axis=0)) for k,x in jk.items()}
        np.savez_compressed(OUT/(tag+'_orbital.npz'),T=T,**v,**{k+'_se':x for k,x in se.items()},
                            **{'jk_'+k:x for k,x in jk.items()},**{'block_'+k:x for k,x in blocks.items()})
        for j,t in enumerate(T):
            row=dict(g_d=gd,L=L,T=t)
            for key,value in v.items():
                row[key if value.ndim==1 else key+'0']=value[j] if value.ndim==1 else value[j,0]
                row[(key if value.ndim==1 else key+'0')+'_se']=se[key][j] if value.ndim==1 else se[key][j,0]
            rows.append(row)
    csv_write(OUT/'orbital_channels.csv',rows)
    (OUT/'orbital_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')


if __name__=='__main__':main()
