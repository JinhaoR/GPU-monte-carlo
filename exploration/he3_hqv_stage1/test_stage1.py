"""Deterministic unit tests of known full-matrix textures; no simulation jobs.

Run from project root: python -B exploration/he3_hqv_stage1/test_stage1.py
Uses only unittest, NumPy and the independent modules in this directory.
The runner saves explicit known-contour checks plus noise/threshold sensitivity.
"""
import csv
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from defects import local_geometry,analyze_loop,rectangle_loop,Thresholds,find_candidates,plaquette_map
import synthetic as s

L=96
CENTERS=((L/3-.5,L/2-.5),(2*L/3-.5,L/2-.5))


def around(center,radius=6):
    x,y=center
    return rectangle_loop(int(np.floor(x-radius)),int(np.ceil(x+radius)),
                          int(np.floor(y-radius)),int(np.ceil(y+radius)))


def raw_winding(values):
    return float(np.angle(np.roll(values,-1)*values.conj()).sum()/(2*np.pi))


def rotation(seed):
    q,_=np.linalg.qr(np.random.default_rng(seed).normal(size=(3,3)))
    q[:,0]*=np.linalg.det(q)
    return q


class StageOneTests(unittest.TestCase):
    def check_charge(self,a,c,expected,label=None):
        r=analyze_loop(a,c)
        self.assertTrue(r['accepted'],r)
        self.assertAlmostEqual(r['phase_charge'],expected,places=10)
        self.assertEqual(r['composite_winding'],int(2*expected))
        self.assertEqual(r['director_parity'],-1 if int(2*expected)%2 else 1)
        if label:self.assertEqual(r['label'],label)
        return r

    def test_uniform_and_smooth_have_no_vortices(self):
        for a in [s.uniform(L),s.smooth_texture(L)]:
            self.check_charge(a,around((47.5,47.5),12),0,'no_net_vortex')

    def test_integer_vortices_both_signs_and_chiralities(self):
        for q in [-1,1]:
            for sigma in [-1,1]:
                self.check_charge(s.integer_vortex(L,charge=q,sigma=sigma),around((47.5,47.5)),q,'integer_vortex')

    def test_integer_pair_individual_and_total(self):
        a=s.integer_pair(L,CENTERS)
        for c,q in zip(CENTERS,[1,-1]):self.check_charge(a,around(c),q,'integer_vortex')
        self.check_charge(a,rectangle_loop(22,73,36,59),0,'no_net_vortex')

    def test_hqv_pair_individual_and_total(self):
        for sigma in [-1,1]:
            a=s.hqv_pair(L,CENTERS,sigma=sigma)
            for c,q in zip(CENTERS,[.5,-.5]):self.check_charge(a,around(c),q,'hqv')
            self.check_charge(a,rectangle_loop(22,73,36,59),0,'no_net_vortex')

    def test_hqv_nested_resolved_contours(self):
        a=s.hqv_pair(L,CENTERS)
        for radius in [3,5,8,12]:
            for c,q in zip(CENTERS,[.5,-.5]):self.check_charge(a,around(c,radius),q)

    def test_periodic_pair_has_opposite_charges(self):
        a,centers=s.periodic_hqv_pair(L)
        for c,q in zip(centers,[-.5,.5]):self.check_charge(a,around(c),q,'hqv')
        self.check_charge(a,rectangle_loop(34,61,12,83),0,'no_net_vortex')
        # No artificial discontinuity on opposite boundary bonds.
        for axis in [0,1]:
            bonds=np.linalg.norm(np.roll(a,-1,axis=axis)-a,axis=(-2,-1))
            seam=np.take(bonds,-1,axis=axis)
            self.assertLess(seam.max(),.15)

    def test_periodic_seam_wrap_and_noncontractible_cycle(self):
        a,centers=s.periodic_hqv_pair(L)
        b=np.roll(a,50,axis=0)
        center=((centers[0][0]+50)%L,centers[0][1])
        c=around(center,6)
        for contour in [c,c%L]:
            r=analyze_loop(b,contour,periodic=True)
            self.assertTrue(r['accepted'],r)
            self.assertEqual(r['phase_charge'],-.5)
        cycle=np.array([(i,0) for i in range(L)])
        r=analyze_loop(s.uniform(L),cycle,periodic=True)
        self.assertFalse(r['accepted'],r)
        self.assertIn('noncontractible_torus_cycle',r['reasons'])

    def test_net_integer_charge_does_not_identify_one_core(self):
        a=s.same_sign_hqv_pair(L,CENTERS)
        for center in CENTERS:self.check_charge(a,around(center),.5,'hqv')
        self.check_charge(a,rectangle_loop(22,73,36,59),1,'integer_vortex')

    def test_lattice_resolution_of_resolved_contours(self):
        for size in [48,96,192]:
            centers=((size/3-.5,size/2-.5),(2*size/3-.5,size/2-.5))
            a=s.hqv_pair(size,centers,core=size/48)
            for center,q in zip(centers,[.5,-.5]):self.check_charge(a,around(center,size/16),q)

    def test_contour_orientation_reverses_charge(self):
        a=s.hqv_pair(L,CENTERS)
        self.check_charge(a,around(CENTERS[0]),.5)
        self.check_charge(a,around(CENTERS[0])[::-1],-.5)

    def test_exact_representational_branch_changes_cancel(self):
        x,y=s.grid(L)
        phi=np.arctan2(y-CENTERS[0][1],x-CENTERS[0][0])-np.arctan2(y-CENTERS[1][1],x-CENTERS[1][0])
        amplitude=s._core_factor(x,y,CENTERS,2)
        original=s.chiral(phi/2,alpha=phi/2,density=amplitude**2)
        sign=np.random.default_rng(310).integers(0,2,size=(L,L))
        d=np.stack((np.sin(phi/2),np.zeros_like(phi),np.cos(phi/2)),axis=-1)
        changed=s.chiral(phi/2+np.pi*sign,density=amplitude**2,director=d*((-1.)**sign)[...,None])
        np.testing.assert_allclose(original,changed,atol=1e-15)
        np.testing.assert_allclose(original,s.hqv_pair(L,CENTERS),atol=1e-15)

    def test_spin_rotation_and_global_phase_preserve_charge(self):
        a=s.hqv_pair(L,CENTERS)
        for seed in [4,17,93]:
            b=s.spin_rotate(a,rotation(seed))*np.exp(1j*(seed+.13))
            for c,q in zip(CENTERS,[.5,-.5]):self.check_charge(b,around(c),q)
            np.testing.assert_allclose(local_geometry(b)['residual'],local_geometry(a)['residual'],atol=2e-14)

    def test_eigenvector_sign_gauge_does_not_change_holonomy(self):
        a=s.hqv_pair(L,CENTERS);g=local_geometry(a)
        signs=np.random.default_rng(18).choice([-1,1],size=(L,L))
        for c,q in zip(CENTERS,[.5,-.5]):
            r=analyze_loop(a,around(c),geometry=g,directors=g['director']*signs[...,None])
            self.assertTrue(r['accepted'],r)
            self.assertEqual(r['director_parity'],-1)
            self.assertEqual(r['phase_charge'],q)

    def test_projection_residual_has_known_frobenius_value(self):
        # Orthogonal spin and opposite orbital chirality give exact squared
        # distance epsilon from the nearest unitary chiral rank-one matrix.
        for epsilon in [0,.02,.1,.3]:
            a=np.zeros((8,8,3,2),complex)
            a[...,2,:]=np.sqrt(1-epsilon)*np.array([1,1j])/np.sqrt(2)
            a[...,0,:]=np.sqrt(epsilon)*np.array([1,-1j])/np.sqrt(2)
            np.testing.assert_allclose(local_geometry(a)['residual'],epsilon,atol=2e-14)

    def test_weak_unrestricted_noise_retains_known_charges(self):
        a=s.hqv_pair(L,CENTERS)
        for strength in [.01,.03,.06]:
            for seed in [12,31,45,77]:
                b=s.add_noise(a,strength,seed)
                for c,q in zip(CENTERS,[.5,-.5]):self.check_charge(b,around(c),q)

    def test_chirality_wall_is_not_an_hqv(self):
        r=analyze_loop(s.chirality_wall(L),around((47.5,47.5),12))
        self.assertFalse(r['accepted'],r)
        self.assertNotEqual(r['label'],'hqv')

    def test_composite_zero_is_not_sufficient(self):
        a=s.composite_zero(L);c=around((47.5,47.5));g=local_geometry(a)
        v=g['composite'][c[:,0],c[:,1]]
        self.assertAlmostEqual(raw_winding(v),1,places=10)
        self.assertGreater(g['density'].min(),.99)
        r=analyze_loop(a,c,geometry=g)
        self.assertFalse(r['accepted'],r)

    def test_chirality_alone_does_not_establish_real_director(self):
        a=s.nonunitary_high_chirality(L);g=local_geometry(a)
        np.testing.assert_allclose(g['chirality'],1,atol=1e-14)
        self.assertGreater(g['residual'].min(),.29)
        self.assertFalse(analyze_loop(a,around((47.5,47.5)))['accepted'])

    def test_density_void_and_amplitude_floor(self):
        a=s.density_void(L)
        self.assertFalse(analyze_loop(a,around((48,48),3))['accepted'])
        self.check_charge(a,around((48,48),10),0)
        b=1e-4*s.hqv_pair(L,CENTERS)
        self.assertFalse(analyze_loop(b,around(CENTERS[0]))['accepted'])
        r=analyze_loop(b,around(CENTERS[0]),thresholds=replace(Thresholds(),min_density=1e-12))
        self.assertTrue(r['accepted'],r)
        self.assertEqual(r['phase_charge'],.5)

    def test_uncompensated_half_fields_are_rejected(self):
        for a in [s.half_phase_without_spin(L),s.half_spin_without_phase(L)]:
            r=analyze_loop(a,around((47.5,47.5)))
            self.assertFalse(r['accepted'],r)

    def test_integer_core_plaquette_alias_is_not_called_hqv(self):
        a=s.integer_vortex(L)
        r=analyze_loop(a,rectangle_loop(47,48,47,48))
        self.assertFalse(r['accepted'],r)
        self.check_charge(a,around((47.5,47.5),5),1)

    def test_continuous_full_field_escape_is_exposed(self):
        # Fill the zero HQV core with an orthogonal spin/orbital component.
        # This smooth full-field excursion leaves outer valid-loop charge but
        # invalidates the core manifold; no absolute C^6 topology is claimed.
        a=s.hqv_pair(L,CENTERS)
        x,y=s.grid(L)
        bump=np.exp(-((x-CENTERS[0][0])**2+(y-CENTERS[0][1])**2)/4)
        a[...,1,0]+=.8*bump
        self.check_charge(a,around(CENTERS[0],8),.5)
        self.assertGreater(local_geometry(a)['residual'][31,47],.12)

    def check_auto(self,a,centers,charges,periodic=False):
        found=[r for r in find_candidates(a,periodic=periodic) if r['accepted']]
        self.assertEqual(len(found),len(charges),found)
        remaining=list(found)
        for center,charge in zip(centers,charges):
            matches=[]
            for r in remaining:
                delta=np.array([r['center_x'],r['center_y']])-np.array(center)
                if periodic:delta-=np.rint(delta/np.array(a.shape[:2]))*np.array(a.shape[:2])
                if r['phase_charge']==charge:matches.append((float(np.linalg.norm(delta)),r))
            self.assertTrue(matches,found)
            distance,r=min(matches,key=lambda item:item[0])
            self.assertLessEqual(distance,1.1)
            remaining.remove(r)

    def test_automatic_uniform_and_smooth_have_no_candidates(self):
        for a in [s.uniform(L),s.smooth_texture(L)]:
            self.assertEqual(find_candidates(a,periodic=True),[])

    def test_automatic_integer_single_and_pair(self):
        for q in [-1,1]:
            self.check_auto(s.integer_vortex(L,charge=q),[(47.5,47.5)],[q])
        self.check_auto(s.integer_pair(L,CENTERS),CENTERS,[1,-1])

    def test_automatic_hqv_pair_and_periodic_pair(self):
        self.check_auto(s.hqv_pair(L,CENTERS),CENTERS,[.5,-.5])
        a,centers=s.periodic_hqv_pair(L)
        self.check_auto(a,centers,[-.5,.5],periodic=True)
        b=np.roll(a,50,axis=0)
        moved=[((cx+50)%L,cy) for cx,cy in centers]
        self.check_auto(b,moved,[-.5,.5],periodic=True)
        # Oversized contours in a tiny torus are retained as explicit rejected
        # checks, rather than raising or falsely certifying a local core.
        tiny,_=s.periodic_hqv_pair(4)
        found=find_candidates(tiny,periodic=True)
        self.assertTrue(found)
        self.assertFalse(any(r['accepted'] for r in found),found)
        self.assertTrue(all(any('contour_too_large_for_box' in loop['reasons']
                                for loop in r['loops']) for r in found))

    def test_automatic_adversaries_have_no_accepted_defects(self):
        cases=s.make_cases(L)
        for name in ['chirality_wall','composite_zero','nonunitary_high_chirality','density_void',
                     'half_phase_without_spin','half_spin_without_phase']:
            case=cases[name]
            found=find_candidates(case['field'],periodic=case['boundary']=='periodic')
            self.assertFalse(any(r['accepted'] for r in found),(name,found))

    def test_automatic_weak_noise_locates_both_known_cores(self):
        a=s.hqv_pair(L,CENTERS)
        for strength in [.03,.06]:
            for seed in [1,12,31,45]:
                self.check_auto(s.add_noise(a,strength,seed),CENTERS,[.5,-.5])

    def test_periodic_elementary_map_has_resolved_neutrality(self):
        a,_=s.periodic_hqv_pair(L);m=plaquette_map(a,periodic=True)
        self.assertEqual(m['composite_winding'].sum(),0)
        self.assertEqual(abs(m['composite_winding']).sum(),2)
        self.assertAlmostEqual(m['lifted_phase_charge'].sum(),0,places=10)
        self.assertAlmostEqual(abs(m['lifted_phase_charge']).sum(),1,places=10)
        self.assertEqual(np.sum(m['director_parity']==-1),2)

    def test_automatic_too_close_pair_remains_unresolved(self):
        a=s.hqv_pair(L,((46.5,47.5),(48.5,47.5)))
        found=find_candidates(a)
        self.assertTrue(found)
        self.assertFalse(any(r['accepted'] for r in found),found)

    def test_production_complex64_known_contour_charges(self):
        for charge in [-1,1]:
            self.check_charge(s.integer_vortex(L,charge=charge).astype(np.complex64),
                              around((47.5,47.5)),charge,'integer_vortex')
        for sigma in [-1,1]:
            a=s.hqv_pair(L,CENTERS,sigma=sigma).astype(np.complex64)
            for center,charge in zip(CENTERS,[.5,-.5]):
                self.check_charge(a,around(center),charge,'hqv')
        a,centers=s.periodic_hqv_pair(L)
        for center,charge in zip(centers,[-.5,.5]):
            self.check_charge(a.astype(np.complex64),around(center),charge,'hqv')

    def test_production_complex64_rotation_and_phase(self):
        a=s.hqv_pair(L,CENTERS).astype(np.complex64)
        for seed in [4,17,93]:
            b=(s.spin_rotate(a,rotation(seed).astype(np.float32))*
               np.complex64(np.exp(1j*(seed+.13)))).astype(np.complex64)
            self.assertEqual(b.dtype,np.complex64)
            for center,charge in zip(CENTERS,[.5,-.5]):
                self.check_charge(b,around(center),charge,'hqv')

    def test_production_complex64_automatic_counts(self):
        self.check_auto(s.integer_pair(L,CENTERS).astype(np.complex64),CENTERS,[1,-1])
        self.check_auto(s.hqv_pair(L,CENTERS).astype(np.complex64),CENTERS,[.5,-.5])
        a,centers=s.periodic_hqv_pair(L)
        self.check_auto(a.astype(np.complex64),centers,[-.5,.5],periodic=True)


def save_known_checks():
    rows=[]
    for name,case in s.make_cases(L).items():
        centers=case['centers'] or [(47.5,47.5)]
        for index,center in enumerate(centers):
            c=around(center,12 if name=='chirality_wall' else 6)
            r=analyze_loop(case['field'],c)
            row=dict(case=name,boundary=case['boundary'],center_x=center[0],center_y=center[1],
                     expected_charge=case['expected_charges'][index] if case['expected_charges'] else '',
                     **{k:r.get(k) for k in ['accepted','label','composite_winding','projected_winding','phase_charge','director_parity','lifted_phase_charge','max_residual','min_density','max_composite_step','max_director_step']},
                     reasons='; '.join(r['reasons']))
            rows.append(row)
    sensitivity=[]
    base=s.hqv_pair(L,CENTERS)
    for strength in [0,.03,.06,.12,.2,.3,.5]:
        for residual_max in [.03,.08,.12,.2]:
            accepted=correct=wrong=0
            for seed in range(10):
                a=s.add_noise(base,strength,seed)
                for center,q in zip(CENTERS,[.5,-.5]):
                    r=analyze_loop(a,around(center),thresholds=replace(Thresholds(),max_residual=residual_max))
                    accepted+=bool(r['accepted'])
                    correct+=bool(r['accepted'] and r['phase_charge']==q)
                    wrong+=bool(r['accepted'] and r['phase_charge']!=q)
            sensitivity.append(dict(noise_relative_rms=strength,max_squared_residual=residual_max,
                                    loops=20,accepted=accepted,correct=correct,wrong_charge_accepted=wrong))
    out=HERE/'results';out.mkdir(parents=True,exist_ok=True)
    for name,values in [('synthetic_validation.csv',rows),('noise_sensitivity.csv',sensitivity)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return rows,sensitivity


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StageOneTests))
    rows,sensitivity=save_known_checks()
    out=HERE/'results'
    (out/'validation.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors],
        known_contours=len(rows),sensitivity_loops=sum(x['loops'] for x in sensitivity),
        notes=['Synthetic construction validates kinematics and classification, not equilibrium defect populations.',
               'All lengths are in lattice sites; no energy minimization or Monte Carlo dynamics was run.',
               'Adversarial textures are rejected rather than assigned an HQV count.']),indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
