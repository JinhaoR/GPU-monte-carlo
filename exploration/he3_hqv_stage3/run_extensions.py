"""Explicit additional cases justified by finite-size and validation results.

Run only to reproduce this report's bounded extension, not an automatic scan:
python -B exploration/he3_hqv_stage3/run_extensions.py
Completed cases are reused. No MC or GPU calculation is performed.
"""
from time import perf_counter
import numpy as np
from run_cases import run_one


def main():
    start=perf_counter()
    for L,Rs in [(128,[16,24,32,48]),(192,[16,24,32,48])]:
        for R in Rs:
            if perf_counter()-start>600:raise RuntimeError('Extension budget reached; no more cases launched')
            run_one(f'L{L}_R{R}_p2',L,R)
    for R in [16,32]:run_one(f'L64_R{R}_rotated_pin',64,R,right_rotation=np.pi/2)
    run_one('L64_R24_released',64,24,release='L64_R24_p2')
    for R in [16,32]:run_one(f'L128_R{R}_gd0001',128,R,gd=.001)
    print('Explicit extension seconds',perf_counter()-start,flush=True)


if __name__=='__main__':main()
