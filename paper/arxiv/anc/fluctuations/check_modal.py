"""Finite modal-row diagnostic for Section 9 of A beginning is worth one limit.
Uses the same seeded stream as the coarsening observations, with an explicit
modal fixed-point check. Period search is bounded; no infinite theorem is tested.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,importlib.util
from pathlib import Path
import numpy as np
import psutil
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('supplied',out/'check_fluctuations.py')
supplied=importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplied)
rng=np.random.default_rng(8)
started=time.perf_counter()
result={}
for rule,eps in ((30,.02),(30,.01),(110,.01)):
    x=rng.random((128,128,16))<.5
    for t in range(3000):
        x,g=supplied.guarded_step(x,rule,eps,rng)
    rows,counts=np.unique(np.packbits(x.reshape(128*128,16),axis=1),axis=0,return_counts=True)
    modal=np.unpackbits(rows[np.argmax(counts)])[:16].astype(bool)
    nxt=supplied.eca(rule,np.roll(modal,1),modal,np.roll(modal,-1))
    result[f'rule{rule}_eps{eps}']={
        'modal_row':''.join('1' if v else '0' for v in modal),
        'modal_share':float(counts.max()/(128*128)),
        'modal_fixed':bool(np.array_equal(modal,nxt)),
        'period_and_preperiod':supplied.period_1d(rule,modal),
        'agreement_d32':float((x==np.roll(x,32,axis=0)).mean())}
    print(rule,eps,result[f'rule{rule}_eps{eps}'],flush=True)
result['seconds']=round(time.perf_counter()-started,2)
mem=psutil.Process().memory_info()
result['peak_working_set_bytes']=getattr(mem,'peak_wset',None)
assert result['peak_working_set_bytes'] is None or result['peak_working_set_bytes']<1_000_000_000
assert result['rule30_eps0.02']['modal_fixed'] and result['rule30_eps0.01']['modal_fixed']
assert not result['rule110_eps0.01']['modal_fixed']
(out/'modal_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8', newline="\n")
