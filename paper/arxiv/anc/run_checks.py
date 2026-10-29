"""Run finite consistency checks sequentially under a measured memory bound."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE=Path(__file__).resolve().parent
scripts=sys.argv[1:] or ['renewal/check_renewal.py','renewal/check_renewal_independent.py',
    'fluctuations/check_fluctuations.py','fluctuations/check_fluctuations_independent.py',
    'fluctuations/check_modal.py']
records=[]
for name in scripts:
    start=time.monotonic()
    peak=0
    print('Running '+name,flush=True)
    target=HERE/Path(name).parent/(Path(name).stem+'_output.txt')
    with target.open('w',encoding='utf-8',newline='\n') as output:
        proc=subprocess.Popen([sys.executable,'-u',str(HERE/name)],cwd=HERE,
            stdout=output,stderr=subprocess.STDOUT)
        monitor=psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                children=[monitor]+monitor.children(recursive=True)
                measured=sum(p.memory_info().rss for p in children if p.is_running())
                measured+=psutil.Process().memory_info().rss
                peak=max(peak,measured)
                if measured>=900000000:
                    for p in reversed(children):p.kill()
                    raise RuntimeError('Measured memory bound exceeded')
            except psutil.NoSuchProcess:
                pass
            if time.monotonic()-start>1800:
                proc.kill()
                raise RuntimeError('Check exceeded time limit')
            time.sleep(.1)
    content=target.read_text(encoding='utf-8')
    target.write_text(content,encoding='utf-8',newline='\n')
    if proc.returncode or 'Traceback' in content or 'FAIL' in content:
        raise RuntimeError('Check failed: '+name)
    records.append(dict(script=name,status='PASS',seconds=round(time.monotonic()-start,3),
                        sampled_peak_total_rss_bytes=peak,sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest()))
    print(name+' PASS',flush=True)
summary=dict(status='PASS',single_threaded=True,memory_limit_bytes=900000000,checks=records,
             memory_method='sampled child-process-tree plus runner RSS; not an OS historical peak', scope='Finite criteria and descriptive simulations are distinguished; no noncomputability certification.')
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
print('ALL FINITE CHECKS PASS',flush=True)
