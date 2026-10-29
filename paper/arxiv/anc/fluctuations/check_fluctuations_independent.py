"""Independent finite checks for A beginning is worth one limit.
Theorem 2.4, Lemma 5.1 and the Section 5.3 quantizer: conditional transfer,
certificate bounds and rational plateaus, with counterexamples and a finite
guard diagnostic. These are finite checks, not universal no-go experiments.
All writes stay beside this script.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, math, time, hashlib, itertools
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_info
sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
START = time.perf_counter()
results = {}

def exact_transfer():
    # A different rule, ring size, noise, and correlated seed from the supplied X1.
    n = 8
    states = np.arange(1 << n)
    bits = ((states[:,None] >> np.arange(n)) & 1).astype(np.int8)
    det = bits ^ np.roll(bits,-1,axis=1) ^ np.roll(bits,1,axis=1)
    p1 = np.where(det,0.89,0.07)
    M = np.ones((len(states),len(states)))
    for i in range(n):
        M *= np.where(bits[None,:,i], p1[:,None,i], 1-p1[:,None,i])
    pi = np.full(len(states),1/len(states))
    for _ in range(600):
        pi = pi @ M
    pi /= pi.sum()
    lam = np.zeros(len(states))
    # A computable mixture of three entire configurations, with nonproduct correlations.
    lam[[0,85,255]] = [0.2,0.3,0.5]
    f = bits[:,0].astype(float)
    err_inside = err_weighted = err_lower = outside = 0.0
    tested = 0
    for lag in range(1,4):
        f = M @ f
        B = [j % n for j in range(-lag,lag+1)]
        keys = bits[:,B] @ (1 << np.arange(len(B)))
        q = 0.0
        for w in range(1 << len(B)):
            mask = keys == w
            pw = float(pi[mask].sum())
            conditional = float(pi[mask] @ f[mask] / pw)
            err_inside = max(err_inside,float(np.max(np.abs(f[mask]-conditional))))
            q += float(lam[mask].sum()) * conditional
            err_lower = max(err_lower,0.07**len(B)-pw)
            tested += 1
        err_weighted = max(err_weighted,abs(q-float(lam @ f)))
        # Intentionally underspecified initial condition: radius zero.
        for symbol in (0,1):
            v = f[bits[:,0] == symbol]
            outside = max(outside,float(v.max()-v.min()))
    assert err_inside < 1e-12 and err_weighted < 1e-12 and err_lower <= 1e-12
    assert outside > 0.05
    return dict(n=n,rule='xor(left,center,right)',p1_if_zero=.07,p1_if_one=.89,
                all_conditioning_patterns_tested=tested,inside_error=err_inside,
                weighted_correlated_seed_error=err_weighted,
                lower_bound_violation=err_lower,undersized_cone_extension_spread=outside,
                stationary_residual=float(np.max(np.abs(pi @ M-pi))))

def constants():
    D = 24
    eps = Q(1,64*D**8)
    c = 16*D**8*eps
    delta = (1+8*D)*eps
    assert c == Q(1,4) and delta < Q(1,16)
    worst = Q(0)
    for k in range(1,129):
        m,L = 8*k+9,64*(k+1)
        h,n0 = (m+1)//2,(L+7)//8
        e = 8*D*m*eps*c**(n0-1)+(4*delta)**h
        target = Q(1,96*4**k)
        worst = max(worst,(2*k+3)*e/target)
        assert (2*k+3)*e <= target
        # Sum the tree counting bound, rather than substitute its largest term.
        A=2*D**2
        for n in (1,2,3,10):
            N=4*(n-1)
            assert 2*D*sum(A**ell for ell in range(N+1)) <= 4*D*A**N
    return dict(D=D,epsilon=str(eps),epsilon_float=float(eps),c=str(c),delta=str(delta),
                m_formula='8*k+9',L_formula='64*(k+1)',k_checked=[1,128],
                maximum_error_budget_ratio=float(worst),
                original_threshold=1/(32*D**8),correct_one_site_coefficient=4*D)

def plateau_quantizer():
    # Continuous finite sum of clipped rational affine maps: no discontinuous rounding.
    def quantize(k,x):
        h=Q(1,2**k)
        return h*sum(min(Q(1),max(Q(0),2*x/h-2*j-Q(1,2))) for j in range(2**k))
    tested=0
    for k in range(1,8):
        h=Q(1,2**k)
        for j in range(2**k+1):
            center=j*h
            for fraction in [Q(-1,4),Q(-1,8),Q(0),Q(1,8),Q(1,4)]:
                x=center+fraction*h
                if 0<=x<=1:
                    assert quantize(k,x)==center
                    tested+=1
    return dict(exact_plateau_checks=tested,grid_step='2**(-k)',
                plateau_radius='grid_step/4',asymptotic_ratio_error_at_most='grid_step/8',
                globally_continuous_computable=True,
                conclusion='dyadic target approximants plus this quantizer give eventual fixed-k equality')

def exact_counterexamples():
    fair_majorities={}
    for m in (1,3,5,9,17,33):
        probability=sum(Q(math.comb(m,j),2**m) for j in range((m+1)//2,m+1))
        assert probability == Q(1,2)
        fair_majorities[str(m)] = str(probability)
    # Conditional independence given a cloned bit is not unconditional decorrelation.
    p0,p1=Q(1,1000),Q(2,1000)
    joint=(p0*p0+p1*p1)/2
    marginal=(p0+p1)/2
    covariance=joint-marginal*marginal
    assert covariance == (p1-p0)**2/4 > 0
    # Guard off, active reference, but a locally incorrect majority gives the correct next bit.
    old,next_bit=0,1  # complement-center rule, ECA 51
    triple=(1,1,0)
    plane_majority=int(sum(triple)>=2)
    guard=len(set(triple))==1
    assert not guard and old != next_bit and plane_majority == next_bit
    # One equality y_i(t+1)=y_i(t) also need not mean permanently frozen content.
    # Full-symbol strict majority differs from plurality.
    categorical=(Q(2,5),Q(7,20),Q(1,4))
    assert sum(categorical)==1 and max(categorical)<Q(1,2)
    return dict(fair_majority_P1=fair_majorities,
                unconditional_error_covariance=str(covariance),
                covariance_persists_at_arbitrary_transverse_separation=True,
                guard_counterexample=dict(rule=51,reference_old=old,reference_next=next_bit,
                                          local_triple=triple,guard_on=guard,output=plane_majority),
                unique_plurality_without_majority=[str(x) for x in categorical])

def step(x,rule,eps,rng,guarded):
    east,north=np.roll(x,-1,axis=0),np.roll(x,-1,axis=1)
    majority=(x & east)|(x & north)|(east & north)
    index=(np.roll(majority,1,axis=2).astype(np.int8)*4
           +majority.astype(np.int8)*2+np.roll(majority,-1,axis=2))
    computed=((rule >> index) & 1).astype(bool)
    if guarded:
        agree=(x == east)&(x == north)
        guard=agree&np.roll(agree,1,axis=2)&np.roll(agree,-1,axis=2)
        computed=np.where(guard,computed,majority)
    return computed ^ (rng.random(x.shape)<eps)

def line_step(row,rule):
    index=np.roll(row,1).astype(np.int8)*4+row.astype(np.int8)*2+np.roll(row,-1).astype(np.int8)
    return ((rule >> index)&1).astype(bool)

def independent_guard_experiment():
    rng=np.random.default_rng(291029)
    row=rng.random(40)<.5
    reference=row.copy()
    initial=np.broadcast_to(row,(40,32,40)).copy()
    naive,guarded=initial.copy(),initial.copy()
    rows=[]
    for t in range(1,101):
        reference=line_step(reference,30)
        naive=step(naive,30,.01,rng,False)
        guarded=step(guarded,30,.01,rng,True)
        if t in (10,25,50,100):
            rows.append(dict(t=t,naive_error=float(np.mean(naive != reference)),
                             guarded_error=float(np.mean(guarded != reference))))
    alternate=np.arange(40)%2==0
    assert np.array_equal(alternate,line_step(alternate,30))
    return dict(seed=291029,shape=list(initial.shape),epsilon=.01,rows=rows,
                alternating_rule30_row_is_fixed=True,interpretation='finite-volume diagnostic, not a no-go theorem')

for label,fn in [('exact_transfer',exact_transfer),('certificate_constants',constants),
                 ('plateau_quantizer',plateau_quantizer),
                 ('counterexamples',exact_counterexamples),('guard_diagnostic',independent_guard_experiment)]:
    results[label]=fn()
    print(label,json.dumps(results[label],ensure_ascii=False),flush=True)
results['threadpools']=[{k:v for k,v in pool.items() if k!='filepath'}
                       for pool in threadpool_info()]
assert all(p.get('num_threads',1)==1 for p in results['threadpools'])
results['seconds']=round(time.perf_counter()-START,3)
info=psutil.Process().memory_info()
results['peak_working_set_bytes']=getattr(info,'peak_wset',None)
assert results['peak_working_set_bytes'] is None or results['peak_working_set_bytes'] < 1_000_000_000
results['primary_source_sha256']=hashlib.sha256((OUT/'check_fluctuations.py').read_bytes()).hexdigest()
results['all_checks_pass']=True
(OUT/'independent_checks.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8', newline="\n")
print('ALL INDEPENDENT CHECKS PASS',results['seconds'],results['peak_working_set_bytes'],flush=True)
