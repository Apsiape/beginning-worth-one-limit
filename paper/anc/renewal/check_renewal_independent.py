"""Independent finite checks for A beginning is worth one limit.
Sections 4, 5 and 8: renewal words, block drift, finite cones, seed covariance,
and exact-cone simulations. Finite horizons do not certify infinite reliability.
All writes stay beside this script; one numerical thread.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import itertools, json, math, time
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
T0=time.perf_counter()
results={}
lines=[]
def log(s):
    print(s,flush=True); lines.append(s)

# A different renewal representation: countdown until the next 1.
def cylinder(p, word):
    mean=sum(g*v for g,v in p.items()); den=1+mean
    G=max(p)
    pi=[Q(1)/den]+[sum(v for g,v in p.items() if g>=j)/den for j in range(1,G+1)]
    w=pi
    for symbol in word:
        selected=[v if (j==0)==(symbol==1) else Q(0) for j,v in enumerate(w)]
        w=[Q(0) for _ in pi]
        for j,v in enumerate(selected):
            if j: w[j-1]+=v
            else:
                for g,prob in p.items(): w[g]+=v*prob
    return sum(w)
def formula(p, word):
    den=1+sum(g*v for g,v in p.items())
    tail=lambda n: sum(v for g,v in p.items() if g>=n)
    pos=[i for i,b in enumerate(word) if b]
    if not pos: return sum(tail(j) for j in range(len(word),max(p)+1))/den
    val=tail(pos[0])*tail(len(word)-1-pos[-1])/den
    for a,b in zip(pos,pos[1:]): val*=p.get(b-a-1,Q(0))
    return val
nr=0
for bs in ([Q(0)]*3,[Q(1)]*3,[Q(1,3),Q(3,4),Q(2,5)]):
    p={}
    for k,b in enumerate(bs,1):
        p[2*k]=b/Q(2**k); p[2*k+1]=(1-b)/Q(2**k)
    z=sum(p.values()); p={g:v/z for g,v in p.items() if v}
    for n in range(1,9):
        for word in itertools.product((0,1),repeat=n):
            assert cylinder(p,word)==formula(p,word),(p,word)
            nr+=1
    for k,b in enumerate(bs,1):
        ce=cylinder(p,(1,)+(0,)*(2*k)+(1,))
        co=cylinder(p,(1,)+(0,)*(2*k+1)+(1,))
        assert ce/(ce+co)==b
results['renewal']={'all_binary_words_length_1_to_8':nr,'pass':True,
    'beta_k_all_zero_interarrival_period':math.gcd(*[2*k+2 for k in range(1,50)])}
log(f'Renewal countdown-chain identities: {nr} exact word comparisons PASS; all-zero beta has period 2.')

# Check the one-sided block compiler against an independently updated radius-r rule.
rng=np.random.default_rng(931817)
comparisons=0
for r in (1,2,3):
    B=2*r; length=B*16
    rule=rng.integers(0,2,2**(2*r+1),dtype=np.uint8)
    x=rng.integers(0,2,length,dtype=np.uint8); h=x.copy()
    for t in range(1,8):
        code=sum(np.roll(x,-j).astype(np.int64)*(2**(j+r)) for j in range(-r,r+1))
        x=rule[code]
        blocks=h.reshape(-1,B)
        pair=np.concatenate((blocks,np.roll(blocks,-1,axis=0)),axis=1)
        nh=np.empty_like(blocks)
        for j in range(B):
            idx=sum(pair[:,j+k].astype(np.int64)*2**k for k in range(2*r+1))
            nh[:,j]=rule[idx]
        h=nh.reshape(-1)
        assert np.array_equal(h,np.roll(x,-r*t))
        comparisons+=length
results['blocking']={'coordinate_comparisons':comparisons,'pass':True}
log(f'One-sided blocking/drift: {comparisons} coordinate comparisons PASS.')

# Six-parent backward-cone geometry; distinct transverse slabs have disjoint noise.
parents={(a,b,i) for a,b in ((0,0),(1,0),(0,1)) for i in (0,1)}
geometry=[]
for t in range(1,9):
    layer={(0,0,0)}; v=0
    for k in range(t):
        assert len(layer)==(k+1)**2*(k+2)//2
        v+=len(layer)
        layer={tuple(x[j]+p[j] for j in range(3)) for x in layer for p in parents}
    shifted={(a+2*t+1,b,i) for a,b,i in layer}
    assert not layer.intersection(shifted)
    geometry.append({'t':t,'fault_vertices':v,'initial_sites':len(layer),'disjoint':True})
results['cones']=geometry
log('Exact six-parent cones and spacing 2t+1: horizons 1 through 8 PASS.')

# Exact countercheck of unconditional decorrelation with a shared Bernoulli row.
eps=Q(1,50); q0=eps/2; q1=eps
p_error=(q0+q1)/2
p_both=(q0*q0+q1*q1)/2
cov=p_both-p_error*p_error
assert cov==eps*eps/16 and cov>0
results['unconditional_error_correlation']={'epsilon':str(eps),'p_error':str(p_error),
 'p_both':str(p_both),'covariance_at_any_separation':str(cov),
 'proposed_decaying_unconditional_covariance_bound_is_false':True}
log(f'Cloned-row covariance counterexample: exact covariance {cov}, independent of separation.')

# Independent positive-rate stacked XOR CA on exact shrinking cones, with no torus.
# Logical F(y)_i=y_i XOR y_(i+1). The full infinite seed law is cloned Bernoulli.
# Each batch shares one logical row per history across all 17 distant redundant cones.
def batch_xor(t,eps,p,batch,m,rng):
    row=rng.random((batch,t+1))<p
    logical=row.copy()
    for _ in range(t): logical=logical[:,:-1]^logical[:,1:]
    truth=logical[:,0]
    x=np.broadcast_to(row[:,None,None,None,:],(batch,m,t+1,t+1,t+1)).copy()
    for _ in range(t):
        a=x[:,:,:-1,:-1,:]; b=x[:,:,1:,:-1,:]; c=x[:,:,:-1,1:,:]
        vote=(a&b)|(a&c)|(b&c)
        x=vote[...,:-1]^vote[...,1:]
        if eps: x^=rng.random(x.shape)<eps
    return x[:,:,0,0,0],truth
rng=np.random.default_rng(910261029)
for t in range(1,7):
    x,truth=batch_xor(t,0,0.31,32,3,rng)
    assert np.all(x==truth[:,None])
results['xor_noiseless_validation']={'horizons':list(range(1,7)),'pass':True}
rows=[]
trials=4096; mmax=17
for t,eps,p in ((2,0.02,0.17),(4,0.002,0.61),(6,0.0004,0.17)):
    bad={m:0 for m in (1,3,5,9,17)}; total_errors=0; true_ones=0
    for start in range(0,trials,128):
        n=min(128,trials-start)
        x,truth=batch_xor(t,eps,p,n,mmax,rng)
        wrong=x!=truth[:,None]
        total_errors+=int(wrong.sum()); true_ones+=int(truth.sum())
        for m in bad:
            decoded=x[:,:m].sum(axis=1)>m/2
            bad[m]+=int(np.count_nonzero(decoded!=truth))
    v=sum((k+1)**2*(k+2)//2 for k in range(t))
    delta=1-(1-eps)**v
    assert delta<0.5
    bounds={m:math.exp(-2*m*(0.5-delta)**2) for m in bad}
    row={'t':t,'epsilon':eps,'seed_p':p,'trials':trials,'endpoints_per_trial':mmax,
        'logical_density':true_ones/trials,'single_endpoint_error':total_errors/(trials*mmax),
        'wrong_majority_counts':bad,'certified_fault_vertices':v,'certified_single_error_bound':delta,
        'hoeffding_bounds':bounds,'observed_rates_below_bounds':all(bad[m]/trials<=bounds[m] for m in bad)}
    rows.append(row)
    log(f'Exact-cone XOR t={t}, eps={eps}, p={p}: site error {row["single_endpoint_error"]:.7f}, wrong majorities {bad}.')
results['independent_xor_monte_carlo']=rows

# A separate small finite-torus OR experiment for phase-selection illustration.
def or_step(z,eps,rng):
    north=np.concatenate((z[1:],z[:1]),axis=0)
    east=np.concatenate((z[:,1:],z[:,:1]),axis=1)
    corrected=(z&north)|(z&east)|(north&east)
    next_i=np.concatenate((corrected[:,:,1:],corrected[:,:,:1]),axis=2)
    return (corrected|next_i)^(rng.random(z.shape)<eps)
rng=np.random.default_rng(6261029)
torus_rows=[]
for kind,p in (('clone',0.08),('clone',0.32),('clone_zero',0.0),('product',0.02),('product',0.32)):
    final=[]; raw=[]; all_zero_seed=0
    for rep in range(8):
        if kind=='product': z=rng.random((16,16,64))<p
        else:
            row=rng.random(64)<p
            all_zero_seed+=int(not row.any())
            z=np.broadcast_to(row,(16,16,64)).copy()
        for _ in range(96): z=or_step(z,0.005,rng)
        final.append(float((z.sum(axis=(0,1))>128).mean())); raw.append(float(z.mean()))
    item={'start':kind,'p':p,'replicates':8,'lattice':[16,16,64],'T':96,'epsilon':0.005,
          'decoded_densities':final,'mean_raw_density':sum(raw)/8,'all_zero_finite_seed_count':all_zero_seed}
    torus_rows.append(item)
    log(f'OR torus {kind} p={p}: 8 decoded densities {final}; raw mean {item["mean_raw_density"]:.7f}.')
results['independent_or_torus']=torus_rows

peak = None  # Optional platform-specific peak; monitored by the runner.
results['runtime_s']=round(time.perf_counter()-T0,4)
results['peak_working_set_bytes']=peak
results['numpy_version']=np.__version__
results['numerical_threads_requested']=1
assert peak is None or peak<1_000_000_000
log(f'Finished in {results["runtime_s"]} s; peak working set {results["peak_working_set_bytes"]} bytes.')
(HERE/'independent_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8', newline="\n")
(HERE/'independent_output.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8', newline="\n")
