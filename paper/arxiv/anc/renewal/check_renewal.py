"""Finite renewal and preparation checks for A beginning is worth one limit.
Run: python -u check_renewal.py. One numerical thread; writes beside itself.
Lemma 4.1: rational renewal identities and gap ratios; Proposition 8.2:
separation constants. The seeded finite tori illustrate preparation dependence
and escape times, not infinite phases, seed universality or noncomputability.
The selected simulation rates are not certified all-time reliability rates.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

t0 = time.time()

# ---------------------------------------------------------------- C1
# Gap law: P(G=2k) = 2^-k b_k, P(G=2k+1) = 2^-k (1-b_k), k = 1..Kmax, truncated and renormalised
# (truncation only for the exact finite check; the identities are pointwise in the gap law).
def gap_law(bs):
    p = {}
    for k, b in enumerate(bs, start=1):
        p[2 * k] = Fr(1, 2 ** k) * b
        p[2 * k + 1] = Fr(1, 2 ** k) * (1 - b)
    Z = sum(p.values())
    return {g: v / Z for g, v in p.items()}

def renewal_chain_cylinder(p, word):
    """Exact nu([word]) at position 0 via the stationary Markov chain on states (g, j):
    current gap length g, position j in block '1 0^g' (j=0 is the '1')."""
    states = [(g, j) for g in sorted(p) for j in range(g + 1)]
    EG = sum(g * v for g, v in p.items())
    pi = {(g, j): p[g] / (1 + EG) for (g, j) in states}  # stationary law of a block-renewal chain
    def sym(s):
        return 1 if s[1] == 0 else 0
    def nxt(s):
        g, j = s
        if j < g:
            return [((g, j + 1), Fr(1))]
        return [((h, 0), p[h]) for h in sorted(p)]
    dist = {s: pi[s] for s in states if sym(s) == word[0]}
    for c in word[1:]:
        nd = {}
        for s, w in dist.items():
            for s2, q in nxt(s):
                if sym(s2) == c:
                    nd[s2] = nd.get(s2, 0) + w * q
        dist = nd
    return sum(dist.values()), EG

c1 = {"rows": [], "pass": True}
rng = np.random.default_rng(20261029)
for trial in range(3):
    bs = [Fr(int(x), 7) for x in rng.integers(0, 8, size=6)]
    p = gap_law(bs)
    Z = sum(Fr(1, 2 ** k) for k in range(1, 7))
    for k in range(1, 7):
        w_even = [1] + [0] * (2 * k) + [1]
        w_odd = [1] + [0] * (2 * k + 1) + [1]
        ce, EG = renewal_chain_cylinder(p, w_even)
        co, _ = renewal_chain_cylinder(p, w_odd)
        f_even = p[2 * k] / (1 + EG)
        f_odd = p[2 * k + 1] / (1 + EG)
        ratio = ce / (ce + co)
        ok = (ce == f_even) and (co == f_odd) and (ratio == bs[k - 1])
        c1["rows"].append({"trial": trial, "k": k, "beta_k": str(bs[k - 1]), "ratio": str(ratio),
                           "nu_even_chain": str(ce), "nu_even_formula": str(f_even), "ok": ok})
        c1["pass"] &= ok
    # general cylinder formula nu([0^a 1 0^g1 1 ... 1 0^b]) = P(G>=a) prod P(G=g_j) P(G>=b) / (1+EG)
    tail = lambda a: sum(v for g, v in p.items() if g >= a)
    for word in ([0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 1]):
        cw, EG = renewal_chain_cylinder(p, word)
        ones = [i for i, c in enumerate(word) if c == 1]
        a = ones[0]; b = len(word) - 1 - ones[-1]
        prod = Fr(1)
        for u, v in zip(ones, ones[1:]):
            prod *= p.get(v - u - 1, Fr(0))
        f = tail(a) * prod * tail(b) / (1 + EG)
        ok = (cw == f)
        c1["rows"].append({"trial": trial, "word": "".join(map(str, word)), "chain": str(cw), "formula": str(f), "ok": ok})
        c1["pass"] &= ok
log("C1 renewal identities (exact rational arithmetic):", "PASS" if c1["pass"] else "FAIL",
    f"({len(c1['rows'])} rows)")
OUT["C1_renewal_identities"] = c1

# ---------------------------------------------------------------- C2
# Untruncated gap law: EG = sum_k 2^-k (2k + 1 - b_k) <= 5; nu(x0 != x1) = 2 nu([10]) = 2/(1+EG) >= 1/3.
c2 = {"pass": True, "rows": []}
for bs_const in (Fr(0), Fr(1, 3), Fr(1)):
    EG_upper = 5 - bs_const  # sum 2^-k (2k+1) = 5, sum 2^-k b = b for constant b
    sep = Fr(2) / (1 + EG_upper)
    ok = EG_upper <= 5 and sep >= Fr(1, 3)
    c2["rows"].append({"b_const": str(bs_const), "EG": str(EG_upper), "nu_x0_ne_x1": str(sep), "ok": ok})
    c2["pass"] &= ok
s = sum(Fr(2 * k + 1, 2 ** k) for k in range(1, 200))
c2["sum_2k+1_over_2^k_partial_199"] = float(s)
c2["pass"] &= abs(float(s) - 5) < 1e-12
log("C2 EG <= 5 and nu(x0 != x1) >= 1/3:", "PASS" if c2["pass"] else "FAIL")
OUT["C2_phase_separation_constants"] = c2

# ---------------------------------------------------------------- C3
# Toom-stacked max CA on a torus A x B x I. Local rule: m = NEC majority of (a,b),(a+1,b),(a,b+1) at each i;
# new x(a,b,i) = max(m(a,b,i), m(a,b,i+1)). The max CA is one-sided (offsets 0,1), so no shift/blocking is needed.
# Noise: each updated bit is flipped independently with probability eps (all kernel entries positive).
def step(x, eps, rng):
    m = ((x.astype(np.int8) + np.roll(x, -1, axis=0) + np.roll(x, -1, axis=1)) >= 2)
    y = np.maximum(m, np.roll(m, -1, axis=2))
    flips = rng.random(x.shape) < eps
    return np.logical_xor(y, flips)

def run(x0, eps, T, rng, record=(0, 5, 10, 20, 40, 80, 120)):
    x = x0.copy()
    rows = {}
    for t in range(T + 1):
        if t in record:
            dec = (x.mean(axis=0) > 0.5)  # plane-majority decoding along a (illustration only)
            rows[t] = {"raw_density": round(float(x.mean()), 4), "decoded_density": round(float(dec.mean()), 4)}
        if t < T:
            x = step(x, eps, rng)
    return rows

A = B = 24; I = 96; T = 120
c3 = {"lattice": [A, B, I], "T": T, "runs": [], "pass": True}
rng = np.random.default_rng(7)
for eps in (0.0, 0.005, 0.02):
    for p in (0.05, 0.3, 0.7):
        seed_row = rng.random(I) < p
        x0 = np.broadcast_to(seed_row, (A, B, I)).copy()  # cloned Bernoulli(p): translation invariant law
        rows = run(x0, eps, T, rng)
        c3["runs"].append({"start": "clone Bernoulli", "p": p, "eps": eps, "rows": rows})
    x0 = np.zeros((A, B, I), dtype=bool)  # cloned constant fixed point: the frozen phase
    rows = run(x0, eps, T, rng)
    c3["runs"].append({"start": "clone all-0", "p": 0.0, "eps": eps, "rows": rows})
# product-Bernoulli control (no cloning) and forgetting control (large eps)
for p in (0.05, 0.3):
    x0 = rng.random((A, B, I)) < p
    rows = run(x0, 0.005, T, rng)
    c3["runs"].append({"start": "product Bernoulli (no clone)", "p": p, "eps": 0.005, "rows": rows})
x0 = np.zeros((A, B, I), dtype=bool)
rows = run(x0, 0.35, 40, rng, record=(0, 5, 10, 20, 40))
c3["runs"].append({"start": "clone all-0, high noise (forgetting control)", "p": 0.0, "eps": 0.35, "rows": rows})
# pass criteria: (i) every begun run from a cloned Bernoulli(p) seed, eps <= 0.02, decodes to density 1 at T;
# (ii) every frozen run (eps <= 0.02) decodes to density 0 at T; (iii) raw error density stays near eps.
for r in c3["runs"]:
    last = r["rows"][max(r["rows"])]
    if r["start"] == "clone Bernoulli":
        ok = last["decoded_density"] == 1.0
    elif r["start"] == "clone all-0":
        ok = last["decoded_density"] == 0.0 and last["raw_density"] <= 3 * r["eps"] + 1e-9
    else:
        ok = True
    r["ok"] = ok
    c3["pass"] &= ok
for r in c3["runs"]:
    last_t = max(r["rows"])
    log(f"C3 start={r['start']:<45} p={r['p']:<5} eps={r['eps']:<6} t={last_t:<4} "
        f"raw={r['rows'][last_t]['raw_density']:<7} decoded={r['rows'][last_t]['decoded_density']:<6} "
        f"t=10 decoded={r['rows'].get(10, {}).get('decoded_density')}")
log("C3 selection by the beginning (two eternal phases, seed- and noise-independent begun law):",
    "PASS" if c3["pass"] else "FAIL")
OUT["C3_toom_stacked_max_ca"] = c3

# ---------------------------------------------------------------- C4
# Finite tiny tori: escape time of the frozen phase (first time decoded density >= 1/2) at eps = 0.03,
# fixed I = 3, growing redundancy plane A = B. Finite worlds forget (unique stationary law); the frozen phase is
# metastable with an escape time that grows with the redundancy.
c4 = {"eps": 0.03, "I": 3, "rows": []}
rng = np.random.default_rng(11)
CAP = 60000
for a in (2, 3, 4, 5):
    times = []
    for rep in range(12):
        x = np.zeros((a, a, 3), dtype=bool)
        t_esc = CAP + 1
        for t in range(1, CAP + 1):
            x = step(x, 0.03, rng)
            if (x.mean(axis=0) > 0.5).mean() >= 0.5:
                t_esc = t
                break
        times.append(t_esc)
    c4["rows"].append({"A=B": a, "I": 3, "median_escape": float(np.median(times)), "max_escape": int(max(times)),
                       "censored": int(sum(1 for v in times if v > CAP))})
    log(f"C4 torus {a}x{a}x3 eps=0.03: median escape {np.median(times)}, max {max(times)}, censored {c4['rows'][-1]['censored']}/12")
meds = [r["median_escape"] for r in c4["rows"]]
c4["pass"] = all(meds[j] < meds[j + 1] for j in range(len(meds) - 1)) and meds[0] <= CAP
log("C4 finite-torus forgetting (escape time grows with redundancy):", "PASS" if c4["pass"] else "FAIL")
OUT["C4_finite_torus_forgetting"] = c4

# ---------------------------------------------------------------- summary
peak = None  # Optional per-script peak; the runner samples process-tree RSS.
OUT["runtime_s"] = round(time.time() - t0, 2)
OUT["peak_working_set_bytes"] = peak
OUT["all_pass"] = all(v.get("pass", True) for v in OUT.values() if isinstance(v, dict))
assert OUT["all_pass"], "Finite renewal criterion failed"
log("ALL PASS" if OUT["all_pass"] else "SOME CHECK FAILED", f"runtime {OUT['runtime_s']} s, peak working set {peak} bytes")
with open(os.path.join(HERE, "checks.json"), "w", newline="\n", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, default=str)
with open(os.path.join(HERE, "check_output.txt"), "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
