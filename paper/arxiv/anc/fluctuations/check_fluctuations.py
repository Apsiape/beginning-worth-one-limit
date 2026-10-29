# Finite fluctuation and coarsening checks for A beginning is worth one limit.
# Run: python -u check_fluctuations.py. Related proofs: Theorem 2.4 and Lemma 6.1.
# Four numerical criteria are distinct from five descriptive observation groups.
# Finite torus branches are metastability; sampled noise is not analytically certified.
# Single-threaded numpy; every random stream is seeded; results -> checks.json next to this file.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, math, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
T0 = time.time()


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------
# X1. Exact fluctuation-transfer identity on a small ring (exact Markov chain, float64).
# 1D PCA on Z_n, radius 1, binary, positive rates.  For the stationary law pi and a ball B of
# radius rho around site 0, P_pi(x_0(t)=1 | x_B(0)=w) must equal the begun probability from ANY
# configuration extending w whenever rho >= t (light cone), and may differ when rho < t.
# ---------------------------------------------------------------------------------------------
def x1():
    n = 11
    N = 1 << n
    # kernel: noisy rule 110 with asymmetric noise (positive rates, not attractive, not symmetric)
    rule = 110
    def p1(l, c, r):
        det = (rule >> (l * 4 + c * 2 + r)) & 1
        return 0.93 if det else 0.04
    states = np.arange(N)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)  # bits[s, i]
    # per-site probability of 1 at next step, for each state
    P1 = np.zeros((N, n))
    for i in range(n):
        l = bits[:, (i - 1) % n]; c = bits[:, i]; r = bits[:, (i + 1) % n]
        idx = l * 4 + c * 2 + r
        table = np.array([p1(*(((k >> 2) & 1), ((k >> 1) & 1), (k & 1))) for k in range(8)])
        P1[:, i] = table[idx]
    # transition matrix M[s, s'] = prod_i P(s'_i | s)
    M = np.ones((N, N))
    for i in range(n):
        q1 = P1[:, i][:, None]
        b = bits[:, i][None, :]
        M *= np.where(b == 1, q1, 1.0 - q1)
    assert np.allclose(M.sum(1), 1.0)
    # stationary law
    pi = np.ones(N) / N
    for _ in range(4000):
        pi = pi @ M
    pi /= pi.sum()
    stat_resid = float(np.abs(pi @ M - pi).max())
    rng = np.random.default_rng(1)
    rows = []
    maxdev_inside = 0.0
    mindev_outside = []
    for rho in (1, 2, 3, 4):
        ball = [(j) % n for j in range(-rho, rho + 1)]
        for t in (1, 2, 3, 4, 5):
            for trial in range(3):
                w = rng.integers(0, 2, size=len(ball))
                mask = np.all(bits[:, ball] == w[None, :], axis=1)
                v = np.where(mask, pi, 0.0)
                pw = v.sum()
                v = v / pw
                for _ in range(t):
                    v = v @ M
                cond = float((v * bits[:, 0]).sum())
                # begun from two different deterministic extensions of w
                begun = []
                for ext in range(2):
                    cfg = rng.integers(0, 2, size=n)
                    cfg[ball] = w
                    s = int((cfg << np.arange(n)).sum())
                    u = np.zeros(N); u[s] = 1.0
                    for _ in range(t):
                        u = u @ M
                    begun.append(float((u * bits[:, 0]).sum()))
                dev = max(abs(cond - begun[0]), abs(cond - begun[1]))
                inside = rho >= t
                if inside:
                    maxdev_inside = max(maxdev_inside, dev)
                else:
                    mindev_outside.append(dev)
                rows.append(dict(rho=rho, t=t, P_ball=pw, cond=cond, begun_ext1=begun[0], begun_ext2=begun[1],
                                 dev=dev, inside_light_cone=inside))
    uncond = float((pi * bits[:, 0]).sum())
    min_ball_prob = min(r["P_ball"] for r in rows)
    res = dict(n=n, stationary_residual=stat_resid, unconditional_P1=uncond,
               max_dev_inside_light_cone=maxdev_inside,
               max_dev_outside_light_cone=float(max(mindev_outside)),
               min_conditioning_probability=min_ball_prob,
               kappa_bound_check=bool(min_ball_prob >= 0.04 ** 9 - 1e-30),
               rows=rows[:12])
    res["PASS"] = bool(maxdev_inside < 1e-10 and max(mindev_outside) > 1e-4 and stat_resid < 1e-12)
    return res


# ---------------------------------------------------------------------------------------------
# Toom NEC helpers (2D torus, boolean, True = '+')
# ---------------------------------------------------------------------------------------------
def nec_step(x, eps, rng):
    e = np.roll(x, -1, axis=0)
    nn = np.roll(x, -1, axis=1)
    m = (x & e) | (x & nn) | (e & nn)
    if eps > 0:
        flip = rng.random(x.shape) < eps
        m ^= flip
    return m


def box_major(x, k):
    # fraction of + in the k x k box with lower-left corner at each site (torus), > 1/2 ?
    s = np.zeros(x.shape, dtype=np.int32)
    xi = x.astype(np.int32)
    row = np.zeros(x.shape, dtype=np.int32)
    for d in range(k):
        row += np.roll(xi, -d, axis=0)
    for d in range(k):
        s += np.roll(row, -d, axis=1)
    return s * 2 > k * k


# ---------------------------------------------------------------------------------------------
# X2. One rule, three readings (Toom NEC, symmetric noise): begun from product seeds, eternal
# phases, and the eternal fluctuation readout (plant = condition on a ball pattern; exact by X1).
# ---------------------------------------------------------------------------------------------
def x2():
    rng = np.random.default_rng(2)
    L, k, T = 128, 9, 400
    res = dict(L=L, k=k, T=T, begun={}, eternal={}, fluct={})
    for eps in (0.02, 0.05):
        row = {}
        for p in (0.1, 0.3, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0):
            vals = []
            for rep in range(3):
                x = rng.random((L, L)) < p
                for _ in range(T):
                    x = nec_step(x, eps, rng)
                vals.append(float(box_major(x, k).mean()))
            row[str(p)] = dict(mean=float(np.mean(vals)), reps=vals)
        res["begun"][str(eps)] = row
        # eternal: long runs from both frozen phases
        et = {}
        for name, init in (("plus", True), ("minus", False)):
            x = np.full((L, L), init)
            for _ in range(1500):
                x = nec_step(x, eps, rng)
            et[name] = dict(box_major=float(box_major(x, k).mean()), density_plus=float(x.mean()))
        res["eternal"][str(eps)] = et
    # fluctuation readout at eps = 0.05: eternal minus state, plant + squares, read center density at lag t
    eps = 0.05
    Lf = 256
    rho = 24
    spacing = 64
    centers = [(a, b) for a in range(32, Lf, spacing) for b in range(32, Lf, spacing)]
    fl = {}
    for t in (2, 5, 10, 20, 40, 80):
        cond_vals, begun_vals = [], []
        for rep in range(6):
            x = np.zeros((Lf, Lf), dtype=bool)
            for _ in range(300):
                x = nec_step(x, eps, rng)  # sample of the eternal minus phase
            for (a, b) in centers:
                x[a - rho:a + rho + 1, b - rho:b + rho + 1] = True  # condition on an all-+ ball
            y = np.ones((Lf, Lf), dtype=bool)  # begun from all + (the conditioned pattern extended)
            for _ in range(t):
                x = nec_step(x, eps, rng)
                y = nec_step(y, eps, rng)
            # safe zone: sites whose depth-t backward cone (NE-cone) lies in the planted ball
            for (a, b) in centers:
                lo, hi = -rho, rho - t
                if hi >= lo:
                    cond_vals.append(float(x[a + lo:a + hi + 1, b + lo:b + hi + 1].mean()))
                else:
                    cond_vals.append(float(x[a, b]))
            begun_vals.append(float(y.mean()))
        cv = float(np.mean(cond_vals)); bv = float(np.mean(begun_vals))
        fl[str(t)] = dict(fluct_readout=cv, begun_from_plus=bv, inside_light_cone=bool(t <= 2 * rho),
                          center_only=None)
    # also the center-site readout far outside the light cone (island erodes)
    far = {}
    for t in (60, 120, 240):
        vals = []
        for rep in range(4):
            x = np.zeros((Lf, Lf), dtype=bool)
            for _ in range(300):
                x = nec_step(x, eps, rng)
            for (a, b) in centers:
                x[a - rho:a + rho + 1, b - rho:b + rho + 1] = True
            for _ in range(t):
                x = nec_step(x, eps, rng)
            vals.append(float(np.mean([x[a, b] for (a, b) in centers])))
        far[str(t)] = float(np.mean(vals))
    res["fluct"] = dict(eps=eps, rho=rho, lags=fl, center_density_beyond_cone=far)
    # Finite numerical criteria on the tested density grid:
    b02 = res["begun"]["0.02"]; b05 = res["begun"]["0.05"]
    tested_density_grid_ok = all(b05[str(p)]["mean"] > 0.99 for p in (0.6, 0.7, 0.8, 0.9, 1.0)) and \
        all(b02[str(p)]["mean"] > 0.99 for p in (0.6, 0.7, 0.8, 0.9, 1.0))
    low_ok = all(b05[str(p)]["mean"] < 0.01 for p in (0.1, 0.3)) and all(b02[str(p)]["mean"] < 0.01 for p in (0.1, 0.3))
    eternal_ok = all(res["eternal"][e]["plus"]["box_major"] > 0.99 and res["eternal"][e]["minus"]["box_major"] < 0.01
                     for e in ("0.02", "0.05"))
    fl_ok = all(abs(fl[str(t)]["fluct_readout"] - fl[str(t)]["begun_from_plus"]) < 0.01 for t in (2, 5, 10, 20, 40))
    res["PASS"] = bool(tested_density_grid_ok and low_ok and eternal_ok and fl_ok)
    res["criteria"] = dict(tested_density_grid_ok=tested_density_grid_ok, low_ok=low_ok, eternal_ok=eternal_ok, fluct_matches_begun=fl_ok)
    return res


# ---------------------------------------------------------------------------------------------
# X3. D-cf numbers: the 3D stacked rule of Gacs Def. 1.5 (plane NEC majority, then an elementary
# rule on lines), cloned random row, product flip noise.  Clock-free decoder: m copies at FIXED
# a-spacing Ls, at times t >> Ls.  Measures pair-error correlation vs distance and decoder failure.
# ---------------------------------------------------------------------------------------------
def eca(rule, l, c, r):
    idx = (l.astype(np.int8) << 2) | (c.astype(np.int8) << 1) | r.astype(np.int8)
    table = np.array([(rule >> j) & 1 for j in range(8)], dtype=bool)
    return table[idx]


def stacked_step(x, rule, eps, rng):
    # x[a, b, i]
    e = np.roll(x, -1, axis=0)
    nn = np.roll(x, -1, axis=1)
    c = (x & e) | (x & nn) | (e & nn)
    y = eca(rule, np.roll(c, 1, axis=2), c, np.roll(c, -1, axis=2))
    if eps > 0:
        y ^= rng.random(x.shape) < eps
    return y


def x3():
    rng = np.random.default_rng(3)
    A, B, I = 60, 20, 64   # A >= m*Ls so that the m copies are distinct (an A=30 first run wrapped m=9, Ls=6)
    rule = 110
    out = {}
    for eps in (0.01, 0.03):
        row0 = rng.random(I) < 0.5
        ref = row0.copy()
        x = np.broadcast_to(row0[None, None, :], (A, B, I)).copy()
        err_series = []
        pair = {L: [] for L in (1, 2, 3, 4, 6, 8, 12, 15)}
        single = []
        dec = {(m, Ls): [] for (m, Ls) in ((5, 3), (9, 3), (5, 6), (9, 6))}
        Tmax = 400
        for t in range(1, Tmax + 1):
            x = stacked_step(x, rule, eps, rng)
            ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
            err = x ^ ref[None, None, :]
            if t % 50 == 0:
                err_series.append((t, float(err.mean())))
            if t >= 100 and t % 10 == 0:
                pe = float(err.mean())
                single.append(pe)
                for L in pair:
                    pair[L].append(float((err & np.roll(err, -L, axis=0)).mean()))
                for (m, Ls) in dec:
                    # majority over m copies at a-offsets 0, Ls, ..., (m-1)Ls for every base site
                    cnt = np.zeros(err.shape, dtype=np.int16)
                    for l in range(m):
                        cnt += np.roll(err, -l * Ls, axis=0)
                    dec[(m, Ls)].append(float((cnt * 2 > m).mean()))
        pe = float(np.mean(single))
        pr = {str(L): dict(P_both=float(np.mean(v)), ratio_to_square=float(np.mean(v)) / pe ** 2) for L, v in pair.items()}
        dd = {}
        for (m, Ls), v in dec.items():
            # independent-binomial prediction with the measured site error rate
            pred = sum(math.comb(m, j) * pe ** j * (1 - pe) ** (m - j) for j in range((m + 1) // 2, m + 1))
            dd[f"m={m},Ls={Ls}"] = dict(failure=float(np.mean(v)), independent_prediction=pred)
        out[str(eps)] = dict(site_error=pe, error_series=err_series, pair=pr, decoder=dd)
    # PASS: (i) site error stays bounded (no growth) ; (ii) pair ratio -> ~1 by L = 12 ; (iii) clock-free decoder failure
    # at t >> Ls is within a factor 3 of the independent prediction or below 1e-4, and decreases with m.
    ok = True
    for e, d in out.items():
        es = [v for _, v in d["error_series"]]
        ok &= max(es) < 3 * min(es) + 1e-3
        ok &= abs(d["pair"]["12"]["ratio_to_square"] - 1) < 0.25
        for key, v in d["decoder"].items():
            ok &= (v["failure"] < 3 * v["independent_prediction"] + 1e-4)
        ok &= d["decoder"]["m=9,Ls=6"]["failure"] <= d["decoder"]["m=5,Ls=6"]["failure"] + 1e-6
    return dict(A=A, B=B, I=I, rule=rule, results=out, PASS=bool(ok))


# ---------------------------------------------------------------------------------------------
# X4. Product seed versus cloned seed for a copy-majority decoder (Section 8's limited product-seed comparison).
# Two decoders at far-apart (a, b) offsets read the same row positions.  Cloned seed: they agree
# with each other and with the noiseless 1D trajectory.  Product seed: they agree only at the
# level of independent samples, and the decoded rows carry no common content.
# ---------------------------------------------------------------------------------------------
def x4():
    rng = np.random.default_rng(4)
    A, B, I = 40, 40, 64
    rule, eps, m, Ls, T = 110, 0.01, 9, 4, 60
    res = {}
    for seedtype in ("cloned", "product"):
        if seedtype == "cloned":
            row0 = rng.random(I) < 0.5
            x = np.broadcast_to(row0[None, None, :], (A, B, I)).copy()
        else:
            x = rng.random((A, B, I)) < 0.5
            row0 = x[0, 0, :].copy()
        ref = row0.copy()
        for t in range(T):
            x = stacked_step(x, rule, eps, rng)
            ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
        def decode(a0, b0):
            cnt = np.zeros(I, dtype=np.int16)
            for l in range(m):
                cnt += x[(a0 + l * Ls) % A, b0, :]
            return cnt * 2 > m
        d1 = decode(0, 0); d2 = decode(0, B // 2)
        res[seedtype] = dict(agree_two_decoders=float((d1 == d2).mean()),
                             agree_with_1D_trajectory=float((d1 == ref).mean()),
                             density_decoded=float(d1.mean()))
    res["PASS"] = bool(res["cloned"]["agree_two_decoders"] > 0.95 and res["cloned"]["agree_with_1D_trajectory"] > 0.95
                       and res["product"]["agree_with_1D_trajectory"] < 0.8)
    return res


# ---------------------------------------------------------------------------------------------
# X5. Does a symmetric product seed self-organize plane-wide redundancy?  Toom NEC from
# Bernoulli(1/2): correlation at distance r over time (coarsening would make it grow).
# ---------------------------------------------------------------------------------------------
def x5():
    rng = np.random.default_rng(5)
    L = 256
    res = {}
    for eps in (0.0, 0.02):
        x = rng.random((L, L)) < 0.5
        series = []
        t = 0
        for target in (1, 10, 100, 1000, 3000):
            while t < target:
                x = nec_step(x, eps, rng); t += 1
            s = x.astype(np.float64) * 2 - 1
            mbar = float(s.mean())
            cors = {}
            for r in (4, 16, 64):
                cors[str(r)] = float(((s * np.roll(s, r, axis=0)).mean() + (s * np.roll(s, r, axis=1)).mean()) / 2 - mbar ** 2)
            series.append(dict(t=t, magnetization=mbar, corr=cors))
        res[str(eps)] = series
    res["observation_recorded"] = True
    return res


# ---------------------------------------------------------------------------------------------
# X6. Self-organized redundancy from a PRODUCT seed: 3D stacked rule (plane NEC majority, then a
# line rule), product Bernoulli(1/2) seed, product flip noise.  Does the world clone itself along
# the redundancy planes?  Measures, over time, the fraction of agreeing sites in rows at (a, b) distance and
# the fraction of columns carrying the modal row.
# ---------------------------------------------------------------------------------------------
def x6():
    rng = np.random.default_rng(6)
    A, B, I = 48, 48, 24
    eps = 0.02
    res = {}
    for rule in (204, 110, 30):
        x = rng.random((A, B, I)) < 0.5
        t = 0
        series = []
        for target in (10, 100, 300, 1000, 3000):
            while t < target:
                x = stacked_step(x, rule, eps, rng); t += 1
            agree_far = float((x == np.roll(np.roll(x, A // 2, axis=0), B // 2, axis=1)).mean())
            rows = np.packbits(x.reshape(A * B, I), axis=1)
            _, counts = np.unique(rows, axis=0, return_counts=True)
            # modal row up to the noise: fraction of columns within Hamming distance 2 of the modal row
            modal = np.unpackbits(_[np.argmax(counts)])[:I].astype(bool)
            ham = (x.reshape(A * B, I) != modal[None, :]).sum(1)
            series.append(dict(t=t, row_agreement_half_torus=agree_far, frac_columns_near_modal_row=float((ham <= 2).mean()),
                               modal_row_density=float(modal.mean())))
        res[str(rule)] = series
    res["observation_recorded"] = True
    return res


# ---------------------------------------------------------------------------------------------
# X7. GUARDED stacking (the specific guard diagnostic): compute on a line only where the NEC triple already agrees
# on the whole line window, otherwise just keep the plane majority (identity).  From a cloned seed
# it must stay reliable; from a product seed: does it self-clone AND compute?
# ---------------------------------------------------------------------------------------------
def guarded_step(x, rule, eps, rng):
    e = np.roll(x, -1, axis=0)
    nn = np.roll(x, -1, axis=1)
    c = (x & e) | (x & nn) | (e & nn)
    agree = (x == e) & (x == nn)
    g = agree & np.roll(agree, 1, axis=2) & np.roll(agree, -1, axis=2)
    y = np.where(g, eca(rule, np.roll(c, 1, axis=2), c, np.roll(c, -1, axis=2)), c)
    if eps > 0:
        y ^= rng.random(x.shape) < eps
    return y, g


def x7():
    rng = np.random.default_rng(7)
    A, B, I = 48, 48, 24
    eps = 0.02
    res = {}
    for rule in (110, 30):
        # (i) cloned seed: reliability of the guarded computer
        row0 = rng.random(I) < 0.5
        ref = row0.copy()
        x = np.broadcast_to(row0[None, None, :], (A, B, I)).copy()
        errs = []
        for t in range(1, 1001):
            x, g = guarded_step(x, rule, eps, rng)
            ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
            if t % 100 == 0:
                errs.append((t, float((x ^ ref[None, None, :]).mean()), float(g.mean())))
        # (ii) product seed
        x = rng.random((A, B, I)) < 0.5
        t = 0
        series = []
        for target in (10, 100, 300, 1000, 3000, 6000):
            while t < target:
                x, g = guarded_step(x, rule, eps, rng); t += 1
            agree_far = float((x == np.roll(np.roll(x, A // 2, axis=0), B // 2, axis=1)).mean())
            flat = x.reshape(A * B, I)
            rows, counts = np.unique(np.packbits(flat, axis=1), axis=0, return_counts=True)
            modal = np.unpackbits(rows[np.argmax(counts)])[:I].astype(bool)
            near = (flat != modal[None, :]).sum(1) <= 2
            series.append(dict(t=t, row_agreement_half_torus=agree_far, frac_columns_near_modal_row=float(near.mean()),
                               guard_on_fraction=float(g.mean())))
        # (iii) is the self-cloned world computing?  follow the modal row for 50 more steps against the 1D rule
        flat = x.reshape(A * B, I)
        rows, counts = np.unique(np.packbits(flat, axis=1), axis=0, return_counts=True)
        modal = np.unpackbits(rows[np.argmax(counts)])[:I].astype(bool)
        ref = modal.copy()
        track = []
        for s in range(1, 51):
            x, g = guarded_step(x, rule, eps, rng)
            ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
            if s % 10 == 0:
                track.append((s, float((x == ref[None, None, :]).mean()), bool((ref != np.roll(ref, 1)).any())))
        res[str(rule)] = dict(cloned_seed_error=errs, product_seed=series, follows_1D_rule_after_selfcloning=track,
                              modal_row_1D_period=period_1d(rule, modal), modal_row=''.join('1' if b else '0' for b in modal))
    res["observation_recorded"] = True
    return res


# ---------------------------------------------------------------------------------------------
# X8. Is guarded self-cloning a coarsening (unbounded domain growth) or a finite-size consensus?
# Row agreement between columns at a-distance d on a large (a, b) torus, over time.
# ---------------------------------------------------------------------------------------------
def x8():
    rng = np.random.default_rng(8)
    A, B, I = 128, 128, 16
    res = {}
    for rule, eps in ((30, 0.02), (30, 0.01), (110, 0.01)):
        x = rng.random((A, B, I)) < 0.5
        t = 0
        series = []
        for target in (30, 100, 300, 1000, 3000):
            while t < target:
                x, g = guarded_step(x, rule, eps, rng); t += 1
            ag = {str(d): float((x == np.roll(x, d, axis=0)).mean()) for d in (2, 8, 32, 64)}
            series.append(dict(t=t, agreement=ag, guard_on=float(g.mean())))
        res[f"rule{rule}_eps{eps}"] = series
    res["observation_recorded"] = True
    return res


# ---------------------------------------------------------------------------------------------
# X9. Diagnostic for X7/X8: on a longer line (I = 48), is the coarsened guarded world computing a
# generic trajectory (long 1D period) or a selected short cycle?  And does a CLEAN cloned start
# keep its own trajectory (reliability) or collapse and re-organize?
# ---------------------------------------------------------------------------------------------
def period_1d(rule, row, tmax=20000):
    seen = {}
    r = row.copy()
    for t in range(tmax):
        key = r.tobytes()
        if key in seen:
            return t - seen[key], seen[key]
        seen[key] = t
        r = eca(rule, np.roll(r, 1), r, np.roll(r, -1))
    return None, None


def x9():
    rng = np.random.default_rng(9)
    A, B, I = 64, 64, 48
    rule, eps = 30, 0.01
    res = {}
    # clean cloned start
    row0 = rng.random(I) < 0.5
    ref = row0.copy()
    x = np.broadcast_to(row0[None, None, :], (A, B, I)).copy()
    clean = []
    for t in range(1, 601):
        x, g = guarded_step(x, rule, eps, rng)
        ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
        if t in (10, 25, 50, 100, 200, 400, 600):
            clean.append((t, float((x ^ ref[None, None, :]).mean()), float(g.mean())))
    res["clean_start_error_vs_own_trajectory"] = clean
    # naive (unguarded) stacking from the same clean start, for comparison
    ref = row0.copy()
    x = np.broadcast_to(row0[None, None, :], (A, B, I)).copy()
    naive = []
    for t in range(1, 601):
        x = stacked_step(x, rule, eps, rng)
        ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
        if t in (10, 50, 200, 600):
            naive.append((t, float((x ^ ref[None, None, :]).mean())))
    res["naive_clean_start_error"] = naive
    res["period_of_initial_row"] = period_1d(rule, row0)
    # product start, long run
    x = rng.random((A, B, I)) < 0.5
    prod = []
    for t in range(1, 4001):
        x, g = guarded_step(x, rule, eps, rng)
        if t in (100, 300, 1000, 2000, 4000):
            prod.append((t, float((x == np.roll(x, A // 2, axis=0)).mean()), float(g.mean())))
    res["product_start_agreement_half_torus"] = prod
    flat = x.reshape(A * B, I)
    rows, counts = np.unique(np.packbits(flat, axis=1), axis=0, return_counts=True)
    modal = np.unpackbits(rows[np.argmax(counts)])[:I].astype(bool)
    res["modal_row_share"] = float(counts.max() / (A * B))
    res["period_of_modal_row_trajectory"] = period_1d(rule, modal)
    ref = modal.copy(); track = []
    for s_ in range(1, 201):
        x, g = guarded_step(x, rule, eps, rng)
        ref = eca(rule, np.roll(ref, 1), ref, np.roll(ref, -1))
        if s_ % 50 == 0:
            track.append((s_, float((x == ref[None, None, :]).mean())))
    res["modal_row_tracking"] = track
    res["observation_recorded"] = True
    return res


if __name__ == "__main__":
    for name, fn in (("X1_exact_transfer", x1), ("X2_one_rule_three_readings", x2), ("X3_clock_free_decoder", x3),
                     ("X4_product_vs_cloned_decoder", x4), ("X5_symmetric_seed_coarsening", x5),
                     ("X6_product_seed_self_cloning", x6),
                     ("X7_guarded_self_cloning", x7),
                     ("X8_guarded_coarsening_scale", x8),
                     ("X9_guarded_diagnostic", x9)):
        t1 = time.time()
        log(f"== {name}")
        r = fn()
        if "PASS" in r:
            assert r["PASS"], name + ": finite numerical criterion failed"
        else:
            assert r.get("observation_recorded") is True
        r["seconds"] = round(time.time() - t1, 1)
        OUT[name] = r
        log(json.dumps({k: v for k, v in r.items() if k not in ("rows",)}, indent=1)[:4000])
        with open(os.path.join(HERE, "checks.json"), "w", newline="\n", encoding="utf-8") as f:
            json.dump(OUT, f, indent=1)
    OUT["ALL_PASS"] = all(v["PASS"] for v in OUT.values() if isinstance(v, dict) and "PASS" in v)
    OUT["total_seconds"] = round(time.time() - T0, 1)
    OUT["criterion_groups"] = 4
    OUT["observation_groups"] = 5
    assert sum("PASS" in v for v in OUT.values() if isinstance(v, dict)) == 4
    assert OUT["ALL_PASS"], "Fluctuation criterion failed"
    OUT["peak_working_set_bytes"] = None
    with open(os.path.join(HERE, "checks.json"), "w", newline="\n", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    log("ALL_PASS", OUT["ALL_PASS"], "total_seconds", OUT["total_seconds"], "peak", OUT.get("peak_working_set_bytes"))
