"""Exact quenched mean-field flow of System C / C''' on the GPU (CuPy only).

Integrates  p_v <- p_v + eta * p_v (1-p_v) * D_v(p, w)  with
D_v = (1/mean w) * sum_{j ∋ v} w_j * s_jv * prod_{other literals} P[false],
and the clause-weight dynamics.  No sampling, no benchmark: this evaluates the
deterministic K -> infinity limit derived in research/system_c/SystemC_3SAT_attack.md.

Weight rules:
  "C"   : S_j += lam if f_j > 0.999 else S_j = mu S_j + (1-mu) f_j   (the historical gate)
  "C3"  : S_j += lam * f_j  when the flow has stalled, plus EMA decay for satisfied clauses
"""
import sys, argparse
import cupy as cp

def planted_3sat(n, r, rng):
    m = int(r * n)
    xs = rng.integers(0, 2, n)
    V = cp.stack([cp.random.permutation(n)[:3] for _ in range(0)]) if False else None
    # vectorised distinct triples: sample and reject collisions
    V = rng.integers(0, n, (m, 3))
    bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    while bool(bad.any()):
        V[bad] = rng.integers(0, n, (int(bad.sum()), 3))
        bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    pats = cp.array([[(k >> i) & 1 for i in range(3)] for k in range(1, 8)])
    T = pats[rng.integers(0, 7, m)]
    S = cp.where(T == 1, xs[V], 1 - xs[V])
    return V, S, xs

def random_3sat(n, r, rng):
    m = int(r * n)
    xs = cp.zeros(n, dtype=cp.int64)          # unknown; placeholder for reporting
    V = rng.integers(0, n, (m, 3))
    bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    while bool(bad.any()):
        V[bad] = rng.integers(0, n, (int(bad.sum()), 3))
        bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    S = rng.integers(0, 2, (m, 3))
    return V, S, xs

def write_dimacs(path, V, S, n):
    Vh = cp.asnumpy(V); Sh = cp.asnumpy(S)
    with open(path, "w") as fh:
        fh.write("p cnf %d %d" % (n, Vh.shape[0]) + chr(10))
        for j in range(Vh.shape[0]):
            lits = [int(Vh[j, i] + 1) if Sh[j, i] == 1 else -int(Vh[j, i] + 1) for i in range(3)]
            fh.write(" ".join(map(str, lits)) + " 0" + chr(10))

def xorsat(n, r, rng):
    m = int(r * n)
    xs = rng.integers(0, 2, n)
    V = rng.integers(0, n, (m, 3))
    bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    while bool(bad.any()):
        V[bad] = rng.integers(0, n, (int(bad.sum()), 3))
        bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
    b = xs[V].sum(1) % 2
    A = cp.array([[(k >> i) & 1 for i in range(3)] for k in range(8)])           # (8,3)
    par = A.sum(1) % 2                                                              # (8,)
    keep = par[None, :] != b[:, None]                                               # (m,8) forbidden
    Vs = cp.repeat(V, 8, axis=0)[keep.ravel()]
    Ss = (1 - cp.tile(A, (m, 1)))[keep.ravel()]
    return Vs, Ss, xs

def flow(V, S, xs, eta=0.05, steps=4000, noise=1e-3, rule="C", lam=0.21015, mu=0.87959,
         report=500, rng=None, stall_tol=1e-4, floor=1e-9):
    n = int(xs.size); m = int(V.shape[0])
    Sj = cp.zeros(m); w = cp.ones(m)
    p = cp.clip(0.5 + noise * rng.standard_normal(n), 1e-6, 1 - 1e-6)
    hist = []; last_sumf = None; stall_count = 0
    for t in range(steps):
        pv = p[V]
        q = cp.where(S == 1, 1 - pv, pv)
        f = q[:, 0] * q[:, 1] * q[:, 2]
        D = cp.zeros(n)
        for i in range(3):
            others = cp.ones(m)
            for k in range(3):
                if k != i:
                    others = others * q[:, k]
            sgn = cp.where(S[:, i] == 1, 1.0, -1.0)
            cp.add.at(D, V[:, i], w * sgn * others)
        D = D / w.mean()
        p = cp.clip(p + eta * p * (1 - p) * D, floor, 1 - floor)
        sumf = float(f.sum())
        if rule == "C":
            frozen = f > 0.999
            Sj = cp.where(frozen, Sj + lam, mu * Sj + (1 - mu) * f)
        elif rule == "C3":
            stalled = last_sumf is not None and abs(last_sumf - sumf) < stall_tol
            Sj = cp.where(f > 0.5, Sj + (lam * f if stalled else 0.0), mu * Sj)
        elif rule == "breakout":
            Sj = cp.where(f > 0.5, Sj + lam * f, mu * Sj)
        elif rule == "breakout_nodecay":
            Sj = Sj + lam * f
        elif rule == "breakout_stall":
            # classical breakout: raise weights of violated clauses only at a (numerical) local minimum
            if last_sumf is not None and abs(last_sumf - sumf) < stall_tol:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= 20:
                Sj = cp.where(f > 0.5, Sj + lam, Sj)
                stall_count = 0
        w = 1 + 5 * Sj
        last_sumf = sumf
        if t % report == 0 or t == steps - 1:
            x = (p > 0.5).astype(cp.int32)
            d = float((x != xs).mean())
            xv = x[V]; lit = cp.where(S == 1, xv, 1 - xv)
            unsat = int((lit.max(1) == 0).sum())
            hist.append((t, round(d, 4), unsat, round(sumf, 2), round(float(w.max()), 1)))
            if unsat == 0:
                break
    return hist, p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="planted", choices=["planted", "xor", "random"])
    ap.add_argument("--dimacs", default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--r", type=float, default=4.26)
    ap.add_argument("--rule", default="C3")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", type=int, default=500)
    ap.add_argument("--floor", type=float, default=1e-9)
    a = ap.parse_args()
    free, tot = cp.cuda.Device(0).mem_info
    print(f"GPU free {free // 2**20} MiB of {tot // 2**20}", file=sys.stderr)
    rng = cp.random.default_rng(a.seed)
    V, S, xs = {"planted": planted_3sat, "xor": xorsat, "random": random_3sat}[a.family](a.n, a.r, rng)
    if a.dimacs:
        write_dimacs(a.dimacs, V, S, a.n)
    print(f"{a.family} n={a.n} r={a.r} m={int(V.shape[0])} rule={a.rule}  (t, frac wrong, #unsat, sum f, max w)")
    hist, p = flow(V, S, xs, steps=a.steps, noise=a.noise, rule=a.rule, report=a.report, rng=rng, floor=a.floor)
    for h in hist:
        print("  ", h)
    x = cp.asnumpy((p > 0.5).astype(cp.int32))
    if a.dimacs:
        with open(a.dimacs + ".assignment", "w") as fh:
            fh.write(" ".join(str(i + 1) if x[i] == 1 else str(-(i + 1)) for i in range(a.n)) + chr(10))
