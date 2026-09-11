"""System C''' — the corrected UMACO-derived SAT algorithm, finite K, GPU only (CuPy).

Derived in UMACO_lineage_reconstruction.md Part IV (not yet published) and research/system_c/SystemC_3SAT_attack.md §6, §11:
  * pheromone tau in R^{n x 2}, sampling exponent alpha = 1, K-normalised deposit prop. to Q^{3/2},
    leaky integrator with evaporation rho, marginal floor (entropy floor) -- the pheromone integrates fitness;
  * per-variable independent construction from the marginals (Theorem 13: exact product wrong-set);
  * local search on the top 20% of the *current* samples: pick a random unsatisfied clause, flip the
    variable of smallest weighted break (System C' rule), F flips, fresh RNG each iteration;
  * clause weights: classical breakout -- raise the weight of clauses unsatisfied by the best-so-far
    assignment only when the best-so-far has not improved for P iterations; no decay;
  * stop when any ant satisfies every clause. Assignment is independently recounted on the host.
No PAQ tensors, economy or homology: no SAT implementation in the lineage ever connected them to the search.
"""
import sys, argparse, time
import cupy as cp
import numpy as np


def load_dimacs(path):
    clauses = []
    nvars = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line[0] in "c%":
                continue
            if line[0] == "p":
                nvars = int(line.split()[2]); continue
            lits = [int(t) for t in line.split()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if lits:
                clauses.append(lits)
    return nvars, clauses


def build(nvars, clauses):
    # pad to width 3 by repeating a literal (repetition is harmless for evaluation and break counts
    # are computed per distinct variable position below)
    V = np.zeros((len(clauses), 3), dtype=np.int32)
    S = np.zeros((len(clauses), 3), dtype=np.int8)
    for j, c in enumerate(clauses):
        c = (c + [c[-1]] * 3)[:3]
        for i, l in enumerate(c):
            V[j, i] = abs(l) - 1
            S[j, i] = 1 if l > 0 else 0
    # variable -> (clause, slot) incidence, CSR
    order = np.argsort(V.ravel(), kind="stable")
    inc_clause = (order // 3).astype(np.int32)
    inc_slot = (order % 3).astype(np.int32)
    counts = np.bincount(V.ravel(), minlength=nvars)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return (cp.asarray(V), cp.asarray(S), cp.asarray(inc_clause), cp.asarray(inc_slot), cp.asarray(ptr))


def solve(nvars, clauses, K=512, F=20, top=0.2, rho=0.1, floor=1e-3, lam=0.21015, P=10,
          max_iter=20000, seed=0, log=200, beta=0.05, elite=0):
    rng = cp.random.default_rng(seed)
    V, S, inc_clause, inc_slot, ptr = build(nvars, clauses)
    m = int(V.shape[0]); n = nvars
    # dense incidence for break counts: pad each variable's incidence list to max degree
    deg = cp.diff(ptr); maxdeg = int(deg.max())
    inc_c = cp.full((n, maxdeg), -1, dtype=cp.int32)
    inc_s = cp.zeros((n, maxdeg), dtype=cp.int32)
    idx = cp.arange(maxdeg)
    for v in range(0, n):  # host loop only at setup
        pass
    # vectorised fill
    lens = deg
    rowpos = cp.repeat(cp.arange(n), lens.tolist())
    colpos = cp.concatenate([cp.arange(int(l)) for l in lens.tolist()]) if n > 0 else cp.zeros(0, cp.int32)
    inc_c[rowpos, colpos] = inc_clause
    inc_s[rowpos, colpos] = inc_slot
    valid = inc_c >= 0

    tau = cp.full((n, 2), 0.5)
    w = cp.ones(m)
    best_unsat = m + 1; best_x = None; since_improve = 0
    ksel = max(1, int(top * K))
    t0 = time.time()

    def lit_true(X):               # X: (K,n) int8 -> (K,m,3) bool
        xv = X[:, V]               # (K,m,3)
        return cp.where(S[None, :, :] == 1, xv == 1, xv == 0)

    elites = None
    for it in range(max_iter):
        p = cp.clip(tau[:, 1] / (tau[:, 0] + tau[:, 1]), floor, 1 - floor)
        X = (rng.random((K, n)) < p[None, :]).astype(cp.int8)
        if elites is not None:
            X[:elites.shape[0]] = elites          # carry the best post-search assignments over
        L = lit_true(X)                           # (K,m,3)
        sat = L.any(2)                            # (K,m)
        unsat_cnt = (~sat).sum(1)                 # (K,)
        # --- local search on the top ksel ants (fewest unsatisfied clauses)
        sel = cp.argsort(unsat_cnt)[:ksel]
        Xs = X[sel].copy()
        for _ in range(F):
            Ls = lit_true(Xs); sats = Ls.any(2)   # (k,m)
            if bool(sats.all(1).any()):
                break
            uns = ~sats
            # random unsatisfied clause per ant
            r = rng.random((ksel, m)) * uns
            cj = cp.argmax(r, axis=1)             # (k,)  (ants with none unsat pick 0; masked below)
            has = uns.any(1)
            cand = V[cj]                          # (k,3) candidate variables
            # weighted break of each candidate: clauses where that variable is the unique true literal
            ntrue = Ls.sum(2)                     # (k,m)
            brk = cp.zeros((ksel, 3))
            for i in range(3):
                vv = cand[:, i]                   # (k,)
                cl = inc_c[vv]                    # (k,maxdeg)
                sl = inc_s[vv]
                vld = valid[vv]
                clc = cp.where(vld, cl, 0)
                rows = cp.arange(ksel)[:, None]
                crit = vld & (ntrue[rows, clc] == 1) & Ls[rows, clc, sl]
                brk[:, i] = (crit * w[clc]).sum(1)
            pick = cp.argmin(brk + rng.random((ksel, 3)) * 1e-6, axis=1)
            fv = cand[cp.arange(ksel), pick]
            rows = cp.arange(ksel)
            Xs[rows[has], fv[has]] = 1 - Xs[rows[has], fv[has]]
        X[sel] = Xs
        L = lit_true(X); sat = L.any(2); unsat_cnt = (~sat).sum(1)
        if elite > 0:
            elites = X[cp.argsort(unsat_cnt)[:elite]].copy()
        # --- best so far
        bi = int(cp.argmin(unsat_cnt)); bu = int(unsat_cnt[bi])
        if bu < best_unsat:
            best_unsat = bu; best_x = X[bi].copy(); since_improve = 0
        else:
            since_improve += 1
        if best_unsat == 0:
            break
        # --- weighted quality and pheromone deposit (K-normalised, Q^{3/2})
        # Boltzmann fitness on the weighted satisfied count: log g_v = beta * (weighted count difference),
        # so the pheromone log-odds move at rate beta * dQ/dp_v (count scale), matching the mean-field flow.
        cnt = (sat * w[None, :]).sum(1)
        Fq = cp.exp(beta * (cnt - cnt.max()))
        dep1 = (Fq[:, None] * X).sum(0) / K
        dep0 = (Fq[:, None] * (1 - X)).sum(0) / K
        tau = (1 - rho) * tau + cp.stack([dep0, dep1], 1)
        # --- breakout: raise weights of clauses violated by the best assignment when stalled
        if since_improve >= P:
            Lb = lit_true(best_x[None, :])[0]
            w = w + lam * (~Lb.any(1))
            since_improve = 0
        if it % log == 0:
            H = float(cp.mean(-(p * cp.log2(p) + (1 - p) * cp.log2(1 - p))))
            print(f"  it {it:6d}  best_unsat {best_unsat:5d}  mean_unsat {float(unsat_cnt.mean()):8.1f}  "
                  f"max_w {float(w.max()):6.1f}  entropy {H:.3f}  {time.time()-t0:6.1f}s", flush=True)
    return best_unsat, (cp.asnumpy(best_x) if best_x is not None else None), it


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf")
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--max_iter", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--elite", type=int, default=0)
    a = ap.parse_args()
    free, tot = cp.cuda.Device(0).mem_info
    print(f"GPU free {free//2**20} MiB of {tot//2**20}", file=sys.stderr)
    nvars, clauses = load_dimacs(a.cnf)
    print(f"{a.cnf}: n={nvars} m={len(clauses)} K={a.K} F={a.F}")
    bu, x, it = solve(nvars, clauses, K=a.K, F=a.F, max_iter=a.max_iter, seed=a.seed, floor=a.floor, P=a.P, beta=a.beta, elite=a.elite)
    # independent host recount
    if x is not None:
        xs = set(i + 1 if x[i] == 1 else -(i + 1) for i in range(nvars))
        rec = sum(1 for c in clauses if not any(l in xs for l in c))
        print(f"RESULT best_unsat={bu} iterations={it} host_recount_unsat={rec}")
