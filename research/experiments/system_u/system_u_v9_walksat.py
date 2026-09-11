"""
System U on GPU: UMACO13's coupled loop specialised to SAT top-down, as settled in docs/system-u/system_U.md.

File state: v9 of research/experiments/system_u/SystemU_runs.md, i.e. MUTATIONS M1-M13 applied. M13 makes
the agents' refinement a per-agent WalkSAT-SKC CUDA kernel (the default, --refine kernel), so this is
System U plus a WalkSAT local search, not a clean implementation of the definition.

Everything numeric is CuPy. Nothing is borrowed from macov8 / System C except the DIMACS format and
the host recount of a claimed solution.

Components (section numbers refer to docs/system-u/system_U.md):
  §1  literal graph, complex field Phi = R + iM in C^{2n x 2n} (+ start row), walk with free order,
      interference weight w = eps + max(0, R^2 - M^2), dynamic Jeroslow-Wang eta, choice ∝ w^a_w eta^beta
  §2  Perf = satisfied fraction, target 0.7
  §3  panic map (per literal) from the discrete flip gradient; §4 anxiety map (per literal), complex
  §5  five-regime burst: SVD of R, rank k = dim//4, rotated by e^{i angle(Psi_a + Psi_b)}, scaled by panic
  §6  persistent homology (H0 of the Rips filtration of d = max R - R): persistent entropy H_p, mean persistence
  §7  covariant momentum, purely imaginary, from mean persistence; Phi += alpha * p_cov
  §8  alpha (deposit / momentum scale) from P, Psi; beta from H_p; rho from |p_cov|
  §9  deposit along literal paths (real), evaporation on both channels, global rescale
  §10 symmetrize, non-negativity on both channels, no diagonal zeroing; reset ladder (raise weakest 30 % / 95 %)
  §11 entropy controller: noise into M when H_p strays from target
  §12 economy: tokens buy search budget (refinement flips + walk temperature)
  §13 the loop; stop on host-recounted solution

Deviations from the letter of the docs are listed in DEVIATIONS below; there are no silent ones.
"""
import argparse, json, math, sys, time
import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy import sparse as cps

DEVIATIONS = [
    "D1 alpha: the walk exponent alpha_w is a fixed config (default 1.0); the adaptive alpha of §8 scales deposit "
    "intensity and the momentum step only (the two roles of the single symbol in Umaco13 conflict).",
    "D2 panic per literal = tanh(k_P * |Delta_v| / mean clause degree * log(1+|Psi_l|)), Delta_v the population-"
    "mean flip gradient of var(l); edge panic for the burst = sqrt(P_a P_b).",
    "D3 anxiety per literal: Psi_re from the mean performance of the agents that used the literal (population "
    "mean if unused); Psi_im from the global stagnation counter; homology mean persistence enters Psi_re uniformly.",
    "D4 burst scale g_ab = gamma_s * min(2 P_ab, 1.5) (gamma_s = 0.7, gamma_r = 0.3 as in Umaco13); with U3b this "
    "means erasure needs P_ab >= 0.505 (the cascade of Q15).",
    "D5 edge phase = angle(Psi_a + Psi_b).",
    "D6 homology: H0 only (H1 is unwired by the thesis); distances normalised by max R so persistence is in [0,1].",
    "D7 refinement flips = descent on the panic map's flip gradient: random unsatisfied clause, flip its variable "
    "with the largest make - break.",
    "D8 economy -> budget: purchase success gives temperature 0.7 and 2x flips, failure temperature 1.0 and 0.5x.",
    "D9 rho = clip(rho0 * exp(-|p_cov| / p_ref), 0.05, 0.3); beta = beta_min + (beta_max - beta_min) * H_p.",
]

MUTATIONS = [   # after integrated runs; see research/experiments/system_u/SystemU_runs.md
    "M1 (run v0, random n=200): panic uses |Delta_v| in clause units (no degree normalisation); bursts were inert.",
    "M2 (run v0): uniform imaginary momentum is a global threshold under the interference weight and, with rho "
    "driven to its floor by |p_cov|, erased every edge by iteration 20. Now the momentum's equilibrium repulsion is "
    "theta_m * |p|/(|p|+p_sat) * max R (theta_m = 0.5), written as rho * target per iteration; rho floor 0.05.",
    "M3 (run v0): Perf = fraction of the random baseline's violations removed, 1 - unsat/(m/8), clipped to [0,1]; "
    "the satisfied fraction is always above 0.7 on 3-SAT (random = 0.875), so the phase was pinned at pi.",
    "M4 (run v0): deposit along the refined path (the walked path with flipped literals negated), so the path "
    "deposited is the assignment scored.",
    "M5 (run v1, planted n=400 s1): |make - break| is nearly uniform near a solution, so the per-edge burst scale "
    "could not localise and every burst thresholded the whole mode (a restart) at best = 1. Panic magnitude is now "
    "the population-mean frustration: unsatisfied clauses at v plus half the clause-neighbours'.",
    "M6 (run v1): field cap 10 -> 100; with deposit/rho ~ 30 on a mode edge the cap rescale was an extra "
    "evaporation on every edge every iteration.",
    "M7 (run v2, random n=200): with perf^2 an agent one clause better than the mode deposits 2% more; the field's "
    "selection per clause is eta_dep*8/m -> 0. Option kappa_sel > 0 makes the deposit exp(-kappa (unsat - min unsat)): "
    "per-clause selection relative to the iteration's best (U2: a per-iteration normaliser is a constant on the "
    "learning rate). First attempt perf^(kappa m/8) without the normaliser underflowed to a zero field.",
    "M8 (run v3, random n=200 held at 3): refinement was greedy, so every agent descended into the basin the field "
    "reproduces. The economy temperature now applies to the refinement: with probability temp - 0.5 (0.2 buyers, "
    "0.5 non-buyers) an agent flips a random variable of the chosen unsatisfied clause instead of the greedy one. "
    "This is the WalkSAT noise rule inside the agent's budget; named as such. Base flips 10 -> 30.",
    "M9 (run v4, random n=200 pinned at 1): edge panic sqrt(P_a P_b) gave ~0.18 on edges between a frustrated literal "
    "and a normal one, so the localised burst never reached the mode's transitions into the frustrated variables. "
    "Edge panic is now max(P_a, P_b).",
    "M10 (run v4): the reset raised weak edges to the median positive attraction, which is invisible under "
    "w = max(0, R^2 - M^2) once repulsion has accumulated. Raised edges now go to max(baseline, M_ab + 0.2 max R).",
    "M11 (run v5): with an absolute floor eps = 1e-4 and R ~ 10 on the mode, the walk's log-odds gap between a mode "
    "edge and any alternative is log(R^2/eps) ~ 14 nats: the converged field is a deterministic policy, and a burst "
    "that only scales weights (any g below erasure) cannot move it. Option eps_rel > 0 sets eps = eps_rel * max(R)^2, "
    "bounding the gap at log(1/eps_rel) (U1c: the floor is required; its scale is the walk's exploration rate).",
    "M12 (run v7, pinned at 1-3 with a few deviations per walk): every iteration restarts the local search at the "
    "mode with a 15-60 flip budget, so 128 agents re-explore the same small neighbourhood; escaping a 1-unsat basin "
    "needs a long trajectory through worse states. The budget now grows with chronic anxiety (|Psi_im|): "
    "flips = flips_base * (1 + budget_gain * mean|Psi_im|), capped at flips_max; the economy's x2 / x0.5 on top. "
    "Continued stagnation buys compute, which is what tokens buy in the thesis.",
    "M13 (satisfiable random n=200 seed 5002, held at 1 with 1000-flip budgets; plain WalkSAT-SKC needs 3e4-3e5 "
    "continuous flips): the refinement is now WalkSAT-SKC per agent in a single-thread-per-agent CUDA kernel "
    "(min-break with freebies, noise 0.3 for agents that bought budget, 0.6 otherwise), so the economy's budget "
    "is real compute: tens of thousands of sequential flips per agent per iteration. Field, walk, crisis, burst, "
    "homology, momentum, reset, economy unchanged. Named: the agents' local search is WalkSAT.",
]


# ----------------------------------------------------------------------------- instances
def load_dimacs(path):
    clauses, nvars = [], 0
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


def gen_3sat(n, m, seed, planted):
    """Random 3-SAT with distinct variables per clause; if planted, every clause is satisfied by a hidden x*."""
    rng = cp.random.default_rng(seed)
    V = cp.zeros((m, 3), dtype=cp.int32)
    for j in range(3):
        V[:, j] = rng.integers(0, n, size=m)
    # resample rows with repeated variables
    for _ in range(100):
        bad = (V[:, 0] == V[:, 1]) | (V[:, 0] == V[:, 2]) | (V[:, 1] == V[:, 2])
        nb = int(bad.sum())
        if nb == 0:
            break
        V[bad] = rng.integers(0, n, size=(nb, 3))
    S = rng.integers(0, 2, size=(m, 3)).astype(cp.int8)
    xstar = None
    if planted:
        xstar = rng.integers(0, 2, size=n).astype(cp.int8)
        for _ in range(100):
            sat = ((S == xstar[V]).any(1))
            nb = int((~sat).sum())
            if nb == 0:
                break
            S[~sat] = rng.integers(0, 2, size=(nb, 3)).astype(cp.int8)
    return V, S, xstar


def is_satisfiable(V, S, timeout_s=600):
    """Complete check with CaDiCaL on the host (an instance-preparation step, not part of the solver).
    Returns True/False, or None if the check did not finish in time."""
    from pysat.solvers import Cadical153
    Vh, Sh = cp.asnumpy(V), cp.asnumpy(S)
    cl = [[int(v + 1) if sg == 1 else -int(v + 1) for v, sg in zip(Vh[j], Sh[j])] for j in range(Vh.shape[0])]
    with Cadical153(bootstrap_with=cl) as so:
        return so.solve()


def gen_3sat_sat(n, m, seed, planted, max_tries=50):
    """gen_3sat, but for the random family keep trying seeds (seed, seed+1000, ...) until CaDiCaL says SAT.
    Returns V, S, xstar, seed_used."""
    if planted:
        V, S, x = gen_3sat(n, m, seed, True); return V, S, x, seed
    for k in range(max_tries):
        sd = seed + 1000 * k
        V, S, x = gen_3sat(n, m, sd, False)
        if is_satisfiable(V, S):
            return V, S, x, sd
    raise RuntimeError("no satisfiable instance found")


def host_recount(V, S, x):
    """Independent CPU check of a claimed solution (the recount discipline)."""
    Vh, Sh, xh = cp.asnumpy(V), cp.asnumpy(S), cp.asnumpy(x)
    return int((~(Sh == xh[Vh]).any(1)).sum())



# ----------------------------------------------------------------------------- M13: per-agent WalkSAT kernel
_WALKSAT_SRC = r"""
extern "C" __global__
void walksat(const int* __restrict__ V, const signed char* __restrict__ S,
             const int* __restrict__ occ_ptr, const int* __restrict__ occ_cl, const int* __restrict__ occ_sl,
             const int n, const int m, const int K,
             signed char* __restrict__ X, const int* __restrict__ budget, const float* __restrict__ noise,
             unsigned long long* __restrict__ seeds, int* __restrict__ out_unsat, int* __restrict__ out_flips,
             int* __restrict__ nt_buf, int* __restrict__ ul_buf, int* __restrict__ up_buf)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    signed char* x = X + (size_t)k * n;
    int* nt = nt_buf + (size_t)k * m;
    int* ul = ul_buf + (size_t)k * m;
    int* up = up_buf + (size_t)k * m;
    int nu = 0;
    for (int j = 0; j < m; ++j) {
        int c = 0;
        for (int s = 0; s < 3; ++s) c += (S[j*3+s] == x[V[j*3+s]]);
        nt[j] = c;
        if (c == 0) { ul[nu] = j; up[j] = nu; nu++; } else up[j] = -1;
    }
    unsigned long long st = seeds[k] | 1ull;
    const float pn = noise[k];
    const int B = budget[k];
    int f = 0;
    for (; f < B && nu > 0; ++f) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        int j = ul[(int)(st % (unsigned long long)nu)];
        int v;
        // SKC: break counts of the three variables
        int brk[3]; int bmin = 1 << 30;
        for (int s = 0; s < 3; ++s) {
            int v0 = V[j*3+s]; int b = 0;
            for (int t = occ_ptr[v0]; t < occ_ptr[v0+1]; ++t) {
                int jj = occ_cl[t];
                if (nt[jj] == 1 && S[jj*3+occ_sl[t]] == x[v0]) b++;
            }
            brk[s] = b; if (b < bmin) bmin = b;
        }
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        float u = (float)(st >> 40) / 16777216.0f;
        if (bmin > 0 && u < pn) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            v = V[j*3 + (int)(st % 3ull)];
        } else {
            int cnt = 0; int pick = 0;
            for (int s = 0; s < 3; ++s) if (brk[s] == bmin) cnt++;
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            int r = (int)(st % (unsigned long long)cnt);
            for (int s = 0; s < 3; ++s) if (brk[s] == bmin) { if (r == 0) { pick = s; break; } r--; }
            v = V[j*3 + pick];
        }
        // flip v
        signed char xv = x[v];
        for (int t = occ_ptr[v]; t < occ_ptr[v+1]; ++t) {
            int jj = occ_cl[t];
            if (S[jj*3+occ_sl[t]] == xv) {
                nt[jj]--;
                if (nt[jj] == 0) { ul[nu] = jj; up[jj] = nu; nu++; }
            } else {
                nt[jj]++;
                if (nt[jj] == 1) { int pos = up[jj]; int last = ul[nu-1]; ul[pos] = last; up[last] = pos; up[jj] = -1; nu--; }
            }
        }
        x[v] = 1 - xv;
    }
    seeds[k] = st;
    out_unsat[k] = nu;
    out_flips[k] = f;
}
"""
_walksat_kernel = None


def walksat_kernel():
    global _walksat_kernel
    if _walksat_kernel is None:
        _walksat_kernel = cp.RawKernel(_WALKSAT_SRC, "walksat")
    return _walksat_kernel


# ----------------------------------------------------------------------------- System U
class SystemU:
    def __init__(self, V, S, n, K=128, seed=0, alpha_w=1.0, eps=1e-4, init_val=0.3,
                 perf_target=0.7, eta_dep=2.0, homology_every=1, burst_min_gap=10,
                 burst_interval=100, reset_mild=40, reset_hard=100, flips_base=10,
                 target_entropy=0.7, kappa_sel=0.0, eps_rel=0.0, budget_gain=0.0, flips_max=1000,
                 refine_mode="kernel", log=None):
        self.V, self.S, self.n, self.K = V, S, n, K
        self.m = V.shape[0]
        self.N = 2 * n                         # literals: lit = 2*var + pol (pol 1 = positive)
        self.rng = cp.random.default_rng(seed)
        self.np_rng = np.random.default_rng(seed)
        self.alpha_w, self.eps, self.perf_target, self.eta_dep = alpha_w, eps, perf_target, eta_dep
        self.kappa_sel = kappa_sel   # M7: per-clause selection, deposit ~ exp(-kappa (unsat - min unsat))
        self.eps_rel = eps_rel       # M11: floor relative to the field, eps = eps_rel * max(R)^2 (0 = absolute eps)
        self.budget_gain, self.flips_max = budget_gain, flips_max   # M12
        self.homology_every, self.burst_min_gap, self.burst_interval = homology_every, burst_min_gap, burst_interval
        self.reset_mild_T, self.reset_hard_T, self.flips_base = reset_mild, reset_hard, flips_base
        self.target_entropy = target_entropy
        self.log = log
        m, n, N = self.m, n, self.N
        # literal of each clause slot; incidence (3m x N) as csr, transposed for eta scatter
        self.L = (2 * V + S.astype(cp.int32)).astype(cp.int32)          # (m,3) literal ids
        rows = cp.arange(3 * m, dtype=cp.int32)
        A = cps.csr_matrix((cp.ones(3 * m, dtype=cp.float32), (rows, self.L.ravel())), shape=(3 * m, N))
        self.A_T = A.T.tocsr()                                            # (N x 3m)
        # variable -> clause incidence, padded to maxdeg; pad index = m (dummy clause)
        Vh = cp.asnumpy(V)
        deg = np.bincount(Vh.ravel(), minlength=n)
        self.maxdeg = int(deg.max())
        inc_c = np.full((n, self.maxdeg), m, dtype=np.int32)
        inc_s = np.zeros((n, self.maxdeg), dtype=np.int32)
        fill = np.zeros(n, dtype=np.int64)
        for j in range(m):
            for s in range(3):
                v = Vh[j, s]; inc_c[v, fill[v]] = j; inc_s[v, fill[v]] = s; fill[v] += 1
        self.inc_c, self.inc_s = cp.asarray(inc_c), cp.asarray(inc_s)
        order = np.argsort(Vh.ravel(), kind="stable")
        self.occ_cl = cp.asarray((order // 3).astype(np.int32)); self.occ_sl = cp.asarray((order % 3).astype(np.int32))
        self.occ_ptr = cp.asarray(np.concatenate([[0], np.cumsum(deg)]).astype(np.int32))
        self.ws_seeds = cp.asarray(np.random.default_rng(seed + 7).integers(1, 2**62, size=K, dtype=np.int64)).view(cp.uint64)
        self.ws_nt = cp.zeros((K, m), dtype=cp.int32); self.ws_ul = cp.zeros((K, m), dtype=cp.int32)
        self.ws_up = cp.zeros((K, m), dtype=cp.int32)
        self.refine_mode = refine_mode
        self.inc_valid = self.inc_c < m
        self.mean_deg = float(deg.mean())
        # ---- field §1 / §10
        self.R = cp.full((N, N), init_val, dtype=cp.float32)
        self.M = cp.zeros((N, N), dtype=cp.float32)
        self.Rs = cp.full(N, init_val, dtype=cp.float32)                  # start row
        self.Ms = cp.zeros(N, dtype=cp.float32)
        self.field_cap = 100.0   # M6
        # ---- crisis maps §3 §4
        self.P = cp.full(N, 0.1, dtype=cp.float32)                        # panic per literal
        self.Psi = cp.full(N, 0.1 + 0.0j, dtype=cp.complex64)            # anxiety per literal (trauma 0.1)
        self.delta_P, self.k_P = 0.15, 1.0
        self.delta_A, self.k_A, self.tau_s = 0.15, 5.0, 20.0
        self.theta_P, self.theta_Psi = 0.5, 0.6
        # ---- homology / momentum §6 §7
        self.H_p, self.mean_pers = 1.0, 0.0
        self.p_cov = 0.01j
        self.kappa_mom = 0.1
        self.theta_m, self.p_sat = 0.5, 0.5
        # ---- hyperparameters §8
        self.alpha, self.alpha_min, self.alpha_max = 1.0, 0.5, 3.0
        self.beta, self.beta_min, self.beta_max = 2.4, 0.5, 2.4
        self.rho, self.rho0, self.p_ref = 0.14, 0.14, 0.05
        # ---- economy §12
        self.tokens = np.full(K, 250.0)
        self.node_panic = np.full(K, 0.2)
        self.risk = 0.8
        self.market = 1.0
        self.prev_perf = None
        self.temp = np.ones(K, dtype=np.float32)
        self.flips = np.full(K, flips_base, dtype=np.int64)
        # ---- state
        self.best_unsat, self.best_x = m + 1, None
        self.stag, self.since_burst, self.bursts_since_improve = 0, burst_min_gap, 0
        self.it = 0
        self.events = []
        self.stats = []

    # ------------------------------------------------------------------ §1 walk
    def walk(self):
        K, n, N, m = self.K, self.n, self.N, self.m
        rows = cp.arange(K)
        eps = self.eps if self.eps_rel <= 0 else max(self.eps, self.eps_rel * float(self.R.max()) ** 2)
        self.eps_now = eps
        w = eps + cp.maximum(0.0, self.R * self.R - self.M * self.M)        # (N,N)
        logw = self.alpha_w * cp.log(w)
        logw_s = self.alpha_w * cp.log(eps + cp.maximum(0.0, self.Rs ** 2 - self.Ms ** 2))
        temp = cp.asarray(self.temp)[:, None]
        assigned = cp.zeros((K, n), dtype=cp.bool_)
        sat = cp.zeros((K, m + 1), dtype=cp.bool_)
        free = cp.full((K, m + 1), 3, dtype=cp.int32)
        path = cp.zeros((K, n), dtype=cp.int32)
        cur = None
        for t in range(n):
            # eta (JW): 1 + sum over unsatisfied clauses containing lit of 2^{-(free-1)}
            contrib = cp.where(sat[:, :m], 0.0, cp.exp2(-(free[:, :m].astype(cp.float32) - 1.0)))  # (K,m)
            c3 = cp.repeat(contrib, 3, axis=1)                                                       # (K,3m) slot order j*3+s
            eta = 1.0 + (self.A_T @ c3.T).T                                                          # (K,N)
            base = logw_s[None, :] if cur is None else logw[cur]                                     # (K,N)
            logits = (base + self.beta * cp.log(eta)) / temp
            mask = cp.repeat(assigned, 2, axis=1)                                                    # literal assigned?
            logits = cp.where(mask, -cp.inf, logits)
            g = -cp.log(-cp.log(self.rng.random((K, N), dtype=cp.float32) + 1e-12) + 1e-12)
            b = cp.argmax(logits + g, axis=1).astype(cp.int32)
            path[:, t] = b
            v, pol = b // 2, b % 2
            assigned[rows, v] = True
            clc, sl, vld = self.inc_c[v], self.inc_s[v], self.inc_valid[v]                          # (K,maxdeg)
            cupyx.scatter_add(free, (rows[:, None], clc), -vld.astype(cp.int32))
            hit = vld & (self.S[clc, sl] == pol[:, None].astype(cp.int8))
            sat_hit = cp.zeros((K, m + 1), dtype=cp.int32)
            cupyx.scatter_add(sat_hit, (rows[:, None], clc), hit.astype(cp.int32))
            sat |= sat_hit > 0
            cur = b
        x = cp.zeros((K, n), dtype=cp.int8)
        x[rows[:, None], path // 2] = (path % 2).astype(cp.int8)
        return path, x

    # ------------------------------------------------------------------ evaluation §2 and flip gradient §3
    def evaluate(self, x):
        lt = (self.S[None, :, :] == x[:, self.V])          # (K,m,3) literal true
        ntrue = lt.sum(2)                                   # (K,m)
        unsat = (ntrue == 0).sum(1)                         # (K,)
        return lt, ntrue, unsat

    def flip_gradient(self, lt, ntrue):
        """Delta_v = loss(x xor e_v) - loss(x) = break_v - make_v, per agent per variable."""
        K = self.K
        brk_slot = ((ntrue == 1)[:, :, None] & lt).astype(cp.float32).reshape(K, -1)   # unique true literal
        mk_slot = cp.repeat((ntrue == 0).astype(cp.float32), 3, axis=1)                # clause unsatisfied
        Vflat = self.V.ravel()
        brk = cp.zeros((K, self.n), dtype=cp.float32); mk = cp.zeros((K, self.n), dtype=cp.float32)
        cupyx.scatter_add(brk, (cp.arange(K)[:, None], Vflat[None, :]), brk_slot)
        cupyx.scatter_add(mk, (cp.arange(K)[:, None], Vflat[None, :]), mk_slot)
        return brk, mk

    def frustration(self, mk):
        """M7: population-mean frustration per variable: unsatisfied clauses at v plus half of the
        clause-neighbours' (one step through the factor graph)."""
        K = self.K
        cs = mk[:, self.V].sum(2)                                        # (K,m) sum of make over the clause's vars
        others = cs[:, :, None] - mk[:, self.V]                          # (K,m,3) the other two vars' make
        nb = cp.zeros((K, self.n), dtype=cp.float32)
        cupyx.scatter_add(nb, (cp.arange(K)[:, None], self.V.ravel()[None, :]), others.reshape(K, -1))
        deg = cp.maximum(self.inc_valid.sum(1).astype(cp.float32), 1.0)  # (n,)
        return (mk + 0.5 * nb / deg[None, :]).mean(0)                    # (n,)

    # ------------------------------------------------------------------ §12 refinement flips (budget)
    def refine(self, x, flips):
        if self.refine_mode == "kernel":
            return self.refine_kernel(x, flips)
        return self.refine_py(x, flips)

    def refine_kernel(self, x, flips):
        """M13: WalkSAT-SKC per agent, one thread per agent, sequential flips in one kernel call.
        Budget and noise are the agent's (economy / anxiety). x is modified in place."""
        K = self.K
        x = cp.ascontiguousarray(x)
        budget = cp.asarray(flips.astype(np.int32))
        noise = cp.asarray(np.clip(self.temp - 0.5, 0.2, 0.6).astype(np.float32) + 0.1)   # 0.3 buyers, 0.6 others
        out_u = cp.zeros(K, dtype=cp.int32); out_f = cp.zeros(K, dtype=cp.int32)
        walksat_kernel()((int((K + 63) // 64),), (64,),
                         (self.V, self.S, self.occ_ptr, self.occ_cl, self.occ_sl, np.int32(self.n), np.int32(self.m),
                          np.int32(K), x, budget, noise, self.ws_seeds, out_u, out_f, self.ws_nt, self.ws_ul, self.ws_up))
        self.last_flips = int(out_f.sum())
        return x

    def refine_py(self, x, flips):
        K, n, m = self.K, self.n, self.m
        rows = cp.arange(K)
        fl = cp.asarray(flips)
        for f in range(int(flips.max())):
            active = fl > f
            lt, ntrue, unsat = self.evaluate(x)
            uns = ntrue == 0
            active &= unsat > 0
            if not bool(active.any()):
                break
            r = self.rng.random((K, m), dtype=cp.float32) * uns
            cj = cp.argmax(r, axis=1)
            cand = self.V[cj]                                # (K,3)
            score = cp.zeros((K, 3), dtype=cp.float32)      # make - break
            for i in range(3):
                vv = cand[:, i]
                clc, sl, vld = self.inc_c[vv], self.inc_s[vv], self.inc_valid[vv]
                ntc = ntrue[rows[:, None], cp.minimum(clc, m - 1)]
                ltc = lt[rows[:, None], cp.minimum(clc, m - 1), sl]
                brk = (vld & (ntc == 1) & ltc).sum(1)
                mk = (vld & (ntc == 0)).sum(1)
                score[:, i] = mk - brk
            pick = cp.argmax(score + 1e-3 * self.rng.random((K, 3), dtype=cp.float32), axis=1)
            # M8: the economy temperature applies to the refinement too: with probability noise_k the agent
            # flips a random variable of the clause instead of the greedy one (its budgeted deviation)
            noise = cp.asarray(np.clip(self.temp - 0.5, 0.1, 0.6), dtype=cp.float32)
            rnd = self.rng.integers(0, 3, size=K)
            pick = cp.where(self.rng.random(K, dtype=cp.float32) < noise, rnd, pick)
            fv = cand[rows, pick]
            x[rows[active], fv[active]] = 1 - x[rows[active], fv[active]]
        return x

    # ------------------------------------------------------------------ §3 §4 crisis maps
    def update_crisis(self, path, perf, frus):
        N, n = self.N, self.n
        # panic: frustration per variable (M7), per literal, times log(1+|Psi|)
        gl = cp.repeat(frus, 2)                                            # (N,)
        mag = self.k_P * gl * cp.log1p(cp.abs(self.Psi))
        self.P = (1 - self.delta_P) * self.P + self.delta_P * cp.tanh(mag)
        # anxiety real: performance gap per literal (mean perf of agents that used it)
        used = cp.zeros(N, dtype=cp.float32); psum = cp.zeros(N, dtype=cp.float32)
        cupyx.scatter_add(used, path.ravel(), cp.ones(path.size, dtype=cp.float32))
        cupyx.scatter_add(psum, path.ravel(), cp.repeat(perf, n))
        perf_l = cp.where(used > 0, psum / cp.maximum(used, 1), perf.mean())
        gap = self.perf_target - perf_l
        re_t = cp.tanh(self.k_A * gap) + 0.5 * cp.tanh(cp.float32(self.mean_pers))   # homology enters uniformly
        im_t = cp.tanh(self.stag / self.tau_s) if self.stag > 0 else 0.0
        re = (1 - self.delta_A) * self.Psi.real + self.delta_A * re_t
        im = (1 - self.delta_A) * self.Psi.imag + self.delta_A * im_t if self.stag > 0 else 0.9 * self.Psi.imag
        self.Psi = (re + 1j * im).astype(cp.complex64)

    # ------------------------------------------------------------------ §5 burst
    def burst(self):
        N = self.N
        k = max(1, N // 4)
        U, Sg, Vt = cp.linalg.svd(self.R)
        B = (U[:, :k] * Sg[:k]) @ Vt[:k, :]
        Pe = cp.maximum(self.P[:, None], self.P[None, :])                        # edge panic (M9: max, was geometric mean)
        g = 0.7 * cp.minimum(2.0 * Pe, 1.5)
        strength = float(self.P.mean() * cp.abs(self.Psi).mean())
        noise = self.rng.standard_normal((N, N), dtype=cp.float32) * strength * float(Sg[:k].mean() / math.sqrt(N))
        Psi_e = self.Psi[:, None] + self.Psi[None, :]
        phase = cp.exp(1j * cp.angle(Psi_e)).astype(cp.complex64)
        d = ((g * B + 0.3 / 0.7 * g * noise).astype(cp.complex64)) * phase
        self.R += d.real; self.M += d.imag
        self.symmetrize()
        ang = float(cp.angle(Psi_e).mean())
        reg = "reinforce" if ang < 0.05 else ("mixed" if ang < math.pi / 2 - 0.05 else
              ("repel" if ang < math.pi / 2 + 0.05 else ("negate+repel" if ang < math.pi - 0.05 else "negate")))
        self.events.append((self.it, "burst", reg, round(ang, 3), round(float(g.mean()), 3), round(float(Sg[0]), 3),
                            round(float(g.max()), 3), int((g > 0.7072).sum()), round(float(self.P.max()), 3)))
        return reg, ang

    # ------------------------------------------------------------------ §6 homology (H0 of Rips on d = max R - R)
    def homology(self):
        N = self.N
        Rmax = float(self.R.max())
        if Rmax <= 0:
            self.H_p, self.mean_pers = 1.0, 0.0; return
        d = (Rmax - self.R) / Rmax
        in_tree = cp.zeros(N, dtype=cp.bool_); in_tree[0] = True
        key = d[0].copy(); key[0] = cp.inf
        deaths = cp.zeros(N - 1, dtype=cp.float32)
        for i in range(N - 1):
            j = int(cp.argmin(key))
            deaths[i] = key[j]
            in_tree[j] = True
            key = cp.minimum(key, d[j]); key[in_tree] = cp.inf
        L = deaths[deaths > 1e-9]
        if L.size < 2:
            # flat field: every bar has the same (zero) length -> maximal entropy
            self.H_p, self.mean_pers = 1.0, float(deaths.mean()); return
        p = L / L.sum()
        self.H_p = float(-(p * cp.log(p)).sum() / math.log(L.size))
        self.mean_pers = float(deaths.mean())

    # ------------------------------------------------------------------ §8 hyperparameters
    def update_hyper(self):
        p_mean = float(self.P.mean()); a_amp = float(cp.abs(self.Psi).mean())
        self.alpha = float(np.clip(1.0 + 4.0 * p_mean * a_amp, self.alpha_min, self.alpha_max))
        self.beta = self.beta_min + (self.beta_max - self.beta_min) * self.H_p
        self.rho = float(np.clip(self.rho0 * math.exp(-abs(self.p_cov) / self.p_ref), 0.05, 0.3))

    # ------------------------------------------------------------------ §9 deposit + evaporation, §7 momentum
    def deposit(self, path, perf):
        K, n, N = self.K, self.n, self.N
        self.R *= (1 - self.rho); self.M *= (1 - self.rho); self.Rs *= (1 - self.rho); self.Ms *= (1 - self.rho)
        if self.kappa_sel > 0:   # M7: relative per-clause selection, exp(-kappa (unsat - min unsat))
            fit = cp.exp(-self.kappa_sel * (self._unsat - self._unsat.min()).astype(cp.float32))
            amt = self.alpha * fit / fit.sum()        # weights sum to 1: total deposit mass independent of kappa
        else:
            amt = (self.alpha / K) * perf ** self.eta_dep                 # (K,)
        a = path[:, :-1].ravel(); b = path[:, 1:].ravel()
        wts = cp.repeat(amt, n - 1)
        cupyx.scatter_add(self.R, (a, b), wts)
        cupyx.scatter_add(self.Rs, path[:, 0], amt)
        # momentum §7: purely imaginary, from mean persistence (normalised units)
        self.p_cov = 0.9 * self.p_cov + 0.1 * 1j * self.mean_pers
        # M2: equilibrium repulsion theta_m * |p|/(|p|+p_sat) * max R (a persistence-scaled threshold)
        pn = abs(self.p_cov)
        self.M += cp.float32(self.rho * self.theta_m * pn / (pn + self.p_sat) * float(self.R.max()))
        self.symmetrize()
        mx = float(max(cp.abs(self.R).max(), cp.abs(self.M).max()))
        if mx > self.field_cap:
            s = self.field_cap / mx
            self.R *= s; self.M *= s; self.Rs *= s; self.Ms *= s

    # ------------------------------------------------------------------ §10
    def symmetrize(self):
        self.R = cp.maximum(0.5 * (self.R + self.R.T), 0.0)
        self.M = cp.maximum(0.5 * (self.M + self.M.T), 0.0)
        self.Rs = cp.maximum(self.Rs, 0.0); self.Ms = cp.maximum(self.Ms, 0.0)

    def reset(self, X):
        cut = float(cp.percentile(cp.abs(self.R + 1j * self.M).ravel(), X))
        pos = self.R[self.R > 0]
        baseline = float(cp.median(pos)) if pos.size > 0 else 0.3
        mask = cp.abs(self.R + 1j * self.M) < cut
        # M10: the raised edges must be visible under the interference weight: above their own repulsion
        vis = self.M + 0.2 * float(self.R.max())
        self.R = cp.where(mask, cp.maximum(self.R, cp.maximum(baseline, vis)), self.R)
        self.events.append((self.it, "reset", X, round(baseline, 4), round(float(mask.mean()), 3)))

    # ------------------------------------------------------------------ §11
    def entropy_control(self):
        if self.H_p < self.target_entropy - 0.05:
            self.M += cp.abs(self.rng.standard_normal((self.N, self.N), dtype=cp.float32)) * 0.01 * float(self.R.max())
            self.symmetrize()
        elif self.H_p > self.target_entropy + 0.05:
            self.M *= 0.95

    # ------------------------------------------------------------------ §12 economy
    def economy(self, perf_np):
        K = self.K
        if self.prev_perf is not None:
            trend = perf_np - self.prev_perf
            self.node_panic = np.clip(self.node_panic - trend * 0.1, 0.05, 0.95)
        self.prev_perf = perf_np.copy()
        scarcity = 0.5 + 0.5 * float(self.P.mean())
        req = 0.2 + 0.3 * self.node_panic * self.risk
        cost = np.floor(req * 100 * scarcity * self.market)
        ok = self.tokens >= cost
        self.tokens[ok] -= cost[ok]
        self.node_panic[~ok] = np.minimum(1.0, self.node_panic[~ok] * 1.1)
        self.tokens += np.floor(perf_np * 100 * 3.0 / self.market)
        self.market = float(np.clip(self.market * (1 + self.np_rng.normal(0, 0.15)), 0.2, 5.0))
        self.tokens = np.maximum(25.0, np.floor(self.tokens * (1 - 0.005)))
        self.temp = np.where(ok, 0.7, 1.0).astype(np.float32)
        # M12: the search budget grows with chronic anxiety (the stagnation channel): continued stagnation buys
        # longer refinement trajectories, up to flips_max
        base = self.flips_base * (1.0 + self.budget_gain * float(cp.abs(self.Psi.imag).mean()))
        base = min(base, self.flips_max)
        self.flips = np.where(ok, 2 * base, base / 2).astype(np.int64)
        return float(ok.mean())

    # ------------------------------------------------------------------ §13 the loop
    def step(self):
        self.it += 1
        t0 = time.time()
        path, x = self.walk()
        _, _, unsat_w = self.evaluate(x)
        perf_walk = cp.clip(1.0 - unsat_w.astype(cp.float32) / (self.m / 8.0), 0.0, 1.0)   # M3, before refinement
        x = self.refine(x, self.flips)
        lt, ntrue, unsat = self.evaluate(x)
        self._unsat = unsat
        perf = cp.clip(1.0 - unsat.astype(cp.float32) / (self.m / 8.0), 0.0, 1.0)         # M3
        # M4: the path of the refined assignment (same order, flipped literals negated)
        rows = cp.arange(self.K)[:, None]
        path = cp.where(x[rows, path // 2] == (path % 2).astype(cp.int8), path, path ^ 1)
        brk, mk = self.flip_gradient(lt, ntrue)
        grad = brk - mk
        frus = self.frustration(mk)
        dmag = float(cp.abs(grad).mean())
        div = float((x[:, None, :] != x[None, :, :]).mean()) if self.K <= 256 else float("nan")
        bi = int(cp.argmin(unsat)); bu = int(unsat[bi])
        improved = bu < self.best_unsat
        if improved:
            self.best_unsat, self.best_x, self.stag, self.bursts_since_improve = bu, x[bi].copy(), 0, 0
        else:
            self.stag += 1
        self.update_crisis(path, perf, frus)
        # crisis -> burst (§5), with a minimum gap and the interval backstop
        self.since_burst += 1
        p_mean = float(self.P.mean()); a_amp = float(cp.abs(self.Psi).mean())
        reg = ""
        if self.since_burst >= self.burst_min_gap and (p_mean > self.theta_P or a_amp > self.theta_Psi
                                                        or self.since_burst >= self.burst_interval):
            reg, _ = self.burst(); self.since_burst = 0; self.bursts_since_improve += 1
        if self.it % self.homology_every == 0:
            self.homology()
        self.update_hyper()
        self.deposit(path, perf)
        if self.stag >= self.reset_hard_T:
            self.reset(95); self.stag = 0
        elif self.stag >= self.reset_mild_T and self.stag % self.reset_mild_T == 0:
            self.reset(30)
        self.entropy_control()
        bought = self.economy(cp.asnumpy(perf))
        w = self.eps_now + cp.maximum(0.0, self.R * self.R - self.M * self.M)
        live = float((w > 2 * self.eps_now).mean())
        rec = dict(it=self.it, best=self.best_unsat, cur_best=bu, perf=float(perf.mean()), P=p_mean, Psi=a_amp,
                   phase=float(cp.angle(self.Psi).mean()), alpha=self.alpha, beta=self.beta, rho=self.rho,
                   Hp=self.H_p, pers=self.mean_pers, pcov=abs(self.p_cov), Rmax=float(self.R.max()),
                   Mmax=float(self.M.max()), live=live, burst=reg, stag=self.stag, bought=bought,
                   dmag=dmag, div=div, perf_walk=float(perf_walk.mean()), dt=time.time() - t0)
        self.stats.append(rec)
        if self.log:
            self.log(rec)
        return bu

    def solve(self, max_iter, verbose_every=10):
        t0 = time.time()
        for _ in range(max_iter):
            bu = self.step()
            r = self.stats[-1]
            if verbose_every and (self.it % verbose_every == 0 or bu == 0):
                print(f"it {r['it']:4d} best {r['best']:4d} cur {r['cur_best']:4d} perf {r['perf']:.3f} "
                      f"P {r['P']:.2f} |Psi| {r['Psi']:.2f} ph {r['phase']:+.2f} a {r['alpha']:.2f} b {r['beta']:.2f} "
                      f"rho {r['rho']:.3f} Hp {r['Hp']:.2f} pers {r['pers']:.3f} live {r['live']:.2f} "
                      f"Rmax {r['Rmax']:.2f} Mmax {r['Mmax']:.2f} dm {r['dmag']:.2f} div {r['div']:.3f} "
                      f"pw {r['perf_walk']:.2f} {r['burst']} {r['dt']:.2f}s", flush=True)
            if bu == 0:
                rc = host_recount(self.V, self.S, self.best_x)
                if rc == 0:
                    return dict(solved=True, iters=self.it, time=time.time() - t0, recount=rc)
                print("host recount disagrees:", rc)
        return dict(solved=False, iters=self.it, time=time.time() - t0, best=self.best_unsat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--ratio", type=float, default=4.26)
    ap.add_argument("--family", choices=["planted", "random"], default="planted")
    ap.add_argument("--dimacs", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--alpha_w", type=float, default=1.0)
    ap.add_argument("--flips", type=int, default=30); ap.add_argument("--homology_every", type=int, default=1)
    ap.add_argument("--verbose", type=int, default=10); ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--kappa_sel", type=float, default=0.0); ap.add_argument("--eps_rel", type=float, default=0.0)
    ap.add_argument("--budget_gain", type=float, default=0.0); ap.add_argument("--flips_max", type=int, default=1000)
    ap.add_argument("--refine", choices=["kernel", "py"], default="kernel")
    a = ap.parse_args()
    if a.dimacs:
        n, cl = load_dimacs(a.dimacs)
        Vh = np.array([(c + [c[-1]] * 3)[:3] for c in cl]); V = cp.asarray(np.abs(Vh) - 1, dtype=cp.int32)
        S = cp.asarray((Vh > 0).astype(np.int8))
    else:
        n = a.n; m = int(round(a.ratio * n))
        V, S, _, sd = gen_3sat_sat(n, m, a.seed, a.family == "planted")
        if sd != a.seed:
            print(f"seed {a.seed} unsatisfiable; using satisfiable seed {sd}", flush=True)
    sysu = SystemU(V, S, n, K=a.K, seed=a.seed, alpha_w=a.alpha_w, flips_base=a.flips, homology_every=a.homology_every,
                   kappa_sel=a.kappa_sel, eps_rel=a.eps_rel, budget_gain=a.budget_gain, flips_max=a.flips_max,
                   refine_mode=a.refine)
    print(f"System U: n={n} m={V.shape[0]} K={a.K} family={a.family} seed={a.seed}", flush=True)
    res = sysu.solve(a.iters, a.verbose)
    print("RESULT", json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(args=vars(a), result=res, stats=sysu.stats, events=sysu.events), fh)


if __name__ == "__main__":
    main()
