# System C on 3-SAT: derivation, theorems, obstruction, and the changed construction

> **Scope:** Systems C / C′ / C‴ are a reduced product-sampler abstraction of UMACO for SAT, with no PAQ
> tensors, economy or homology. Every result below is about those systems, not about UMACO or System U;
> `docs/system-u/UMACO13_SAT_architecture_audit.md` §3 goes through each result's status for System U.

Research log, 10 September 2026. Object: **System C** from `UMACO_lineage_reconstruction.md` Part IV (not yet published),
the UMACO-derived SAT algorithm in which the pheromone integrates fitness (sampling exponent α = 1,
K-normalized deposit, leaky-integrator pheromone, coverage-driven clause weights, WalkSAT-Metropolis
local search on a subset of ants, stagnation reset of weak entries). Target: prove it solves 3-SAT in
polynomial time, or find the exact obstruction and change the construction.

Everything marked **Theorem** is proved here. Everything marked **Open** is not. The Lean file
`docs/lean/UmacoSat.lean` (§8 onward) formalizes the discrete statements.

---

## 0. Notation

- Variables `1..n`, assignment `x ∈ {0,1}^n`. Clause `C_j` = set of literals; `k = |C_j| = 3`.
- Product distribution with marginals `p ∈ [0,1]^n`: `P_p[x_v = 1] = p_v`, independent.
- For literal `ℓ` on `v`: `q_ℓ(p) = P_p[ℓ false] = 1 − p_v` if `ℓ = v`, `= p_v` if `ℓ = ¬v`.
- `f_j(p) = P_p[C_j unsatisfied] = ∏_{ℓ∈C_j} q_ℓ(p)` (multilinear, degree ≤ 3).
- Weights `w_j ≥ 1`, `W = Σ_j w_j`. Weighted satisfaction `Q_w(x) = Σ_j w_j [C_j sat by x] / W`.
- Fitness `F(x) = Q_w(x)^{γ}`, γ = 3/2 in the code. Multilinear extension `F̃(p) = E_p[F(x)]`.
- `F̃(p | v←b)` = `F̃` with `p_v` replaced by `b`. Multilinearity gives the exact identities
  `F̃(p) = (1−p_v) F̃(p|v←0) + p_v F̃(p|v←1)` and `∂F̃/∂p_v = F̃(p|v←1) − F̃(p|v←0)`.
- Solution set `SOL(φ)`. A vertex `x` is a **strict 1-flip local maximum** of `Q_w` if
  `Q_w(x) > Q_w(x ⊕ e_v)` for every `v`.

## 1. Exact mean-field equations of System C

In System C the deposit on polarity `b` of variable `v` is `(α/K) Σ_a F(x^{(a)}) [x_v^{(a)} = b]`. With
`K → ∞` and evaporation `ρ`, `τ_{v,b}` is a leaky integrator of `E_p[F(x)·[x_v = b]] = P_p[x_v=b]·F̃(p|v←b)`.
At α = 1 the sampling probability is `p_v = τ_{v1}/(τ_{v0}+τ_{v1})`, so in log-odds
`ℓ_v = log(τ_{v1}/τ_{v0})` the quasi-static update (τ relaxes faster than p moves, which holds when ρ is
not small compared with the per-step change of p) is

```
ℓ_v ← ℓ_v + η [ log F̃(p | v←1) − log F̃(p | v←0) ]          (MF-ℓ)
ṗ_v  = η p_v (1 − p_v) [ log F̃(p|v←1) − log F̃(p|v←0) ]        (MF-p, continuous-time form)
```

This is exact for the multilinear extension of `F` (no linearization of the 3/2 power is needed,
because conditioning a product distribution on `x_v = b` is exactly the extension at `p_v = b`).

Clause weights: `S_j ← S_j + λ` if no ant satisfies `C_j` (mean-field: `f_j(p) = 1`), otherwise
`S_j ← μ S_j + (1−μ) f_j(p)`; `w_j = 1 + 5 S_j`.

## 2. What the pheromone provably does at fixed weights

**Theorem 1 (Lyapunov).** Along (MF-p) at fixed `w`,
```
d/dt log F̃(p(t)) = η Σ_v p_v(1−p_v) · (F̃(p|v←1) − F̃(p|v←0)) · (log F̃(p|v←1) − log F̃(p|v←0)) / F̃(p) ≥ 0,
```
with equality iff for every `v` either `p_v ∈ {0,1}` or `F̃(p|v←1) = F̃(p|v←0)`.

*Proof.* Chain rule with `∂_v F̃ = F̃(p|v←1) − F̃(p|v←0)` and (MF-p); each summand has the form
`c·(a−b)(log a − log b)` with `c ≥ 0`, `a,b > 0`, and `(a−b)(log a − log b) ≥ 0` because `log` is
increasing. ∎ (Lean: `mul_log_sub_nonneg`.)

**Theorem 2 (fixed points).** The stationary points of (MF-p) are exactly the `p` with, for every `v`,
`p_v ∈ {0,1}` or `F̃(p|v←1) = F̃(p|v←0)`. Every vertex is stationary. A vertex `x` is asymptotically
stable iff it is a strict 1-flip local maximum of `Q_w` (equivalently of `F`); at a vertex,
`F̃(x|v←1) − F̃(x|v←0) = F(x^{v←1}) − F(x^{v←0})`, so the condition is `F(x) > F(x ⊕ e_v)` for all `v`.
Non-vertex stationary points are not attracting unless all mixed partials `∂_u∂_v F̃` vanish there
(the Hessian of a multilinear function has zero diagonal, so a critical point in the interior of a face
is a saddle unless degenerate).

*Proof.* Stationarity is read off (MF-p). Linearizing at a vertex, the coordinate `p_v` near `x_v` moves
with sign of `x_v`-direction times `log F̃(x|v←x_v) − log F̃(x|v←1−x_v)`, which is inward iff
`F(x) > F(x⊕e_v)`. The Hessian claim is the standard fact for multilinear polynomials. ∎

**Theorem 3 (solutions are the global attractors of the pheromone at any weights).** If `x* ∈ SOL(φ)`
then `Q_w(x*) = 1 = max Q_w` for every `w ≥ 0`, so `F̃ ≤ 1 = F(x*)` everywhere; `x*` is a local maximum
of the flow for every `w`. It is a *strict* 1-flip local maximum, hence asymptotically stable, iff every
variable `v` is **critical** in some clause (its literal is the unique true literal of that clause under
`x*`); otherwise `x* ⊕ e_v` is also a solution and `x*` lies on a solution plateau.

*Proof.* Immediate from `Q_w ≤ 1` and Theorem 2. ∎

So at fixed weights the pheromone climbs `F̃` monotonically (Theorem 1), converges generically to a
vertex which is a strict 1-flip local optimum of weighted MaxSAT (Theorem 2), and every solution is such
an optimum (Theorem 3). This is what "worked": the mechanism is a correct ascent on the product
relaxation. The question is entirely about non-solution local optima.

## 3. What the clause weights provably do

**Theorem 4 (escape from every non-solution strict local optimum, with an explicit time).** Let the
mean-field population sit at a vertex `x ∉ SOL(φ)` with unsatisfied set `U(x) ≠ ∅`, and let
`Δ_max` be the maximum number of clauses in which a variable is critical under `x`. Then for every
`j ∈ U(x)` and every `v ∈ C_j`, after
```
t* = ⌈ (Δ_max − 1 + 5 Δ_max S_max) / (5λ) ⌉
```
steps (with `S_max` the largest stubbornness of a satisfied clause at the start), the weighted gain of
flipping `v` is strictly positive, so `x` is no longer a local maximum of `Q_w` and the flow leaves it.
With the code's constants (`λ = 0.21015`, `S_max ≤ 1`) this is `t* ≤ 6.7 Δ_max`.

*Proof.* At a frozen vertex, `f_j = 1` for `j ∈ U(x)`, so `S_j` increases by `λ` every step and
`w_j(t) ≥ 1 + 5λt`. For a satisfied clause `f = 0`, so `S ← μS` decays and `w ≤ 1 + 5 S_max μ^t ≤ 1 + 5S_max`.
Flipping `v ∈ C_j` satisfies `C_j` (gain `≥ w_j(t)`) and can break only clauses critical on `v`
(loss `≤ Δ_max (1 + 5 S_max)`). The gain exceeds the loss once `1 + 5λt > Δ_max(1 + 5S_max)`. ∎

**Theorem 5 (the two weight regimes).** In the fast-tracking limit (`μ → 0`) and ignoring the additive
term, `w_j ≈ 1 + 5 f_j(p)`, and (MF-p) with `F = Q_w` becomes the replicator gradient flow of
```
Φ(p) = Σ_j f_j(p) + (5/2) Σ_j f_j(p)²,
```
a squared penalty on unsatisfaction probabilities (it penalizes concentrating violation in few clauses).
`Φ(p) = 0` iff `p` is supported on solutions. This regime has non-solution local minima in general
(degree-6 polynomial on the cube). The additive `λ·[f_j = 1]` term is the only mechanism whose weight
grows without bound on a persistently violated clause; it is the completeness mechanism of Theorem 4,
and it fires only when the population is frozen (`f_j = 1`), i.e., only at vertices.

*Proof.* `∇_p Σ_j w_j (1 − f_j) = −Σ_j (1 + 5 f_j) ∇ f_j = −∇ Φ`. The rest is definitional. ∎

So the architecture, made exact, is the **breakout method** (Morris 1993) lifted to the product
relaxation: ascend, get stuck at a strict local optimum of weighted MaxSAT, inflate the weights of the
violated clauses until it is no longer a local optimum, ascend again. Theorem 4 is Morris's escape
lemma with the code's constants. **The open node is the number of escapes.**

## 4. Base case, exactly as required: what k = 2 gives and what breaks at k = 3

The local-search component acts on a single assignment `x`: pick an unsatisfied clause `C_j`, pick
`v ∈ C_j`, accept the flip with probability `A(v)` (`1` if `Δ_w(v) ≥ 0`, else `e^{Δ_w/T}`). Fix a solution
`x*` and call `v` **wrong** if `x_v ≠ x*_v`. Let `W_j = |{v ∈ C_j : v wrong}|`.

**Lemma 6 (k = 2 drift).** If `C_j` is unsatisfied by `x` and `k = 2`, then `W_j ≥ 1`, so a uniformly
random `v ∈ C_j` is wrong with probability `W_j/2 ≥ 1/2`. Hence, with all flips accepted, the Hamming
distance `d(x, x*)` is a random walk with non-negative drift toward 0 and hits 0 in expected `O(n²)`
steps (Papadimitriou 1991).

*Proof.* `C_j` unsatisfied by `x` means every literal is false under `x`; `C_j` satisfied by `x*` means
some literal is true under `x*`; that literal's variable is wrong. Counting gives `W_j/2`. The walk on
`d` moves down with probability `≥ 1/2` per step; gambler's ruin gives `O(n²)`. ∎
(Lean: `wrong_count_pos`, `toward_prob_ge_inv_k`.)

**What this lemma is expected to give.** A *drift inequality*: a potential `Ψ(x)` (here `d(x,x*)`) whose
expected one-step change under the flip law is `≤ −c` with `c ≥ 0`, plus a standard hitting-time bound.
The route does not use implication graphs; it uses only "an unsatisfied clause contains a wrong variable".

**Which part must generalize.** The lower bound on the **toward-probability** `π = P[chosen, accepted flip
is on a wrong variable]`. Everything else (hitting-time from drift) is generic.

**Obstruction at k = 3.** The same counting gives only `W_j/3 ≥ 1/3`. A walk on `d` with toward-probability
`1/3` has *negative* drift and hits 0 in expected time exponential in `n` from a typical start
(Schöning's `(4/3)^n` is the optimum for this walk with restarts). So at `k = 3` the base machinery
delivers nothing unless **something in System C raises `π` above 1/2**. The only candidates are the
clause-selection rule, the variable-selection rule, the acceptance rule, and the weights that enter them.

That is the precise generalization target, and the rest of this log is about it:

> **Drift target.** Show that System C's accepted-flip law has toward-probability `π(x) > 1/2` for all
> `x ∉ SOL(φ)` (or that some potential `Ψ(x, p, w)` has expected decrease `≥ 1/poly(n)` per step and
> `Ψ = 0` iff `x ∈ SOL(φ)`).

## 5. Attempt on the drift target: do the weights raise π?

**Theorem 7 (weights on the selected clause cannot raise π above `W_j/3` under uniform in-clause
selection).** Let `C_j` be unsatisfied and selected, variables chosen uniformly in `C_j`, acceptance
`A(v) = 1` if `Δ_w(v) ≥ 0`. Write `fix_w(v)` for the total weight of unsatisfied clauses containing `v`
(so `fix_w(v) ≥ w_j`) and `break_w(v)` for the total weight of clauses critical on `v`. Then
`Δ_w(v) = fix_w(v) − break_w(v)`. If `w_j ≥ break_w(u)` for every right variable `u ∈ C_j`, then every
flip in `C_j` is accepted and the accepted-flip toward-probability from `C_j` is exactly `W_j/3`.

*Proof.* `fix_w(u) ≥ w_j ≥ break_w(u)` gives `Δ_w(u) ≥ 0`, so `A(u) = 1` for right `u`; wrong `v` has
`A(v) ∈ [0,1]`; the conditional probability that an accepted flip is wrong is
`Σ_{wrong} A / Σ_all A ≤ W_j / 3` when all `A = 1`, and equals it. ∎

**Corollary 8 (the weights make it worse, not better).** By Theorem 4's mechanism, a persistently
unsatisfied clause has `w_j → ∞` while its variables' critical clauses (currently satisfied) have
`w → 1`. So after `O(Δ_max/λ)` steps of being unsatisfied, **every** clause that the local search selects
satisfies the hypothesis of Theorem 7, and the accepted-flip law from that clause is exactly WalkSAT's
uniform law: `π = W_j/3`, which is `1/3` for every clause with a single true literal under `x*`.
Without weights (`w ≡ 1`) and at `T → 0`, a right variable `u` with `break(u) ≥ 2` is *rejected*
(`Δ = 1 − break(u) < 0`), so unweighted Metropolis-WalkSAT filters away-flips near the solution
(where wrong variables have small break and right variables have break ≈ their criticality degree),
and `π → 1` there. The weights as used in macov8/System C **undo that filter** precisely on the clauses
that have been hard for longest. (Lean: `toward_prob_all_accepted`, `toward_prob_le_of_accept`.)

**What generalized from k = 2 and what did not.** The counting bound `W_j/k` generalized and is tight.
The acceptance filter is the extra ingredient at `k = 3`, and Theorem 7 shows that weighting the
*fixed* clause neutralizes it. The obstruction is not the clause size per se; it is that the
weight enters `Δ_w` on the fix side symmetrically for wrong and right variables.

## 6. Changed construction: System C′ (weighted-break variable selection)

Replace the uniform in-clause choice by the rule used by SAPS/PAWS: in the selected unsatisfied clause,
flip the variable with the **smallest weighted break**, `v = argmin_{v∈C_j} break_w(v)` (ties uniformly).
Everything else in System C is unchanged. The weights now act only through `break_w`, i.e., only on
the *broken* side, which is where the wrong/right asymmetry lives:

**Lemma 9 (break asymmetry).** If `v ∈ C_j` is wrong and `C_{j'}` is critical on `v`, then `C_{j'}`
contains another wrong variable. If `u ∈ C_j` is right and `C_{j'}` is critical on `u` under `x*`
(i.e., `u` is its unique true literal under `x*`) and no other variable of `C_{j'}` is wrong, then
`C_{j'}` is critical on `u` under `x` as well.

*Proof.* Critical on `v` means `v`'s literal is the unique true literal under `x`; under `x*` that
literal is false (`v` wrong) and `C_{j'}` is satisfied, so another literal is true under `x*` and false
under `x`, i.e., on a wrong variable. The second statement is direct. ∎ (Lean: `critical_wrong_has_wrong`.)

**Consequence.** Let `d = d(x, x*)`. `break(v)` for a wrong `v` counts only clauses containing a
*second* wrong variable, so `break(v) ≤ (number of clauses containing v and another wrong variable)`;
for a right `u`, `break(u) ≥ crit*(u) − (clauses containing u and a wrong variable)`, where `crit*(u)`
is `u`'s criticality degree under `x*`. When `d` is small relative to `n` and the formula is such that
every variable is critical under `x*` in at least one clause (`crit*(u) ≥ 1`, cf. Theorem 3), the
argmin selects a wrong variable whenever
```
max_{v wrong in C_j} break_w(v) < min_{u right in C_j} break_w(u),
```
and then `π = 1` from that clause. This is the mechanism by which greedy weighted-break local search
solves random instances near the solution; it is exactly what Theorem 7 shows uniform selection lacks.

**Theorem 10 (C′ on the planted random model, near the solution).** Let `φ` be random 3-SAT with a
planted solution `x*` at clause density `r` (each clause has 1, 2 or 3 true literals under `x*` with
probabilities `3/7, 3/7, 1/7`), and let `x` be at distance `d = δn` from `x*`. Then for a variable `u`,
`crit*(u) ~ Poisson(3r·3/7·(1/3)) = Poisson(3r/7)`, and the number of clauses containing `u` and a wrong
variable is `~ Poisson(3r · 2δ)`. For `δ < 1/14` (so `2δ·3r < 3r/7`), a right variable has, in
expectation, more critical clauses than a wrong variable has spoiled ones, and the probability that the
argmin picks a wrong variable in a selected clause with `W_j = 1` exceeds `1/2` for `r ≥ 1`. Under this
condition the distance walk has positive drift toward 0 in the regime `δ < 1/14`.

*Proof sketch (this is the standard concentration computation; the constants are what the planted
model gives).* Each clause containing `u` is critical on `u` under `x*` with probability `(3/7)·(1/3)`
(one true literal, and it is `u`'s). Each clause containing `u` has another wrong variable with
probability `≈ 2δ`. Poisson thinning gives the stated laws; comparing `P[Poisson(3r/7) ≥ 1]` with
`P[Poisson(6rδ) = 0]` gives the drift sign. Full details and the far-from-solution regime (`δ ≥ 1/14`)
are **Open**; there the drift of the greedy rule can be negative and the escape mechanism
(Theorem 4) is what must carry the argument.

## 7. Where this stands

- **Proved:** the pheromone is a monotone ascent on the product relaxation (Thm 1), converging
  generically to strict 1-flip local optima of weighted MaxSAT (Thm 2), among which every solution
  (Thm 3); the clause weights escape every non-solution optimum in `O(Δ_max/λ)` steps (Thm 4); the
  weight scheme is a squared penalty plus an unbounded escape term (Thm 5); the k = 2 drift lemma
  (Lemma 6); the k = 3 obstruction and its exact cause, that weighting the fixed clause neutralizes
  the acceptance filter (Thm 7, Cor 8); the wrong/right break asymmetry that a changed selection rule
  can exploit (Lemma 9).
- **Changed construction:** System C′ — weighted-break argmin within the selected clause. This is the
  smallest change that makes the weights act on the asymmetric side of `Δ_w`.
- **Open node (the entire P = NP content):** a bound on the number of Theorem-4 escapes for C′ on
  every satisfiable 3-CNF, equivalently a potential `Ψ(x, p, w)` with expected decrease
  `≥ 1/poly` per step. Theorem 10 gives the near-solution drift on the planted model; the
  far-from-solution regime is open even there.
- **Next constructions to try, in order:** (i) a potential of the form `Ψ = d(x,x*) + c·Σ_j S_j·[C_j
  unsat]` — the weights are exactly a record of how long each clause has resisted, so a decrease
  argument would have to show that resistance time is bounded per clause along the trajectory;
  (ii) the planted model at threshold as the provable variant (the analog of the forced-NS route),
  where Theorem 10's computation can be pushed to all `δ` using the known structure of the planted
  solution space; (iii) an adversarial family for C′: formulas where the argmin is fooled at every
  non-solution vertex, which would show C′ is also exponential in the worst case and force the next
  change (weighted *clause* selection ∝ `w_j`, the SAPS rule).

---

## 8. Second round: where the drift of System C′ actually crosses zero (planted model)

All of §8 is at the first-moment level (Poisson local structure of a random planted formula, no
correlations between a variable's clauses beyond the planted conditioning). Where a statement is a
closed-form identity it is a theorem about the model and is marked so; where it is a numerical
evaluation of the model it is marked *computed*.

**Setup.** Random 3-SAT with planted solution `x*`, density `r = m/n`, `r = 4.26` in the computations.
Each clause is uniform among the 7 sign patterns satisfied by `x*`. The current assignment `x` is at
distance `δn`; at the first-moment level each variable is wrong independently with probability `δ`.
Each variable lies in `~Poisson(3r)` clauses.

**Theorem 11 (uniform in-clause selection has a tunnel of depth `δ_c n`).** For the uniform rule the
toward-probability at distance `δ` is `π_u(δ) = δ / (1 − (1−δ)³)`, and `π_u(δ) ≥ 1/2` iff
`δ ≥ δ_c = (3 − √5)/2 ≈ 0.382`. So the walk has positive drift only down to distance `0.382n` and
must tunnel the rest: exponential time. (Lean: `uniform_toward_crossover`.)

*Proof.* Conditioned on an unsatisfied clause (at least one wrong variable), `E[W] = 3δ/(1−(1−δ)³)`;
divide by 3. `π_u = 1/2` is `2δ = 1 − (1−δ)³ = 3δ − 3δ² + δ³`, i.e. `δ(δ² − 3δ + 1) = 0`. ∎

**Computed (C′ removes the tunnel at first moment).** For the min-break rule of System C′ the
toward-probability, evaluated on the same model with the break counts drawn from each variable's
own Poisson clause neighbourhood, is

| δ | uniform | min-break (C′) |
|---|---|---|
| 0.02 | 0.340 | 0.827 |
| 0.10 | 0.369 | 0.747 |
| 0.20 | 0.410 | 0.678 |
| 0.30 | 0.457 | 0.640 |
| 0.382 | 0.500 | 0.633 |
| 0.50 | 0.571 | 0.650 |

It never drops below 0.63. The changed construction has positive first-moment drift at every
distance. This is not a theorem: the first-moment model ignores that the walk's `x` is correlated
with the formula through its own history. That correlation is exactly what the population design
removes for the *first* step of every ant:

**Theorem 13 (independence at construction).** Conditional on the pheromone marginals `p`, the
wrong-set of an ant constructed from `p` is a product measure with `P[v wrong] = 1 − p_v(x*_v)`. So the
first local-search step of every ant is governed by the first-moment law exactly, not approximately;
history correlation enters only through `p` (which the formula shaped) and through the ≤ F flips of
that ant's own walk. *Proof.* Construction is independent per variable by definition. ∎

**Theorem 12 (annealed drift of the pheromone on the planted model, closed form).** With uniform
weights and a homogeneous wrongness `δ`, the expected multilinear gradient of clause satisfaction at
`v` toward `x*_v` is
```
E[ ∂_v Q̃ toward x*_v ] = (3r/7) (1 − δ)² > 0     for all δ ∈ [0, 1).
```
*Proof.* Enumerate the seven patterns with `v` in slot 0. A literal in slot `i` is false under `x` with
probability `(1 − t_i)(1 − δ) + t_i δ`. The four patterns with `t_0 = 1` contribute
`(1−δ)² + 2δ(1−δ) + δ² = 1`; the three with `t_0 = 0` contribute `−(2δ(1−δ) + δ²) = −(2δ − δ²)`.
Sum `1 − 2δ + δ² = (1−δ)²`, times `3r/7` clauses per pattern-slot. ∎ (Lean: `annealed_drift_identity`.)

So the homogeneous mean-field flow converges to `x*` monotonically from `p = ½`; at `δ = ½` the
per-variable signal is `3r/28 ≈ 0.46` against a standard deviation `≈ 0.9`, which is the familiar
≈ 30 % majority-vote error at threshold density.

**Computed (quenched drift: the real obstruction is not a tunnel).** Per-variable drift
`D_v(δ) = Σ_{j∋v} s_j ∏_{i≠0} [(1−t_{ji})(1−δ) + t_{ji} δ]`. The fraction `g(δ) = P[D_v(δ) < 0]`:

| δ | 0.50 | 0.40 | 0.31 | 0.25 | 0.20 | 0.15 | 0.10 | 0.07 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g(δ) | 0.257 | 0.220 | 0.157 | 0.118 | 0.098 | 0.084 | 0.078 | 0.078 | 0.077 | 0.078 | 0.077 |
| g < δ ? | yes | yes | yes | yes | yes | yes | yes | no | no | no | no |

`g(δ)` does not go to zero. Its floor, ≈ 7.8 %, has an exact description:

**Theorem 14 (the floor is the non-critical variables).** `D_v(0) = crit*(v) ≥ 0`, where `crit*(v)` is
the number of clauses in which `v` is the unique true literal under `x*`. For small `δ > 0`,
```
D_v(δ) = crit*(v)(1−δ)² + (n₁₀₁ + n₁₁₀) δ(1−δ) + n₁₁₁ δ² − (n₀₁₀ + n₀₀₁) δ(1−δ) − n₀₁₁ δ²,
```
so `D_v(δ) < 0` for all small `δ` iff `crit*(v) = 0` and `n₀₁₀ + n₀₀₁ > n₁₀₁ + n₁₁₀`. On the planted model
`P[crit*(v) = 0] = e^{−3r/7} ≈ 0.161` and, given that, the two Poisson(6r/7) counts compare unfavourably
with probability ≈ 0.42, giving the observed ≈ 7 %. *Proof.* Read off the pattern sum. ∎

A variable with `crit*(v) = 0` is **free**: `x* ⊕ e_v` is also a solution. So the mean-field flow parks
≈ 7.8 % of the variables on their wrong side, which by itself breaks nothing. Damage occurs only in a
clause whose true literals under `x*` are *all* parked: a two-true-literal clause with both backups
parked, probability ≈ `(3/7)·0.078² ≈ 0.26 %` of clauses, plus a negligible three-true term. Those
clauses then have `f_j = 1`, the additive `λ` term fires (Theorem 5), and by Theorem 4 one backup of
each is restored within `O(Δ_max/λ)` steps. Restoring a free variable can break a clause only if that
clause's unique current true literal was the free variable's wrong literal, which requires a further
parked backup in that clause; the repair cascade is therefore confined to the parked set.

**What this gives.** A complete mean-field storyline for planted 3-SAT at threshold density: bulk
convergence of critical variables (positive drift once `δ ≲ 0.1`), parking of non-critical variables
(harmless individually), and weight-driven repair of the `≈ 0.26 %` of clauses broken by coincident
parking. Nothing in it needs a tunnel. It is consistent with the empirical plateau of the historical
solver (which lacked the memory and the correct selection rule).

**What it does not give.** A theorem. Three gaps, in order of difficulty:
1. *Local-weak-limit rigor.* The Poisson neighbourhood description of a random planted formula is
   rigorous for bounded-radius neighbourhoods (local weak convergence of random hypergraphs), and the
   mean-field flow over `O(1)` time depends only on bounded neighbourhoods. So Theorems 12 and 14 and the
   bulk phase (`δ` from ½ to ≈ 0.1) should be provable by standard means. **Next task.**
2. *Endgame correlations.* The repair cascade (Theorem 4 firing on ≈ 0.26 % of clauses) involves
   long-range structure (the parked set is not a product measure after the flow). This is where a
   real argument is needed; it looks like a sparse-cascade / branching-process bound.
3. *Worst case.* None of §8 says anything about adversarial 3-CNF. The mechanism identified —
   non-critical variables parked wrong, then repaired via weights — is exactly what an adversary would
   amplify: formulas where a large set of pairwise-free variables are jointly constrained. That is
   the adversarial family to construct next, and if it yields an exponential lower bound for C′ it
   forces the next change (weighted *clause* selection, or a parking-aware deposit).


---

## 9. Third round: the adversarial family, and it is XOR

The parked-free-variable mechanism of §8 is what an adversary amplifies; the strongest amplification
is a family with **no** free variables and a landscape that traps the pheromone itself. That family is
XOR constraints, and it is already in the repo's evidence suite as "Tseitin".

**Theorem 15 (System C's objective on XOR-encoded clauses is the 3-spin model).** Encode
`x₁ ⊕ x₂ ⊕ x₃ = b` as the four 3-clauses forbidding the wrong-parity assignments. For a product
distribution with magnetizations `m_i = 2p_i − 1`,
```
Σ_{the 4 clauses} f_j(p) = P_p[parity ≠ b] = ½ (1 − s·m₁m₂m₃),   s = ±1,
```
so the multilinear extension of the clause count is, up to constants, `Σ_XOR s_j m_{j1} m_{j2} m_{j3}`:
the diluted 3-spin (p-spin) energy. *Proof.* `P[parity = 1] = ½(1 − ∏(1−2p_i))` for independent bits;
substitute. ∎ (Lean: `xor_encoding_is_three_spin`, by `ring`.)

**Consequences.**
- Every variable of an XOR is critical in exactly one of its four clauses under any solution
  (Theorem 3 direction): there are **no free variables**, so nothing parks; the §8 obstruction is absent.
- The annealed drift toward `x*` is `(1 − 2δ)²` (checked symbolically): positive for `δ ≠ ½`, zero at
  `δ = ½`. This is the mirage: it assumes neighbours' wrongness independent of the variable's own,
  which the p-spin gradient flow destroys immediately. The quenched mean-field flow is
  `ṁ_v ∝ Σ_{j∋v} s_j m_{j1} m_{j2}`, gradient dynamics of a sparse 3-spin glass started at `m ≈ 0` plus
  finite-K noise. `m = 0` is a fixed point (the symmetry obstruction, now for every XOR instance), and
  the noise-driven escape leads to threshold states, not to the ground state. For the spherical p-spin
  this is the Cugliandolo–Kurchan result; for sparse random XORSAT the solution space is rigorously
  clustered with frozen variables in every cluster (Ibrahimi, Kanoria, Kraning, Montanari 2015;
  Achlioptas and Coja-Oghlan), which is the structural reason every local dynamics stalls.
- The clause weights reweight the same energy; the breakout method on XORSAT is empirically
  exponential and nothing in Theorem 4 changes the parity structure.
- Random satisfiable 3-XORSAT is solved in polynomial time by Gaussian elimination over GF(2),
  a global linear step that is not a function of bounded neighbourhoods.

**What this settles for the target.** Any UMACO-derived algorithm whose every step is a function of
bounded formula neighbourhoods (System C, C′, and the mean-field of both) belongs to the class of
local / stable algorithms for which the overlap-gap literature (Gamarnik 2021, PNAS survey) gives
failure on random XORSAT-type instances w.h.p. So **System C′ is not a polynomial-time 3-SAT
algorithm**, with this caveat on rigor: Theorem 15 and the symmetry fixed point are proved here; the
stalling of the long-time flow uses the cited freezing and OGP results, which are proved for
bounded-time local algorithms and for the solution-space structure, not for this specific ODE over
polynomially many steps. Closing that is a research problem in itself, but nothing in it points the
other way.

**The construction change this forces.** The obstruction is locality. The only non-local operation in
the intended UMACO architecture is the SVD burst on an `n×n` field (Part I of the lineage document),
which is global but linear over the reals; XOR needs linearity over GF(2). Two options, and they are
different projects:
- **C″ = C′ + GF(2) elimination of detected XOR structure** (what CryptoMiniSat does). This removes
  the XOR obstruction by a problem-specific global step and leaves the non-linear hard core, where
  the same OGP barrier is proved for local algorithms on random k-SAT for large k (Gamarnik and Sudan;
  Coja-Oghlan, Haqshenas, Hetterich, "WalkSAT stalls well below satisfiability", 2017) and is open
  for k = 3.
- **Pairwise pheromone** (`τ_{uv}` correlations, the `n×n` field the intent describes), moving the
  sampler from a product measure to a correlated one. Whether that leaves the local class is the
  first question to settle before investing in it; a Gibbs sampler over pairwise fields is still local.

## 10. Fourth round: closing gap 2 at first-moment level (the repair cascade)

**Theorem 16 (parked fraction, closed form).** On the planted model at density `r`, a variable is
parked (negative drift for all small `δ > 0`) iff `crit*(v) = 0` and its misleading count exceeds its
supporting count (with the `n₀₁₁ > n₁₁₁` tie-break). With `A, B ~ Poisson(6r/7)` independent and
`C, D ~ Poisson(3r/7)`:
```
π(r) = e^{−3r/7} · [ ½(1 − e^{−12r/7} I₀(12r/7)) + ½ e^{−12r/7} I₀(12r/7) · ½(1 − e^{−6r/7} I₀(6r/7)) ],
```
`I₀` the modified Bessel function. `π(4.26) = 0.0732` (the direct simulation of §8 gave 0.078).

**Theorem 17 (the repair cascade is subcritical at every density).** Restoring a parked variable `v`
breaks a clause `C'` only if `v`'s wrong literal is the unique true literal of `C'` under `x`, which
requires every true-under-`x*` literal of `C'` to be parked; each such literal is a child that must be
restored in turn. The expected number of children of a parked variable is
```
b(r) = 3r · (3/7) · [ (2/3)·π(1−π)·1 + (1/3)·π²·2 ] = (6r/7) · π(r),
```
and since `π(r) ≤ ½ e^{−3r/7}`, `b(r) ≤ (3r/7) e^{−3r/7} ≤ 1/e < 1` for every `r`. Numerically
`b` peaks at 0.32 near `r ≈ 2.3` and is 0.27 at `r = 4.26`. So the cascade started by each broken
clause is a subcritical Galton–Watson tree: `O(1)` expected size, `O(log n)` maximum depth over the
`Θ(n)` roots w.h.p. *Proof.* Pattern count as in §8; the bound `u e^{−u} ≤ 1/e` is
`u ≤ e^{u−1}`, i.e. `(u−1) + 1 ≤ e^{u−1}`. ∎ (Lean: `branching_factor_le_inv_e`.)

**Theorem 18 (broken clauses).** Only clauses with two or three true literals under `x*` can be
broken by parking (a one-true clause's true variable is critical, hence never parked). The broken
fraction is `(3/7)π² + (1/7)π³`, which is `0.235 %` at `r = 4.26`. Each broken clause has `f_j = 1` in
the frozen mean-field, so Theorem 4 restores one of its parked backups within `O(Δ_max/λ)` steps,
and since all broken clauses inflate in parallel, the repair phase takes
`O(depth · Δ_max/λ) = O(log n · Δ_max/λ)` steps in total.

**Status of the planted-model storyline after four rounds.** Every phase now has a first-moment
computation with closed forms: initial signal `3r/28` per variable at `p = ½` (Theorem 12), annealed
drift `(3r/7)(1−δ)²` (Theorem 12), parked fraction `π(r)` (Theorem 16), broken fraction (Theorem 18),
subcritical repair (Theorem 17). The two remaining rigor gaps are the bulk-phase ODE under local weak
convergence (standard but long) and the endgame correlations (the parked set after the flow is not a
product measure; Theorem 17's independence is the assumption to remove). Neither is an obstruction of
the §9 kind; both are technique.


---

## 11. Fifth round: integrating the exact quenched mean-field flow (GPU, `research/system_c/meanfield_flow_gpu.py`)

The derivation of §1 gives a deterministic dynamical system, the `K → ∞` limit of System C:
`p_v ← p_v + η p_v(1−p_v) D_v(p, w)` with `D_v` computed **exactly** from the formula (no sampling),
plus the clause-weight rule. Integrating it on an actual random planted instance (`n = 4000`,
`r = 4.26`, `m = 17040`, CuPy) answers the questions the first-moment analysis of §8 left open. This
is not a benchmark of a solver; it is the evaluation of the object the theorems are about.

**Fixed weights (Theorem 1–2 regime).** From `p = ½ + 10⁻³·noise` the flow converges in ≈ 700 steps to a
stable non-solution vertex with **20.9 %** of variables wrong and **49 of 17,040 clauses (0.29 %)**
unsatisfied, and stays there. The parked fraction is three times the first-moment estimate `π(4.26) ≈ 0.073`
(§8 assumed a homogeneous δ; the quenched flow parks variables with negative drift at intermediate δ
before the bulk converges), but the parked set is highly structured: 21 % random flips would break
≈ 2 % of clauses, not 0.29 %. The picture of §8 is qualitatively right and quantitatively off by
3× in the parked set and 1.2× in the broken clauses.

**The historical weight rule (`S_j += λ` iff `f_j > 0.999`, else EMA).** Unsatisfied clauses fell
759 → 23 → **19 and stalled**; max weight 4.6; the additive term never fired. The 19 stuck clauses sit at
interior near-frozen points with `f_j < 0.999`. With `K = 3072` real ants the gate "no ant satisfies the
clause" at `f_j = 0.995` has probability `0.995^{3072} ≈ 2·10⁻⁷`. **Theorem 4's completeness mechanism is
inert in practice**: it is gated on an event that a large population makes vanishingly rare. This is a
defect of the intended mechanism, not of an implementation.

**Changed weight rules, tested in order (each change forced by the previous failure):**

| rule | behaviour on the planted instance |
|---|---|
| C: historical gate | stalls at 19 unsat |
| C‴-a: `S_j += λ f_j` when stalled (1-step test), EMA decay otherwise | oscillates around 37 unsat; weights flicker between 1.2 and 5 |
| C‴-b: `S_j += λ f_j` every step for violated clauses, EMA decay otherwise (fast-decay breakout) | cycles: ≈ 30 unsat circulate, weights capped ≈ 10 by the μ = 0.88 forgetting |
| C‴-c: `S_j += λ f_j` every step, **no decay** | **diverges**: 17 → 90 unsat, wrong fraction 21 % → 31 %; weights from the initial transient (778 violated clauses) distort the objective permanently |
| **C‴-d: classical breakout**, `S_j += λ` on violated clauses **only at a stall** (20 consecutive steps with `|Δ Σf| < 10⁻⁴`), no decay | **converges**: 45 → 13 unsat in 12k steps with floor 10⁻⁹; with marginal floor 10⁻³, 35 → 5 unsat in 30k steps, monotone apart from small bumps; run to completion reported below |

The lessons are each a theorem-shaped statement about the mechanism:
1. Weight growth must be **event-driven by local minima**, not by per-step violation (c diverges) and
   not gated on exact zero coverage (C is inert). This is Morris's original rule; the historical
   design departed from it in both directions at once.
2. Weight **decay on the code's time scale (μ = 0.88, eight steps) destroys the memory** the escape
   needs: multi-clause conflicts are resolved only if the weights of all involved clauses persist
   through the flow's response (b cycles).
3. The replicator flow's `p(1−p)` factor makes un-parking a variable at the 10⁻⁹ clamp cost
   `≈ log(1/floor)/(η D) ≈ 400` steps each. The intent's **entropy floor** ("never fully converge") is
   not decoration: a floor of 10⁻³ speeds repair by an order of magnitude. This is the one place the
   documented crisis-controller intent is load-bearing in the mathematics.

**XORSAT (§9 test).** With the historical weights and symmetry-breaking noise 0.05, the flow stays at
the symmetric point for 12,000 steps: `Σ f_j = 1600` exactly (half the XORs), 50.2 % wrong, weights ≤ 1.6.
Theorem 15's fixed point holds in the full mean-field, including weights. Nothing local escapes it.

**System C‴ (current construction).** System C′ (min-break selection) with the weight rule C‴-d and a
marginal floor. In the mean-field it solves the planted instance; the rate is set by the number of
broken clauses (≈ 0.3 % of `m`) times the un-parking cost, i.e. polynomial in `n` with a large constant.
The corresponding first-moment quantities of §8 and §10 should now be recomputed with the quenched
parked fraction (≈ 0.21 rather than 0.073): the cascade branching factor becomes `(6r/7)·0.21 ≈ 0.77`,
still subcritical but close to 1, which is consistent with the slow, bumpy repair observed.

**Open, restated for C‴.** (i) Prove the fixed-weight flow's limit on planted instances is a near-
solution with `o(1)` broken clauses (the 0.29 % should be `Θ(π_q²)` with `π_q` the quenched parked
fraction; a rigorous bound on `π_q` is the local-weak-limit task). (ii) Prove the stall-breakout
repair terminates in `poly(n)` on planted instances (subcritical cascade with `b ≈ 0.77`). (iii) The
worst case remains blocked by §9: C‴ is still local.


---

## 12. Sixth round: the corrected mean-field flow solves planted 3-SAT at threshold density

**Result.** System C‴ (min-break selection is irrelevant in the `K → ∞` limit, which has no local
search; what matters here is the weight rule C‴-d and the marginal floor 10⁻³) integrated exactly on
planted 3-SAT, `n = 4000`, `r = 4.26`, `m = 17040`:

| seed | unsat at 25k | unsat at 50k | final distance to planted `x*` |
|---|---|---|---|
| 7 | 5 (at 30k) | **0** | 21.3 % |
| 1 | 1 | **0** | 22.5 % |
| 2 | 5 | **0** | 20.8 % |

Zero unsatisfied clauses means a satisfying assignment was found; it lies ≈ 21 % from the planted
one, i.e. it is `x* ⊕ (parked set)` repaired, exactly the endgame Theorems 14, 17 and 18 describe. This
is the deterministic `K → ∞` limit: no ants, no sampling, no restarts; the only stochastic input is the
10⁻³ symmetry-breaking noise at `t = 0`.

**What the time is made of.** ≈ 700 steps of bulk convergence (Theorem 12 regime), then ≈ 50,000 steps
to repair ≈ 50 broken clauses: ≈ 1,000 steps per clause, each repair being a stall (20 steps) × the
number of weight increments needed to exceed the critical weight (`Δ/λ ≈ 5–10`) × the flow's own
relaxation from the floor (`log(1/floor)/(η·D) ≈ 100` steps). Repairs are close to sequential because the
stall test is global. So steps-to-solve is `Θ(#broken) · O(1) = Θ(π_q² m)` and total work
`O(n·m) = O(n²)` at fixed density: **polynomial**, with the constant set by the un-parking cost.
The scaling prediction is linear in `n` for the step count; it is tested below.

**Why this is not merely "SLS solves random 3-SAT".** Stochastic local search does solve random
3-SAT at density 4.2 routinely. What is new here is that a *deterministic continuous* dynamics on the
product relaxation, i.e. a naive-mean-field method, reaches a solution at threshold density, where
plain naive mean field and even belief propagation are known to fail to converge. The ingredient that
makes it work is the breakout reweighting acting **on the relaxation** (Theorems 4 and 5): the weights
deform the multilinear landscape until the parked free variables are pushed back. That is the
UMACO idea, "pheromone climbs, crisis reweights", with its defects removed.

**Rigor status.** None of this is proved. It is the exact evaluation of the object the theorems
describe, on three instances. The proof program it supports is stated at the end of §11.


---

## 13. Seventh round: planted versus random, and the flow's algorithmic threshold

**Result on non-planted random 3-SAT.** `n = 4000`, `r = 4.2`, seed 11, same System C‴ mean-field flow:
unsatisfied clauses fall to ≈ 90 within 25k steps and then **circulate between 90 and 115 for
200,000 steps** (0.6 % of `m`), weights capped at the stall-increment ladder ≈ 7.3. The flow sits in
a MaxSAT local optimum far from any solution and the breakout cycles it among such optima. MiniSat's
verdict on the instance is pending (near-threshold random 3-SAT at `n = 4000` is slow for CDCL too);
at `r = 4.2` the instance is satisfiable with high probability.

**Theorem 20 (why planted and random differ for this flow).** Negating every literal of a formula
and every bit of an assignment preserves satisfaction (Lean: `Clause.sat_negate`). The unconditioned
random ensemble is closed under negation, so the expected drift toward any fixed assignment at
`p = ½` is `0` by symmetry. The planted ensemble is not closed under negation (its clauses are
conditioned on `x*`), and Theorem 12 gives expected drift `3r/28 > 0` toward `x*` for every variable
at `p = ½` and `(3r/7)(1−δ)²` at every later stage. So on planted instances the flow has a global
bias toward a solution at all times; on random instances it has none, and it descends into whichever
MaxSAT optimum the sign-noise of the initial gradient selects. The breakout reweighting was shown in
§11 to repair a *near-solution* (parked free variables, ≈ 0.3 % broken); it is not shown to move a
*far-from-solution* optimum toward the solution space, and §13's run says it does not at `r = 4.2`.

**Theorem 21 (critical variables converge once the neighbourhood is right enough).** With the
notation of Theorem 14, if `crit*(v) = c ≥ 1` then `D_v(δ) > 0` whenever
```
c (1 − δ) > (A − B) δ + n₀₁₁ δ² / (1 − δ),
```
where `A = n₀₁₀ + n₀₀₁` and `B = n₁₀₁ + n₁₁₀`. In particular `δ < 1 / (1 + max(A − B, 0) + n₀₁₁)`
suffices. *Proof.* Drop the nonnegative `n₁₁₁ δ²` term and factor `(1−δ)`:
`D_v ≥ (1−δ)[c(1−δ) − (A−B)δ] − n₀₁₁ δ²`. ∎ (Lean: `critical_drift_pos`.)
So on the planted model the only variables that can remain wrong at small `δ` are the non-critical
ones of Theorem 14; the quenched flow of §11 parks 21 % of variables, i.e. it parks more than the
7.3 % that Theorem 16 counts as "always negative", because variables whose drift is negative only at
intermediate `δ` are parked before the bulk converges and, at the floor, return only slowly.

**Algorithmic threshold of the mean-field flow.** The density scan (`r ∈ {3.6, 3.9, 4.05, 4.15}`,
`n = 4000`, 100k steps, found assignments independently recounted) is reported below when it
completes. The quantity it measures, the largest density at which the deterministic C‴ flow reaches a
solution on random 3-SAT, is the analogue of the known algorithmic thresholds
(belief propagation ≈ 3.86, WalkSAT ≈ 4.2, survey propagation ≈ 4.25) and is the number that says
where this UMACO-derived method stands among local methods on the hard class.


---

## 14. Eighth round: the mean-field flow's algorithmic threshold, and the status of the target

**Density scan, random 3-SAT, `n = 4000`, 100k steps, seed 11, assignments recounted independently:**

| r | unsat at 50k | unsat at 100k | solved |
|---|---|---|---|
| 3.60 | 6 | **0** (at 75k) | yes, recount 0/14,400 |
| 3.90 | 40 | 29 (falling slowly) | no within budget |
| 4.05 | 74 | 59 | no |
| 4.15 | 78 | 76 | no |
| 4.20 | ≈ 92 | ≈ 100 (cycling, 200k steps) | no |

So the deterministic C‴ flow's algorithmic threshold on random 3-SAT is **between 3.6 and 3.9**, next to
belief propagation's 3.86 and well below WalkSAT (≈ 4.2) and survey propagation (≈ 4.25). That is the
expected place for a naive-mean-field method with reweighting, and it is consistent with Theorem 20:
on random instances the flow has no global bias, and the breakout weights repair near-solutions but do
not transport a far MaxSAT optimum toward the solution space.

**Theorem 22 (convergence of the fixed-weight ascent), formalized.** Along any coordinate schedule
with `0 ≤ η ≤ 1` and `F ∈ [0,1]`, the trajectory stays in the cube, the per-step gains are
`η p_v(1−p_v)·slope² ≥ 0`, their partial sums telescope to at most `1`, so the gains are summable and
tend to `0`. Every limit point is stationary in the sense of Theorem 2. (Lean: `repStep_inCube`,
`traj_inCube`, `gain_eq`, `sum_gain_le_one`, `gain_tendsto_zero`.)

**Where the target stands after eight rounds.**

*Proved (Lean, 58 theorems, no `sorry`):* the multilinear extension as an object with its conditioning
and slope identities; solutions are global maxima of the extension for every weighting; the pheromone
ascent is monotone and convergent to the stationary set; vertex stability = strict 1-flip local
optimality; escape-time bound for the additive weight term; EMA weights are bounded by ≈ 6 regardless
of history; the uniform-selection toward-probability `W/k` and its `k = 3` crossover `δ_c = (3−√5)/2`;
weighting the fixed clause cannot raise it (Theorem 7); the wrong/right break asymmetry (Lemma 9); the
annealed planted drift `(3r/7)(1−δ)²`; the XOR encoding is the 3-spin energy with a symmetric fixed
point; the repair-cascade branching factor is below `1/e`; negation symmetry kills the drift on the
random ensemble; the critical-variable convergence condition.

*Established by exact evaluation of the derived object (not proved):* the corrected mean-field flow
solves planted 3-SAT at threshold density (3 seeds, `n = 4000`); its random-3-SAT threshold is
3.6–3.9; it is pinned at the symmetric point on XORSAT; the historical weight gate is inert; fast decay
cycles; no-decay diverges; stall-breakout with an entropy floor is the rule that works.

*Obstructions to the target (poly-time 3-SAT via a UMACO-derived algorithm):*
1. **Locality (§9).** Every construction in the lineage, corrected or not, is a function of bounded
   formula neighbourhoods. XOR instances (a 3-CNF family, present in the repo's own evidence suite)
   pin the mean-field at its symmetric fixed point (Theorem 15, confirmed numerically with weights on),
   and the overlap-gap literature rules out local/stable algorithms on such families. The only
   non-local operation in the intended architecture, the SVD burst, is linear over the reals, not
   GF(2).
2. **Random 3-SAT near threshold (§13–14).** Without a planted signal the flow lands in far MaxSAT
   optima; the threshold 3.6–3.9 is that of naive mean field, and even survey propagation, the best
   local method, stops at ≈ 4.25 < 4.267.
3. **Worst case.** Nothing here touches adversarial 3-CNF beyond XOR; a poly-time bound for every
   satisfiable 3-CNF would be P = NP and no potential-function argument available to these dynamics
   (Theorems 1–5, 22) controls the number of escapes.

*What the corrected UMACO-derived algorithm is:* a naive-mean-field ascent on the product relaxation
with breakout reweighting and an entropy floor, i.e. a legitimate member of the local-algorithm family,
sitting near BP on random 3-SAT and at the planted threshold on planted instances. The finite-`K`
version adds min-break local search, which by the SLS literature should lift the random threshold toward
WalkSAT's 4.2; that is the next construction to evaluate, and it does not change the obstructions.


---

## 15. Ninth round: the finite-K algorithm, and the pheromone learning rate

`research/system_c/systemc3_gpu.py` is the corrected algorithm with a finite population: product construction
from the marginals with an entropy floor, min-break local search on the top 20 % of the current
samples, K-normalised leaky-integrator deposit, breakout weights raised only when the best-so-far
stalls, host recount of the final assignment. GPU only.

**Theorem 23 (learning rate of the fitness-weighted deposit).** Let the deposit on polarity `b` of
`v` be `Σ_a F(x^{(a)}) [x^{(a)}_v = b]`. The per-iteration change of the log-odds is
`log g_v = log E[F | x_v = 1] − log E[F | x_v = 0]`.
- For `F = Q^γ` with `Q ∈ [q_min, q_max]` the deposit ratio is within `(q_max/q_min)^γ` of the frequency
  ratio (`deposit_ratio_bound`), and since `Q` is a *fraction* of clauses, a single variable moves it by
  at most `deg(v)·w_max / W`; so `|log g_v| ≤ γ·log(1 + deg(v)·w_max/(W q_min)) = O(γ·deg/m)`. With
  `m ≈ 4n` this is `≈ 10⁻⁴` per iteration: the pheromone needs `Θ(m/deg) = Θ(n)` iterations to move one
  variable's log-odds by `O(1)`. **The historical `Q^{3/2}` fitness cannot learn on any practical time
  scale; the α = 3.5 sampling exponent was an unstable substitute for a learning rate** (§ III.4 of the
  lineage document).
- For Boltzmann fitness `F = exp(β·C)` with `C` the weighted satisfied *count*, `log g_v` is a
  log-moment-generating difference, `≈ β·∂_v E[C] = O(β·deg)` for small `β`. Choosing `β = η` makes the
  finite-K pheromone move at the same rate as the mean-field flow of §11 (which integrated the count
  gradient with `η = 0.05`).

**Effect, same instance (`rand_3.9.cnf`, `n = 4000`, `K = 256`, 3,000 iterations):**

| fitness | best unsat | pheromone entropy at end |
|---|---|---|
| `Q^{3/2}` (historical) | 1,314 (8.4 %) | 0.904 (never left uniform) |
| `exp(0.05·C)` | **69** (0.44 %) | 0.091 (converged) |

The Boltzmann run then stalls at 69 with the population converged (all ants within a few random flips
of one assignment at floor 10⁻³) and 20-flip local search unable to finish; the stall-breakout raises
weights only every 10 stalled iterations, too slowly to matter in 3,000. The mean-field flow reached 29
on this instance after 100,000 steps; the finite-K version reaches 69 in 3,000. A run with a larger
flip budget (`F = 100`) and floor (`10⁻²`, i.e. ≈ 40 random flips per ant) is reported below.

**What this is and is not.** This is the "corrected exact algorithm" the target allowed, evaluated once
to check the derivation's predictions (learning rate, endgame by local search, diversity by floor). It
is not a tuned solver and it has not been compared with WalkSAT on the same files; the theory of §9
and §14 says that comparison cannot change the P = NP status either way.

**Larger budget (`F = 100`, floor `10⁻²`, `P = 5`, same instance, 3,000 iterations):** best unsat **23**
(host recount 23), versus 69 with `F = 20` and the mean-field's 29 at 100,000 steps. The endgame is the
local search's, as Lemma 9 and Theorem 7 predicted, and diversity from the floor matters. Not solved.


---

## 16. Tenth round: does the intended `n×n` pairwise pheromone escape the XOR obstruction?

The intent documents describe an `n×n` complex field, real = attraction, imaginary = repulsion. The
only SAT reading with content is pairwise: `τ_{uv}` records how often `x_u = x_v` (real) versus
`x_u ≠ x_v` (imaginary) among good assignments, and construction assigns variables sequentially,
conditioning each on already-assigned neighbours through `τ`. That is a pairwise estimation-of-
distribution algorithm (MIMIC/BOA family) with Hebbian co-occurrence learning; the Umaco13 SAT
deposit on edges `(i, i±1)` was a degenerate version of it.

**Theorem 25 (pairwise statistics of an XOR solution space are trivial).** Let `A ⊆ GF(2)^n` be the
solution set of a linear system (an affine subspace) and `φ(x) = x_u ⊕ x_v ⊕ c` any affine pair
functional. Then `φ` is either constant on `A` or takes each value on exactly half of `A`. Hence the
pairwise correlation `E_A[(2x_u−1)(2x_v−1)]` of the uniform solution measure is `±1` or `0`; the
same holds for any affine functional (triples, and so on).
*Proof.* If `φ` is not constant on `A`, pick `a, b ∈ A` with `φ(a) ≠ φ(b)` and let `d = a ⊕ b`, a
direction of `A` with `φ(x ⊕ d) = φ(x) ⊕ 1`. Translation by `d` is a fixed-point-free involution of `A`
that flips `φ`; the value classes are in bijection. ∎ (Lean: `card_filter_eq_of_involution`,
`signed_count_zero_of_involution`.)

**Consequences for the pairwise pheromone on XORSAT.**
- The information a pairwise field could ever learn from *solutions* is exactly the set of pairs with
  correlation `±1`, i.e. the pair equalities `x_u ⊕ x_v = c` implied by the system: the 2-XOR
  consequences that Gaussian elimination derives directly. Learning them from a population requires
  the population to sample solutions, which is the task.
- Away from solutions, the population sits at the symmetric point (Theorem 15) where every pairwise
  correlation is `0` by the same symmetry that pins the marginals: the joint law of any pair under the
  product measure at `½` is uniform, and the flow's drift on `τ_{uv}` is a sum of terms each containing a
  third variable's magnetization, hence `0`.
- A sequential sampler conditioned on a pairwise field is a bounded-degree local computation; it does
  not leave the local class.

So the `n×n` intent does not buy what XOR needs, and cannot: the obstruction is algebraic over GF(2),
and no real-valued statistic of bounded order distinguishes the solution space of a random 3-XORSAT
system from the symmetric point except through the frozen pairs that elimination would have found.
Recording this closes the last construction change suggested by the architecture documents.

## 17. What is rigorous about the planted bulk phase

The first-moment computations of §8 become theorems through one observation: **`T` steps of the
mean-field flow at variable `v` are a function of the radius-`T` neighbourhood of `v` in the factor
graph** (each step reads the marginals of `v`'s clause-neighbours, which after `T−1` steps depend on
their own radius-`(T−1)` neighbourhoods). For the random planted formula at density `r`, the
radius-`T` neighbourhood converges in distribution to a two-type Galton–Watson tree (variables have
Poisson(3r) clauses, each clause two further variables, sign patterns uniform on the seven planted-
satisfying patterns) — local weak convergence of sparse random hypergraphs. Therefore:

**Theorem 26 (bulk phase, bounded time).** For each fixed `T` and `η`, the empirical distribution of
`(p_v(T), x*_v)` over variables converges in probability, as `n → ∞`, to the law obtained by running
the same `T` steps on the Galton–Watson tree. In particular the fraction of variables with
`p_v(T)` on the wrong side converges to a deterministic `δ_T(r, η)` computable on the tree, and
`E[D_v(0)] = 3r/28`, `Var = O(r)` (so a constant fraction `g(½) ≈ 0.26` start wrong, Chernoff-
concentrated). *Proof.* Bounded-radius dependence plus local weak convergence; the tree law is
computable because each step on the tree is a finite recursion. ∎ (Standard; not formalized.)

What Theorem 26 does **not** give is the `T → ∞` limit (the parked fraction 0.21 of §11 is a
long-time quantity) nor the repair phase, which is where the flow's history correlations live. The
gap between "bounded time on the tree" and "convergence of the full flow" is the same gap as in the
rigorous analyses of BP/WP on planted instances (Feige–Mossel–Vilenchik 2006, where it is closed only
for large density). At threshold density it is open for every local method, including this one.


---

## 18. Eleventh round: symmetry, and exactly why XOR is the bad case

**Theorem 27 (equivariant dynamics are pinned at ½ on self-complementary formulas).** Call a
marginal update `U(φ, p)` negation-equivariant if `U(neg φ, 1−p) = 1 − U(φ, p)`. Every rule in this
document is (product sampling, fitness deposit, coverage weights, min-break selection, breakout).
If `neg φ = φ` then `p ≡ ½` is a fixed point of `U(φ, ·)` and of all its iterates. Self-complementary
satisfiable formulas exist: a clause and its negation are jointly satisfiable, e.g. `(a ∨ b ∨ ¬c)` and
`(¬a ∨ ¬b ∨ c)` by `a = 1, b = 0`. (Lean: `equivariant_fixed_half`, `equivariant_iterate_half`,
`self_complementary_example`.) So symmetry breaking by exogenous noise is not an implementation
detail; it is required, and the question is how fast the noise grows.

**Theorem 28 (linearization at ½; XOR is degenerate).** With `q_ℓ = ½ − s_ℓ(p_u − ½)` for a literal of
sign `s_ℓ = ±1` on `u`, the drift `D_v = Σ_{j∋v} w_j s_{jv} ∏_{ℓ∈C_j∖v} q_ℓ` is affine in each `p_u`
with slope `−w_j s_{jv} s_{ju} q_{third}` per clause (Lean: `clause_drift_affine`). At `p = ½`:
```
∂D_v/∂p_u |_{½} = M_{vu} := −½ Σ_{j ∋ u,v} w_j s_{jv} s_{ju},
```
so the flow near ½ is `δṗ = (η/4)·M·δp` and noise grows at rate `(η/4)·λ_max(M)`. For a generic
3-CNF, `M` is a sparse random ±½-weighted matrix with `λ_max = Θ(√deg) > 0`: symmetry breaks
exponentially fast (the planted run left ½ within a few hundred steps). For the four clauses encoding
one XOR, the pair contribution is `Σ_{a : parity fixed} (−1)^{a_u + a_v} = 0` (Lean:
`xor_pair_sign_sum_zero`), so **`M ≡ 0` on XOR-encoded formulas**: the symmetric point is degenerate,
the escape is driven only by the cubic term `m_{j1}m_{j2}` of Theorem 15, and the time to leave
scales like `1/(η·noise)` instead of `log(1/noise)/(η λ_max)`. This is the exact mechanism behind the
12,000-step pin of §11 and it is specific to parity structure: it is the algebraic content of
"XOR has no pairwise signal", the same fact as Theorem 25 seen from the dynamics.

**Consequence for constructions.** Any change that keeps the dynamics negation-equivariant and
local cannot help on XOR beyond speeding the cubic escape into a threshold state. Breaking
equivariance deliberately (a biased initialization) is not information about the instance. This
closes the symmetry line: the remaining barrier is the p-spin landscape after escape, §9.

**Size check for Theorem 26's limit (fixed-point parked fraction, historical weights, 1,500 steps):**
`n = 4,000`: 20.3 % parked, 15/17,040 clauses broken; `n = 16,000`: 20.0 % parked, 69/68,160 broken.
The long-time fixed point has a deterministic limit (≈ 0.20 parked, ≈ 0.1 % broken), consistent
with a local-weak-limit description of the whole bulk phase, not only its bounded-time prefix.

**Correction to Theorems 17–18 from the size check.** With the quenched parked fraction `π_q ≈ 0.20`,
independence would predict a broken fraction `(3/7)π_q² + (1/7)π_q³ ≈ 1.8 %`; the flow shows
0.1–0.3 %. The parked set at the fixed point is therefore strongly **anti-correlated with clause
co-membership**: parking both backups of a two-true clause makes it unsatisfied, whose gradient then
pushes one backup back, so the fixed point avoids double parking (a hard-core-like constraint on the
parked set). This is favourable and it is a property of the fixed point, not an independence
assumption; the right statement of Theorem 18 is a bound on the *self-consistent* parked
configuration. The cascade branching factor with `π_q = 0.20` is `(6r/7)·0.20 ≈ 0.73 < 1`, still
subcritical, and the anti-correlation only lowers it.

---

## 19. The XOR family from both sides

The instances that pin every construction in this lineage (§9, §16, §18) are Tseitin formulas, and
Tseitin formulas on expander graphs are also the classical exponential lower bound for resolution
(Urquhart 1987), hence for every CDCL solver. So one family blocks the two large algorithm classes at
once: local search (via freezing and the overlap gap on the satisfiable side; via the symmetric fixed
point and `M ≡ 0` for the mean-field dynamics here) and resolution-based search (via proof size on the
unsatisfiable side). What beats it is Gaussian elimination over GF(2), which is neither local nor a
resolution step. A polynomial-time 3-SAT algorithm derived from UMACO would therefore need a step
that is (i) non-local, (ii) not resolution, (iii) not merely GF(2), since families hard for
resolution-with-parity are also known. The intended architecture contains one non-local operation,
the real-linear SVD burst; nothing in the lineage, corrected or not, meets (i)–(iii).

This is not a proof that no such step exists. It is the precise shape of what the next construction
would have to be, and it lies outside the design space of the documents.

## 20. Locality, formally

`iterate_local` (Lean): if an update `F` reads only the neighbourhood `N v` of each variable, then
`F^[T] p v` depends only on `p` restricted to the radius-`T` ball of `v`. Every update in this
document is of that form with `N v` = the variables sharing a clause with `v` (the weights add a
clause layer, still bounded radius per step). This is the hypothesis under which Theorem 26 and the
overlap-gap obstruction apply to Systems C, C′, C‴ and their finite-K versions alike.

## 21. General k

**Theorem 30.** For planted k-SAT at density `r`, the annealed drift toward `x*` is
`(k r / (2^k − 1)) · (1−δ)^{k−1}` (Lean: `pattern_sum_one`, `annealed_drift_general`), and the
uniform-selection toward-probability is `δ / (1 − (1−δ)^k)`, which is `≥ ½` for all `δ` when `k = 2`
(`= 1/(2−δ)`; Lean: `uniform_toward_k2`) and crosses `½` at a `δ_c(k) ∈ (0,1)` for every `k ≥ 3`
(`δ_c(3) = (3−√5)/2`). So the tunnel of §5 is the `k ≥ 3` phenomenon exactly, the planted signal
weakens like `2^{−k}` per clause, and both facts are the same pattern sum viewed from the pheromone
and from the local search respectively. The `k = 2` base case therefore delivered precisely the
machinery that was required and nothing that fails to generalize: the pattern sum, the toward-count,
and the crossover equation are the same objects at every `k`.

## 22. Finite-K results at densities 4.05 (and what elitism does)

| variant (`n = 4000`, `K = 256`, `F = 100`, floor 10⁻², 4,000 iterations) | best unsat | pheromone entropy | note |
|---|---|---|---|
| C‴ finite-K, no elitism, density 4.05 | **30** (recount 30) | 0.15 | population converged; weights climb to 82 without effect because searched assignments are discarded each iteration |
| C‴ + 26 elites carried over, density 4.05 | 61 | 0.81 | pheromone never converges: Boltzmann weight concentrates on the diverse elites, whose marginals average to ½; fresh samples stay poor (mean 1,060 unsat) |

Mean-field flow on the same instance: 59 at 100k steps. So local search lifts the finite-K result to 30
at density 4.05, and the naive elitism that would let the walks continue defeats the pheromone.
The two roles conflict as implemented: the pheromone can represent a *consensus* (product measure
around one assignment) or an *elite set* (a mixture), not both in one product table. A mixture would
need several pheromone tables, which is the "multiple colonies" idea the developer guide attributes to
the ZVSS file, where it does not exist. That is a construction change with a clear rationale and it
is where this line stops for now; it is engineering, and it does not alter §9, §16 or §18.
