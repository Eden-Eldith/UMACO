# System U: theorems and derivations

Companion to `system_U.md` (the definition). Everything here is about System U as settled on
2026-09-11: literal-graph walk with free order, interference weight `w = max(0, Re(Φ²))`, dynamic
heuristic `η`, performance = satisfied fraction, panic and anxiety as maps, the five-regime burst.
Nothing from `research/system_c/SystemC_3SAT_attack.md` is assumed.

Notation. Literals `ℓ ∈ L = {1..n} × {T, F}`, `2n` of them; `var(ℓ)`, `¬ℓ`. Field `Φ ∈ ℂ^{2n×2n}`
(plus a start row `Φ_{s,·}`). Edge weight `w_{ab} = ε + max(0, Re(Φ_{ab}²))` with a floor `ε > 0`
(see U1c for why the floor is not optional). A **partial assignment** `σ` is a set of literals with
distinct variables; `U(σ)` = literals of unassigned variables. Heuristic `η(b | σ) > 0`.

---

## U1. The law of the walk on a fixed field

An agent's walk is the Markov chain on states `(a, σ)` (current literal, partial assignment) with

```
p(b | a, σ) = w_{ab}^α · η(b|σ)^β / Z(a, σ),      Z(a,σ) = Σ_{b' ∈ U(σ)} w_{ab'}^α η(b'|σ)^β,   b ∈ U(σ),
```
started at `(s, ∅)`, stopped after `n` steps. A **path** is `π = (b_1, …, b_n)`, each `b_t ∈ U(σ_{t−1})`,
`σ_t = σ_{t−1} ∪ {b_t}`; it determines the complete assignment `x(π) = σ_n` and the order.

**U1a (path law).** `P_Φ(π) = ∏_{t=1}^{n} p(b_t | b_{t−1}, σ_{t−1})` with `b_0 = s`. The assignment law
is the mixture over orders: `P_Φ(x) = Σ_{π : x(π) = x} P_Φ(π)`, a sum over the `n!` orderings of `x`'s
literals. *Proof.* Markov property by construction. ∎

**U1b (sanity: the flat field is uniform).** If `w ≡ c` and `η ≡ 1`, then `p(b | a, σ) = 1/(2(n−t))` at
step `t+1`, every path has probability `∏_t 1/(2(n−t)) = 1/(2^n n!)`, and `P_Φ(x) = n!/(2^n n!) = 2^{−n}`.
So the walk starts uniform, as the product sampler did, but for a different reason (all orders
equally likely, all polarities equally likely). ∎ (Lean: `flat_walk_uniform`.)

**U1c (support, and why the floor is required).** With `ε > 0`, every `p(b|a,σ) > 0`, so every
assignment has positive probability under every field: the walk can always reach every solution.
With `ε = 0`, an assignment `x` is reachable iff the literals of `x` admit an ordering `b_1…b_n` with
`w_{b_{t−1} b_t} > 0` for all `t`, i.e. iff the subgraph of `x`'s literals with positive-weight edges has
a Hamiltonian path from `s`. Repulsion (`Im Φ ≥ Re Φ` on an edge) can therefore **disconnect**
assignments, including solutions, and deciding reachability is itself hard. So the interference weight
needs the floor: repulsion may make an edge arbitrarily unlikely, never impossible. This is the
architecture's "target entropy / never fully converge" intent, in the form it takes on the literal
graph. ∎

**U1d (what the field row is).** `p(· | a, σ)` depends on `Φ` only through the row `Φ_{a,·}`
restricted to `U(σ)`, and on the formula only through `η(· | σ)`. So the walk is *field-row local* and
*partial-assignment global*: the field couples `a` to every literal it has ever been followed by, and
`η` sees every clause touched by `σ`. Neither is bounded-radius in the factor graph. ∎

---

## U2. What the deposit–evaporation pair ascends (the score-function identity)

Parameterize the transition weights by `θ_{ab} = α·log w_{ab}` so that
`p(b|a,σ) = exp(θ_{ab} + β log η(b|σ)) / Z(a,σ)`. Let `f(π) ≥ 0` be a performance of the completed
assignment (System U uses `Perf = satisfied fraction`, raised to `η_dep` in the deposit). Define the
**objective** `J(θ) = E_θ[f(π)]`, the expected performance of a sampled walk.

**U2a (gradient of the walk's expected performance).**
```
∂J/∂θ_{ab} = E_θ[ f(π) · Σ_{t=1}^{n} ( 1[b_{t−1} = a, b_t = b] − 1[b_{t−1} = a] · p(b | a, σ_{t−1}) ) ].
```
*Proof.* Score-function identity: `∂_θ E[f] = E[f · ∂_θ log P_θ(π)]`, and
`log P_θ(π) = Σ_t [θ_{b_{t−1}b_t} + β log η − log Z(b_{t−1}, σ_{t−1})]`, with
`∂ log Z(a,σ)/∂θ_{ab} = p(b|a,σ)·1[b ∈ U(σ)]`. ∎ (Lean: `softmax_score_identity`, one step.)

**U2b (the ACO update is a biased stochastic gradient step on J).** The deposit
`ΔΦ_{ab} = I·f(π)^{η_dep}` for each edge `(a,b)` used by the agent is, in the log-weight coordinates,
the first term of U2a for one sample (up to the monotone reparameterization `f ↦ f^{η_dep}` and the
map `Φ ↦ θ = α log w`). The evaporation `Φ ← (1−ρ)Φ` on every edge stands in for the second term,
`−f · E[usage of (a,b) | visits to a]`, which is the baseline that would make the estimator unbiased.
So: **System U's field update is stochastic gradient ascent on the expected performance of the walk,
with evaporation as a uniform surrogate for the expected-usage baseline.** This is the model-based-
search reading of ant systems (Zlochin, Birattari, Meuleau, Dorigo 2004), instantiated on the
literal graph. ∎

Consequences, none of which need a product measure:
- **U2c (stationary fields).** Fixed points of the exact (unbiased) update are stationary points of
  `J`. `J` is maximized by fields that put the walk's mass on maximum-`f` paths; on a satisfiable
  formula those are paths through solutions (`f = 1`).
- **U2d (non-concavity).** `J` is a smooth non-concave function of `θ` (a softmax mixture), so
  stationary points that are not solutions exist in general: fields concentrated on a set of high-`f`
  non-solution paths. This is the System-U analogue of the local optima of Systems C, but on
  *paths*, not on product marginals, and the escape mechanism is the burst (U3), not clause weights.
- **U2e (where the interference weight enters).** `θ = α log(ε + max(0, Re(Φ²)))`. The additive
  updates (deposit, burst, momentum) act on `Φ`; the walk sees `θ`. Repulsion moves `θ` down
  smoothly through `Re(Φ²)`; the floor keeps `θ > −∞`. The burst's five regimes are therefore
  moves in `θ`-space of a specific shape: rank-`k` in `Φ`, hence *not* rank-`k` in `θ`, because of the
  square and the log; this is what U3 has to compute.

---

## U3. The burst as an operator

Write `Φ = R + iM` entrywise. The burst (system_U.md §5) is
```
Φ ← Φ + g ⊙ e^{iφ} ⊙ B,      B = U_k Σ_k V_kᵀ  (rank-k part of R),   g = γ_s · f_scale(P) ≥ 0,   φ = angle(Ψ),
```
with `g` and `φ` maps (per edge) since panic and anxiety are maps. Everything after the SVD is
entrywise, so the whole regime table is one identity in `(g, cos φ, sin φ)`; the SVD only decides
*which* entries are "captured" (`B_ab ≈ R_ab`). The random part `γ_r ΔΦ_rand` is treated in U3e.

**U3a (entrywise burst identity).** On an edge with `M_ab = 0` and `B_ab = R_ab = r` (a captured
edge), the new weight is
```
w' = ε + max(0, r² · F(g, φ)),        F(g, φ) = (1 + g cos φ)² − (g sin φ)² = 1 + 2g cos φ + g² cos 2φ.
```
At full strength `g = 1`: `F(1, φ) = 2 cos φ (1 + cos φ) = 4 cos φ cos²(φ/2)`. In log-weight
coordinates the burst is a **uniform shift** `Δθ = α log F` on every captured edge (for `r² ≫ ε`),
i.e. a tempering of the walk's current mode by the factor `F^α`, not a per-edge noise. On a
partially captured edge (`B_ab ≠ R_ab`, or `M_ab ≠ 0`) the same formula reads
`w' = ε + max(0, (R_ab + g cos φ B_ab)² − (M_ab + g sin φ B_ab)²)`. ∎ (Lean: `burst_factor`,
`burst_on_captured`, `burst_full_factor`.)

**U3b (the five regimes, computed).** At `g = 1` on captured edges:

| regime | φ | `F(1, φ)` | effect on `w` | left behind in `M` |
|---|---|---|---|---|
| reinforce | 0 | 4 | `w ← ε + 4r²` (`Δθ = +2α log 2`) | 0 |
| "mixed" | (0, π/2) | `4 cos φ cos²(φ/2)` ∈ (0, 4) | **net reinforcing for φ < 68.5°** (`cos φ > (√3−1)/2`), suppressing only on (68.5°, 90°) | `r sin φ` |
| repel | π/2 | 0 | erased to the floor `ε` | `r` (threshold) |
| negate + repel | (π/2, π) | ≤ 0 | erased to the floor; residual attraction `(1+cos φ) r` remains | `r sin φ` |
| negate | π | 0 | erased to the floor | 0 |

So at full strength the captured edges are erased **iff** `cos φ ≤ 0`, i.e. iff `φ ≥ π/2`
(`burst_full_erase_iff`). The sector `(0, π/2)`, described in the thesis table as "part reinforce,
part convert to repulsion", is *net reinforcing* over most of its range (`burst_full_net_reinforce_iff`):
the first-order term `2g cos φ` is attraction, repulsion enters only at second order. At general
strength the crossover is `cos φ* = (−1 + √(1 + 2g²)) / (2g)`, which tends to `π/2` as `g → 0`.
Two further facts: **no phase erases anything when `g < 1/√2`** (`F > 0` for all φ,
`burst_no_erase_of_small`; at `φ = 3π/4`, `F = 1 − √2 g` exactly, `burst_factor_three_quarter`, so the
threshold is sharp). Below that strength the burst only re-scales the mode; above it, erasure
is possible and at `g = 1` it happens for the whole closed half-plane `φ ∈ [π/2, π]`. ∎

**U3c (what distinguishes π from π/2).** Both erase the captured edges at `g = 1`. The
difference is the memory. In the π regime `M` is untouched: the erased edge sits at the floor
and the very next deposit rebuilds it from `ε` (memoryless erase). In the π/2 regime `M_ab = r`
persists, and by `w = ε + max(0, R² − M²)` the edge stays at the floor until its rebuilt
attraction exceeds its old strength: `w = ε ⟺ R_ab² ≤ M_ab²` (`iw_eq_floor_iff`). Repulsion is
therefore a **per-edge threshold** that the population must out-deposit before the old mode becomes
visible again; it is not a subtraction. In `(π/2, π)` the edge keeps a shrunken attraction
`(1 + cos φ) r` under a threshold `r sin φ`; both decay by evaporation, the threshold only if
evaporation acts on `M` too (it does: `Φ ← (1−ρ)Φ` is complex). This is the exact content of
"discourages already explored areas" (thesis §3.2) on the literal graph. ∎

**U3d (what the SVD captures on a field concentrated on paths).** Two exact cases.

1. *A single path.* `R = c · A_π` with `A_π` the 0/1 edge matrix of one path is a partial
   permutation matrix: all `n` non-zero singular values equal `c`. The rank-`k` truncation is
   **not unique** (any `k`-dimensional subspace of the degenerate space), so on a single sharp path the
   burst erases an SVD-routine-dependent `k` of the `n` edges. The "dominant structure" the burst is
   meant to find does not exist in a single path.
2. *A population agreeing on an assignment `x` and varying the order.* Deposits then fall on all
   edges among the `n` literals of `x`; in the limit of all orders `R` restricted to those literals is
   the all-ones block with the diagonal removed, `J − 1`. It decomposes as `(n−1)·P − (1 − P)` with
   `P = J/n` an orthogonal projection (`mode_block_decomp`, `mode_projection_idem`), so its singular
   values are `n − 1` (once) and `1` (`n − 1` times) and its rank-1 part is `(n−1)·P`. A full π
   burst with `k ≥ 1` leaves every edge of the mode at `1/n` (`mode_block_after_burst`): the weight
   drops from `ε + 1` to `ε + 1/n²`, the residual energy is a `1/n` fraction of the original.

So the SVD finds **unordered co-selection**, which literals the good walks use together, not the
order they use them in. The population's agreement on an assignment is a rank-1 block on that
assignment's literals, and that block is exactly what the burst removes (π) or thresholds (π/2). ∎

**U3e (directedness: descent, not noise).** Let `‖·‖` be the Frobenius norm, so the field's energy
`‖R‖² = Σ_i σ_i²`. `B` is the orthogonal projection of `R` onto the span of its top-`k` singular
directions, hence `⟨R, B⟩ = ‖B‖² = Σ_{i≤k} σ_i²` and
```
‖R − B‖² = ‖R‖² − ‖B‖² = Σ_{i>k} σ_i²                     (burst_energy_drop)
```
and every edge weight after a full π burst is at most `ε + Σ_{i>k} σ_i²` (`entry_sq_le_frob`). Within
the captured subspace, `B` is the perturbation of its size most aligned with `R`, so `−B` is the
steepest energy-descent direction there (`projected_best_direction`); across all rank-`k` directions
this is the Ky Fan maximum principle (cited). A perturbation orthogonal to the field, which is the
mean behaviour of the isotropic random part `γ_r ΔΦ_rand`, *raises* the energy:
`‖R − Δ‖² = ‖R‖² + ‖Δ‖²` (`orthogonal_noise_energy_rise`). So the structured burst is a directed
step and the random part is exploration; they are not the same kind of move.

On a field of rank `≤ k` the full π burst leaves `R = 0`; all weights equal `ε`, and the walk's law
becomes `η^β / Σ η^β`, independent of `ε` and `α` (`flat_law_indep`): the walk returns in one step to
the heuristic-guided flat walk of U1b. ∎

**U3f (what "escape from the current mode of J" means, and what it does not).** By U2, a mode of `J`
is a field concentrated on a set of high-`f` paths. The burst is *not* a gradient step on `J`:
at a stationary point the U2a gradient on the mode's edges is ≈ 0 (usage ≈ expected usage), while
the burst steps along `−B`, the mode's own direction, by a definite amount (U3a: a uniform
`Δθ = α log F` on the captured edges). It is descent out of the mode in the energy/entropy sense of
U3e, and it lowers `J` if the mode was a strict local maximum, since every direction does.

The limit of what it can do: the burst acts on **edges**, i.e. on order and adjacency structure,
while `f` depends only on the **assignment**, and the map path ↦ assignment is `n!`-to-1 (U1a).
Erasing the mode's edges removes the walk's preference for *that assignment's literals following
each other*; by U1c the assignment stays reachable through every order, at floor weight. After a
burst the search is a restart into the `η`-guided flat walk (U3e) with memory only in `M`
(π/2 regime: per-edge thresholds on the old mode's literal block) or in the residual `R − B`
(π regime). The question this leaves, and the one that decides anything about polynomial time, is
the number of burst cycles needed: how the sequence of erased or thresholded blocks covers the
assignment space. That is a statement about the chain (walk → deposit → mode → burst → walk) and is
target 5/6 (XOR, planted, random) for System U.

**Q15, settled (author, 2026-09-11).** The net reinforcement of the "below target, stagnating" sector is
intended: an escalation cascade in which mild failure first reinforces the mode (exploit and saturate
it), and accumulated anxiety and panic rotate and strengthen the burst until `F(g, φ)` crosses zero.
U3b is therefore the exact map of that cascade: the crossing is `cos φ* = (−1 + √(1 + 2g²))/(2g)` and
no crossing exists below `g = 1/√2`. Negation happens only once continued commitment is itself the
evidence of failure.

---

## U4. Symmetry

Let `σ` be a permutation of the `2n` literals that commutes with negation (`σ(¬ℓ) = ¬σ(ℓ)`): a
permutation of the variables composed with negations of some of them. Transport everything along it:
the formula `F ↦ σF` (relabel every clause), the field `Φ ↦ P_σ Φ P_σᵀ` (and the start row), the
panic and anxiety maps likewise.

**U4a (System U is equivariant).** The law of the whole System U trajectory started from
`(σF, P_σ Φ₀ P_σᵀ)` is the transport of the law started from `(F, Φ₀)`. *Proof, component by component.*
Walk: one step is equivariant because the transition probability is a ratio of transported
quantities over a transported candidate set (`walk_step_equivariant`); the path law is the product
(U1a). `η`: a function of the partial assignment and the clauses, both transported. Deposit: along
transported paths, same amounts. Evaporation: scalar. Interference weight: entrywise. Burst: the SVD
of `P R Pᵀ` is `(PU) Σ (PV)ᵀ`, so the rank-`k` part is transported (up to tie-breaking on degenerate
spectra, which is a routine's choice, not architecture). Homology: built from the distance matrix of
`Re Φ`, a permutation-invariant construction, so the diagrams, entropy and persistence are unchanged.
Percentile reset: invariant. Economy: reads performances only. ∎

**U4b (invariance does not force flatness; the pin does not exist for U).** Suppose `F` is invariant
under a group `G` of such `σ` and `Φ₀` is `G`-invariant. By U4a the expected dynamics keep `Φ`
`G`-invariant. For the product samplers this was fatal: a per-variable field invariant under a flip of
that variable has marginal `1/2`, so on formulas whose symmetry group moves every variable (XOR
systems) the expected dynamics never left the uniform distribution. On the literal graph a
`G`-invariant field is merely one in the `G`-invariant subspace of literal-pair weights, and that
subspace contains correlation structure: for two variables `x, y` and the pair flip, the field
`w(x,y) = w(¬x,¬y) = 2`, `w(x,¬y) = w(¬x,y) = 1` is invariant and says "x and y agree"
(`invariant_field_not_flat`). So the fixed points of U's expected dynamics on a symmetric formula
are *not* confined to the flat field, and the equivariance argument gives no obstruction for U. The
question of what U's expected dynamics do on XOR has to be answered by computing the deposit signal
directly, which is U5. ∎

---

## U5. XOR for System U: what the literal field receives

Take a system of `m` XOR constraints on `n` variables, each on three distinct variables, encoded as
four 3-clauses each (`xorCNF` for one constraint). `f(x) = 1 − (violated constraints)/(4m)` up to the
constant: one clause per violated constraint.

**U5a (pairwise balance: the field gets no first-moment signal).** Under the flat walk (U1b, `η ≡ 1`)
every edge `(a, b)` between literals of distinct variables is used with the same probability
`1/(4n)`, so the expected deposit on `(a, b)` is proportional to `E[f(x) | x ∋ a, x ∋ b]` under the
uniform assignment. For a single constraint the satisfied-clause count summed over the free variable
is `7` for every fixing of the other two, whichever pair is fixed (`xor_pair_balance`). For the system:
every constraint has a variable outside `{var(a), var(b)}`, so given the two literals its parity is
uniform (`parity_given_two_uniform`) and it is violated with probability exactly `1/2`; by linearity
`E[#violated | x_a, x_b] = m/2` for every edge. Hence **the expected deposit is the same on every edge
and the flat field is an exact fixed point of the expected deposit–evaporation dynamics on every
3-XOR system.** Not because of symmetry (U4b): because the literal-to-literal field is a *pairwise*
object and parity has no pairwise component. An OR clause does have one (`or_clause_pair_signal`),
which is why planted and random 3-SAT are different problems for U. ∎

**U5b (where a second-moment signal can live).** With a nonlinear deposit `f^{η_dep}`, `η_dep ≠ 1`,
the signal on `(a, b)` depends on the *distribution* of `#violated` given the pair, not just its mean.
That distribution is the Hamming-weight distribution of a coset of the span of the free columns of the
constraint matrix, and different value pairs can give different cosets when constraints touching
`a` and `b` share their free variables (e.g. constraints `{a, z, u}` and `{b, z, u}`: fixing `x_a = x_b`
gives 0 or 2 violations, fixing `x_a ≠ x_b` gives exactly 1). So there is a pairwise signal at second
moment, and it is a *local* structural one (pairs of constraints sharing two variables), which
random sparse 3-XOR systems have only in `O(1)` places. This is the loophole, stated exactly; it does
not change the conclusion below for random systems.

**U5c (what does the work on XOR: η, as a soft force).** With the field flat the walk is
`η^β / Σ η^β` (`flat_law_indep`). When two variables of a constraint are assigned, exactly one of the
four clauses is still unsatisfied and its last free literal `z` gets `η = 2` against `η = 1` for `¬z`
(the Jeroslow–Wang weight of a unit clause). So when the walk assigns that variable it takes the
forced polarity with odds `2^β : 1`, i.e. probability `2^β/(1+2^β) < 1` (`soft_force_lt_one`); `L`
consecutive forced variables are all taken correctly with probability at most `(2^β/(1+2^β))^L`
(`soft_force_pow_le`). Unit propagation is recovered only as `β → ∞`, and even then the walk is
unit propagation on a self-chosen order **without backtracking**: on a random 3-XOR system above the
2-core threshold a variable becomes forced two different ways with constant probability per core
variable, and the walk cannot undo. The burst has nothing to erase from a flat field (U3e), and the
field concentrates only by drift on modes that are random with respect to `f`.

**U5d (the XOR obstruction for System U, stated).** On 3-XOR systems the field carries no
first-moment information (U5a), the burst acts on nothing (U3e on a flat field), and the search is
carried by `η` alone as soft, non-backtracking unit propagation (U5c), whose per-walk success
probability on a system with a linear-size core is exponentially small in the core size. A
polynomial number of walks therefore fails on random 3-XOR above the core threshold, and on the
unique-solution full-rank instances in the same way. This is System U's own version of the
obstruction, re-proved for U: **a pairwise pheromone is blind to parity at first moment**, whatever
the sampler; the product samplers were blind at degree one, the literal field is blind at degree two.
The honest exception is U5b, second-moment signal on constraints sharing two variables, which is
structurally local and rare in random systems.

*What this does and does not say about 3-SAT.* Nothing about planted or random 3-SAT: OR clauses
have first-moment pairwise signal (`or_clause_pair_signal`), so the field learns, the burst has modes
to erase, and the cascade of U3 is live. Those are targets 6 and are next, and for them the finite
computation of U5a has to be replaced by the actual chain (walk → deposit → mode → burst), which is
where GPU evaluation becomes the tool for a specific question rather than a benchmark.

---

