# System U — UMACO specialized to SAT, derived top-down

> **Status: experimental reconstruction (2026).** System U joins the February 2025 SAT literal walk to
> the March 2025 crisis architecture (§0). The two were never joined in the historical code, so this is
> not the solver that produced the 2025 benchmark corpus. Runs and code: `research/experiments/system_u/`.

Living definition. Built from the thesis (equations 3.1–3.19), `umaco/Umaco13.py`, and the recovered
history in `old_files/` (358 Python files, January–March 2025). Each component records its provenance:
where it was first defined, where its wiring was lost, and what System U takes as the definition.
Questions still open for the author are marked **Q**. Nothing here is a theorem; theorem targets are
listed at the end.

Companion documents: `UMACO13_SAT_architecture_audit.md` (the component-by-component audit that
produced this), `UMACO_lineage_reconstruction.md` (which mechanisms persist across versions; not yet
published), `research/system_c/SystemC_3SAT_attack.md` (results about the earlier bottom-up systems,
which do **not** transfer).

---

## 0. Provenance timeline (from `old_files/`, by file date)

| date | file | what appears |
|---|---|---|
| 2025-01-17 | `U/Universal Knowledge Capsule (2).py` | earliest file in the corpus |
| 2025-02-21 | `M/macov2.py`, `macov3.py` | **SAT as a layered literal graph; walk with `τ^α · η^β`; dynamic clause heuristic η** (§1) |
| 2025-02-21 | `M/macoproposedv4no-2.py` | clause weights / stubbornness first appear |
| 2025-02-22 | `M/macov5no-*` | GPU port: walk flattened to per-variable choice; η becomes a **static** occurrence table with a sign bug; entropy controller, partial reset, local search appear |
| 2025-02-23 | `M/macov6no-*` | **η dropped from the kernel; β kept as a dead parameter** (stays dead through macov8, the benchmarks and the paper); quantum burst appears (as a hyperparameter kick) |
| 2025-02-24 | `M/macov7`, `M/macov8no-1` | the benchmark solver lineage |
| 2025-03-06 | `U/Umaco.py`, `M/maco_direct_train.py` | SVD burst, panic, anxiety first appear (the PAQ core) |
| 2025-03-07 | `U/Umaco5.py`, `U/Umaco6.py` | persistent homology; complex field; covariant momentum |
| 2025-03-09 | `M/maco_direct_train4.py` | token economy |
| 2025-09 | `umaco/Umaco13.py` | canonical core; SAT adapter reads the diagonal, which the clamp zeroes |

Two lineages, three weeks apart: the **SAT walk** (February) and the **crisis architecture** (March).
They were never joined in code. System U is the join, done on paper first.

---

## 1. The search space and the walk (from macov2, 21 Feb 2025)

**Object.** A layered directed graph: one layer per variable, two nodes per layer (`var_i = True`,
`var_i = False`), a `start` node, edges from every node of layer `i` to both nodes of layer `i+1`.
Pheromone lives on **edges**: `τ[(node_a, node_b)]`. Every path from `start` through all layers is a
complete assignment. (`macov2.py` l.31–54.)

In matrix form this is a field over literals, `Φ ∈ ℂ^{2n×2n}` (thesis §3.2 allows "variable pairs
(a,b)"), band-structured by the layer order in macov2. **Settled (author, 2026-09-11): the walk chooses where
to go next.** From the current literal an agent picks any *unassigned* variable's literal (Umaco13's
path rule over literals, as over cities in TSP); the assignment order is itself learned; the field is
a full `2n×2n` literal-to-literal matrix, not a band.

**Walk.** From the current node, the next node is chosen with probability
```
P(next = b | current = a) ∝ τ[(a,b)]^α · η(a,b)^β                     (macov2 l.60–85)
```
This is the architecture's construction rule (Umaco13's COMBINATORIAL_PATH, l.853–884, is the same
rule with `η = 1/distance`). Construction is therefore **sequential and conditional**, not a product
measure. Consequence: none of the product-sampler theorems (Systems C–C‴) apply to U.

**η, the problem heuristic (macov2 `calculate_heuristic`, l.86–113).** Collect the partial assignment
carried by the path up to `a`, add `b`'s literal, count the clauses satisfied by that partial
assignment, return the fraction. Dynamic, partial-assignment-aware, cheap.

*What happened to it.* macov5 (22 Feb) replaced it by a precomputed table
`heuristic_true = |pos − neg| / total`, `heuristic_false = 1 − heuristic_true`, which discards the
majority direction (the absolute value) and is static; macov6 (23 Feb) dropped the table but kept
`β` in the kernel signature. Every benchmark in the repo ran with β dead.

**System U takes** macov2's semantics with one sharpening: count the clauses **newly** satisfied by
`b` given the partial assignment (the already-satisfied ones do not depend on the choice), weighting
each by how close it was to being violated:
```
η(a,b) = 1 + Σ_{C ∋ lit(b), C unsatisfied by the partial assignment} 2^{−(free literals in C − 1)}
```
so a clause with `lit(b)` as its last free literal contributes 1 (a forced literal), a clause with two
free literals contributes ½, and so on. This is macov2's clause count made marginal and near-forced-
weighted (the Jeroslow–Wang form). **Q1-status: answered by macov2; the sharpening is the only choice
left, and macov2's exact fraction is acceptable instead.**

**Reading both channels.** Umaco13 constructs from `Re Φ` only (l.825); the thesis says `Im Φ` is
repulsion that "discourages exploration of unfavourable or already explored areas". **Settled (author): attraction and repulsion must interfere, "yin and yang".** The edge value is a complex
amplitude and the walk reads its **interference term**, the real part of its square:
```
w(Φ_ab) = max(0, Re(Φ_ab²)) = max(0, (Re Φ_ab)² − (Im Φ_ab)²)
```
Pure attraction: full weight. Equal attraction and repulsion: exact cancellation (neutral edge). More
repulsion than attraction: avoided. Small repulsion barely dents strong attraction and kills weak
attraction. The burst's phases act literally on this: phase π subtracts attraction (destructive),
phase π/2 adds repulsion that cancels it on the affected edges, phase 0 reinforces. The walk uses
`w^α · η^β`.

---

## 2. Evaluation and performance

Loss = number of unsatisfied clauses (`sat_loss`, Umaco13 l.1090). Performance for the crisis
system: thesis 3.2 compares `Perf` to `Perf_target = 0.7`, so `Perf` must live in `[0,1]` with typical
good values above 0.7. For SAT, `Perf` = fraction of clauses satisfied. **Settled (author): yes.** This
choice decides which burst regimes (§5) are reachable: with `Perf = 1/(1+loss)` as in the code, the
gap is always positive and the negation regime never occurs.

No clause weights in U: Umaco13 has none, the thesis has none; the crisis system is the memory of
resistance (panic, §3). The February clause-weighting is a System-C ingredient, not part of U.
**Q3-status: default taken (no clause weights); overridable.**

---

## 3. Panic tensor (thesis 3.1; Umaco13 l.533–590)

```
P(t+1) = (1 − δ_P) P(t) + δ_P · tanh( k_P · |∇L| · log(1 + |Ψ|) )
```
`∇L` on a CNF is the **discrete flip gradient**: `Δ_v(x) = loss(x ⊕ e_v) − loss(x)` (make minus
break), averaged over the population; for the `n×n` tensor, off-diagonal entries take the pair flip
gradient on pairs sharing a clause. Umaco13's `ε = 10⁻⁶` finite difference is this formula with the
step truncated to zero on integers (so panic was identically zero on SAT). Panic is a **frustration
map**: high where flipping changes many clauses. **Q4-status: answered by 3.1 (magnitude).**
Indexed `ij` in the thesis, so a field, not a scalar. **Settled (author): maps**, per position (per
literal, and per edge where the burst rotation needs a phase).

---

## 4. Anxiety wavefunction (thesis 3.2–3.3; Umaco13 l.491–497, overwritten l.701–704)

```
Ψ_re(t+1) = (1 − δ) Ψ_re + δ · tanh( k (Perf_target − Perf) )        acute: performance gap
Ψ_im(t+1) = (1 − δ) Ψ_im + δ · tanh( k · stagnation_counter )         chronic: stagnation; ×0.9 decay when not stagnating
```
Umaco13 overwrites `Ψ` each iteration with a positive scalar from the persistence image, so the phase
is always 0 and the imaginary part never grows; that is the wiring loss that made the burst
degenerate. System U keeps both recursions. `Ψ` is a **map** (settled), with the homology term entering the real part uniformly.

---

## 5. Quantum burst (thesis 3.4–3.6; Umaco13 l.592–632)

```
Φ_re = U Σ V*;   ΔΦ_struct = U_{:,1:k} Σ_{1:k} V*_{1:k,:}           (k = n/4 in the code)
ΔΦ_burst = ( γ_s ΔΦ_struct + γ_r ΔΦ_rand ) · e^{ i·angle(Ψ) } · f_scale(‖P‖)
Φ ← Φ + ΔΦ_burst
```
Trigger: `mean(P) > θ_P` or `‖Ψ‖ > θ_Ψ` (crisis); the thesis also allows a fixed interval.
The operator acts on the **whole literal field**; it is the architecture's non-local step.
Its direction is set by `angle(Ψ)`, i.e. jointly by the performance gap and the stagnation duration:

| regime | `Ψ` | phase | effect on the field |
|---|---|---|---|
| below target, moving | `+re, 0` | 0 | reinforce the dominant structure of `Re Φ` |
| below target, stagnating | `+re, +im` | (0, π/2) | part reinforce, part convert to repulsion |
| chronic stagnation | `≈0, +im` | π/2 | the dominant structure becomes **repulsive** (moves to `Im Φ`) |
| above target, moving | `−re, 0` | π | **negate** the dominant structure (this is the "recursive negation" regime) |
| above target, stagnating | `−re, +im` | (π/2, π) | negate and repel |

**Computed effect (SystemU_theorems.md U3, 2026-09-10).** On a captured edge the weight is scaled by
`F(g, φ) = 1 + 2g cos φ + g² cos 2φ`; at full strength `F = 2 cos φ (1 + cos φ)`, so the edge is
erased to the floor iff `φ ≥ π/2`. The `(0, π/2)` row is net *reinforcing* for `φ < 68.5°`; only the
rows with `φ ≥ π/2` suppress the mode. π/2 leaves a persistent threshold `Im = Re` (the edge stays
invisible until re-attracted past its old strength); π erases without memory. Nothing is erased at
strength `g < 1/√2`. This is the intended escalation cascade (Q15, §14).

On the literal field, the top singular components of `Re Φ` are the dominant blocks of literal
co-selection in good paths (the population's current mode): a population agreeing on an assignment
gives a rank-1 block on that assignment's literals (U3d); a single sharp path has a flat spectrum and
no dominant structure. None of the five regimes was reachable
in any implementation: macov6–8's "burst" is a hyperparameter kick (`σ×3, α×0.7`), and Umaco13's has
phase 0 by the anxiety overwrite. System U implements the operator as written.

---

## 6. Persistent homology (thesis 3.9–3.10; Umaco13 l.638–720)

```
Diagrams = Rips(Φ_re);   H_p = PersistentEntropy(Diagrams)   ("Formula 10")
```
`Φ_re` is a similarity; the code passes it as a distance, so the sign is fixed (`d = max − Φ_re`).
On the literal field: `H₀` bars = blocks of literals that co-select strongly; `H₁` bars = cycles in
the strong-coupling graph. `H_p` (entropy of the diagram) measures how many independent blocks the
population has organised into. Feedback, per the thesis: `H_p → β` (3.18) and mean persistence →
covariant momentum (3.11). No role is assigned to `H₁` by the thesis; it is recorded as the one
statistic in the architecture that sees cyclic structure, and left unwired.

---

## 7. Covariant momentum (thesis 3.11–3.12; Umaco13 l.504, 714–716, 979)

```
p_cov(t+1) = (1 − δ) p_cov + δ · i · MeanPersistence(Diagrams);    Φ ← Φ + α · p_cov
```
**Purely imaginary by design**: it injects repulsion in proportion to how persistent the landscape's
features are. Kept exactly. (An earlier proposal to make it a momentum of the field was mine and is
withdrawn.) The repulsion it writes is read by the walk through `w(Φ) = max(0, Re − Im)` (§1).

---

## 8. Crisis-driven hyperparameters (thesis 3.17–3.19; Umaco13 l.758–785)

```
|α| = f( mean P, mean |Ψ| );   β = f( H_p );   ρ = f( ‖p_cov‖ )
```
`α` scales deposit and the momentum step; `β` is the heuristic exponent of the walk (§1), which now
has content; `ρ` is the evaporation (§9) — the code computes it and then evaporates at a fixed 0.1, a
wiring loss. `α` is kept bounded below so the deposit cannot vanish (Umaco13's overwrite to ≈ 0.05 on
the first step is what made its field inert).

---

## 9. Deposit and evaporation (thesis 3.7–3.8; macov2 l.115–140; Umaco13 l.353–361)

```
Φ(t+1) = (1 − ρ) Φ(t) + Σ_k ΔΦ_k(t);     ΔΦ_k[a,b] = I · perf_k^η_dep   for consecutive (a,b) ∈ path_k
```
Agents deposit **real** attraction along the literal path they walked. `η_dep` is the performance
exponent (1.3 core, 1.5 LLM, 2 in Umaco13); `I` is the intensity, scaled by `α`. Agents never write
repulsion; the imaginary channel is written only by the burst (§5), the momentum (§7) and the
entropy noise (§11). Evaporation on both channels. Global rescale (not per-entry clipping) if the
field's max magnitude exceeds a cap, per the LLM variant.

---

## 10. Symmetrize / clamp, partial reset

Symmetrize both channels (a pair edge is undirected in the literal graph); **no diagonal zeroing**
(the diagonal is unused on a literal-edge field anyway); non-negativity on both channels.
Partial reset after `partial_reset_threshold` stagnant iterations: entries of `|Φ|` below the weakest-
`X` % percentile are **raised** to a baseline (the intent's "reintroduce options"), attraction channel
only; repulsion on those edges is left to the crisis system. `X` is a dial: **30 % (mild, the guide's
value) by default; 95 % as a hard restart** (the macov8 accident that kept a 5 % backbone and
re-randomised the rest, which is what carried the February benchmarks), used only after the burst has
fired repeatedly without progress. Escape ladder: burst (structured) → mild reset → hard reset.
**Q10-status: settled (raise; 30 / 95 dial).**

---

## 11. Entropy controller (Umaco9 l.406–415, dropped in Umaco13; also macov5 onward on the SAT side)

If persistent entropy strays from `target_entropy`, inject small noise into `Im Φ` (repulsion).
Restored from Umaco9. The SAT-side controller of macov5–8 (entropy of the marginals → α, ρ, noise)
was found to have inverted signs (`research/system_c/SystemC_3SAT_attack.md` C2); System U uses the Umaco9 form only.

---

## 12. Economy and agents (thesis 3.13–3.16; Umaco13 l.372–450)

Tokens buy "computational resources (compute time, memory)". In System U the purchased resource is an
agent's **search budget** for the iteration: how many refinement flips it may make on its completed
assignment, and how far it may deviate from the field during the walk (a per-agent temperature on
the walk's choice). Cost from requested power, market value and scarcity (`scarcity = 0.5 + 0.5·mean
P`); reward from performance; balance floored. Agents become heterogeneous in budget.
**Q11-status: answered by the thesis (compute time and memory).**

---

## 13. The loop of System U

```
for t:
  1 each agent walks the literal graph (§1) with its budget (§12), reading w(Φ) = max(0, Re − Im), η, α, β
  2 loss, Perf = satisfied fraction (§2)
  3 flip gradient → panic (§3);  performance gap, stagnation → anxiety (§4)
  4 if crisis: burst (§5) — SVD of Re Φ, phase from Ψ, scale from P
  5 homology of Re Φ → H_p, mean persistence (§6)
  6 α, β, ρ from P, Ψ, H_p, p_cov (§8)
  7 deposit along paths, evaporate (§9);  Φ += α·p_cov (§7)
  8 symmetrize; reset if stagnant (§10);  entropy noise (§11)
  9 economy: costs, rewards, budgets (§12)
  stop when an agent's assignment satisfies every clause (host recount)
```

---

## 14. Open questions (for the author)

**Q15 (from U3b) — settled (author, 2026-09-11): keep the construction as written.** Below-target
stagnation must not force repulsion at once. Intended behaviour is an **escalation cascade**: mild
failure first reinforces the dominant mode so the system exploits and saturates the structure it has
("grab the bull by the horns"); continued stagnation and degrading performance raise anxiety and
panic, which rotate (`φ` grows) and strengthen (`g` grows) the burst until it crosses from
reinforcement into suppression/repulsion. The intervention point is emergent from accumulated crisis,
not triggered by stagnation alone: the mode is negated only once continued commitment has itself
become evidence of failure. U3b gives the crossing exactly: `cos φ* = (−1 + √(1 + 2g²))/(2g)`, erasure
iff `g ≥ 1/√2` and `φ` past the zero of `F(g, φ)`.

Settled 2026-09-11: Q-order (walk chooses), Q12 (interference `Re(Φ²)`), Q13
(satisfied fraction), Q14 (maps).

Answered: Q1 (η, by macov2), Q3 (no clause weights), Q10 (raise, 30/95 dial), Q4 (panic = magnitude), Q5–Q6 (phase regimes,
by 3.2–3.5), Q7 (Formula 10 = persistent entropy), Q8 (imaginary momentum is intended), Q9 (path
deposit), Q11 (budget).

---

## 15. Theorem targets for System U (none inherited)

1. **Stationary law of the walk on a fixed field.** With fixed `Φ`, `η`, `α`, `β`, the walk is a Markov
   chain on the layered graph; its law over complete assignments is a chain-structured distribution
   (product of transition kernels along the layer order). Compute it; it is the analogue of the
   product measure in Systems C, and the first object every later statement is about.
2. **What the deposit–walk pair ascends.** Deposit along paths of good agents, evaporation, walk from
   the field: find the functional of `Φ` this is a (noisy) ascent of, if any. The analogue of
   Theorem 1 for C. Expected to be an ascent on a *path-level* objective, not a product relaxation.
3. **The burst as an operator.** For each of the five regimes, what does adding, rotating or
   subtracting the rank-`k` part of `Re Φ` do to the walk's law? In particular, whether the π regime
   moves the law out of the basin of the current mode (the escape claim) is a statement that can be
   made precise on the chain-structured law of target 1.
4. **Symmetry.** Negating the formula relabels literals; the literal field is *not* pinned by the
   equivariance argument that pinned the product samplers, because the walk conditions on earlier
   choices. State exactly what symmetric initial states are fixed points of U, if any.
5. **XOR.** With the literal walk and the global burst, the earlier obstruction does not apply by
   declaration; it has to be decided for U. The expected route is target 1 on an XOR instance.
6. **Planted and random 3-SAT.** Same three families, same recount discipline, GPU only, after 1–3.

Until 1–3 exist on paper, no code.
