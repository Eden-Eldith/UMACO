# UMACO13 → SAT: architecture-preservation audit

Purpose. Derive the SAT specialization **downward** from `umaco/Umaco13.py`'s full loop, component by
component, preserving each component's mathematical object, its information access, and its coupling
to the others. This replaces the earlier route (February solver → corrections → System C‴), which
produced theorems about progressively narrower objects and called them limits of UMACO.

Rules used here:
- The **object** each component acts on is kept exactly (the `n×n` complex field, not a population
  matrix; the field's SVD, not a population PCA; persistent homology of the field, not dropped).
- "Actual code semantics" cites line numbers in `Umaco13.py`. "Intended semantics" cites the
  documents. "Proposed SAT semantics" is mine, marked **PROPOSED**, and is the minimal reading that
  gives the component *content* on a CNF while keeping its information access.
- Locality columns are answered for the **proposed** component, not inherited from any previous system.
- Where the intended semantics is ambiguous and the answer changes the mathematics, a question is
  raised for the author (collected in §4). Nothing is built until those are settled or explicitly
  defaulted.

Nothing in this document is a theorem. The derived system is named **System U** (for UMACO13-SAT) and
is not System C, C′ or C‴; no result about those transfers to U without a new proof (§3).

---

## 1. The loop, as it executes (`UMACO.optimize`, l.907–1017)

```
for i in range(max_iter):
  1  candidates  = _construct_solutions(agents)                      # l.917, reads Re τ only
  2  losses, performances = loss_fn(candidates), 1/(1+loss)          # l.920–923
  3  grad        = _compute_finite_difference_gradients(...)         # l.927 (ε = 1e-6, no-op on ints)
     _panic_backpropagate(grad)                                      # l.928: panic ← .85 panic + .15 tanh(|grad|·log1p|anx|)
  4  if mean(panic) > .7 or ‖anxiety‖ > 1.7: _quantum_burst()        # l.932: SVD(Re τ) top-k·0.7 + noise·0.3, ×e^{i·arg anx}, added
  5  _persistent_homology_update()                                   # l.936: ripser(Re τ as distance) → anxiety ≡ scalar; momentum += i·lifetime
  6  _update_hyperparams()                                           # l.937: α.re = mean(panic)·mean|anx|; ρ ← .9ρ+.1e^{−‖mom‖}; β = .1·PE
  7  deposit(paths, performances, α.re)                              # l.940–976: τ *= .9; τ[a,b] += α.re·perf² along paths
  8  τ += α.re · covariant_momentum                                  # l.979
  9  _symmetrize_and_clamp()                                         # l.980: Re τ ← sym, diag ← 0, ≥ 0
 10  best / stagnation                                               # l.983–992
 11  _check_stagnation_and_burst(i)                                  # l.995: partial_reset (weakest 30 %) after 40; scheduled burst / 100
 12  economy.update_market_dynamics(); agent.propose_action(...)     # l.998–1002: tokens; return discarded
```

The coupling graph the author cares about is the chain 2 → 3 → 4 → 5 → 6 → 7 → 8: the landscape
(losses) shapes panic; panic and anxiety gate and scale the burst; homology of the field sets
anxiety and momentum; those set α, β, ρ; α scales the deposit and the momentum step; the field
feeds construction. Every component below is placed on that chain.

---

## 2. Component audit

Columns: **code** (what `Umaco13.py` computes) · **intent** (documents) · **PROPOSED SAT** · **information
available** · **spatial locality** (bounded factor-graph radius per step?) · **temporal locality**
(memoryless per step?) · **population-global?** · **formula-global?** · **status**.

### C1. Pheromone field `τ ∈ ℂ^{n×n}` (NeuroPheromoneSystem, l.332–370)

- **code:** complex64, init `0.3·U + 0.3j·U`; only `Re τ` is read by construction; `Im τ` is written by the
  momentum step and the burst and read by nothing that decides; real diagonal zeroed each iteration (l.795).
- **intent:** one field carrying attraction (Re) and repulsion (Im); "complex arithmetic semantics must be
  preserved" (SYSTEM_ARCHITECTURE); "memory of successful paths" (core_concepts).
- **PROPOSED SAT:** rows and columns are variables. **Diagonal** `τ_{vv}`: unary evidence about `v`
  (the decoder already reads the diagonal, l.892; the bug is the clamp that zeroes it). **Off-diagonal**
  `τ_{uv}`: pair evidence about `(u, v)`. To give one complex number per pair the content of a 2×2 joint
  table, use the two channels as the intent names them: `Re τ_{uv}` = evidence that `u` and `v` **agree**
  (same polarity) in good assignments, `Im τ_{uv}` = evidence that they **disagree**; on the diagonal
  `Re τ_{vv}` = evidence for `x_v = 1`, `Im τ_{vv}` = evidence for `x_v = 0`. This keeps the field
  complex, keeps attraction/repulsion as the two channels, and makes the diagonal decoder meaningful.
  **Q1** (§4): is this the intended reading of the matrix for a discrete problem, or is the matrix
  positional (rows = variables, columns = an abstract lattice as in the continuous case)?
- **information:** all `(u,v)` pairs; history through evaporation.
- **spatial:** the object is global (all pairs); which entries get *written* depends on the deposit (C10).
- **temporal:** memory (evaporation `0.9`).
- **population-global:** yes (all agents write the one field). **formula-global:** through the deposit.
- **status:** **preserved** as object; semantics assigned (proposal); the diagonal clamp is dropped as a bug.

### C2. Construction (`_construct_solutions`, SAT branch l.886–895)

- **code:** `true_probs = minmax(diag(Re τ))`, independent Bernoulli per variable; clauses never read.
- **intent:** "agents explore the problem's solution space guided by the field"; the path type
  (COMBINATORIAL_PATH, l.853–884) constructs **sequentially, conditioning each step on the field row of the
  current position** with `τ^α · η^β`, `η` = problem heuristic.
- **PROPOSED SAT:** use the COMBINATORIAL_PATH form, which is the architecture's own construction rule,
  not the diagonal-Bernoulli special case: order the variables (a path through the variable graph, itself
  sampled from `τ`, as in TSP), and assign `x_v` conditioned on the already-assigned variables through
  the pair channels: `P(x_v = 1) ∝ (Re τ_{vv})^α · η_v^β · ∏_{u assigned} g(τ_{uv}, x_u)` with
  `g(τ_{uv}, x_u) = |Re τ_{uv}|` if `x_u = 1` (agreement channel) and `|Im τ_{uv}|` if `x_u = 0`
  (disagreement channel), and `η_v` the problem heuristic that macov8 never used: the fraction of
  clauses containing `v` that the current partial assignment leaves satisfiable only through `v`
  (unit-propagation strength). This is the TSP rule transcribed; `β` then has content on SAT.
  **Q2:** should `η` be unit-propagation strength, clause counts, or something else you had in mind?
- **information:** field row of `v` (all `u`), the partial assignment, the clauses of `v` (through `η`).
- **spatial:** **not bounded** in the factor graph: the field row couples `v` to every `u` the field has
  ever related it to. If `τ_{uv}` is only ever written for pairs sharing a clause (C10, option A), the
  effective radius is 1; under option B it is unbounded.
- **temporal:** memoryless given `τ`. **population-global:** no (per agent). **formula-global:** via `τ`.
- **status:** **altered** from the code's product sampler to the architecture's sequential conditional
  sampler. This is the single most consequential change relative to System C: the sampler is no longer
  a product measure, and Theorems 1–3, 13 and the equivariance pin (Theorem 27) do not apply as stated.

### C3. Evaluation (l.920–923; `sat_loss` l.1090–1101)

- **code:** loss = number of unsatisfied clauses; performance = `1/(1+loss)`.
- **intent:** same.
- **PROPOSED SAT:** unweighted unsatisfied count as the *loss* the PAQ sees; the deposit (C10) uses
  performance as in the code. No clause weights: Umaco13 has none, and adding macov8's would be
  importing System C. **Q3:** do you want clause weights in the architecture, or is the crisis system
  meant to play that role (weights are a *local* memory of resistance; panic is the architecture's
  spatial memory of resistance)?
- **information:** the whole formula. **spatial:** global sum. **temporal:** memoryless.
- **status:** preserved.

### C4. Gradient → panic tensor (l.533–590)

- **code:** forward finite difference with `ε = 1e-6` per coordinate; on integer assignments the
  perturbation truncates to a no-op, so `grad ≡ 0` and panic decays to 0 (§II.A D13-3). 2-D PAQ: the
  gradient is placed on the **diagonal** of an `n×n` panic tensor; off-diagonal panic never moves.
  `panic ← 0.85·panic + 0.15·tanh(|grad|·log1p|anxiety|)`.
- **intent:** "panic tensor tracks local crisis states (NOT random stress)"; "high gradients = crisis";
  a spatial map of where the landscape is hard.
- **PROPOSED SAT:** the discrete finite difference is the **flip gradient**
  `Δ_v(x) = loss(x ⊕ e_v) − loss(x)` (make minus break at `v`), averaged over the candidates; for the
  `n×n` tensor, off-diagonal entries take the **pair flip gradient**
  `Δ_{uv}(x) = loss(x ⊕ e_u ⊕ e_v) − loss(x)` restricted to pairs that share a clause (others are 0 to
  first order). Panic then becomes a map of *frustration*: high on variables (and pairs) whose flip
  changes many clauses, i.e. exactly the variables that are critical in many clauses under the current
  population. This is the code's formula with the `ε` bug fixed and the off-diagonal filled by the
  same rule. **Q4:** agree that panic should be *frustration* (magnitude of the flip effect), not
  *unsatisfaction* (which clauses are violated)? The code's `|grad|` says magnitude.
- **information:** the candidates and, through `Δ`, their clause neighbourhoods. **spatial:** radius 1 in
  the factor graph per candidate. **temporal:** EMA memory. **population-global:** averaged over agents.
  **formula-global:** no.
- **status:** preserved (bug fixed; off-diagonal filled by the same rule).

### C5. Anxiety wavefunction (l.491–497 init; overwritten at l.701–704 / l.750)

- **code:** complex, same shape as panic; **overwritten every iteration** by one scalar: the persistence-
  image mean (real), or `mean(Re τ) + i·std(Re τ)` in the fallback. No spatial structure, phase 0 in the
  ripser path.
- **intent:** "complex field mapping risk: real = immediate concerns, imaginary = potential future
  challenges"; real part rises with stagnation, imaginary with *prolonged* stagnation (guide); its
  **phase rotates the burst** ("recursive negation", Umaco9 header).
- **PROPOSED SAT:** keep it a field over variables (and pairs, matching panic's shape). Real part:
  EMA of "v is in a clause violated by the current population" (immediate risk). Imaginary part: EMA of
  the *duration* for which that has been true (accumulated, slow). The phase `arg(anxiety_v)` then
  measures how *chronic* the trouble at `v` is; `π/2` means all-chronic. Homology (C7) contributes to the
  real part as in the code (a landscape-shape term) but does not overwrite the field. **Q5:** is the
  phase meant to be *used* as a rotation angle (as the burst code does), so that chronic trouble
  literally rotates the burst from "reinforce" toward "negate"? That is the reading that makes the
  burst's `e^{i·arg}` meaningful; it is also what "recursive negation" would mean.
- **information:** violation status of `v`'s clauses over time; field topology via C7.
- **spatial:** radius 1 per step. **temporal:** two memories (fast, slow). **population-global:** yes.
  **formula-global:** through C7.
- **status:** **restored** (the code's overwrite-by-scalar discards the intended field; the proposal
  keeps the field and gives the two parts the documented meanings).

### C6. Quantum burst (`_quantum_burst`, l.592–632)

- **code:** trigger `mean(panic) > 0.7 or ‖anxiety‖_F > 1.7`, plus a schedule every 100 iterations.
  Effect: `U,S,V = svd(Re τ)`; `structured = U_k S_k V_kᵀ` with `k = n/4`; `combined = 0.7·structured +
  0.3·(N + iN)·strength`, `strength = mean(panic)·mean|anxiety|`; `final = combined · e^{i·arg(anxiety)}`;
  `τ += final`; symmetrize/clamp. With the code's anxiety (phase 0) this **adds** 0.7 of the field's own
  rank-`k` part back: reinforcement.
- **intent:** "SVD-based structured escape", "not random noise", "anxiety-directed leap", and in Umaco9's
  header "Applies Formula 8's **recursive negation** via SVD-based pheromone explosion; phase rotation
  from anxiety imaginary components". README: "triggered by crisis, not on a schedule".
- **PROPOSED SAT:** keep the operation exactly: SVD of the **real `n×n` field** (the agreement channel),
  rank-`k` reconstruction, phase-rotated by the anxiety field (C5), added. Its SAT meaning follows from
  C1: the top singular components of the agreement matrix are the dominant **blocks of co-varying
  variables** in the population's good assignments (the population's current "mode" of correlation).
  With phase 0 (acute, recent trouble) the burst reinforces that mode; with phase near `π` (chronic
  trouble) it **subtracts** it, an anti-mode restart that pushes the field away from the correlation
  structure it has been stuck in. Trigger: crisis only (drop the schedule, per README). **Q6:** is
  "recursive negation" this anti-mode operation (subtract the dominant structure when trouble is
  chronic)? If yes, the code's phase-0 reinforcement is a bug of the anxiety overwrite, not of the burst.
- **information:** the **entire field**. **spatial: global** (a singular vector at `v` depends on all
  entries). **temporal:** acts on accumulated `τ`. **population-global:** yes. **formula-global:** yes,
  through `τ`.
- **status:** preserved as object and operation; the trigger schedule dropped per intent; the phase
  now comes from a real anxiety field. **This is the non-local component.** Its effect on the
  obstructions of §3 must be analysed, not assumed.

### C7. Persistent homology (`_persistent_homology_update`, l.638–720; fallback l.722–752)

- **code:** ripser on `Re τ` **treated as a distance matrix** (symmetrized, zero diagonal); persistence
  image mean → anxiety (scalar, overwrite); mean finite lifetime → `covariant_momentum ← 0.9·mom + 0.1·i·lifetime`
  (pure imaginary); `beta = 0.1·persistent_entropy(Re τ)` (l.777, called on the matrix, not a diagram;
  falls to `beta *= 0.99` on exception).
- **intent:** "understand the shape of the landscape: connected components, loops, voids"; persistent
  entropy high = many basins = exploration; low = converged; feeds `β` and anxiety; "sheaf cohomology"
  momentum.
- **PROPOSED SAT:** with C1, `Re τ` is a *similarity* between variables; the code's use of it as a
  distance is a sign convention to fix (`d = max − Re τ`, or `1/(ε + Re τ)`). Then: **H₀** bars = clusters
  of strongly co-varying variables (blocks); **H₁** bars = cycles in the strong-coupling graph. On a CNF,
  cycles of couplings are exactly what parity-type structure produces (a Tseitin instance's variable
  couplings run around the graph's cycles), so H₁ is the one statistic in the architecture that sees
  *cyclic* constraint structure rather than local structure. Persistent entropy of the H₀ diagram
  measures how many independent blocks the population has organised into. Feedback as in the code:
  entropy → `β` (weight of the problem heuristic in construction), lifetimes → the momentum magnitude,
  a landscape-shape term → the real part of anxiety. **Q7:** you wrote "Formula 10 via Rips complex
  analysis" in Umaco9; what is Formula 10, and did you intend H₁ (loops) to drive anything in
  particular? The dead thesis link is where I would have looked.
- **information:** the **entire field**. **spatial: global.** **temporal:** on accumulated `τ`.
  **population-global:** yes. **formula-global:** through `τ`.
- **status:** **restored** (the code's output is one scalar; the proposal keeps the diagrams as the
  object and fixes the distance sign); the feedback targets are the code's.

### C8. Covariant momentum (l.504 init `0.01i`; l.714–716 / 752 update; l.979 applied)

- **code:** purely imaginary on every path; `τ += α.re · momentum` therefore never changes `Re τ`;
  magnitude from homology lifetimes.
- **intent:** "momentum preserving sheaf cohomology (NOT SGD)"; "topology-respecting momentum" that
  carries the field's recent change direction, scaled by how persistent the landscape's features are.
- **PROPOSED SAT:** a momentum on the **field update**: `mom ← 0.9·mom + 0.1·(τ_t − τ_{t−1})` (complex,
  both channels), with its *magnitude* modulated by the homology lifetime as in the code (persistent
  features → carry momentum; short-lived features → damp it). Applied as `τ += α.re·mom`. This keeps the
  object (a complex `n×n` momentum), makes it a momentum of something (the field's own change), and
  keeps the homology coupling. **Q8:** was the momentum meant to be a momentum of the *field* (as here)
  or of the *anxiety* (a momentum of the risk map)?
- **information:** field history; homology lifetimes. **spatial:** per entry. **temporal:** memory.
  **population-global:** yes. **formula-global:** through `τ`.
- **status:** **altered** (from an imaginary scalar to a momentum of the field); coupling preserved.

### C9. Crisis hyperparameters (`_update_hyperparams`, l.758–785)

- **code:** `α.re ← mean(panic)·mean|anxiety|` (overwrites 3.5 on step 1; typical 0.01–0.1);
  `ρ ← 0.9ρ + 0.1·e^{−‖mom‖}` (unused); `β ← 0.1·PE` (or decay).
- **intent:** α = pheromone influence, rises under stress; β = heuristic weight from entropy; ρ =
  evaporation, slows when momentum is high.
- **PROPOSED SAT:** keep all three formulas and **wire them**: α scales both the deposit and the
  momentum step (as in code); β is the heuristic exponent in C2 (now meaningful); ρ is the evaporation
  in C10 (the code computes it and then evaporates at a fixed 0.1, a wiring bug). Keep `α` bounded
  below so the deposit does not vanish (the code's overwrite to ≈ 0.05 is what made Umaco13 inert).
- **information:** global summaries of panic, anxiety, momentum, homology. **spatial: global scalars.**
  **temporal:** EMA. **status:** preserved, wired.

### C10. Deposit (`deposit`, l.353–361; SAT path l.964–976)

- **code:** `τ *= 0.9`; per agent, `deposit = α.re·perf²` added along consecutive pairs of a path; SAT path
  = `[i, i±1]` per variable ("True connects forward, False backward"), so only band entries `(i, i±1)`
  are ever written; deposit is real (adds to `Re τ`).
- **intent:** reinforce "the moves that led to a good solution"; complex deposit (attraction/repulsion).
- **PROPOSED SAT:** deposit the assignment's **pair and unary evidence** into the channels of C1:
  diagonal `Re τ_{vv} += d·[x_v = 1]`, `Im τ_{vv} += d·[x_v = 0]`; off-diagonal
  `Re τ_{uv} += d·[x_u = x_v]`, `Im τ_{uv} += d·[x_u ≠ x_v]`, with `d = α·perf²` as in the code.
  Which pairs: **option A**, pairs sharing a clause (radius 1, `O(m)` per agent); **option B**, all
  pairs (`O(n²)` per agent, the literal field). **Q9:** A or B? A keeps construction local; B is the
  architecture's literal object and makes construction and the SVD genuinely global.
- **information:** the agent's assignment and performance. **spatial:** A: radius 1; B: global.
  **temporal:** memory via evaporation. **population-global:** all agents write.
- **status:** **altered** from the band encoding (a placeholder) to the pair-evidence encoding;
  performance exponent and evaporation preserved.

### C11. Symmetrize and clamp (l.791–797)

- **code:** `Re τ ← (Re τ + Re τᵀ)/2`, `diag ← 0`, `≥ 0`; `Im τ` untouched.
- **intent:** "maintain physical interpretation as distances or similarities".
- **PROPOSED SAT:** symmetrize both channels (pair evidence is symmetric); **do not zero the diagonal**
  (it is the unary channel); non-negativity on both channels (they are evidence counts).
- **status:** altered (diagonal kept). This is the bug fix that makes the decoder work.

### C12. Partial reset (`partial_reset`, l.363–370; trigger l.799–813)

- **code:** after 40 non-improving iterations, entries with `|τ|` below the 30th percentile `×= 0.1`.
- **intent:** reintroduce options that have evaporated.
- **PROPOSED SAT:** as coded, on `|τ|` over both channels. (For the polarity table this was inert; for
  the pair field the weakest entries are the never-reinforced pairs, and damping them further is
  harmless; the *intent* is better served by raising them to a floor. **Q10:** damp or raise?)
- **spatial:** global percentile. **status:** preserved.

### C13. Economy and agents (l.372–450; l.998–1002)

- **code:** tokens, market value, scarcity `= 0.5 + 0.5·mean(panic)`; `propose_action` computes a
  resource request `0.2 + 0.3·panic·risk` and buys; return value discarded; no effect on the search.
- **intent:** "agents MUST compete for resources or diversity collapses"; tokens buy "the ability to
  explore far from the current solution", "an extensive local search", "a very divergent random jump".
- **PROPOSED SAT:** give the purchased resource the meaning the guide states: an agent's **search
  budget** for the iteration — how many construction steps it may take conditioned on the field versus
  freely (divergence), and how many refinement flips it may make on its candidate (depth). Tokens earned
  by performance, so successful agents dig deeper; the minimum balance keeps poor agents exploring
  cheaply; scarcity rises with mean panic. Agents therefore become heterogeneous in *budget*, not in
  temperature. **Q11:** is budget the intended currency, or should tokens gate the right to *deposit*
  (influence on the field)?
- **information:** global scarcity, own tokens and panic. **spatial:** n/a. **population-global:** yes.
- **status:** **restored** (from inert to budget allocation), per the guide's own examples.

### C14. Entropy controller (Umaco9 l.406–415; absent from Umaco13)

- **code (9):** if persistent entropy strays from target, `τ += 0.01i·N(0,1)`.
- **intent:** maintain diversity.
- **PROPOSED SAT:** keep as in Umaco9, injecting into the disagreement channel (imaginary), which is
  the intended "repulsion" reading. **status:** restored from Umaco9.

---

## 3. What does and does not carry over from the System C theorems

Because C2 is no longer a product sampler and C6/C7 are field-global, the following are **not**
theorems about System U and must be re-derived or replaced:

| System C result | status for U | why |
|---|---|---|
| Thm 1–3 (multilinear ascent, vertex stability, solutions maximal) | **not applicable** | U's construction is sequential-conditional on a pair field; the population law is not a product measure and there is no multilinear extension to ascend |
| Thm 13 (product wrong-set at construction) | not applicable | same |
| Thm 7 / Cor 8 (uniform selection collapses to W/3) | not applicable | U has no in-clause selection rule; refinement is budgeted (C13) |
| Thm 15 (XOR = 3-spin) | applies to the **unary channel only** | the pair channels carry second-order statistics the 3-spin form does not include |
| Thm 25 (pair statistics of an affine solution space are ±1/0) | **applies to U's target, not to U's dynamics** | it bounds what the pair field could learn *from solutions*; U learns from the population |
| Thm 27 (equivariance pins ½) | **does not pin U** | negating the formula leaves the pair agreement channel invariant and swaps the unary channels; the state "diag = ½, off-diagonal = accumulated agreement" is not the symmetric point, and sequential conditioning breaks marginal symmetry as soon as any pair entry is nonzero |
| Thm 28 (`M ≡ 0` on XOR) | applies to the unary linearization only | the pair field's linearization is a different operator; unknown |
| `iterate_local` (radius-T dependence) | **does not apply** | C6 and C7 read the whole field; C2 under option B reads whole rows |
| OGP / stable-algorithm barrier | **not established for U** | requires U to be stable under instance perturbation; the SVD step is stable away from singular-value crossings and unstable at them; homology is discontinuous at persistence thresholds. This has to be proved or refuted for U, and the unstable points may be exactly where U differs from local methods |

So the obstruction status of System U on XORSAT, on random 3-SAT near threshold, and in the worst case
is **open**, and the honest statement is that the earlier "blocked by locality" conclusion was about
Systems C–C‴ and does not extend to U by declaration.

Two things that do survive because they are about the target, not the method: Theorem 25 (any
pair-statistic learner must, on XORSAT, end up representing the frozen-pair structure that GF(2)
elimination computes), and Urquhart's lower bound (U produces no resolution proofs, so on the
unsatisfiable side U says nothing; on the satisfiable side that bound is irrelevant).

---

## 4. Questions for the author (each changes the mathematics of U)

| # | component | question | default if unanswered |
|---|---|---|---|
| Q1 | C1 | Is the `n×n` matrix for a discrete problem meant as unary-on-diagonal, pairwise-off-diagonal (my reading), or positional? | unary/pairwise |
| Q2 | C2 | What is the SAT "heuristic" `η` that `β` weights? unit-propagation strength, clause counts, other? | unit-propagation strength |
| Q3 | C3 | Clause weights (macov8) or no clause weights (Umaco13), with panic as the resistance memory? | no clause weights; panic is the memory |
| Q4 | C4 | Panic = frustration (flip-effect magnitude) rather than violation? | frustration |
| Q5 | C5 | Is the anxiety **phase** meant to rotate the burst from reinforce toward negate as trouble becomes chronic? | yes |
| Q6 | C6 | Is "Formula 8's recursive negation" the anti-mode burst (subtract the dominant SVD structure when chronic)? | yes |
| Q7 | C7 | What is "Formula 10"? Should H₁ (loops) drive anything specific? | H₀ entropy → β, lifetimes → momentum magnitude, H₁ count → anxiety real part |
| Q8 | C8 | Momentum of the field, or of the anxiety map? | field |
| Q9 | C10 | Deposit on clause-sharing pairs (A, local) or all pairs (B, global)? | B, to keep the field's global character; A as a control |
| Q10 | C12 | Reset: damp weakest entries (code) or raise them (intent)? | raise |
| Q11 | C13 | Tokens buy search budget (divergence + depth), or the right to deposit? | budget |

(Superseded by §6: the thesis at `finalized-work/fixed-thesis-maco/` settles Q1, Q3–Q9, Q11.)

---

## 5. Order of work once §4 is settled

1. Write System U's loop as a transition system (state: `τ ∈ ℂ^{n×n}`, panic `∈ ℝ^{n×n}`, anxiety
   `∈ ℂ^{n×n}`, momentum `∈ ℂ^{n×n}`, `α, β, ρ`, tokens, best).
2. Derive its population law under C2 (a sequential conditional sampler on a pair field is a
   chain-structured model along the construction order; its marginals are computable), and re-ask
   the Theorem 1 question for that law: what functional, if any, does the deposit-construction pair
   ascend? This is the first real theorem target for U.
3. Analyse C6 as an operator on the field: what does adding or subtracting the rank-`k` part of the
   agreement matrix do to the population law? This is where U is genuinely non-local, and where the
   XOR question for U is decided.
4. Only then implement, on the GPU, top-down from this document, and evaluate on the same three
   families (planted, random, XOR) with the same recount discipline. Name every deviation.


---

## 6. Revisions after the thesis (equations 3.1–3.19, `finalized-work/fixed-thesis-maco`)

The thesis is the authoritative statement of intent. Its equations resolve most of §4 and overturn two
of my proposals. Recorded here rather than by rewriting §2, so that the change of reading is visible.

### 6.1 What the equations say

| eq. | statement | consequence for the audit |
|---|---|---|
| 3.1 | `P(t+1) = (1−δ_P)P + δ_P tanh(k_P |∇L| log(1+|Ψ|))` | C4 confirmed: panic is driven by the **magnitude** of the loss gradient (frustration), coupled to anxiety amplitude. Q4 answered. |
| 3.2 | `Ψ_re(t+1) = (1−δ)Ψ_re + δ tanh(k (Perf_target − Perf))` | C5: the real part is the **performance gap** to a target, not a violation map. Negative when performing above target. |
| 3.3 | `Ψ_im(t+1) = (1−δ)Ψ_im + δ tanh(k · stagnation_counter)`, decays ×0.9 when not stagnating | C5: the imaginary part is **stagnation duration**. Q5 answered: the two parts are exactly "acute gap" and "chronic stagnation". |
| 3.4–3.6 | `ΔΦ_struct = U_{:,1:k} Σ_{1:k} V*_{1:k,:}`; `ΔΦ_burst = (γ_s ΔΦ_struct + γ_r ΔΦ_rand)·e^{i·angle(Ψ)}·f_scale(‖P‖)`; `Φ += ΔΦ_burst` | C6 confirmed as an operation. The phase is `angle(Ψ)` with `Ψ = (gap, stagnation)`: **the burst's direction in the complex plane is set jointly by the performance gap and the stagnation duration.** See 6.2. Q6: "recursive negation" is not defined in the text; its operative content is 3.5. |
| 3.7–3.8 | `Φ(t+1) = (1−ρ)Φ + Σ_k ΔΦ_k`; `ΔΦ_k[a,b] = I·perf_k^η` for `(a,b) ∈ path_k` | C10: deposit is **along the agent's path**, on consecutive pairs `(a,b)`; real-valued. Neither my option A nor B: the pairs are the agent's *traversal order*. Q9 answered, see 6.3. |
| 3.9–3.10 | `Diagrams = Rips(Φ_real)`; `H_p = PersistentEntropy(Diagrams)` | C7: "Formula 10" is persistent entropy. Q7 answered. Feedback is entropy → β (3.18). No specific role is assigned to H₁. |
| 3.11–3.12 | `p_cov(t+1) = (1−δ)p_cov + δ·i·MeanPersistence(Diagrams)`; `Φ += α·p_cov` | C8: **the covariant momentum is intentionally purely imaginary**: it injects repulsion proportional to the mean persistence of the landscape's features. My proposal to make it a momentum of the field was flattening; **withdrawn**. C8 status returns to *preserved as coded*. Q8 answered. |
| 3.13–3.16 | tokens buy "computational resources (compute time, memory)"; cost from requested power, market value, scarcity; balance floored | C13 confirmed: **budget**. Q11 answered. |
| 3.17–3.19 | `|α| = f(mean P, mean|Ψ|)`; `β = f(H_p)`; `ρ = f(‖p_cov‖)` | C9 confirmed; α is complex with magnitude from panic·anxiety. |
| §3.2 text | `Φ ∈ ℂ^{N×N}`, "indexed by spatial positions **or variable pairs** (a,b) representing coordination landmarks"; "real = attraction (exploitation), imaginary = repulsion, discouraging exploration of unfavorable or already explored areas" | C1: pairs are an intended reading. Q1 answered in principle; which pairs is 6.3. |
| §5 | SAT section truncated; "domain-specific mechanisms" | Discrete construction is **not specified** in the thesis. Q2 stays open. |

### 6.2 The burst's phase, read correctly

With `Ψ = Ψ_re + iΨ_im`, `Ψ_re = tanh(k(Perf_target − Perf))`, `Ψ_im = tanh(k·stagnation) ≥ 0`:

| regime | `Ψ` | `angle(Ψ)` | what `e^{i·angle}·ΔΦ_struct` does to the field |
|---|---|---|---|
| below target, not stagnating | `+re, 0` | 0 | adds the dominant structure to **Re Φ**: reinforce the current attraction mode |
| below target, stagnating | `+re, +im` | `(0, π/2)` | splits it: part reinforces attraction, part becomes **repulsion** on the same pairs |
| chronic stagnation | `≈0, +im` | `π/2` | moves the dominant structure entirely into **Im Φ**: the pairs the field has organised around become repulsive |
| above target, not stagnating | `−re, 0` | `π` | **subtracts** the dominant structure from Re Φ: the negation |
| above target, stagnating | `−re, +im` | `(π/2, π)` | subtract attraction and add repulsion |

So the burst is a single operator whose direction encodes two global state variables. "Recursive
negation" fits the `π` regime (the field's own principal structure is negated), and the `π/2` regime is
the conversion of attraction into repulsion. Neither was reachable in Umaco13 because the code
overwrote `Ψ` with a positive scalar (phase 0), and because with `Perf = 1/(1+loss)` and loss = clause
count, `Perf ≈ 0`, so the gap is always positive. **The performance definition therefore selects
which burst regimes exist.** For SAT, `Perf` = fraction of clauses satisfied (in `[0,1]`, typically
`> 0.7`) makes the negation regime the normal one and the reinforcement regime the early one. **Q13**
(new): is `Perf_target = 0.7` with `Perf` = satisfied fraction the intended reading for SAT?

### 6.3 The deposit is along the path; what is the path on a CNF?

Equation 3.8 deposits on consecutive pairs of the agent's traversal. The COMBINATORIAL_PATH
construction in Umaco13 is the architecture's own traversal rule (next node chosen from the field row
of the current node, weighted by the heuristic). Read downward, the SAT agent **traverses literals**:
from the current literal it chooses the next unassigned variable's literal with probability
`∝ w(Φ_{cur,ℓ})^α · η_ℓ^β`, assigns it, and continues; the path is the sequence of chosen literals;
the deposit goes on consecutive literal pairs. This gives:

- **Object:** `Φ ∈ ℂ^{2n×2n}` over literals (pairs `(a,b)` = "after literal a, literal b was chosen on a
  good path"). Unary evidence for a literal is its column mass. This is the classical literal-graph
  encoding, and it is exactly 3.8 with "path" = construction order. It replaces my proposed
  variable-pair agreement/disagreement channels, which were an invention.
- **Channels:** `Re Φ` is written by agents (3.8, attraction). `Im Φ` is written **only by the crisis
  machinery**: the burst's rotation (3.5), the covariant momentum (3.11), and Umaco9's entropy
  noise. Agents never deposit repulsion; the system does.
- **Construction must read both channels.** Umaco13 reads `Re Φ` only (l.825), so repulsion never
  acts, which is why the imaginary part is inert in every version. The reading that gives the
  thesis's words content: transition weight `w(Φ_{ab}) = max(0, Re Φ_{ab} − Im Φ_{ab})` (repulsion
  cancels attraction), or `Re Φ_{ab}·e^{−Im Φ_{ab}}`. **Q12** (new): which, or another form?
- **Locality:** a row of `Φ` over literals has `2n` entries; after path deposits it is dense along the
  pairs good agents have used consecutively. Construction reads a full row per step: **not bounded**
  in the factor graph. The SVD (3.4) and Rips (3.9) act on the whole `2n×2n` matrix.

### 6.4 Revised component statuses

| component | §2 status | after thesis |
|---|---|---|
| C1 field | preserved, my channel semantics | preserved; object = literal-pair field `ℂ^{2n×2n}`; Re = agent attraction, Im = crisis repulsion |
| C2 construction | sequential-conditional on variable pairs | **literal walk** (COMBINATORIAL_PATH rule on the literal graph), reading both channels |
| C4 panic | frustration | confirmed (3.1) |
| C5 anxiety | violation / duration field | **gap (3.2) and stagnation (3.3)**; in the code these are global scalars; whether the thesis intends a *field* or scalars is not stated (the code broadcasts a scalar). Q14: scalar or per-literal? |
| C6 burst | anti-mode when chronic | the five-regime operator of 6.2; crisis-triggered; scheduled fallback allowed by the thesis ("or at fixed intervals") |
| C7 homology | blocks and cycles | as coded on `Re Φ` (distance sign to fix); entropy → β; no H₁ role assigned |
| C8 momentum | altered to field momentum | **reverted**: purely imaginary scalar repulsion from mean persistence (3.11) |
| C10 deposit | pair evidence, A or B | **path deposit** on consecutive literals (3.8) |
| C13 economy | budget | confirmed |

### 6.5 Questions that remain (reduced)

| # | question | default |
|---|---|---|
| Q2 | the SAT heuristic `η` that `β` weights | unit-propagation strength of the literal under the current partial assignment |
| Q10 | reset: damp or raise the weakest entries | raise |
| Q12 | how construction combines Re (attraction) and Im (repulsion) | `max(0, Re − Im)` |
| Q13 | `Perf` for SAT = satisfied fraction, target 0.7 | yes |
| Q14 | panic and anxiety as fields over literals, or global scalars as the code broadcasts | fields (3.1 is written with indices `ij`) |

Everything else in §4 is settled by the thesis.

### 6.6 What this does to §3

Unchanged in direction, sharpened in content: System U's sampler is a **literal walk** on a dense
`2n×2n` field, its burst is a five-regime global operator, its repulsion channel is written by the
crisis machinery and read by construction. None of the System C theorems apply; the first theorem
targets for U are (i) the stationary law of the literal walk on a fixed field, (ii) what the
deposit–walk pair ascends, (iii) the burst as an operator on the field in each of the five regimes.
