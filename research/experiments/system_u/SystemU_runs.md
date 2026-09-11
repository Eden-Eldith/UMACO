# System U: integrated runs (build → run → break → explain → mutate)

Implementation: `system_u_v9_walksat.py` in this folder (CuPy only; formerly `examples/system_u_gpu.py`;
its current state is v9 below, with M13's per-agent WalkSAT-SKC refinement). Sweeps: `system_u_scaling.py`.
Raw logs and per-iteration state: `benchmarks/system_u/`. Every deviation from `docs/system-u/system_U.md` is
listed in `DEVIATIONS` in the script; every change made after a run is listed in `MUTATIONS` and here.

## v0 — System U as settled (2026-09-11)

Instances: planted 3-SAT at `m/n = 4.26` (n = 100, 200), random 3-SAT n = 200 (seed 1). K = 128 agents,
10 refinement flips base (economy: 20 on a successful purchase, 5 on failure).

| run | result | iterations | notes |
|---|---|---|---|
| planted n=100 ×3 | solved | 2 | refinement flips alone |
| planted n=200 ×3 | solved | 13, 15, 20 | one or two bursts, all `negate+repel`; seed 2 ended with 0 % live edges |
| random n=200 s1 | **not solved** (best 7 unsat) | 150 | best reached at it 20, then flat; 0 % live edges from it 20 |

### Break: what the state log shows on the random instance

1. **Momentum–evaporation loop erases the field.** `ρ = ρ₀·exp(−|p_cov|/p_ref)` is driven to its floor
   (0.02) by the momentum norm, while the covariant momentum writes a *uniform* imaginary value to
   every edge. Under `w = ε + max(0, R² − M²)` a uniform `M` is a **global threshold**: every edge with
   `R < M` is invisible. Equilibrium `M* = α·κ·|p|·R_max/ρ ≈ 4.6·R_max`, so by iteration 20 the fraction
   of live edges was 0.00 and stayed there.
2. **Collapse of the population onto one path.** With the field dead the walk is η-only, agents take
   nearly the same greedy path, and all 128 deposits land on the same ~200 edges: `R_max` climbed from
   0.25 to the cap (10) while everything else stayed at the floor. Mode collapse *caused by* repulsion.
3. **Panic inert.** The flip gradient normalised by the mean clause degree gave `P ≈ 0.06–0.09`, so the
   burst strength was `g ≈ 0.1`: by U3b (`F = 1 + 2g cos φ + g² cos 2φ`) that changes weights by ~10 %
   and never erases (erasure needs `g ≥ 1/√2`). The escalation cascade of Q15 could not escalate.
4. **Phase pinned at π.** A random 3-SAT assignment satisfies 7/8 of the clauses, so "satisfied
   fraction, target 0.7" is exceeded from the first iteration; `Ψ_re < 0` always; only the negate
   regimes were ever reachable.

### Mutations → v1

- **M1** panic uses `|Δ_v|` in clause units (no degree normalisation).
- **M2** momentum repulsion written as `ρ·θ_m·|p|/(|p|+p_sat)·R_max` per iteration, so its equilibrium
  is a persistence-scaled fraction (`θ_m = 0.5`) of the field maximum instead of a multiple of it;
  `ρ` floor raised to 0.05. Under the interference weight this is still a threshold, now one that
  prunes edges below half the strongest, scaled by persistence. (The uniform-scalar momentum of the
  thesis has no other reading on the literal graph; a structure-proportional momentum would be a
  different object and is not introduced.)
- **M3** `Perf = 1 − unsat/(m/8)` clipped to [0,1]: the fraction of the random baseline's violations
  removed. Random = 0, solution = 1, target 0.7 kept. *For the author:* this replaces "satisfied
  fraction" by its affine rescaling to the SAT scale; without it the five regimes are unreachable.
- **M4** deposit along the refined path (walked order, flipped literals negated), so the path that is
  deposited is the assignment that is scored.

## v1 — after M1–M4 (2026-09-11)

| run | result | notes |
|---|---|---|
| random n=200 s1 | not solved, best 7 (reached it 50) | panic rose to 0.76; 12 bursts at `g` 0.6–1.0 through negate+repel → repel → mixed; 3 resets (30/30/95). Population collapsed onto a mode by it 30 (diversity 0.12), bursts dispersed it (0.41) and it never re-formed a better one. |
| planted n=400 s0, field on | **solved, it 29** | v0 had failed on this instance (best 3 at it 200) |
| planted n=400 s0, **field off** (`alpha_w = 0`) | not solved, best 18 | the field is doing the work |
| planted n=400 s1, field on | not solved, best 1 (reached it 38) | 16 bursts (6 negate+repel, 10 repel), 4 resets, 160 iterations without closing the last clause |

### Break: what the seed-1 trajectory shows

The mode at it 38 was one clause from the planted solution. Every subsequent burst fired in the repel
regime (phase ≈ π/2: above target, stagnating), and by U3c/U3d it thresholded the mode's whole literal
block, which is a full restart; the population then re-formed a worse mode, was thresholded again, and
so on. The burst could not localise because its per-edge scale is `√(P_a P_b)` and panic, defined
from `|make − break|`, is nearly uniform near a solution: every variable is critical for ~1.3 clauses,
so `|Δ_v| ≈ 1.5` everywhere and the map carries no information about *where* the frustration is.
The refinement flips (greedy on make − break, 20 per agent) cannot close a 1-unsat local minimum
that needs a 2–3 flip move.

### Mutations → v2

- **M5** panic magnitude = population-mean frustration at the variable (unsatisfied clauses at `v`
  plus half of its clause-neighbours'). Far from a solution this is everywhere (global burst); near
  one it is supported on the frustrated region, so `√(P_a P_b)` restricts the burst to the mode's
  edges around it: the same operator, now a local perturbation when the mode is nearly right and a
  restart when it is not. This is the reading of "panic is a frustration map" (thesis 3.1) that has
  content on SAT.
- **M6** field cap 10 → 100 (the rescale was acting as extra evaporation every iteration).

## v2 — after M5–M6 (2026-09-11)

| run | result | notes |
|---|---|---|
| random n=200 s1 | not solved, best 5 (v1: 7) | population stays converged through bursts (diversity 0.07–0.10, was dispersed to 0.4 in v1): the localised burst is a perturbation now, not a restart. The mode improves 7 → 6 → 5 and then holds. |
| planted n=400 s1 | not solved, best 2 (v1: 1) | same picture: converged population, mode holds at 2 from it ~100 |
| planted n=400 s1, field off | not solved, best 20 | |

### Break: the mode holds

Both runs end with a converged population sitting at a mode a few clauses from a solution, the bursts
perturbing it locally without improving it. From U2 the reason is quantitative: the deposit is
`perf^{η_dep}` with `η_dep = 2`, and on the M3 scale an agent one clause better than the mode has
`perf` larger by `8/m`, so it deposits `(1 + 8/m)² − 1 ≈ 16/m ≈ 1 %` more. The field's selection per
clause is `η_dep·8/m → 0` with `n`. The population's variance around the mode (agents differ in 7–10 %
of variables) is not converted into field movement.

### Mutation → v3

- **M7** deposit `∝ exp(−κ·(unsat_k − min_k unsat))`, `κ = 1`: per-clause selection relative to the
  iteration's best. In U2's terms the objective becomes `E[exp(−κ·unsat)]` up to a per-iteration
  normaliser (a constant on the learning rate); the exponent is the thesis's free performance exponent,
  made instance-scaled and relative. (A first attempt with `perf^{κ m/8}` and no normaliser underflowed
  to a zero field: 0.75^106 ≈ 10⁻¹⁴.)

## v3 — after M7 (2026-09-11)

| run | result | notes |
|---|---|---|
| random n=200 s1 | not solved, best 3 (v2: 5, v1: 7) | population converged (diversity 0.03), mode holds at 3 from it ~80 |
| random n=200 s2 | not solved, best 3 | same |

Two implementation errors on the way, both recorded: `perf^{κm/8}` without a normaliser underflowed to
a zero field; `exp(−κ(u − u_min))` without weight normalisation starved the field (only near-best agents
deposited, total mass collapsed, the M2 threshold then killed every edge). The rule that works is the
normalised one: weights `∝ exp(−κ(u_k − u_min))` summing to one.

### Break: the mode holds, again, lower

Selection now moves the mode down to 3 unsatisfied, then the population converges (agents differ in
3 % of variables) and every agent's refinement is greedy, so every agent descends into the same basin,
which the field then reproduces. The localised bursts perturb the frustrated region but the greedy
descent returns to the basin. Nothing in the loop deviates from the greedy step.

### Mutation → v4

- **M8** the economy temperature applies to the refinement: with probability `temp − 0.5` (0.2 for
  agents that bought budget, 0.5 for those that did not) an agent flips a random variable of the chosen
  unsatisfied clause instead of the greedy one. This is the WalkSAT noise rule inside the agent's
  budget, and is named as such. Base flips 10 → 30 (60 buyers / 15 non-buyers).
  Control run: same with the field off (`alpha_w = 0`), to attribute any gain.

## v4 — after M8 (2026-09-11)

| run | result | notes |
|---|---|---|
| planted n=400 s1 | **solved, it 10** | v2 had held at 2 for 200 iterations |
| random n=200 s1 | not solved, best 1 (v3: 3) | pinned at 2 from it 20 (diversity 0.035), a single lucky refinement found 1 at it 60 |
| random n=200 s2 | not solved, best 1 | pinned at 1 from it 20 |
| random n=200 s1, **field off** | not solved, best 5 | same noisy refinement; the field is worth 4 clauses here |

### Break: pinned, and neither escape does anything

The seed-1 trajectory (`system_u_inspect.py`): from it 20 the population reproduces the same
assignment every iteration (`cur = 2`, diversity 0.03), `R_max` grows to 12 on its edges. Every burst
(every 10 iterations, regime negate+repel) fires at mean edge strength `g ≈ 0.05–0.07`: with the
localised panic, `√(P_a P_b)` is small on every edge that joins a frustrated literal to a normal one,
and those are all the mode's transitions into and out of the frustrated variables. Three 30 % resets
and one 95 % reset changed the diversity by nothing (0.035 before and after): the raised edges went to
the median positive attraction, which is invisible under `w = max(0, R² − M²)` once repulsion has
accumulated on the field. So the two escape mechanisms of the architecture were both inert on a pinned
mode, for two separate reasons, both consequences of the interference weight.

### Mutations → v5

- **M9** edge panic `max(P_a, P_b)` instead of `√(P_a P_b)`: the burst covers every edge touching a
  frustrated literal.
- **M10** reset baseline `max(median positive R, M_ab + 0.2·R_max)`: a reintroduced option is one the
  walk can see.

## v5 — after M9–M10; v6 — after M11 at 1 % (2026-09-11)

| run | result | notes |
|---|---|---|
| v5 random n=200 s1, s2 | not solved, best 1, 1 | bursts now at `g_max` 0.8–0.87 with 1,600–6,300 edges at erasure strength (M9 works); resets now disperse the population to diversity 0.35–0.40 (M10 works); **the population re-converges to the same mode within ~15 iterations every time** |
| v6 random n=200 s1, s2 (`eps_rel = 0.01`) | best 4, 2 at it 100, stopped | never converged (diversity 0.38): a floor at 1 % of the maximum weight against 400 alternatives per step makes the mode edge a 1-in-5 choice; panic went global, the bursts thresholded the whole field (`M_max > R_max`) |

### Break, both directions

The converged field is an attractor of the loop itself: the mode's non-frustrated edges keep `R ≈ 10`
through every burst and reset (bursts are localised by design after M5/M9; the reset only *raises*),
so after any dispersal the walk re-finds the same basin and the deposit re-strengthens it. The
deviation rate of the walk from a mode is set by the floor: per step, `Σ_alternatives ε / w_mode ≈
400·ε/R_max²`. With `ε = 10⁻⁴` absolute that is ~0.08 deviations per 200-step walk (pinned); with
`ε = 0.01·R_max²` it is ~800 (random). The walk needs a few deviations per walk, i.e. a relative
floor near `10⁻⁴`. U1c said the floor is required; this fixes its scale as the exploration rate.

### Mutation → v7

- **M11** (scaled) `ε = ε_rel·R_max²` with `ε_rel ∈ {10⁻⁴, 3·10⁻⁴}`: a few guided (η-weighted)
  deviations per walk, from a converged field that keeps its mode.

## Correction (2026-09-11): the random n=200 instances were UNSAT

CaDiCaL (host, 0.25 s each): random n=200 at 4.26, seeds 1 and 2, are **unsatisfiable**. Ratio 4.26 is
the threshold; at n = 200 about half of the instances have no solution, and I never checked. So from
v4 on, "pinned at best = 1" was System U **holding the optimum** of an unsatisfiable formula, and the
runs v5–v8 were chasing an impossible target. What stands from those runs: the *state-log* findings
(bursts inert at `√(P_a P_b)`, resets invisible under the interference weight, the deviation-rate
arithmetic of the floor, the restart-to-mode structure of an iteration) are true statements about the
loop and M9/M10 fixed real defects; M11/M12 are not justified by any failure and are kept as options
only (default off). The field-off controls (best 5 on the same instances) are still informative: the
field finds the optimum, the same local search alone does not.

Fix: `gen_3sat_sat` verifies random instances with CaDiCaL and advances the seed until one is
satisfiable (an instance-preparation step; the solver itself stays GPU-only). Planted instances are
satisfiable by construction. All random results below use verified-satisfiable instances.

## v4-config on verified-satisfiable instances (2026-09-11)

Config: M1–M10 on, `kappa_sel = 1`, absolute floor, base flips 30 (M11/M12 off).

| run | result | notes |
|---|---|---|
| planted n=800 s0, s1 | **solved**, it 16 and 30 (55 s, 94 s) | no bursts needed |
| random n=200 (seeds 1001, 5002, 3; CaDiCaL-verified SAT) | solved 2/3: it 134 (12 bursts, 3 resets), it 15; seed 5002 **held at 2** from it 14 to 200 (18 bursts, 5 resets) | |
| random n=400 s0 (SAT) | not solved, best 5 (reached it 88), 17 bursts, 4 resets | |

These are the real failures: early convergence to a mode a few clauses short, then holding through
every burst and reset. Tests running on exactly these instances: M12 (budget from chronic anxiety,
up to 1000 flips), with and without M11 (relative floor 1e-4).

### M12 tests (py refinement, budget to 1000): seed 5002 reached 1 (was 2), n=400 reached 3 (was 5) by it 50, then held; ~10 s/iteration.

### Break: the agents' local search is not compute at the landscape's scale

Instance check (host, plain WalkSAT-SKC, noise 0.5): seed 5002 is solved in 33k / 103k / 352k
*continuous* flips. System U's refinement restarts every agent at the field's mode each iteration and
gives it a 15–1000 flip segment; 6 M such flips did not solve it. The vectorised-over-agents Python
refinement (15 kernel launches per flip) cannot go longer. Since the walk reproduces the mode (walk
perf 0.98), an iteration is "restart at the best basin, search 1000 flips"; a plateau escape needs
10⁴–10⁵.

### Mutation → v9

- **M13** the refinement is WalkSAT-SKC per agent inside one CUDA kernel (one thread per agent,
  sequential flips; min-break with freebies; noise 0.3 for agents that bought budget, 0.6 otherwise).
  The economy's budget (`flips_base = 2000`, M12 growth to `flips_max = 50000`) is now real compute.
  Field, walk, η, crisis maps, burst, homology, momentum, hyperparameters, reset, economy: unchanged.
  Named plainly: the agents' local search is WalkSAT. Runs: random n=200 seed 5002, n=400 seeds 0/1
  (4.26), n=800 seed 0 (4.2), all CaDiCaL-verified.
