# The paper's P = NP argument, in Lean

Source: `benchmarks/benchmark_analysis/UMACO_Polynomial_SAT_Scaling_Paper.md` §5.2–5.3.
Lean: `docs/lean/UmacoSat.lean` §26 (theorems `polyBounded_*`, `inP_of_reduction`,
`np_subset_p_of_sat_in_p`, `sat_in_P_of_solver`, `paper_argument`, `fixed_budget_poly`), building on
§3–4 (`soundness`, `cost_linear`, `Solver`, `Complete`, `decision_rule_correct_iff`).
Build: clean, 107 theorems, no `sorry`, standard axioms only (`propext`, `Classical.choice`, `Quot.sound`).

## The argument as written in the paper

1. MACO returns a full satisfying assignment on every satisfiable instance (100 %, verified), and
   fails to reach 100 % on unsatisfiable ones.  ⟹ "run MACO; 100 % ⟺ SAT" is a decision procedure.
2. MACO's running time is polynomial in the instance size.
3. SAT is NP-complete.  ⟹ a polynomial-time decision procedure for SAT gives P = NP.

## The argument as a theorem

```
paper_argument :
  CookLevin NP →                       -- (3) every problem in NP reduces to SAT in poly time
  (S : Solver N) → 0 < N →             -- a sound solver with a finite seed set (O2 built into Solver)
  PolyBoundedBy CNF.size bound →       -- (2) per-run cost ≤ c·(size+1)^k          (O1)
  0 < δ → Complete S δ →               -- (1) on every satisfiable φ a δ-fraction of seeds reach 100 %  (O3)
  ∀ A ∈ NP, InP A                      -- NP ⊆ P
```

| paper step | Lean | status |
|---|---|---|
| 100 % ⟹ satisfiable (soundness, the recount) | `soundness`; built into `Solver.sound` | **proved** |
| decision rule correct on UNSAT | `decide_unsat_correct` | **proved** |
| decision rule correct on SAT ⟺ completeness | `decision_rule_correct_iff` | **proved** (equivalence) |
| deterministic decider from a complete solver over a finite seed set | `sat_in_P_of_solver` | **proved** |
| fixed-budget run cost polynomial (linear) in `n + m` | `cost_linear`, `fixed_budget_poly` | **proved** |
| poly ∘ poly is poly; reductions compose | `polyBounded_comp`, `polyBounded_add`, `inP_of_reduction` | **proved** |
| SAT ∈ P ⟹ NP ⊆ P | `np_subset_p_of_sat_in_p` | **proved** from `CookLevin` |
| Cook–Levin | `CookLevin NP` | hypothesis (standard theorem, not re-proved) |
| **O3: completeness with some δ > 0** | `Complete S δ` | **hypothesis** |

So the spine of the argument is a theorem. Given Cook–Levin, everything reduces to the single
hypothesis `Complete S δ` for a solver whose per-run cost is polynomial. Nothing else is missing.

## What `Complete S δ` says, exactly

`Solver N` has a finite seed space `Fin N` and `Complete S δ` says: for **every** satisfiable CNF,
at least a `δ` fraction of the `N` seeds reach 100 %. Two points that the formalization makes
precise:

- The seed set is **fixed in advance and independent of the instance**. A randomized solver with a
  fresh seed per run satisfies a different statement (success probability ≥ δ per run), which
  gives SAT ∈ RP and hence NP = RP, not P = NP directly. The step from "≥ δ per run" to a fixed seed
  set that works for all instances is a derandomization assumption; `paper_argument` takes the
  deterministic form. Under `Complete S δ` with δ > 0 the decider runs all `N` seeds and answers
  SAT iff one succeeds; cost `N·bound φ`, polynomial when `N` is a constant (or polynomial in size).
- The cost bound `bound φ` must hold **for the budget that achieves completeness**. `cost_linear`
  shows any fixed budget `(T, K, F)` is linear in `n + m`; the content of O1 is that a budget
  polynomial in `n` suffices for O3, i.e. that `δ` does not decay faster than any polynomial as `n`
  grows.

## Where the benchmark corpus sits

The corpus is the evidence for `Complete S δ` in the tested range: satisfiable instances on which
MACO produced a witness that was then verified (the protocol the author describes: MiniSat's initial
attempt, the BREAKTHROUGH record of MACO's witness, MiniSat's verification of that witness). Under
that protocol, each such run is one instance of `S.run φ s = some a` with `φ.Satisfies a`, i.e. one
data point for the hypothesis. The theorem says what those data points would have to extend to:
every satisfiable instance, at every size, with a seed fraction bounded below and a budget bounded
by a polynomial. That is the proof obligation that remains, and it is a statement about the
solver's dynamics, not about the corpus. It is obligation O3 in the table above; this document adds
the theorem that O3 is *sufficient*.

## What would discharge it

A proof of `Complete S δ` for a polynomial budget is a hitting-time bound for the solver's Markov
chain on every satisfiable 3-CNF. The System U work (`docs/system-u/SystemU_theorems.md` U1–U5,
`research/experiments/system_u/SystemU_runs.md`) is the attempt to build a solver whose dynamics admit
such a bound; its state is recorded there.
