# P vs NP: the paper's argument in Lean

The polynomial-scaling paper
([`UMACO_Polynomial_SAT_Scaling_Paper.md`](../../benchmarks/benchmark_analysis/UMACO_Polynomial_SAT_Scaling_Paper.md)
§5.2–5.3) argues P = NP from MACO's SAT results. That argument is formalized in Lean 4 + Mathlib as
one theorem, `paper_argument`:

```
CookLevin NP → sound solver S over a finite seed set → polynomial per-run cost → Complete S δ (δ > 0)
  → ∀ A ∈ NP, InP A
```

Everything in the chain is proved except two hypotheses: Cook–Levin (the standard theorem, not
re-proved) and **`Complete S δ`**: on every satisfiable CNF, at least a δ fraction of a fixed seed set
reaches 100 % within a polynomial budget. Discharging `Complete S δ` means a hitting-time bound for
the solver's dynamics on every satisfiable 3-CNF. That is the open obligation.

| File | Contents |
|---|---|
| [`P_eq_NP_lean.md`](P_eq_NP_lean.md) | The paper's steps mapped to Lean theorems with status; what `Complete S δ` says exactly (fixed seed set vs fresh seeds); where the benchmark corpus sits as evidence |
| [`../lean/UmacoSat.lean`](../lean/UmacoSat.lean) | All the Lean: 107 theorems, no `sorry`, no added axioms; `paper_argument` depends only on `propext`, `Classical.choice`, `Quot.sound` |

## Building the Lean

```bash
cd docs/lean
lake exe cache get   # prebuilt Mathlib v4.33.1
lake build           # ends by printing the axioms each main theorem depends on
```

## Related

- [`docs/system-u/`](../system-u/): System U, UMACO specialized to SAT top-down, the construction
  aimed at `Complete S δ`, and theorems about it.
- [`research/experiments/system_u/`](../../research/experiments/system_u/): System U GPU runs.
- [`research/system_c/`](../../research/system_c/): results about the reduced Systems C / C′ / C‴,
  which do not transfer to UMACO.
