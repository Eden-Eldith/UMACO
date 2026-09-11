# Systems C / C′ / C‴

A reduced abstraction of UMACO for SAT, studied bottom-up from the February 2025 solver lineage:
per-variable marginals sampled independently (a product distribution), a pheromone that integrates
fitness, clause weights, and local search. There are no PAQ tensors, economy or homology.

Everything here is about these systems. None of it is a result about UMACO or System U. System U
([`docs/system-u/`](../../docs/system-u/)) walks a literal graph sequentially over a pairwise complex
field, and its burst and homology act on the whole field, so the product-sampler obstructions proved
here do not apply to it without new proofs.
[`UMACO13_SAT_architecture_audit.md`](../../docs/system-u/UMACO13_SAT_architecture_audit.md) §3 gives
each result's status for System U.

| File | Contents |
|---|---|
| [`SystemC_3SAT_attack.md`](SystemC_3SAT_attack.md) | Research log (10 Sep 2026): derivation, Theorems 1–28, planted-model results, the XOR obstruction, mean-field flow, the finite-K algorithm |
| [`meanfield_flow_gpu.py`](meanfield_flow_gpu.py) | Exact quenched mean-field flow (the K → ∞ limit) of System C / C‴ on planted, random and XOR instances |
| [`systemc3_gpu.py`](systemc3_gpu.py) | System C‴ at finite K: min-break local search, breakout-at-stall clause weights, host recount of any claimed solution |

Both scripts are CuPy-only. Lean statements of the discrete results are in
[`docs/lean/UmacoSat.lean`](../../docs/lean/UmacoSat.lean) (§8 onward). The document that first
defines System C (`UMACO_lineage_reconstruction.md`, Part IV) is not yet published.
