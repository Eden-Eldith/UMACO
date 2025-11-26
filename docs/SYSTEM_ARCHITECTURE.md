# UMACO System Architecture Blueprint

## Canonical Guiding Principles

- **GPU-first execution**: All cognitive dynamics assume CuPy arrays living on GPU. CPU compatibility is an explicit opt-in via `UMACO_ALLOW_CPU=1` and is considered a degraded mode for diagnostics only.
- **PAQ triad integrity**: Panic, Anxiety, and Quantum escape mechanisms are tightly coupled; touching one without considering the others destabilizes emergent behavior.
- **Complex-valued stigmergy**: Pheromone tensors are complex matrices whose real and imaginary channels encode exploitation and exploration pressure respectively. Any refactor must preserve complex arithmetic semantics.
- **Economy-enforced diversity**: Token dynamics keep agents heterogeneous. Resource flows are a first-class control surface, not an optional add-on.
- **Crisis-driven adaptation**: Hyperparameters (`alpha`, `beta`, `rho`, trauma factors, entropy targets) evolve from internal emotional state; they are not static knobs.

## Core Subsystems and Authoritative Modules

| Subsystem | Purpose | Canonical Implementation | Key Artifacts |
|-----------|---------|--------------------------|---------------|
| **UMACO Core** | Abstract cognitive optimizer, PAQ triad logic, stigmergic field, crisis hyperparameters, solver orchestration. | `Umaco13.py` | `UMACO`, `OptimizationResult`, `BaseEconomy`, `BaseNeuroPheromoneSystem`, `BaseUniversalNode`, `create_umaco_solver`, `_ensure_cupy_runtime_ready` |
| **NeuroPheromone System** | Complex pheromone lattice, diffusion, covariant momentum, topological analysis hooks. | `Umaco13.py` (`NeuroPheromoneSystem` class) | `pheromone_real/imag`, `covariant_momentum`, persistent homology utilities |
| **Universal Economy** | Token market, scarcity feedback, agent incentives. | `Umaco13.py` (`QuantumEconomy`) | `buy_resources`, `reward_performance`, market state buffers |
| **Cognitive Agents** | Crisis-aware decision units providing proposals/actions per iteration. | `Umaco13.py` (`CognitiveNode`) | `propose_action`, panic/anxiety state tracking |
| **Problem Facades** | Domain-specific glue for canonical solvers (continuous, combinatorial, SAT). | `Umaco13.py` helper functions, plus per-domain scripts like `TSP-MACO.py`, `ultimate_pf_simulator-v2-n1.py` | `create_umaco_solver`, `loss_fn` adapters |
| **LLM Specialization Harness** | Implements PAQ interfaces for language model fine-tuning with Quantum Economy extensions. | `maco_direct_train16.py` | `EnhancedQuantumEconomy`, `NeurochemicalPheromoneSystem`, `CognitiveNodeLLM`, training loop |
| **Visualization & Diagnostics** | Dashboards, tensor snapshots, panic/pheromone plots. | `visualize_umaco13.py`, `circuit_optimizer_gui.py` | Matplotlib dashboards, GUI bindings |
| **Benchmarking & Tests** | Regression baselines, continuous integration guardrails. | `benchmark_umaco13.py`, `test_umaco13.py`, `benchmark_results/` | Automated comparisons vs CMA-ES/SciPy, unit tests |

## Execution Flow (High Level)

1. **Solver Construction**: `create_umaco_solver` validates GPU runtime, instantiates `UMACO` with configs, synthesizes agent roster from `BaseUniversalNode` subclasses, and wires in the selected `BaseEconomy` & `BaseNeuroPheromoneSystem` implementation.
2. **Iteration Loop** (`UMACO.optimize`):
   - Agents call `propose_action` using panic, anxiety, pheromone context, and economy signals.
   - Economy arbitrates resource access and rewards via token flows.
   - Pheromone system deposits complex values, applies diffusion, and couples with covariant momentum.
   - PAQ triad updates panic/anxiety tensors and triggers `quantum_burst` escapes when thresholds trip.
   - Crisis-adaptive hyperparameters adjust (`alpha`, `beta`, `rho`, `trauma_factor`).
   - Loss callback transformed per `SolverType` evaluates solutions directly on GPU input tensors.
3. **Topology Feedback**: Optional persistent homology analysis produces `homology_report` steering anxiety gradients.
4. **Result Packaging**: Artifacts are wrapped in `OptimizationResult` (pheromones, panic history, loss trajectory, best solution/score).

## Domain Specializations

- **LLM Training (`maco_direct_train16.py`)**
  - Extends `BaseEconomy` → `EnhancedQuantumEconomy` (log-loss awareness, specialization metrics, trading).
  - Extends `BaseNeuroPheromoneSystem` with neurochemical modulation and GPU-native tensors.
  - Introduces LoRA/transformer integration, gradient/perplexity analytics, W&B telemetry.
- **Combinatorial Path (TSP)**
  - `TSP-MACO.py` uses `SolverType.COMBINATORIAL_PATH`, projecting pheromone lattice into tour probabilities and converting `OptimizationResult.best_solution` into route orderings.
- **SAT / Discrete Assignments**
  - `Umaco13.py` helpers convert pheromone diagonals into boolean assignments; `macov8no-3-25-02-2025.py` showcases SAT-specific harnessing.
- **Simulation/Physics**
  - Scripts like `ultimate_pf_simulator-v2-n1.py` embed UMACO loops inside domain simulators, consuming GPU tensors for state evolution.

## File Role Reference

| File | Role Synopsis |
|------|---------------|
| `Umaco13.py` | Canonical core framework (must be loaded for any downstream extension). |
| `maco_direct_train16.py` | Fine-tuning harness implementing `Umaco13` abstract bases for LLM workloads. |
| `UmacoFORCTF-v3-no1.py` | Cryptanalysis-focused variant leveraging same PAQ core. |
| `ultimate_zvss-v4-n1.py` | ZVSS simulation demonstrating SIMULATION mode integration. |
| `visualize_umaco13.py` | GPU-friendly visualization pipeline consuming `OptimizationResult`. |
| `benchmark_umaco13.py` / `benchmark_results/` | Regression & comparative performance tracking. |
| `test_umaco13.py` | Test suite validating core invariants and public APIs. |
| `ultimate_pf_simulator-v2-n1.py` | Protein folding exemplar showing domain adapter pattern. |
| `basic_optimization.py` | Minimal continuous example with default agents and economy. |
| `NeuroPheromonebasicv5.py` | Experimental neuro-pheromone prototypes (legacy insights). |

## GPU Purity Guardrails

- CuPy (`cp`) is the authoritative array backend; convert external NumPy data via `cp.asarray` before computation.
- `asnumpy` and `to_numpy_scalar` exist for visualization or logging only; they should not appear inside core optimization loops.
- `UMACO_ALLOW_CPU=1` is a temporary escape hatch for environments lacking CUDA runtime; do **not** enable by default in committed code.
- Any new modules must import shared GPU utilities (to be centralized in `umaco_gpu_utils.py`) instead of duplicating `_resolve_gpu_backend` logic.

## Interdependency Highlights

- `maco_direct_train16.py` imports abstract bases from `Umaco13.py`, meaning structural signatures in the core must remain stable.
- Visualization and benchmarking scripts depend on the `OptimizationResult` dataclass layout; altering field names/types requires synchronized updates.
- Domain adapters lean on `SolverType` semantics—adding new solver types demands updates to factory methods, agent projections, and downstream loss adapters.

## Pending Consolidation Targets

1. **Shared GPU utilities**: `_resolve_gpu_backend`, `asnumpy`, `to_numpy_scalar` duplicates exist in several scripts; these will migrate to a new `umaco_gpu_utils.py`.
2. **Type hints & docstrings**: `Umaco13.py` and derivatives need systematic type annotations and structured docstrings to ease AI comprehension.
3. **Canonical dependency graph**: A script will be added (`tools/dependency_map.py`) to visualize intra-repo imports for quick impact analysis.

Use this document as the opening context block for every AI-assisted session. It encodes the non-negotiable architectural intent and establishes the authoritative location of each subsystem.
