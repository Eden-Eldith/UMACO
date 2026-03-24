# UMACO: Universal Multi-Agent Cognitive Optimization

<div align="center">

![UMACO Logo](https://github.com/user-attachments/assets/11a19c53-e374-497b-8903-30a9c20ddf91)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-13.0-red)](https://github.com/EdenEldith/umaco)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Created By](https://img.shields.io/badge/created%20by-Eden%20Eldith-purple)](https://github.com/EdenEldith)
![AI-First Framework](https://img.shields.io/badge/AI--First-Framework-blueviolet)
![GPU-First](https://img.shields.io/badge/GPU--First-Architecture-green)

</div>

## A Note on P vs NP

Yes, this repo contains a paper arguing that MACO exhibits polynomial-time scaling on NP-complete SAT instances. It's in [`benchmarks/benchmark_analysis/UMACO_Polynomial_SAT_Scaling_Paper.md`](benchmarks/benchmark_analysis/UMACO_Polynomial_SAT_Scaling_Paper.md). The benchmark logs backing it up — 5,183 runs across 27.1 GPU-hours — are in [`benchmarks/`](benchmarks/).

The data shows polynomial scaling (R² = 0.897) where exponential is expected, and MACO outperforming MiniSat by 4.6x on hard instances. I believe the results are real. The code is verified. The evaluation is correct at every layer.

I don't have the confidence to push on this and get it formally verified. I'm a self-taught developer on disability benefits who got D's in GCSEs. Every time I've tried to pursue this, I've been talked out of it or had the work undermined. So the paper is here, the benchmarks are here, the code is here. Read it if you want. Run it if you want. Tell me I'm wrong if you want. But I'm not going to pretend it doesn't exist anymore.

If you're a researcher and you think this warrants investigation, my ORCID is 0009-0007-3961-1182.

---

## What is UMACO?

UMACO (Universal Multi-Agent Cognitive Optimization) is a GPU-first, AI-collaborative optimization framework built to solve any problem domain through emergent collective intelligence. Unlike traditional libraries you install and call, UMACO is designed from the ground up to be understood, refactored, and specialized by Large Language Models (LLMs) and human developers alike.

The framework combines four interconnected cognitive systems — the **PAQ Triad** (Panic-Anxiety-Quantum), **Topological Stigmergic Fields**, a **Universal Economy**, and **Crisis-Driven Hyperparameters** — into a single architecture that adapts to continuous optimization, combinatorial path problems, satisfiability, and simulation domains.

This project represents not just a technical achievement, but a personal one: I created UMACO without formal computer science education, teaching myself programming from scratch in just over 3 months with the assistance of AI tools. My hope is that this framework demonstrates that meaningful contributions to complex fields can come from non-traditional backgrounds.

### Developer Guide

Looking to dive into the architecture, agent design, or code details?

[Read the UMACO Developer Guide (PDF)](https://github.com/Eden-Eldith/UMACO/blob/master/docs/UMACO%20Developer%20Guide.pdf)

## Table of Contents

- [Quick Start](#quick-start)
- [How UMACO Works](#how-umaco-works)
- [Core Architecture](#core-architecture)
- [Problem Types](#problem-types)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Requirements & Installation](#requirements--installation)
- [Using UMACO with AI](#using-umaco-with-ai)
- [Visualizations](#visualizations)
- [Benchmarks](#benchmarks)
- [Potential Applications](#potential-applications)
- [Technical Overview](#technical-overview)
- [Why Sponsor UMACO?](#why-sponsor-umaco)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Quick Start

### Using the Universal Solver (Recommended)

The `UMacoSolver` facade provides the simplest entry point. It infers problem type, constructs loss functions, and configures the engine automatically.

```python
from universal_solver import UMacoSolver
import numpy as np

# Travelling Salesperson Problem — just provide a distance matrix
distances = np.random.rand(15, 15)
distances = (distances + distances.T) / 2
solver = UMacoSolver(distance_matrix=distances, max_iter=200)
result = solver.optimize()
print(f"Best tour length: {result.score}")

# Custom continuous objective
def my_loss(matrix):
    return float(np.sum((matrix - 0.5) ** 2))

solver = UMacoSolver(custom_loss=my_loss, problem_dim=32, n_ants=12)
result = solver.optimize()

# SAT problem — provide clauses with 1-based literal indices
clauses = [[1, -2, 3], [-1, 2, 4]]
solver = UMacoSolver(clauses=clauses)
result = solver.optimize()
```

### Using the Core Engine Directly

```python
from umaco.Umaco13 import create_umaco_solver, rosenbrock_loss

# Create solver for any problem type
solver, agents = create_umaco_solver(
    problem_type='CONTINUOUS',       # or 'COMBINATORIAL_PATH', 'SATISFIABILITY', 'SIMULATION'
    dim=32,                          # Pheromone matrix resolution
    max_iter=100,                    # Optimization iterations
    n_ants=8                         # Number of cognitive agents
)

# Run optimization
result = solver.optimize(agents, rosenbrock_loss)
print(f"Best score: {result.best_score}")
print(f"Final panic: {result.panic_history[-1]:.3f}")
```

## How UMACO Works

### The AI-Collaborative Workflow

```
+-------------------+    +--------------------+    +-------------------+
| UMACO Framework   |    | Problem Statement  |    |                   |
| Base Code         | +  | in Natural Language | -> | LLM Processing    | ->
|                   |    |                    |    |                   |
+-------------------+    +--------------------+    +-------------------+

+------------------------+    +-----------------------+
| Custom UMACO Variant   |    |                       |
| Specialized for the    | -> | Problem Solution      |
| Specific Problem       |    |                       |
+------------------------+    +-----------------------+
```

Rather than being a static library, UMACO serves as a sophisticated template that LLMs can understand and customize:

1. **Problem Description**: Describe your optimization problem to an LLM in natural language
2. **Framework Presentation**: Provide the UMACO code to the LLM
3. **AI Refactoring**: The LLM refactors UMACO to create a specialized variant
4. **Implementation**: Run the customized code to solve your specific problem

### Example Use Cases

1. **Financial Portfolio Optimization** — Present UMACO + portfolio constraints to an LLM; get back a customized optimizer with risk-adaptive agents

2. **Neural Network Hyperparameter Tuning** — Describe network architecture + constraints; receive a UMACO variant specialized for hyperparameter search

3. **Supply Chain Optimization** — Provide logistics constraints and objectives; obtain a modified UMACO with specialized pheromone interpretations

## Core Architecture

UMACO's architecture is designed to be both powerful and comprehensible to AI, allowing for effective refactoring across diverse domains.

### PAQ Core (Panic-Anxiety-Quantum)

The heart of UMACO is the PAQ Core, which gives the system its adaptive intelligence:

- **Panic Tensor**: A 2D field that tracks local "crisis states," identifying regions where optimization is struggling. Updated via finite-difference gradients coupled to anxiety amplitude.
- **Anxiety Wavefunction**: A complex field that maps existential risk gradients, with real parts representing immediate concerns and imaginary parts representing potential future challenges.
- **Quantum Burst**: An SVD-based mechanism for escaping local minima that combines structured perturbations with controlled randomness, modulated by anxiety. Triggered by crisis, not on a schedule.

### Topological Stigmergic Field

The TSF implements indirect coordination through environment modification, inspired by how ants communicate via pheromone trails:

- **Complex Pheromones**: Represented as complex numbers where:
  - Real part = attractive forces (exploitation)
  - Imaginary part = repulsive forces (exploration)
- **Persistent Homology**: Uses topological data analysis (ripser/persim) to understand the "shape" of the optimization landscape
- **Covariant Momentum**: Implements momentum that preserves information about the topological structure of the search space

<div align="center">
<img src="https://github.com/user-attachments/assets/81bf1f54-70a5-4f1a-998a-0fffee57567a" alt="Topological Field Visualization" width="600"/>
</div>

### Universal Economy

The economy provides a regulatory framework for resource allocation among agents:

- **Token-Based Resource Management**: Agents receive tokens based on their contribution to optimization progress
- **Dynamic Market Forces**: Market value fluctuates based on resource scarcity, trading activity, and random volatility
- **Multi-Agent Trading**: Agents can trade tokens, enabling specialization in different aspects of the optimization

### Crisis-Driven Hyperparameters

All key hyperparameters dynamically respond to the system's internal state:

- **Alpha**: Controls the influence of covariant momentum, scaling with panic and anxiety
- **Beta**: Modulates the balance between attraction and repulsion based on persistent entropy
- **Rho**: Controls the rate of change, scaling inversely with momentum magnitude

These are the system's emotional state, not static configuration values.

## Problem Types

UMACO13 supports four unified problem types via the `SolverType` enum:

| Type | Description | Input | Example |
|------|-------------|-------|---------|
| `CONTINUOUS` | Function optimization | Pheromone distributions | Rosenbrock, Rastrigin, sphere |
| `COMBINATORIAL_PATH` | Sequence/tour construction | Distance matrix | TSP, routing |
| `SATISFIABILITY` | Binary constraint satisfaction | CNF clauses | 3-SAT |
| `SIMULATION` | UMACO itself is the solution | Custom loss | Ecosystem modelling, plague spread |

## Repository Structure

```
UMACO/
|-- umaco/                          # Core package
|   |-- __init__.py                 # Package exports and version
|   |-- Umaco13.py                  # UMACO13 — the canonical GPU-first implementation
|   |-- umaco_gpu_utils.py          # Centralized GPU/CPU backend switching
|   |-- maco_direct_train16.py      # LLM training specialization (MACO-LLM)
|   |-- test_umaco13.py             # 16 comprehensive test cases
|   |-- benchmark_umaco13.py        # Performance comparison vs CMA-ES, SciPy
|   |-- visualize_umaco13.py        # Matplotlib dashboards and pheromone heatmaps
|   |-- Umaco9.py                   # Legacy reference implementation
|   \-- umaco10.py                  # Intermediate version (historical)
|
|-- universal_solver.py             # High-level AI-friendly solver facade
|-- test_universal_solver.py        # Tests for the universal solver
|-- setup.py                        # Package installation script
|-- pyproject.toml                  # PEP 621 build configuration
|
|-- examples/
|   |-- basic_optimization.py       # 2D Rosenbrock with visualization
|   |-- TSP-MACO.py                 # Traveling Salesperson Problem
|   |-- circuit_optimizer_gui.py    # Component placement with Tkinter GUI
|   |-- llm_training.py             # Fine-tuning LLMs with MACO-LLM
|   |-- NeuroPheromonebasicv5.py    # Experimental neuro-pheromone prototypes
|   |-- macov8no-3-25-02-2025.py    # GPU-accelerated SAT solver
|   |-- UmacoFORCTF-v3-no1.py      # Cryptanalysis for SPEEDY-7-192
|   |-- ultimate_pf_simulator-v2-n1.py  # Protein folding simulator
|   \-- ultimate_zvss-v4-n1.py      # ZVSS integrated simulator
|
\-- docs/
    |-- core_concepts.md            # PAQ, TSF, economy, crisis-driven hyperparameters
    |-- SYSTEM_ARCHITECTURE.md      # Authoritative architecture blueprint
    |-- adapting_to_llm.md          # Guide for LLM adaptation
    \-- UMACO Developer Guide.pdf   # Comprehensive PDF developer guide
```

### Key Files

| File | Purpose |
|------|---------|
| `umaco/Umaco13.py` | The canonical UMACO13 implementation — GPU-first with PAQ, TSF, economy, and all four problem types |
| `universal_solver.py` | Zero-guessing facade that infers problem type from inputs and constructs appropriate loss functions |
| `umaco/umaco_gpu_utils.py` | Centralized CuPy/NumPy backend resolution with `UMACO_ALLOW_CPU=1` fallback |
| `umaco/maco_direct_train16.py` | MACO-LLM extension for fine-tuning language models with enhanced quantum economy |

## Key Features

### UMACO13 Core Framework

- **Panic-Anxiety-Quantum Triad (PAQ Core)**: Biomimetic crisis response — when agents encounter difficult terrain, panic triggers exploration while anxiety enhances focus on promising regions. Quantum bursts use SVD-based structured escapes driven by crisis.

- **Topological Stigmergic Field (TSF)**: Complex pheromone tensors where real components encode attraction and imaginary components encode repulsion. Persistent homology analysis guides the search through landscape topology.

- **Crisis-Driven Hyperparameters**: Alpha, beta, and rho evolve based on the system's internal emotional state (panic levels, anxiety amplitude, momentum magnitude) — not static tuning.

- **Universal Economy**: Token-based market system where agents compete for resources. High performers earn tokens, underperformers shift strategies. Market dynamics (inflation, volatility) prevent stagnation.

- **GPU-First Architecture**: All core operations use CuPy for GPU acceleration. CPU fallback available via `UMACO_ALLOW_CPU=1` but degrades performance.

### Universal Solver Facade

- **Automatic problem type inference** from inputs (distance matrix -> TSP, clauses -> SAT, custom loss -> continuous)
- **Built-in objectives**: sphere, rastrigin, rosenbrock
- **Input validation**: Strict checks prevent silent misconfiguration
- **Expert overrides**: Surgical configuration control via `config_overrides`
- **Structured results**: `SolverResult` with solution, score, full diagnostic history, and optimizer reference

### MACO-LLM Extensions

- **Enhanced Quantum Economy**: Advanced trading with specialized agent roles (Explorers, Exploiters, Evaluators) and dynamic market conditions
- **NeuroPheromone System**: Neural pathway formation inspired by biological neurochemicals
- **Loss-Aware Metrics**: Specialized tracking for learning rate, regularization, and architecture optimization
- **LoRA/QLoRA Support**: Parameter-efficient fine-tuning integration

### Testing & Benchmarking

- **16 comprehensive tests** covering PAQ core, economy, topology, solution construction, and full optimization loops
- **Benchmark suite** comparing against CMA-ES and SciPy optimizers on standard functions

## Requirements & Installation

### Core Requirements

```
numpy>=1.20.0
matplotlib>=3.5.0
ripser>=0.6.0
persim>=0.3.0
scipy>=1.7.0
```

### Installation

Clone the repository and install locally:

```bash
pip install .
```

Optional features:

```bash
pip install ".[gpu]"   # CUDA GPU acceleration (CuPy)
pip install ".[llm]"   # LLM training tools (PyTorch, Transformers, PEFT, W&B)
```

### GPU Setup

UMACO is GPU-first. For full performance, install CuPy matching your CUDA version:

```bash
pip install cupy-cuda11x   # For CUDA 11.x
pip install cupy-cuda12x   # For CUDA 12.x
```

For CPU-only usage (reduced performance), set the environment variable:

```bash
export UMACO_ALLOW_CPU=1
```

## Using UMACO with AI

### Crafting Effective Prompts

1. **Be Specific About Your Problem**: Clearly define variables, constraints, and objectives
2. **Request Specific Adaptations**: Ask for customized loss functions and domain-appropriate agent specializations
3. **Provide Sample Data**: Include example data formats, variable ranges, and typical scenarios

### Example Prompts

#### Hyperparameter Optimization

```
I have the UMACO framework code below, and I need to optimize hyperparameters for a
neural network that predicts stock prices. Please refactor UMACO to create a specialized
variant for neural network hyperparameter optimization.

[UMACO CODE HERE]

The neural network has these hyperparameters to optimize:
- Learning rate (range: 0.0001 to 0.1)
- Batch size (range: 16 to 512)
- Hidden layer sizes (1-3 layers, 32-512 neurons each)
- Dropout rate (range: 0 to 0.5)

Please customize the UMACO agents to specialize in different aspects of neural network
optimization, and adapt the pheromone interpretation to represent hyperparameter
configurations.
```

#### Logistics Optimization

```
Here's the UMACO framework. I need to optimize a delivery routing system for a fleet
of 20 vehicles serving 150 customers across a city.

[UMACO CODE HERE]

Key constraints: vehicle capacity limits, time windows, driver work hours, fuel
efficiency. Minimize total distance while satisfying all delivery requirements.
```

### Usage Examples

#### Continuous Optimization with UMACO13

```python
from umaco.Umaco13 import create_umaco_solver, rosenbrock_loss
import numpy as np

# Create solver
solver, agents = create_umaco_solver(
    problem_type='CONTINUOUS',
    dim=32,
    max_iter=200,
    n_ants=8
)

# Run optimization
result = solver.optimize(agents, rosenbrock_loss)
pheromone_real = result.pheromone_real
panic_history = result.panic_history
loss_history = result.loss_history
```

#### TSP with the Universal Solver

```python
from universal_solver import UMacoSolver
import numpy as np

# Create a symmetric distance matrix for 20 cities
n_cities = 20
distances = np.random.rand(n_cities, n_cities)
distances = (distances + distances.T) / 2
np.fill_diagonal(distances, 0)

solver = UMacoSolver(distance_matrix=distances, max_iter=500)
result = solver.optimize()

print(f"Best tour length: {result.score}")
print(f"Diagnostic history keys: {list(result.history.keys())}")
```

#### 3-SAT Solving

```python
from universal_solver import UMacoSolver

# Define clauses (1-based literals, negatives = NOT)
clauses = [[1, -2, 3], [-1, 2, 4], [2, -3, -4], [1, 3, 4]]
solver = UMacoSolver(clauses=clauses, max_iter=400)
result = solver.optimize()

print(f"Unsatisfied clauses: {result.score}")
```

#### Training an LLM with MACO-LLM

```python
from umaco.maco_direct_train16 import MACAOConfig

config = MACAOConfig(
    model_name="microsoft/phi-2",
    output_dir="./macao_output",
    training_data_file="your_training_data.jsonl",
    n_agents=8,
    num_epochs=3,
    batch_size=1
)

# See examples/llm_training.py for a complete example
```

## Visualizations

UMACO13 includes a visualization suite (`umaco/visualize_umaco13.py`) for real-time monitoring:

- Panic level evolution over iterations
- Loss trajectory and convergence
- Pheromone field heatmaps (real and imaginary components)
- Hyperparameter evolution (alpha, beta, rho)
- Token distribution across agents
- Quantum burst event markers

<div align="center">
<img src="https://github.com/user-attachments/assets/81bf1f54-70a5-4f1a-998a-0fffee57567a" alt="Pheromone Evolution" width="600"/>
</div>

*Dynamic Evolution of UMACO's Pheromone Tensor:* This animation visualizes the real (Attraction) and imaginary (Repulsion) components of the pheromone field over optimization iterations. Bright regions indicate areas of high pheromone concentration where agents converge on promising solutions.

## Benchmarks

UMACO13 includes a benchmark suite (`umaco/benchmark_umaco13.py`) for comparison against standard optimizers:

| Optimizer | Status | Installation |
|-----------|--------|--------------|
| UMACO13 | Available | Included |
| CMA-ES | Available | `pip install cma` |
| SciPy | Available | `pip install scipy` |

**Sample Results (Sphere Function, 2D):**
- **UMACO13**: Loss = 0.0019 (converged in ~0.6s)
- **CMA-ES**: Loss = ~0.0000 (converged to machine precision)

UMACO13 shows competitive performance while maintaining universal problem applicability across continuous, combinatorial, and satisfiability domains.

## Potential Applications

UMACO can be adapted for numerous applications through AI refactoring:

- **Machine Learning**: Hyperparameter optimization, neural architecture search
- **Financial Modelling**: Portfolio optimization, risk management
- **Drug Discovery**: Molecular optimization, protein folding
- **Engineering Design**: Multi-objective optimization, constraint satisfaction
- **Logistics**: Complex scheduling, route optimization
- **Game Development**: Procedural generation, AI behaviour tuning
- **AI Research**: Meta-learning, transfer learning, reinforcement learning
- **Energy Management**: Grid optimization, resource allocation
- **Cryptography**: Cryptanalysis via multi-agent search with quantum burst escapes

## Technical Overview

### System Architecture

```
UMACO13
   |
   |--- PAQ Core
   |     |- panic_tensor
   |     |- anxiety_wavefunction
   |     |- quantum_burst()
   |     \- panic_backpropagate()
   |
   |--- Topological Stigmergic Field
   |     |- complex pheromones (real + imag)
   |     |- covariant_momentum
   |     \- persistent_homology_update()
   |
   |--- Universal Economy
   |     |- tokens
   |     |- market_value
   |     |- buy_resources()
   |     \- reward_performance()
   |
   |--- Cognitive Agents (UniversalNode)
   |     |- panic_level
   |     |- risk_appetite
   |     \- propose_action()
   |
   \--- Crisis-Driven Hyperparameters
         |- alpha (pheromone influence, evolves with panic)
         |- beta  (heuristic influence, evolves with entropy)
         \- rho   (evaporation rate, evolves with momentum)
```

| Component | Description |
|-----------|-------------|
| **UMACO13** | Canonical GPU-first optimizer with PAQ triad, TSF, economy, and four problem types |
| **UMacoSolver** | High-level facade with automatic problem inference and input validation |
| **MACO-LLM** | LLM training specialization with enhanced quantum economy and agent trading |
| **PAQ Core** | Biomimetic crisis-handling module (Panic-Anxiety-Quantum Triad) |
| **TSF** | Complex pheromone field (real=attraction, imaginary=repulsion) with topology analysis |
| **Universal Economy** | Token-based market system for computational resource allocation |
| **umaco_gpu_utils** | Centralized GPU backend (CuPy) with CPU fallback switching |

### Tuning Guide

**Safe to Tune:**
- `n_dim`: Match your problem size (16-512)
- `max_iter`: More iterations = better solutions (50-1000+)
- `n_ants`: 4-32 range, more for harder problems
- `target_entropy`: Higher = more exploration (0.5-0.9)

**Leave Alone (unless you deeply understand UMACO):**
- The PAQ triad logic (panic, anxiety, quantum bursts)
- Complex-valued pheromone/momentum structures
- SVD-based quantum burst implementation
- Topology-momentum coupling

### Why This Architecture Works with LLMs

1. **Clear Component Separation**: Each system is well-isolated with defined interfaces
2. **Explicit Conceptual Mapping**: Components map to intuitive concepts (panic, economy, agents)
3. **Self-Documenting Code**: Extensive docstrings explain purpose and functionality
4. **Flexible Extension Points**: Easy-to-identify places for domain-specific customization
5. **Standardized Interfaces**: Consistent patterns for interaction between components

### Legacy Implementations

- **Umaco9.py** — The original UMACO implementation. Retained for reference and education. Use UMACO13 for all new projects.
- **umaco10.py** — Intermediate version bridging Umaco9 and Umaco13. Historical interest only.

## Scientific Background

UMACO draws inspiration from several scientific fields:

- **Stigmergy**: Self-organization through environment modification
- **Topological Data Analysis**: Using topological features to understand data
- **Quantum Computing**: Concepts of superposition and structured perturbation
- **Complex Systems**: Emergence and self-organization in multi-agent systems
- **Neuroeconomics**: Decision-making under uncertainty and resource constraints

[Full UMACO Architecture](https://garden-backend-three.vercel.app/fixed-thesis-maco/)

## Why Sponsor UMACO?

UMACO offers several compelling advantages for potential investors and sponsors:

1. **Novel Approach**: A fundamentally new approach to optimization, combining multi-agent systems with AI-collaborative design
2. **AI-First Design**: Specifically built to leverage modern LLMs for rapid problem specialization
3. **Universal Adaptation**: Rapid customization for diverse domains without requiring optimization expertise
4. **Reduced Time-to-Solution**: Natural language problem specification accelerates the development cycle
5. **Emerging Field**: Sits at the intersection of LLM-based programming, multi-agent AI, and topological data analysis
6. **Commercial Applications**: Direct applications in finance, pharmaceuticals, logistics, and more
7. **Inspiring Story**: Created by a self-taught developer who learned coding from scratch in just over 3 months

Your sponsorship would help:
- Expand research into new domains and applications
- Create specialized UMACO variants for high-value industrial problems
- Develop improved documentation and example libraries
- Build a community of practitioners
- Support neurodivergent developers in continuing this innovative work

## Contributing

[![Developer Guide](https://img.shields.io/badge/Docs-Developer_Guide-blue)](https://github.com/Eden-Eldith/UMACO/blob/master/docs/UMACO%20Developer%20Guide.pdf)

UMACO was created by Eden Eldith, a self-taught developer who built this entire framework without formal computer science education. The project welcomes contributions from people of all backgrounds, especially those who might not fit the traditional developer profile:

1. **Example Prompts**: Create and share effective prompts for different domains
2. **Domain Adaptations**: Share successful UMACO variants for different problems
3. **Documentation**: Improve guides on how to effectively describe problems to LLMs
4. **Testing**: Test the framework on different types of problems and LLMs
5. **Community**: Help answer questions, mentor new contributors
6. **Accessibility**: Help make the project more accessible to neurodivergent developers and beginners

See [The UMACO discussions page](https://github.com/Eden-Eldith/UMACO/discussions) for detailed guidelines.

[Read the full UMACO Developer Guide (PDF)](https://github.com/Eden-Eldith/UMACO/blob/master/docs/UMACO%20Developer%20Guide.pdf) for architecture details, agent economy, PAQ triad, topological stigmergy, hands-on code examples, and tips for adapting UMACO to your own projects.

## Citation

If you use this code in your research, please cite:

```
@software{umaco2025,
  author = {Eden Eldith},
  title = {UMACO: Universal Multi-Agent Cognitive Optimization},
  year = {2025},
  url = {https://github.com/Eden-Eldith/UMACO}
}
```

## License

This project is licensed under the RCL License - see the [LICENSE](LICENSE) file for details.

> **Entanglement Notice:**
> Any use, inspiration, or derivative work from this repo, in whole or in part, is subject to the Recursive Entanglement Doctrine (RED.md) and Recursive Cognitive License (RCL.md).
> Non-attribution, commercial exploitation, or removal of epistemic provenance is a violation of licensing and will be treated accordingly.

## Acknowledgements

UMACO was developed entirely by Eden Eldith, who taught themselves programming from scratch in just over three months while facing significant personal challenges including OCD, ADHD, autism, anxiety, and chronic pain. This project demonstrates how innovation can come from unexpected places and how AI assistance can help bridge accessibility gaps in technology development.

Special thanks to the developers of the open-source libraries that made this project possible, and to the AI tools that helped make this development journey accessible to a self-taught programmer with no formal qualifications.

---

<div align="center">
<p>UMACO: AI-Collaborative Optimization Framework</p>
<p>Created by Eden Eldith | 2025</p>
</div>
