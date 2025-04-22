# UMACO: Universal Multi-Agent Cognitive Optimization

UMACO is a novel optimization framework that combines principles from quantum physics, multi-agent systems, topological analysis, and economic mechanisms to solve complex optimization problems that defeat traditional approaches.

This repository contains two main implementations:

1. **UMACO9**: A foundational framework for general-purpose optimization problems using the PAQ Core (Panic-Anxiety-Quantum Triad System) to dynamically respond to challenging optimization landscapes
2. **MACO-LLM**: A specialized implementation for training and fine-tuning Large Language Models using multi-agent systems with an economic marketplace to efficiently allocate computational resources

## System Architecture

| Component | Description |
|-----------|-------------|
| **UMACO9** | Core agent framework for non-convex optimization with crisis detection |
| **MACO-LLM** | Specialized LLM trainer with economic adaptation and trading mechanisms |
| **PAQ Core** | Biomimetic crisis-handling module (Panic-Anxiety-Quantum Triad) |
| **TSF** | Advanced pheromone field using complex tensors (real=attraction, imaginary=repulsion) |
| **Universal Economy** | Market-based computational resource allocation system |
| **ZVSS** | Zero-value state search acceleration for GPU-powered optimization |
| **NeuroPheromone** | Neural pathway formation inspired by biological neurochemicals |

## Problem-Solving Domains

UMACO solves high-complexity problems across domains like optimization, AI, and cryptography:

* **Optimization**: Tackles challenging non-convex optimization problems with numerous local minima that traditional methods struggle with. UMACO's multi-agent approach efficiently navigates complex landscapes that would trap conventional gradient-based methods.

* **Artificial Intelligence**: Implements sophisticated multi-agent systems that collaborate through stigmergic communication (environment-mediated information sharing) to make complex decisions in uncertain environments. This approach mimics how social insects solve collective problems.

* **Complex Systems**: Provides robust analysis of dynamic optimization landscapes through topological methods. The framework adapts resource allocation in changing environments by understanding the "shape" of the solution space.

* **Satisfiability (SAT) Solving**: Accelerates Large Language Model training by reformulating optimization challenges as SAT problems. UMACO adapts traditional optimization frameworks to AI-specific challenges through its crisis-driven hyperparameter tuning.

* **Software Development**: Enhances package dependency resolution and offers seamless integration with external ML libraries including PyTorch, Transformers, and PEFT. The framework resolves complex dependency graphs using the same principles it applies to other optimization problems.

* **Cryptography**: Enables advanced cryptanalysis of modern ciphers like SPEEDY-7-192 through multi-agent search techniques with dynamic pheromone-guided exploration. The quantum burst mechanism helps escape local optima when searching for cryptographic weaknesses.

## Scripts and Documents

* `macov8no-3-25-02-2025.py`: **GPU-Accelerated SAT Solver** - High-performance SAT solver combining MACO with Zero-Value State Search (ZVSS) on GPU hardware. Achieves up to 8.4x speedup over traditional solvers on complex problems by distributing computation across thousands of GPU threads.

* `setup_revised.py`: **Setup File** - Comprehensive installation script that handles all dependencies, CUDA configurations, and optional visualization tools based on your system capabilities.

* `docs/adapting_to_llm.md`: **Adapting UMACO to LLM Training** - Detailed guide for applying UMACO to Large Language Model training, including specific hyperparameter recommendations and performance comparisons with traditional training approaches.

* `docs/core_concepts.md`: **UMACO Core Concepts** - In-depth explanation of UMACO's foundational principles, including the PAQ Core, Topological Stigmergic Field, and Universal Economy. Explains how these components interact to create an effective optimization system.

* **Examples:**
	1. `basic_optimization.py`: **Basic UMACO9 Optimization** - Step-by-step example demonstrating how to configure UMACO9 for the 2D Rosenbrock function. Includes visualization of pheromone evolution and panic responses during optimization.

	2. `llm_training.py`: **LLM Training with MACO** - Complete implementation showing how to use MACO-LLM to fine-tune language models like Phi-2. Features the enhanced quantum economy and multi-agent system with specialized agent roles.

	3. `NeuroPheromonebasicv5.py`: **Self-Optimizing Brain Model** - Neural network implementation that uses MACO to create a self-optimizing brain model. Demonstrates how pheromone values evolve to minimize a loss function through emergent cognitive pathways.

	4. `TSP-MACO.py`: **TSP with MACO** - Efficient solution for the classic Traveling Salesman Problem using a specialized version of the MACO algorithm that outperforms traditional approaches on large-scale problems.

	5. `ultimate_pf_simulator-v2-n1.py`: **Protein Folding with MACO** - Advanced implementation for protein structure prediction combining MACO, ZVSS, and GPU acceleration to achieve results up to 3.8x faster than conventional methods.

	6. `ultimate_zvss-v4-n1.py`: **MACO-ZVSS Integrated** - Comprehensive implementation of MACO with ZVSS optimization for zombie swarm simulation. Features GPU-accelerated parameter optimization and interactive Pygame visualization.

	7. `UmacoFORCTF-v3-no1.py`: **MACO Cryptanalysis for SPEEDY-7-192** - Specialized MACO implementation for cryptographic challenges featuring entropy-based parameter adaptation and quantum burst mechanisms for escaping local search traps.

![pheromone_evolution](https://github.com/user-attachments/assets/81bf1f54-70a5-4f1a-998a-0fffee57567a)
*Dynamic Evolution of UMACO's Pheromone Tensor:* This animation visualizes the real (Attraction) and imaginary (Repulsion) components of the pheromone field over optimization iterations. You can observe how pathways of attraction emerge and adapt as the optimization progresses, guiding the multi-agent search process. Bright regions indicate areas of high pheromone concentration where agents converge on promising solutions.

## 🌟 Key Features

### UMACO9 Core Framework
- **Panic-Anxiety-Quantum Triad System (PAQ Core)**: Biomimetic crisis response system that dynamically adapts to optimization challenges. When agents encounter difficult terrain, the PAQ Core triggers panic responses that increase exploration, while anxiety states enhance focus on promising regions.

- **Topological Stigmergic Field (TSF)**: Sophisticated pheromone-based communication network using complex sheaf sections. Unlike simple pheromone trails in ant colony optimization, TSF encodes both attraction and repulsion information in a complex tensor that evolves throughout the optimization process.

- **Crisis-Driven Hyperparameters**: Intelligent parameter tuning based on system state and optimization progress. Parameters automatically adjust when the system detects stagnation, with specific crisis response patterns for different optimization challenges.

- **Universal Economy**: Market-based system for computational resource allocation and agent coordination. High-performing agents receive more resources, while underperforming agents shift strategies to explore new approaches.

### MACO-LLM Extensions
- **Enhanced Quantum Economy**: Advanced trading mechanisms with specialized agent roles (Explorers, Exploiters, Evaluators) and dynamic market conditions. The economy includes token inflation/deflation mechanics based on global performance metrics.

- **NeuroPheromone System**: Neural-network-inspired pathway formation with behavior patterns modeled after neurochemicals. Pheromones evolve through reinforcement mechanisms similar to dopamine rewards in biological systems.

- **Loss-Aware Performance Metrics**: Specialized tracking metrics tailored to different aspects of LLM training, with dedicated agents focusing on learning rate, regularization, and architecture optimization.

- **Visualization Tools**: Real-time monitoring of economic activity, agent performance, and optimization progress through interactive dashboards and detailed tensor field visualizations.

## 📋 Requirements

```
numpy
cupy
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
wandb
matplotlib
ripser
persim
```

Additional dependencies may be required depending on specific usage. For example, GPU-accelerated examples require CUDA toolkit installation, and visualization components may need additional Python packages.

## 🚀 Getting Started

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/umaco.git
cd umaco
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional: Install the package in development mode:
```bash
pip install -e .
```

### Usage Examples

<details>
<summary><code>Using UMACO9 for Optimization</code></summary>

```python
from umaco.Umaco9 import UMACO9, UMACO9Config, UniversalEconomy, EconomyConfig, NeuroPheromoneSystem, PheromoneConfig, UniversalNode, NodeConfig

# Configure the economy with 8 agents and 250 tokens per agent
economy_config = EconomyConfig(n_agents=8, initial_tokens=250)
economy = UniversalEconomy(economy_config)

# Configure the pheromone system for a 64-dimensional space
pheromone_config = PheromoneConfig(n_dim=64, initial_val=0.3)
pheromones = NeuroPheromoneSystem(pheromone_config)

# Create the UMACO9 solver with optimized hyperparameters
config = UMACO9Config(
    n_dim=64,
    panic_seed=np.random.random((64, 64)),
    trauma_factor=0.5,
    alpha=0.1, 
    beta=0.2,
    rho=0.3,
    max_iter=1000
)

solver = UMACO9(config, economy, pheromones)

# Define your optimization function
def loss_function(x):
    # Your objective function here
    return np.sum(x**2)

# Create agents
agents = [UniversalNode(i, economy, NodeConfig()) for i in range(8)]

# Run optimization
pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
    agents, loss_function
)
```
</details>

<details>
<summary><code>Training an LLM with MACO-LLM</code></summary>

```python
from umaco.maco_direct_train16 import MACAOConfig

# Configure the training process for a Phi-2 model
config = MACAOConfig(
    model_name="microsoft/phi-2",
    output_dir="./macao_output",
    training_data_file="your_training_data.jsonl",
    n_agents=8,
    num_epochs=3,
    batch_size=1
)

# Run the training script (see examples/llm_training.py for a complete example)
```
</details>

## 📊 Visualizations

MACO-LLM includes real-time visualization of the agent economy, token distribution, and performance metrics using matplotlib and Weights & Biases integration. These visualizations help you understand:

1. How agents are performing relative to each other
2. How tokens are distributed throughout the economy
3. Which agent strategies are most effective
4. How the pheromone field evolves during optimization

You can monitor these visualizations in real-time during training or save them for later analysis.

## 🧠 Understanding the Approach

UMACO is designed to be highly adaptable to different domains. The repository demonstrates this by showing:

1. **UMACO9**: The core framework applied to general optimization problems, from simple benchmarks to complex multi-dimensional challenges.

2. **MACO-LLM**: An adaptation specifically for LLM training, showcasing how the core principles can be tailored to specialized AI domains.

This demonstrates how the fundamental concepts can be tailored to specific applications while maintaining the core principles:

1. **Multi-agent cognitive architecture**: Distributed problem-solving with specialized agent roles and emergent collective intelligence, similar to how insect colonies solve complex problems.

2. **Economic resource management**: Market-based allocation of computational resources that rewards successful strategies and penalizes ineffective approaches, creating a self-organizing system.

3. **PAQ Core for crisis detection and response**: Dynamic adaptation to challenging optimization landscapes through coordinated panic, anxiety, and quantum burst responses.

4. **Topological analysis for landscape understanding**: Advanced mathematical techniques that analyze the "shape" of optimization landscapes to guide the search process more effectively.

See the documentation in the [`docs/`](https://github.com/Eden-Eldith/UMACO/tree/master/docs) directory for detailed explanations of these concepts and adaptation strategies.

- 📄  [Full UMACO Architecture](https://garden-backend-three.vercel.app/fixed-thesis-maco/)


## 📝 Citation

If you use this code in your research, please cite:

```
@software{umaco2025,
  author = {Eden Eldith},
  title = {UMACO: Universal Multi-Agent Cognitive Optimization},
  year = {2025},
  url = {https://github.com/Eden-Eldith/UMACO}
}
```

## 📄 License

This project is licensed under the Recursive Cognitive License - see the LICENSE file for details.
---
> :warning: **Entanglement Notice:**  
> Any use, inspiration, or derivative work from this repo, in whole or in part, is subject to the Recursive Entanglement Doctrine (RED.md) and Recursive Cognitive License (RCL.md).  
> Non-attribution, commercial exploitation, or removal of epistemic provenance is a violation of licensing and will be treated accordingly.
