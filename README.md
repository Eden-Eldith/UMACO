# UMACO: Universal Multi-Agent Cognitive Optimization

<div align="center">

![UMACO Logo](https://github.com/user-attachments/assets/11a19c53-e374-497b-8903-30a9c20ddf91)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-10.0-blue)](https://github.com/EdenEldith/umaco)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Created By](https://img.shields.io/badge/created%20by-Eden%20Eldith-purple)](https://github.com/EdenEldith)
![AI-First Framework](https://img.shields.io/badge/AI--First-Framework-blueviolet)

</div>

## What is UMACO?

UMACO (Universal Multi-Agent Cognitive Optimization) is an AI-first framework specifically designed to be refactored and adapted by Large Language Models (LLMs) to solve domain-specific optimization problems. Unlike traditional libraries that you install and use directly, UMACO serves as a template that LLMs can understand and modify to address your unique optimization challenges.

This project represents not just a technical achievement, but a personal one: I created UMACO without formal computer science education, teaching myself programming from scratch in just over 3 months with the assistance of AI tools. My hope is that this framework demonstrates that meaningful contributions to complex fields can come from non-traditional backgrounds.

### Key Innovation

The core innovation of UMACO lies in its **AI-collaborative design** — it's built from the ground up to be understood, refactored, and specialized by AI models. By combining four interconnected systems in a way that's comprehensible to LLMs, it enables rapid creation of tailored optimization solutions through natural language problem descriptions.

## Table of Contents

- [UMACO: Universal Multi-Agent Cognitive Optimization](#umaco-universal-multi-agent-cognitive-optimization)
  - [What is UMACO?](#what-is-umaco)
  - [Table of Contents](#table-of-contents)
  - [How UMACO Works](#how-umaco-works)
    - [The AI-Collaborative Workflow](#the-ai-collaborative-workflow)
    - [Example Use Cases](#example-use-cases)
  - [Core Architecture](#core-architecture)
    - [PAQ Core (Panic-Anxiety-Quantum)](#paq-core-panic-anxiety-quantum)
    - [Topological Stigmergic Field](#topological-stigmergic-field)
    - [Universal Economy](#universal-economy)
    - [Crisis-Driven Hyperparameters](#crisis-driven-hyperparameters)
  - [System Implementations](#system-implementations)
  - [Problem-Solving Domains](#problem-solving-domains)
  - [Repository Structure](#repository-structure)
    - [Scripts and Documents](#scripts-and-documents)
  - [Key Features](#key-features)
    - [UMACO9 Core Framework](#umaco9-core-framework)
    - [MACO-LLM Extensions](#maco-llm-extensions)
  - [Requirements](#requirements)
  - [Using UMACO with AI](#using-umaco-with-ai)
    - [Crafting Effective Prompts](#crafting-effective-prompts)
    - [Example Prompts](#example-prompts)
    - [Usage Examples](#usage-examples)
  - [Visualizations](#visualizations)
  - [Understanding the Approach](#understanding-the-approach)
  - [Potential Applications](#potential-applications)
  - [Technical Overview](#technical-overview)
    - [System Architecture](#system-architecture)
    - [Why This Architecture Works with LLMs](#why-this-architecture-works-with-llms)
  - [Why Sponsor UMACO?](#why-sponsor-umaco)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## How UMACO Works

### The AI-Collaborative Workflow

UMACO introduces a new paradigm in optimization framework design:

```
┌─────────────────┐    ┌────────────────────┐    ┌───────────────────┐
│ UMACO Framework │    │ Problem Statement  │    │                   │
│ Base Code       │ + │ in Natural Language │ → │ LLM Processing    │ →
│                 │    │                    │    │                   │
└─────────────────┘    └────────────────────┘    └───────────────────┘

┌────────────────────────┐    ┌───────────────────────┐
│ Custom UMACO Variant   │    │                       │
│ Specialized for the    │ → │ Problem Solution      │
│ Specific Problem       │    │                       │
└────────────────────────┘    └───────────────────────┘
```

Rather than being a library you install, UMACO serves as a sophisticated template that LLMs can understand and customize. The process typically involves:

1. **Problem Description**: You describe your optimization problem to an LLM in natural language
2. **Framework Presentation**: You provide the UMACO code to the LLM
3. **AI Refactoring**: The LLM refactors UMACO to create a specialized variant for your problem
4. **Implementation**: You run the customized code to solve your specific problem

This approach dramatically reduces the time from problem conceptualization to solution implementation.

### Example Use Cases

1. **Financial Portfolio Optimization**
   - Present UMACO + portfolio constraints to LLM
   - Get back a customized optimizer with risk-adaptive agents

2. **Neural Network Hyperparameter Tuning**
   - Describe network architecture + constraints
   - Receive a UMACO variant specialized for hyperparameter search

3. **Supply Chain Optimization**
   - Provide logistics constraints and objectives
   - Obtain a modified UMACO with specialized pheromone interpretations

## Core Architecture

UMACO's architecture is designed to be both powerful and comprehensible to AI, allowing for effective refactoring across diverse domains.

### PAQ Core (Panic-Anxiety-Quantum)

The heart of UMACO is the PAQ Core, which gives the system its adaptive intelligence:

- **Panic Tensor**: A 2D field that tracks local "crisis states," identifying regions where optimization is struggling
- **Anxiety Wavefunction**: A complex field that maps existential risk gradients, with real parts representing immediate concerns and imaginary parts representing potential future challenges
- **Quantum Burst**: A mechanism for escaping local minima that combines structured perturbations (via SVD) with controlled randomness, modulated by anxiety

### Topological Stigmergic Field

The TSF implements indirect coordination through environment modification, inspired by how ants communicate via pheromone trails:

- **Complex Pheromones**: Represented as complex numbers where:
  - Real part = attractive forces (exploitation)
  - Imaginary part = repulsive forces (exploration)
- **Persistent Homology**: Uses topological data analysis to understand the "shape" of the optimization landscape
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

All key hyperparameters in UMACO dynamically respond to the system state:

- **Alpha Parameter**: Controls the influence of the covariant momentum, scaling with panic and anxiety
- **Beta Parameter**: Modulates the balance between attraction and repulsion based on the persistent entropy
- **Rho Parameter**: Controls the rate of change, scaling inversely with momentum magnitude

## System Implementations

This repository contains two main implementations:

1. **UMACO9**: A foundational framework for general-purpose optimization problems using the PAQ Core (Panic-Anxiety-Quantum Triad System) to dynamically respond to challenging optimization landscapes
2. **MACO-LLM**: A specialized implementation for training and fine-tuning Large Language Models using multi-agent systems with an economic marketplace to efficiently allocate computational resources

## Problem-Solving Domains

UMACO solves high-complexity problems across domains like optimization, AI, and cryptography:

* **Optimization**: Tackles challenging non-convex optimization problems with numerous local minima that traditional methods struggle with. UMACO's multi-agent approach efficiently navigates complex landscapes that would trap conventional gradient-based methods.

* **Artificial Intelligence**: Implements sophisticated multi-agent systems that collaborate through stigmergic communication (environment-mediated information sharing) to make complex decisions in uncertain environments. This approach mimics how social insects solve collective problems.

* **Complex Systems**: Provides robust analysis of dynamic optimization landscapes through topological methods. The framework adapts resource allocation in changing environments by understanding the "shape" of the solution space.

* **Satisfiability (SAT) Solving**: Accelerates Large Language Model training by reformulating optimization challenges as SAT problems. UMACO adapts traditional optimization frameworks to AI-specific challenges through its crisis-driven hyperparameter tuning.

* **Software Development**: Enhances package dependency resolution and offers seamless integration with external ML libraries including PyTorch, Transformers, and PEFT. The framework resolves complex dependency graphs using the same principles it applies to other optimization problems.

* **Cryptography**: Enables advanced cryptanalysis of modern ciphers like SPEEDY-7-192 through multi-agent search techniques with dynamic pheromone-guided exploration. The quantum burst mechanism helps escape local optima when searching for cryptographic weaknesses.

## Repository Structure

### Scripts and Documents

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

## Key Features

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

## Requirements

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

## Using UMACO with AI

### Crafting Effective Prompts

To get the best results when using UMACO with LLMs, follow these guidelines:

1. **Be Specific About Your Problem**:
   - Clearly define the variables to be optimized
   - Specify constraints and objectives
   - Provide context about domain-specific considerations

2. **Request Specific Adaptations**:
   - Ask for customized loss functions
   - Request domain-appropriate agent specializations
   - Specify what outputs you need from the optimization

3. **Provide Sample Data or Scenarios**:
   - Include example data formats
   - Mention typical ranges for variables
   - Describe typical use scenarios

### Example Prompts

#### Hyperparameter Optimization Prompt

```
I have the UMACO framework code below, and I need to optimize hyperparameters for a neural network that predicts stock prices. Please refactor UMACO to create a specialized variant for neural network hyperparameter optimization.

[UMACO CODE HERE]

The neural network has these hyperparameters to optimize:
- Learning rate (range: 0.0001 to 0.1)
- Batch size (range: 16 to 512)
- Hidden layer sizes (1-3 layers, 32-512 neurons each)
- Dropout rate (range: 0 to 0.5)

The dataset consists of 5 years of daily stock data with features including price, volume, and various technical indicators. The evaluation metric is mean squared error on a validation set, and each training run takes approximately 10 minutes.

Please customize the UMACO agents to specialize in different aspects of neural network optimization, and adapt the pheromone interpretation to represent hyperparameter configurations.
```

#### Logistics Optimization Prompt

```
Here's the UMACO framework. I need to optimize a delivery routing system for a fleet of 20 vehicles serving 150 customers across a city. Please refactor UMACO to create a specialized variant for this vehicle routing problem.

[UMACO CODE HERE]

Key constraints include:
- Vehicle capacity limits
- Time windows for deliveries
- Driver work hour restrictions
- Fuel efficiency considerations

The objective is to minimize total distance traveled while satisfying all delivery requirements. Please adapt the pheromone system to represent routes between locations, and create specialized agents for different aspects of route planning.
```



# Usage Examples

## Using UMACO9 for Optimization

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

### Training an LLM with MACO-LLM

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

### Visualizations

MACO-LLM includes real-time visualization of the agent economy, token distribution, and performance metrics using matplotlib and Weights & Biases integration. These visualizations help you understand:

1. How agents are performing relative to each other
2. How tokens are distributed throughout the economy
3. Which agent strategies are most effective
4. How the pheromone field evolves during optimization

You can monitor these visualizations in real-time during training or save them for later analysis:

![pheromone_evolution](https://github.com/user-attachments/assets/81bf1f54-70a5-4f1a-998a-0fffee57567a)
*Dynamic Evolution of UMACO's Pheromone Tensor:* This animation visualizes the real (Attraction) and imaginary (Repulsion) components of the pheromone field over optimization iterations. You can observe how pathways of attraction emerge and adapt as the optimization progresses, guiding the multi-agent search process. Bright regions indicate areas of high pheromone concentration where agents converge on promising solutions.

### Understanding the Approach

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

### Potential Applications

UMACO can be adapted for numerous applications through AI refactoring:

- **Machine Learning**: Hyperparameter optimization, neural architecture search
- **Financial Modeling**: Portfolio optimization, risk management
- **Drug Discovery**: Molecular optimization, protein folding
- **Engineering Design**: Multi-objective optimization, constraint satisfaction
- **Logistics**: Complex scheduling, route optimization
- **Game Development**: Procedural generation, AI behavior tuning
- **AI Research**: Meta-learning, transfer learning, reinforcement learning
- **Energy Management**: Grid optimization, resource allocation

### Technical Overview

### System Architecture

UMACO is implemented as a modular Python framework with the following key components:

```
UMACO10
   |
   |--- PAQ Core
   |     |- panic_tensor
   |     |- anxiety_wavefunction
   |     |- quantum_burst()
   |     \- panic_backpropagate()
   |
   |--- Topological Field
   |     |- pheromones
   |     |- covariant_momentum
   |     \- persistent_homology_update()
   |
   |--- Universal Economy
   |     |- tokens
   |     |- market_value
   |     |- buy_resources()
   |     \- reward_performance()
   |
   |--- Cognitive Agents
   |     |- UniversalNode
   |     \- propose_action()
   |
   \--- Crisis-Driven Hyperparameters
         |- alpha
         |- beta
         \- rho
```

| Component | Description |
|-----------|-------------|
| **UMACO9** | Core agent framework for non-convex optimization with crisis detection |
| **MACO-LLM** | Specialized LLM trainer with economic adaptation and trading mechanisms |
| **PAQ Core** | Biomimetic crisis-handling module (Panic-Anxiety-Quantum Triad) |
| **TSF** | Advanced pheromone field using complex tensors (real=attraction, imaginary=repulsion) |
| **Universal Economy** | Market-based computational resource allocation system |
| **ZVSS** | Zombie Virus Swarm Simulator: A idea I had to use pheremones as guides for a swarm as well as hyperparamter tuning|
| **NeuroPheromone** | Neural pathway formation inspired by biological neurochemicals |

### Why This Architecture Works with LLMs

UMACO's design is particularly well-suited for AI refactoring because:

1. **Clear Component Separation**: Each system is well-isolated with defined interfaces
2. **Explicit Conceptual Mapping**: Components map to intuitive concepts (panic, economy, agents)
3. **Self-Documenting Code**: Extensive docstrings explain purpose and functionality
4. **Flexible Extension Points**: Easy-to-identify places for domain-specific customization
5. **Standardized Interfaces**: Consistent patterns for interaction between components

These characteristics make it easy for LLMs to understand the codebase and refactor it appropriately for different domains.

## Why Sponsor UMACO?

UMACO offers several compelling advantages for potential investors and sponsors:

1. **Novel Approach**: UMACO represents a fundamentally new approach to optimization, combining multi-agent systems with AI-collaborative design

2. **AI-First Design**: Unlike traditional frameworks, UMACO is specifically built to leverage the capabilities of modern LLMs

3. **Universal Adaptation**: The architecture enables rapid customization for diverse domains without requiring domain expertise in optimization algorithms

4. **Reduced Time-to-Solution**: By enabling natural language problem specification, UMACO dramatically accelerates the optimization solution development cycle

5. **Emerging Field**: UMACO sits at the intersection of several rapidly growing fields (LLM-based programming, multi-agent AI, topological data analysis)

6. **Commercial Applications**: The technology has direct applications in numerous high-value industries (finance, pharmaceuticals, logistics, etc.)

7. **Inspiring Story**: UMACO was created by a self-taught developer who learned coding from scratch in just over 3 months, demonstrating the framework's accessibility and potential for innovation from unexpected sources

Your sponsorship would help accelerate development in the following ways:
- Expand research into new domains and applications
- Create specialized UMACO variants for high-value industrial problems
- Develop improved documentation and example libraries
- Build a community of practitioners using the AI-collaborative approach
- Create interfaces for non-programmers to leverage UMACO's capabilities
- Support neurodivergent developers in continuing this innovative work

## Contributing

UMACO was created by Eden Eldith, a self-taught developer who built this entire framework without formal computer science education. The project welcomes contributions from people of all backgrounds, especially those who might not fit the traditional developer profile. Here's how you can help:

1. **Example Prompts**: Create and share effective prompts for different domains
2. **Domain Adaptations**: Share successful UMACO variants for different problems
3. **Documentation**: Improve guides on how to effectively describe problems to LLMs
4. **Testing**: Test the framework on different types of problems and LLMs
5. **Community**: Help answer questions, mentor new contributors
6. **Accessibility**: Help make the project more accessible to neurodivergent developers and beginners

If you're interested in contributing, please see [The UMACO discussions page](https://github.com/Eden-Eldith/UMACO/discussions) for detailed guidelines. The project aims to be inclusive and supportive, especially for first-time contributors and those from non-traditional backgrounds.

## Scientific Background

UMACO draws inspiration from several scientific fields:

- **Stigmergy**: Self-organization through environment modification
- **Topological Data Analysis**: Using topological features to understand data
- **Quantum Computing**: Concepts of superposition and entanglement
- **Complex Systems**: Emergence and self-organization in multi-agent systems
- **Neuroeconomics**: Decision-making under uncertainty and resource constraints
- **Prompt Engineering**: Techniques for effective communication with LLMs


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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> :warning: **Entanglement Notice:**  
> Any use, inspiration, or derivative work from this repo, in whole or in part, is subject to the Recursive Entanglement Doctrine (RED.md) and Recursive Cognitive License (RCL.md).  
> Non-attribution, commercial exploitation, or removal of epistemic provenance is a violation of licensing and will be treated accordingly.

## Acknowledgements

UMACO was developed entirely by Eden Eldith, who taught themselves programming from scratch in just over three months while facing significant personal challenges including OCD, ADHD, autism, anxiety, and chronic pain. This project demonstrates how innovation can come from unexpected places and how AI assistance can help bridge accessibility gaps in technology development.

Special thanks to the developers of the open-source libraries that made this project possible, and to the AI tools that helped make this development journey accessible to a self-taught programmer with no formal qualifications.

---

<div align="center">
<p>UMACO: AI-Collaborative Optimization Framework</p>
<p>Created by Eden Eldith • 2025</p>
</div>
