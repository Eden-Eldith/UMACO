# UMACO: Universal Multi-Agent Cognitive Optimization
![pheromone_evolution](https://github.com/user-attachments/assets/81bf1f54-70a5-4f1a-998a-0fffee57567a)
*Dynamic Evolution of UMACO's Pheromone Tensor:* This animation visualizes the real (Attraction) and imaginary (Repulsion) components of the pheromone field over optimization iterations.  Observe how pathways of attraction emerge and adapt, guiding the multi-agent search process.

UMACO is a novel optimization framework that combines principles from quantum physics, multi-agent systems, topological analysis, and economic mechanisms to solve complex optimization problems.

This repository contains two main implementations:

1. **UMACO9**: A foundational framework for general-purpose optimization problems using the PAQ Core (Panic-Anxiety-Quantum Triad System)
2. **MACO-LLM**: An implementation specialized for training and fine-tuning Large Language Models using multi-agent systems with an economic marketplace

## 🌟 Key Features

### UMACO9 Core Framework
- **Panic-Anxiety-Quantum Triad System (PAQ Core)**: Dynamic crisis response system that adapts to optimization challenges
- **Topological Stigmergic Field (TSF)**: Pheromone-based communication between agents using complex sheaf sections
- **Crisis-Driven Hyperparameters**: Adaptive parameter tuning based on system state
- **Universal Economy**: Token-based system for resource allocation and agent coordination

### MACO-LLM Extensions
- **Enhanced Quantum Economy**: Trading mechanisms, specialized agent roles, and market dynamics
- **NeuroPheromone System**: Adaptive pathway formation with neurochemical-inspired behavior
- **Loss-Aware Performance Metrics**: Specialized performance tracking based on agent focus areas
- **Visualization Tools**: Real-time economic monitoring and agent performance tracking

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

Additional dependencies may be required depending on specific usage.

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

### Using UMACO9 for Optimization

```python
from umaco.Umaco9 import UMACO9, UMACO9Config, UniversalEconomy, EconomyConfig, NeuroPheromoneSystem, PheromoneConfig, UniversalNode, NodeConfig

# Configure the economy
economy_config = EconomyConfig(n_agents=8, initial_tokens=250)
economy = UniversalEconomy(economy_config)

# Configure the pheromone system
pheromone_config = PheromoneConfig(n_dim=64, initial_val=0.3)
pheromones = NeuroPheromoneSystem(pheromone_config)

# Create the UMACO9 solver
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

# Configure the training
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

## 📊 Visualizations

MACO-LLM includes real-time visualization of the agent economy, token distribution, and performance metrics using matplotlib and Weights & Biases integration.

## 🧠 Understanding the Approach

UMACO is designed to be highly adaptable to different domains. The repository demonstrates this by showing:

1. **UMACO9**: The core framework applied to general optimization problems
2. **MACO-LLM**: An adaptation specifically for LLM training

This demonstrates how the fundamental concepts can be tailored to specific applications while maintaining the core principles:

1. Multi-agent cognitive architecture
2. Economic resource management
3. PAQ Core for crisis detection and response
4. Topological analysis for landscape understanding

See the documentation in the `docs/` directory for detailed explanations of these concepts and adaptation strategies.
📄  !([Full UMACO Architecture](https://garden-backend-three.vercel.app/fixed-thesis-maco/)


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
