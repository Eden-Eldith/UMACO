"""
UMACO: Universal Multi-Agent Cognitive Optimization
==================================================

A framework that combines principles from quantum physics, multi-agent systems,
topological analysis, and economic mechanisms to solve complex optimization problems.

Main Implementations
-------------------
* Umaco9.py - The UMACO9 optimization framework for general-purpose optimization
* maco_direct_train16.py - MACO implementation for LLM training

The repository contains both original implementations to demonstrate how the
core concepts can be adapted to different domains.
"""

__version__ = "0.1.0"

# Make imports available at package level if needed
try:
    from .Umaco9 import (
        UMACO9, UMACO9Config,
        UniversalEconomy, EconomyConfig,
        UniversalNode, NodeConfig,
        NeuroPheromoneSystem, PheromoneConfig
    )
    HAS_CORE = True
except ImportError:
    HAS_CORE = False

# Make LLM components optional to avoid requiring those dependencies
try:
    from .maco_direct_train16 import MACAOConfig
    HAS_LLM_SUPPORT = True
except ImportError:
    HAS_LLM_SUPPORT = False
