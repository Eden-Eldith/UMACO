#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACO-LLM with Enhanced Quantum Economy - Trading & Visualization - FULL WORKING SCRIPT
=====================================================================================

Incorporates the requested fixes for:
1) A more robust performance metric using logarithmic loss and better reward scaling
2) A threshold of 3% for meaningful LR changes
3) Specialized performance metrics for different agent focus areas (learning_rate, regularization, etc.)
4) Gradient norm and perplexity tracking for better LLM metrics
5) Loss-aware damping to avoid wild oscillations as loss improves

It also retains the previous fixes for:
 - buy_resources to respect the min_token_balance
 - reward_performance only giving full reward if the agent contributed to training
 - update_market_dynamics ensuring adequate token supply
 - plus the main training loop changes to pass `influenced_training` to reward_performance
"""

import os
import logging
import numpy as np
import torch
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import wandb
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
import json

# =========================================================================================
#   Configuration
# =========================================================================================

@dataclass
class MACAOConfig:
    """Configuration for MACAO (Multi-Agent Cognitive Architecture with Organic pathways)"""
    model_name: str = "microsoft/phi-2"
    output_dir: str = "./macao_output"
    training_data_file: str = "training_data.jsonl"

    n_agents: int = 8
    initial_tokens: int = 250
    token_reward_factor: float = 3
    min_token_balance: int = 24  # <--- Agents must stay above this token balance

    market_volatility: float = 0.15
    inflation_rate: float = 0.005
    enable_trading: bool = True
    trading_fee: float = 0.01
    trade_offer_lifetime: int = 120

    initial_pheromone: float = 0.2
    evaporation_rate: float = 0.14
    pheromone_intensity: float = 3.5

    myrcene_factor: float = 0.7
    limonene_factor: float = 1.2
    pinene_factor: float = 0.9
    linalool_factor: float = 0.8

    noise_std: float = 0.11
    target_entropy: float = 0.68
    partial_reset_threshold: int = 40
    quantum_burst_interval: int = 200

    batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    num_epochs: int = 3
    max_seq_length: int = 256

    lr_min: float = 1e-6
    lr_max: float = 1e-3
    lora_dropout_min: float = 0.0
    lora_dropout_max: float = 0.2

    enable_visualization: bool = True
    visualization_interval: int = 50
    export_economy_data: bool = True

    wandb_project: str = "macao_llm"
    wandb_run_name: str = "quantum_economy_trading"
    log_interval: int = 10

# =========================================================================================
#   Enhanced Performance Metrics and Reward System
# =========================================================================================

class EnhancedQuantumEconomy:
    """
    Enhanced Quantum Economy with dynamic trading mechanisms and market forces.
    Manages computational resources where cognitive nodes purchase and trade
    resources (GPU memory, compute cycles) using tokens.

    Incorporates:
      - Logarithmic performance metric
      - Hyperbolic tangent scaling for rewards
      - Specialized performance weighting for different agent focus areas
      - Memory of previous loss (improvement factor) and gradient norms
    """
    def __init__(self, config: MACAOConfig):
        self.config = config
        self.tokens = {i: config.initial_tokens for i in range(config.n_agents)}
        self.performance_history = {i: [] for i in range(config.n_agents)}

        self.market_value = 1.0
        self.inflation_rate = config.inflation_rate
        self.market_volatility = config.market_volatility
        self.resource_scarcity = 0.5

        self.outstanding_offers = []
        self.completed_trades = []
        self.agent_specializations = {}

        # Cumulative trade counter
        self.total_trades = 0

        self.history = {
            'token_distribution': [],
            'market_value': [],
            'resource_scarcity': [],
            'trade_volume': [],
            'agent_performance': [],
            'timestamp': []
        }

        # Enhanced tracking for improved performance metric
        self.initial_loss = None
        self.loss_ewma = None
        self.ewma_alpha = 0.1
        self.perplexity_history = []
        self.specialization_metrics = {
            'learning_rate': [],
            'regularization': [],
            'architecture': [],
            'data_focus': []
        }

        self.resource_state = self._get_resource_state()
        self._initialize_specializations()
        self._record_current_state(0)

    def _initialize_specializations(self):
        specialization_types = [
            "computation", "memory", "optimization",
            "exploration", "exploitation", "communication"
        ]
        for i in range(self.config.n_agents):
            primary = np.random.choice(specialization_types)
            remaining = [s for s in specialization_types if s != primary]
            secondary = np.random.choice(remaining)
            self.agent_specializations[i] = {
                "primary": primary,
                "secondary": secondary,
                "efficiency": np.random.uniform(0.8, 1.2),
                "production": np.random.uniform(0.9, 1.3)
            }

    def _get_resource_state(self) -> Dict[str, float]:
        try:
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_usage = gpu_memory_allocated / gpu_memory_total
                try:
                    # Not all CUDA versions have utilization() method
                    gpu_utilization = torch.cuda.utilization() / 100.0
                except:
                    gpu_utilization = 0.5
            else:
                gpu_memory_usage = 0.5
                gpu_utilization = 0.5
            cpu_usage = os.getloadavg()[0] / os.cpu_count() if hasattr(os, 'getloadavg') else 0.5
            resource_state = {
                "gpu_memory": gpu_memory_usage,
                "gpu_util": gpu_utilization,
                "cpu_util": cpu_usage,
                "time_slice": 1.0 - (0.7 * gpu_memory_usage + 0.3 * cpu_usage)
            }
            self.resource_scarcity = 0.7 * gpu_memory_usage + 0.3 * (1 - self.market_value)
            return resource_state
        except Exception as e:
            logging.warning(f"Error getting resource state: {e}")
            return {
                "gpu_memory": 0.5,
                "gpu_util": 0.5,
                "cpu_util": 0.5,
                "time_slice": 0.5
            }

    # ----------------------------------------------------------
    #  Calculate Performance: Logarithmic Loss + Specialization
    # ----------------------------------------------------------
    def calculate_performance(self, loss_val, agent_type=None, gradient_norm=None, 
                              previous_loss=None, agent_id=None):
        """
        Enhanced performance calculation using negative log loss with normalization,
        plus perplexity and agent specializations.
        """
        # Initialize initial loss if not set
        if self.initial_loss is None:
            self.initial_loss = loss_val
            self.loss_ewma = loss_val
        
        if self.loss_ewma is None:
            self.loss_ewma = loss_val

        # Update exponential weighted moving average of loss
        self.loss_ewma = self.ewma_alpha * loss_val + (1 - self.ewma_alpha) * self.loss_ewma
        
        # Calculate perplexity
        perplexity = np.exp(loss_val)
        self.perplexity_history.append(perplexity)
        
        # Base performance: negative log loss (bounded)
        raw_perf = -np.log(loss_val + 1e-8)
        
        # Normalize by initial performance to keep values in a reasonable range
        initial_perf = -np.log(self.initial_loss + 1e-8)
        normalized_perf = raw_perf / (initial_perf if initial_perf != 0 else 1e-8)
        
        # Apply bounds to prevent extreme values
        bounded_perf = min(3.0, max(0.1, normalized_perf))
        
        # Calculate improvement factor if previous_loss is available
        improvement = 0
        if previous_loss is not None:
            # Positive when loss decreases, negative when loss increases
            improvement = (previous_loss - loss_val) / previous_loss
            # Apply a sigmoid-like function to bound improvement factor
            improvement = np.tanh(improvement * 5)  # Range approx [-1, 1]
        
        # Agent specialization weighting
        if agent_type == 'learning_rate':
            # Learning rate agents get rewarded more when actively decreasing loss
            specialist_factor = 0.7 + 0.3 * max(0, improvement)
            # If things get worse, penalize
            if improvement < -0.05:
                specialist_factor *= 0.5
        elif agent_type == 'regularization':
            # Regularization is more important early or when loss plateaus
            training_progress = len(self.perplexity_history) / 1000.0
            plateau_factor = 1.0
            if previous_loss is not None:
                # Detect plateaus: small absolute changes
                plateau_factor = np.exp(-10 * abs(improvement))
            specialist_factor = 0.6 + 0.4 * max(0, np.exp(-3 * training_progress) + plateau_factor)
        elif agent_type == 'architecture':
            # Architecture is crucial when gradient norm is high
            if gradient_norm is not None:
                norm_factor = min(1.0, gradient_norm / 10.0)  # Normalize to [0,1]
                specialist_factor = 0.7 + 0.3 * norm_factor
            else:
                specialist_factor = 0.8
        elif agent_type == 'data_focus':
            # Data focus is more important when loss is relatively high
            relative_loss = loss_val / self.initial_loss
            specialist_factor = 0.6 + 0.4 * min(1.0, relative_loss * 2)
        else:
            specialist_factor = 1.0
        
        final_perf = bounded_perf * specialist_factor
        
        # Track specialization-specific performance metrics
        if agent_type and agent_type in self.specialization_metrics:
            self.specialization_metrics[agent_type].append(final_perf)
        
        # Log internal metrics
        wandb.log({
            "metrics/raw_performance": raw_perf,
            "metrics/normalized_perf": normalized_perf,
            "metrics/bounded_perf": bounded_perf,
            "metrics/final_perf": final_perf,
            "metrics/specialist_factor": specialist_factor,
            "metrics/perplexity": perplexity,
            "metrics/loss_ewma": self.loss_ewma
        })
        
        if agent_id is not None:
            wandb.log({f"agent/{agent_id}/performance": final_perf})
            
        return final_perf, improvement

    # ----------------------------------------------------------
    #  Reward System w/ Bounded Scaling & Loss Improvement
    # ----------------------------------------------------------
    def reward_performance(self, node_id: int, loss_val: float, influenced_training: bool = False, 
                           loss_improved: bool = False, previous_loss: float = None,
                           agent_type: str = None, gradient_norm: float = None):
        """
        Enhanced reward system:
         - Bounded scaling with tanh
         - Emphasis on actual loss reduction
         - Market factor scaled down to avoid extreme values
        """
        performance, improvement = self.calculate_performance(
            loss_val, agent_type, gradient_norm, previous_loss, node_id
        )
        
        # If not explicitly provided, check from previous_loss
        if previous_loss is not None and not loss_improved:
            loss_improved = previous_loss > loss_val
        
        # Base reward with hyperbolic tangent
        base_reward = np.tanh(performance) * 50 * self.config.token_reward_factor
        
        # Production multiplier from specialization
        specialization = self.agent_specializations[node_id]
        production_multiplier = specialization["production"]
        
        # Market multiplier is scaled to 0.3 + 0.7*market_value
        market_multiplier = 0.3 + 0.7 * min(1.5, max(0.5, self.market_value))
        
        if influenced_training:
            if loss_improved:
                improvement_factor = 1.5
                logging.info(f"Agent {node_id}: Loss improved, 1.5x bonus.")
            else:
                improvement_factor = 0.7
                logging.info(f"Agent {node_id}: Proposal used, but loss did not improve.")
            
            reward = int(base_reward * production_multiplier * market_multiplier * improvement_factor)
            self.tokens[node_id] += reward
            logging.info(f"Agent {node_id}: Full reward of {reward} tokens (proposal contributed).")
        else:
            # Small participation reward of 10% if not used
            reward = int(base_reward * production_multiplier * market_multiplier * 0.1)
            self.tokens[node_id] += reward
            logging.info(f"Agent {node_id}: Participation reward of {reward} tokens.")
        
        # Ensure min token balance
        if self.tokens[node_id] < self.config.min_token_balance:
            self.tokens[node_id] = self.config.min_token_balance
        
        self.performance_history[node_id].append((performance, reward, improvement))
        
        # Token overflow -> partial offering
        if self.tokens[node_id] > self.config.initial_tokens * 1.6:
            excess = self.tokens[node_id] - self.config.initial_tokens
            self._create_trade_offer(node_id, int(excess * 0.25), "offer_tokens")
        
        return reward, performance

    # =========================================================================================
    #   Market & Resource Management
    # =========================================================================================

    def _record_current_state(self, global_step):
        self.history['token_distribution'].append(self.get_token_distribution())
        self.history['market_value'].append(self.market_value)
        self.history['resource_scarcity'].append(self.resource_scarcity)
        recent_trades = [t for t in self.completed_trades if t['timestamp'] > global_step - 10]
        self.history['trade_volume'].append(len(recent_trades))

        avg_performance = {}
        for node_id, perfs in self.performance_history.items():
            if perfs:
                # average last 5 performance values if available
                last_perf = [p[0] for p in perfs[-5:]]
                avg_performance[node_id] = sum(last_perf) / len(last_perf)
            else:
                avg_performance[node_id] = 0
        self.history['agent_performance'].append(avg_performance)
        self.history['timestamp'].append(global_step)

    def get_token_distribution(self) -> Dict[int, float]:
        total = sum(self.tokens.values()) or 1.0
        return {k: v/total for k, v in self.tokens.items()}

    def update_market_dynamics(self, global_step: int):
        """
        Market dynamics updated to prevent total token starvation,
        with emergency stimulus if total tokens dip too low.
        """
        self.resource_state = self._get_resource_state()
        market_change = np.random.normal(0, self.market_volatility)
        scarcity_effect = 0.1 * (self.resource_scarcity - 0.5)
        trade_volume = len(self.completed_trades[-10:]) if self.completed_trades else 0
        velocity_effect = -0.01 * trade_volume / max(1, self.config.n_agents)
        self.market_value *= (1 + market_change + scarcity_effect + velocity_effect)
        self.market_value = max(0.2, min(5.0, self.market_value))

        token_sum = sum(self.tokens.values())
        avg_token = token_sum / self.config.n_agents
        min_token = min(self.tokens.values())

        # Check if total token supply is below 70% of the initial baseline
        min_total_tokens = self.config.n_agents * self.config.initial_tokens * 0.7
        if token_sum < min_total_tokens:
            tokens_to_add = int(min_total_tokens - token_sum)
            logging.info(f"Token supply critically low ({token_sum}). Adding {tokens_to_add} tokens to economy.")
            poor_agents = [agent_id for agent_id, bal in self.tokens.items() if bal < avg_token]
            if poor_agents:
                tokens_per_agent = tokens_to_add // len(poor_agents)
                for agent_id in poor_agents:
                    self.tokens[agent_id] += tokens_per_agent
                    logging.info(f"Emergency tokens: Agent {agent_id} received {tokens_per_agent} tokens.")
            else:
                # Distribute evenly
                tokens_per_agent = tokens_to_add // self.config.n_agents
                for agent_id in self.tokens:
                    self.tokens[agent_id] += tokens_per_agent
            wandb.log({"economy/emergency_stimulus": tokens_to_add})
        else:
            # Normal inflation
            inflation_adjustment = 1.0 - (
                self.inflation_rate * (token_sum / (self.config.n_agents * self.config.initial_tokens))
            )
            for agent_id in self.tokens:
                self.tokens[agent_id] = max(
                    self.config.min_token_balance,
                    int(self.tokens[agent_id] * inflation_adjustment)
                )

        self._record_current_state(global_step)

        wandb.log({
            "economy/market_value": self.market_value,
            "economy/resource_scarcity": self.resource_scarcity,
            "economy/token_supply": token_sum,
            "economy/min_token": min_token,
            "economy/avg_token": avg_token,
            "economy/trade_volume": trade_volume,
            "economy/total_trades": self.total_trades,
            "economy/trade_rate": trade_volume / 10.0 if trade_volume else 0.0
        })

    def buy_resources(self, node_id: int, required_power: float) -> bool:
        """
        Purchase resources, factoring in agent specialization, market value, and
        ensuring the agent does not fall below minimum token balance.
        """
        self.resource_state = self._get_resource_state()
        specialization = self.agent_specializations[node_id]
        adjusted_power = required_power / specialization["efficiency"]
        available_power = 1.0 - self.resource_state["gpu_memory"]
        base_cost = int(adjusted_power * 350)
        scarcity_multiplier = np.exp(2 * self.resource_scarcity) - 0.5
        market_multiplier = self.market_value
        cost = int(base_cost * scarcity_multiplier * market_multiplier)

        # 20% discount if agent's primary matches heavily used resource
        if ((specialization["primary"] == "memory" and self.resource_state["gpu_memory"] > 0.7) or
            (specialization["primary"] == "computation" and self.resource_state["gpu_util"] > 0.7)):
            cost = int(cost * 0.8)

        if available_power >= adjusted_power and self.tokens[node_id] >= cost:
            # Ensure we don't drop below min_token_balance
            if self.tokens[node_id] - cost >= self.config.min_token_balance:
                self.tokens[node_id] -= cost
                logging.info(f"Agent {node_id}: Bought resources. Cost={cost}, Tokens left={self.tokens[node_id]}")
                return True
            else:
                needed = cost - (self.tokens[node_id] - self.config.min_token_balance)
                logging.info(f"Agent {node_id}: Purchase would go below min_balance. Need {needed} more tokens.")
                self._create_trade_offer(node_id, needed, "need_tokens")
                return False
        else:
            if self.tokens[node_id] < cost and available_power >= adjusted_power:
                needed = cost - self.tokens[node_id]
                logging.info(f"Agent {node_id}: Not enough tokens. Creating 'need_tokens' offer for {needed}.")
                self._create_trade_offer(node_id, needed, "need_tokens")
            else:
                logging.info(f"Agent {node_id}: Cannot buy resources. Cost={cost}, Tokens={self.tokens[node_id]}")
            return False

    def _create_trade_offer(self, agent_id: int, amount: int, offer_type: str):
        """
        Creates a trade offer. The agent can offer to buy or sell tokens.
        """
        current_time = len(self.history['timestamp']) if self.history['timestamp'] else 0
        offer = {
            'agent_id': agent_id,
            'amount': amount,
            'type': offer_type,
            'exchange_rate': 1.0 if offer_type == 'need_tokens' else 1.3,
            'created_at': current_time
        }
        self.outstanding_offers.append(offer)
        logging.info(f"Agent {agent_id} created trade offer: {offer_type} for {amount} tokens.")

    def process_trades(self):
        """
        Match outstanding buy/sell offers to process trades.
        """
        current_time = len(self.history['timestamp']) if self.history['timestamp'] else 0
        self.outstanding_offers = [
            o for o in self.outstanding_offers
            if current_time - o['created_at'] < self.config.trade_offer_lifetime
        ]

        need_tokens = [o for o in self.outstanding_offers if o['type'] == 'need_tokens']
        offer_tokens = [o for o in self.outstanding_offers if o['type'] == 'offer_tokens']
        trades_completed = []

        for need in need_tokens:
            for offer in offer_tokens:
                if need['agent_id'] != offer['agent_id']:
                    trade_amount = min(need['amount'], offer['amount'])
                    if trade_amount > 0:
                        buyer_id = need['agent_id']
                        seller_id = offer['agent_id']
                        # Buyer gains tokens, seller loses tokens
                        self.tokens[buyer_id] += trade_amount
                        self.tokens[seller_id] -= trade_amount

                        trade = {
                            'buyer': buyer_id,
                            'seller': seller_id,
                            'amount': trade_amount,
                            'rate': offer['exchange_rate'],
                            'timestamp': current_time
                        }
                        self.completed_trades.append(trade)
                        trades_completed.append((need, offer, trade_amount))

                        logging.info(
                            f"Trade completed: Agent {seller_id} -> Agent {buyer_id}, amount={trade_amount}"
                        )

                        need['amount'] -= trade_amount
                        offer['amount'] -= trade_amount

        self.total_trades += len(trades_completed)

        # Remove completed offers
        for need, offer, amount in trades_completed:
            if need['amount'] <= 0 and need in self.outstanding_offers:
                self.outstanding_offers.remove(need)
            if offer['amount'] <= 0 and offer in self.outstanding_offers:
                self.outstanding_offers.remove(offer)

    def trigger_quantum_burst(self):
        """
        Emergency token redistribution when resource pressure is high.
        """
        logging.info("Quantum Economy: Triggering quantum burst due to resource constraints.")
        wandb.log({"economy/quantum_burst": True})
        total_tokens = sum(self.tokens.values())
        base_allocation = total_tokens * 0.3 / len(self.tokens)
        performance_allocation = total_tokens * 0.5
        specialization_allocation = total_tokens * 0.2

        # Weighted performance scores
        performance_scores = {}
        for node_id in self.performance_history:
            if self.performance_history[node_id]:
                recent_perfs = [x[0] for x in self.performance_history[node_id][-10:]]
                weights = [0.7 ** i for i in range(len(recent_perfs))]
                performance_scores[node_id] = sum(p*w for p,w in zip(recent_perfs, weights)) / sum(weights)
            else:
                performance_scores[node_id] = 0.0
        total_perf = sum(performance_scores.values()) or 1.0

        # Basic specialization scores
        specialization_scores = {}
        for node_id, spec in self.agent_specializations.items():
            if spec["primary"] == "memory" and self.resource_state["gpu_memory"] > 0.7:
                specialization_scores[node_id] = 1.5
            elif spec["primary"] == "computation" and self.resource_state["gpu_util"] > 0.7:
                specialization_scores[node_id] = 1.5
            elif spec["primary"] == "optimization" and self.resource_scarcity > 0.7:
                specialization_scores[node_id] = 1.3
            else:
                specialization_scores[node_id] = 1.0
        total_spec = sum(specialization_scores.values()) or 1.0

        new_tokens = {}
        for node_id in self.tokens:
            perf_share = (performance_scores.get(node_id, 0) / total_perf) * performance_allocation
            spec_share = (specialization_scores.get(node_id, 0) / total_spec) * specialization_allocation
            new_tokens[node_id] = int(base_allocation + perf_share + spec_share)
        self.tokens = new_tokens

        self.market_value = 0.8 * self.market_value + 0.2 * 1.0
        self.outstanding_offers = []

    def get_resource_pressure(self) -> float:
        self.resource_state = self._get_resource_state()
        return max(
            self.resource_state["gpu_memory"],
            self.resource_state["gpu_util"],
            self.resource_state["cpu_util"]
        )

    # =========================================================================================
    #   Visualization
    # =========================================================================================

    def visualize_economy(self, save_path=None):
        if not self.history['timestamp']:
            return
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))

        ax = axs[0]
        time_steps = self.history['timestamp']
        for agent_id in range(self.config.n_agents):
            values = [dist.get(agent_id, 0) for dist in self.history['token_distribution']]
            ax.plot(time_steps, values, label=f"Agent {agent_id}")
        ax.set_title("Token Distribution by Agent")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Token Share")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axs[1]
        ax.plot(time_steps, self.history['market_value'], 'g-', label="Market Value")
        ax.plot(time_steps, self.history['resource_scarcity'], 'r-', label="Resource Scarcity")
        ax.plot(time_steps, self.history['trade_volume'], 'b-', label="Trade Volume")
        ax.set_title("Market Conditions")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axs[2]
        for agent_id in range(self.config.n_agents):
            values = [perf.get(agent_id, 0) for perf in self.history['agent_performance']]
            ax.plot(time_steps, values, label=f"Agent {agent_id}")
        ax.set_title("Agent Performance")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Performance Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return save_path
        else:
            return fig

    def visualize_current_state(self, current_step=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        token_values = [self.tokens[i] for i in range(self.config.n_agents)]
        labels = [f"Agent {i}" for i in range(self.config.n_agents)]

        ax1.pie(token_values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title('Current Token Distribution')

        agent_ids = list(range(self.config.n_agents))
        performance_values = []
        for i in range(self.config.n_agents):
            if self.performance_history[i]:
                # each entry is (perf, reward, improvement)
                perf = self.performance_history[i][-1][0]
            else:
                perf = 0
            performance_values.append(perf)

        x = np.arange(len(agent_ids))
        width = 0.35
        ax2.bar(x - width/2, token_values, width, label='Tokens')
        ax2.bar(x + width/2, [p * 1000 for p in performance_values], width, label='Perf × 1000')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_title('Agent Metrics')
        ax2.legend()

        market_info = (
            f"Market Value: {self.market_value:.2f}\n"
            f"Resource Scarcity: {self.resource_scarcity:.2f}\n"
            f"Recent Trades: {len(self.completed_trades[-10:]) if self.completed_trades else 0}\n"
            f"Step: {current_step if current_step is not None else len(self.history['timestamp'])}"
        )
        fig.text(0.02, 0.02, market_info, fontsize=12)
        plt.tight_layout()
        return fig


# =========================================================================================
#   NeuroPheromone System
# =========================================================================================

class NeuroPheromoneSystem:
    """
    Adaptive NeuroPheromone system with dynamic pathway formation
    and neurochemical-inspired behavior regulation.
    """
    def __init__(self, config: MACAOConfig, dimensions: int):
        self.config = config
        self.dimensions = dimensions
        self.pheromone_tensor = cp.ones((dimensions, dimensions), dtype=cp.complex128) * config.initial_pheromone
        self.anxiety_wavefunction = cp.zeros(dimensions, dtype=cp.complex128)
        self.panic_tensor = cp.zeros(dimensions, dtype=cp.float64)
        self.pathway_graph = cp.ones((dimensions, dimensions), dtype=cp.float64) * 0.01
        self.stagnation_counter = 0
        self.best_performance = 0.0
        self.quantum_burst_countdown = config.quantum_burst_interval

    def update_anxiety(self, current_performance: float, target: float = 0.7):
        performance_gap = target - current_performance
        anxiety_factor = cp.tanh(performance_gap * 3)
        self.anxiety_wavefunction.real = 0.8 * self.anxiety_wavefunction.real + 0.2 * anxiety_factor
        if self.stagnation_counter > 10:
            creativity_potential = cp.tanh(self.stagnation_counter / 20)
            self.anxiety_wavefunction.imag = 0.7 * self.anxiety_wavefunction.imag + 0.3 * creativity_potential
        else:
            self.anxiety_wavefunction.imag *= 0.9
        
        # Update panic_tensor based on performance crisis
        crisis_level = cp.maximum(performance_gap, 0) * cp.abs(anxiety_factor)
        self.panic_tensor = 0.85 * self.panic_tensor + 0.15 * cp.tanh(crisis_level + self.stagnation_counter * 0.01)

    def apply_neurochemical_effects(self):
        stability_factor = cp.exp(-cp.abs(self.anxiety_wavefunction.real) * self.config.myrcene_factor)
        self.pheromone_tensor.real *= stability_factor

        creativity_factor = cp.tanh(cp.abs(self.anxiety_wavefunction.imag) * self.config.limonene_factor)
        if cp.mean(creativity_factor) > 0.6:
            self.pheromone_tensor += self._generate_quantum_burst() * creativity_factor

        pathway_clarity = cp.exp(-cp.var(self.pheromone_tensor, axis=0) * self.config.pinene_factor)
        self.pheromone_tensor *= pathway_clarity

        smoothing_factor = cp.tanh(cp.abs(self.anxiety_wavefunction.imag) * self.config.linalool_factor)
        self.pheromone_tensor.imag *= smoothing_factor

    def _generate_quantum_burst(self) -> cp.ndarray:
        burst = cp.random.normal(0, 0.1, (self.dimensions, self.dimensions)) + \
                1j * cp.random.normal(0, 0.1, (self.dimensions, self.dimensions))
        try:
            u, s, vh = cp.linalg.svd(self.pheromone_tensor)
            top_k = max(1, self.dimensions // 4)
            structured_burst = u[:, :top_k] @ cp.diag(s[:top_k]) @ vh[:top_k, :]
            return 0.3 * burst + 0.7 * structured_burst
        except Exception as e:
            logging.warning(f"Error in quantum burst: {e}")
            return burst

    def deposit_pheromones(self, agent_paths: List[List[int]], performances: List[float]):
        self.pheromone_tensor *= (1.0 - self.config.evaporation_rate)
        for path, perf in zip(agent_paths, performances):
            if len(path) < 2:
                continue
            deposit_amount = self.config.pheromone_intensity * (perf ** 1.5)
            for i in range(len(path) - 1):
                from_idx, to_idx = path[i], path[i+1]
                self.pheromone_tensor[from_idx, to_idx] += deposit_amount
                self.pathway_graph[from_idx, to_idx] += 0.1 * perf
        max_pheromone = cp.max(cp.abs(self.pheromone_tensor))
        if max_pheromone > 10.0:
            self.pheromone_tensor *= (10.0 / max_pheromone)

    def get_next_actions(self, current_positions: List[int]) -> List[int]:
        next_positions = []
        for pos in current_positions:
            pheromone_strengths = cp.abs(self.pheromone_tensor[pos, :])
            pathway_strengths = self.pathway_graph[pos, :]
            combined_strength = pheromone_strengths * pathway_strengths
            combined_strength += cp.random.normal(0, self.config.noise_std, self.dimensions)
            combined_strength = cp.maximum(0, combined_strength)
            sum_strength = cp.sum(combined_strength)
            if sum_strength > 0:
                probs = combined_strength / sum_strength
                probs_np = cp.asnumpy(probs)
                next_pos = np.random.choice(self.dimensions, p=probs_np)
            else:
                next_pos = np.random.randint(0, self.dimensions)
            next_positions.append(int(next_pos))
        return next_positions

    def check_stagnation(self, current_performance: float) -> bool:
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
        if self.stagnation_counter >= self.config.partial_reset_threshold:
            self._apply_partial_reset()
            self.stagnation_counter = 0
            return True
        return False

    def _apply_partial_reset(self):
        logging.info("NeuroPheromone system: Applying partial reset.")
        flat_pheromones = cp.abs(self.pheromone_tensor.flatten())
        sorted_indices = cp.argsort(flat_pheromones)
        reset_count = len(sorted_indices) // 5
        for idx in sorted_indices[:reset_count]:
            row = int(idx) // self.dimensions
            col = int(idx) % self.dimensions
            self.pheromone_tensor[row, col] = self.config.initial_pheromone * (0.5 + 0.5 * cp.random.random())
        # Reduce weak pathways
        pathway_array = cp.asnumpy(self.pathway_graph)
        threshold = np.percentile(pathway_array, 30)
        weak_pathways = cp.where(self.pathway_graph < threshold)
        self.pathway_graph[weak_pathways] *= 0.5
        # Create some new random connections
        new_connections = cp.random.randint(0, self.dimensions, (self.dimensions // 10, 2))
        for i, j in new_connections:
            self.pathway_graph[int(i), int(j)] = 0.5 + 0.5 * cp.random.random()

    def check_quantum_burst(self) -> bool:
        # Check panic levels for emergency burst
        panic_mean = cp.mean(self.panic_tensor)
        anxiety_magnitude = cp.mean(cp.abs(self.anxiety_wavefunction))
        emergency_threshold = 0.7
        
        self.quantum_burst_countdown -= 1
        if self.quantum_burst_countdown <= 0 or panic_mean > emergency_threshold or anxiety_magnitude > 1.5:
            self._apply_quantum_burst()
            self.quantum_burst_countdown = self.config.quantum_burst_interval
            return True
        return False

    def _apply_quantum_burst(self):
        logging.info("NeuroPheromone system: Applying quantum burst.")
        burst_pattern = self._generate_quantum_burst()
        # Scale burst by panic and anxiety levels
        panic_strength = float(cp.mean(self.panic_tensor))
        anxiety_strength = float(cp.mean(cp.abs(self.anxiety_wavefunction)))
        burst_scale = 0.5 + 0.5 * (panic_strength + anxiety_strength)
        self.pheromone_tensor += burst_pattern * self.config.limonene_factor * burst_scale
        # Reset anxiety and panic after burst
        self.anxiety_wavefunction *= 0.2
        self.panic_tensor *= 0.3

# =========================================================================================
#   Enhanced Cognitive Node (Agent)
# =========================================================================================

class EnhancedCognitiveNode:
    """
    An enhanced cognitive node that acts as an agent in the quantum economy,
    with advanced trading capabilities, plus specialized LR proposals and
    loss-aware damping for learning rate changes.
    """
    def __init__(self, node_id: int, economy: EnhancedQuantumEconomy, config):
        self.node_id = node_id
        self.economy = economy
        self.config = config
        self.focus_areas = ['learning_rate', 'regularization', 'architecture', 'data_focus']
        self.focus = self.focus_areas[node_id % len(self.focus_areas)]
        self.current_strategy = self._initialize_strategy()
        self.performance_history = []
        self.panic_level = 0.2
        self.last_resource_request = 0.0
        self.trade_history = []
        self.risk_appetite = np.random.uniform(0.5, 0.95)
        self.cooperation_tendency = np.random.uniform(0.7, 0.98)
        self.innovation_drive = np.random.uniform(0.5, 0.9)

        # Additional tracking for loss changes
        self.last_loss = None
        self.initial_loss = None
        self.last_lr_direction = 0
        self.consecutive_improvements = 0
        self.consecutive_regressions = 0

    def _initialize_strategy(self) -> Dict[str, Any]:
        if self.focus == 'learning_rate':
            return {
                'lr': self.config.learning_rate,
                'lr_decay': 'cosine',
                'warmup_steps': 100
            }
        elif self.focus == 'regularization':
            return {
                'weight_decay': 0.01,
                'dropout': 0.1,
                'lora_dropout': 0.05
            }
        elif self.focus == 'architecture':
            return {
                'lora_rank': self.config.lora_rank,
                'lora_alpha': self.config.lora_alpha,
                'target_modules': ["q_proj", "k_proj", "v_proj", "dense"]
            }
        elif self.focus == 'data_focus':
            return {
                'batch_priority': 'default',
                'sequence_length': self.config.max_seq_length,
                'augmentation': 0.0
            }
        return {}

    def propose_update(self, current_loss: float, iteration: int, previous_loss: float = None,
                       gradient_norm: float = None) -> Dict[str, Any]:
        """
        Enhanced proposal with:
          - Consecutive improvement/regression tracking
          - Loss-aware LR changes
          - Memory of successful directions
        """
        # Initialize initial_loss if needed
        if self.initial_loss is None:
            self.initial_loss = current_loss

        # Track improvement
        improvement = 0
        loss_improved = False
        if previous_loss is not None:
            improvement = (previous_loss - current_loss) / previous_loss
            loss_improved = improvement > 0

            if loss_improved:
                self.consecutive_improvements += 1
                self.consecutive_regressions = 0
            else:
                self.consecutive_improvements = 0
                self.consecutive_regressions += 1

        # Evaluate performance for internal node logic
        performance = -np.log(current_loss + 1e-8)
        self.performance_history.append(performance)

        # Adjust panic
        if len(self.performance_history) >= 3:
            recent_trend = self.performance_history[-1] - self.performance_history[-3]
            relative_loss = current_loss / self.initial_loss
            target_panic = 0.2 + 0.8 * min(1.0, relative_loss * 1.5)

            if self.consecutive_improvements >= 3:
                target_panic *= 0.7
            elif self.consecutive_regressions >= 3:
                target_panic *= 1.3

            self.panic_level = 0.8 * self.panic_level + 0.2 * target_panic
            self.panic_level = max(0.1, min(0.9, self.panic_level))

        # Resource request scales with panic
        confidence_factor = 1.0
        if self.consecutive_improvements >= 2:
            confidence_factor = 1.2
        elif self.consecutive_regressions >= 2:
            confidence_factor = 0.8

        required_power = (0.2 + 0.3 * self.panic_level) * (0.8 + 0.4 * self.risk_appetite) * confidence_factor
        resource_granted = self.economy.buy_resources(self.node_id, required_power)
        self.last_resource_request = required_power

        if not resource_granted and np.random.random() < self.cooperation_tendency:
            self._initiate_trade_if_needed()

        new_strategy = copy.deepcopy(self.current_strategy)

        # Learning rate logic
        if self.focus == 'learning_rate':
            market_value = self.economy.market_value
            adjustment_factor = 0.12 if resource_granted else 0.03

            # Less market influence
            if market_value > 1.5 or market_value < 0.5:
                adjustment_factor *= 1.1

            # Direction depends on improvements/regressions
            if self.consecutive_improvements >= 2:
                direction = np.sign(self.last_lr_direction) * (0.8 + 0.2 * self.innovation_drive)
            elif self.consecutive_regressions >= 2:
                direction = -np.sign(self.last_lr_direction) * (1.0 + 0.5 * self.innovation_drive)
            elif previous_loss is not None:
                if loss_improved:
                    direction = np.sign(self.last_lr_direction or 0.1) * (0.6 + 0.2 * self.innovation_drive)
                else:
                    direction = -np.sign(self.last_lr_direction or 0.1) * (0.7 + 0.3 * self.innovation_drive)
            else:
                # default exploration
                direction = -0.5 + np.random.random() * self.innovation_drive

            # Loss-aware damping
            loss_ratio = min(1.0, current_loss / self.initial_loss)
            loss_factor = 0.5 + 0.5 * loss_ratio
            noise = np.random.normal(0, 0.05 * self.innovation_drive)
            lr_change = adjustment_factor * direction * loss_factor * (1.0 + noise)

            self.last_lr_direction = lr_change
            new_strategy['lr'] *= (1.0 + lr_change * 0.7)

            new_strategy['lr'] = max(self.config.lr_min, min(self.config.lr_max, new_strategy['lr']))

            wandb.log({
                f"agent/{self.node_id}/lr_proposal": new_strategy['lr'],
                f"agent/{self.node_id}/lr_change": lr_change,
                f"agent/{self.node_id}/direction": direction,
                f"agent/{self.node_id}/loss_factor": loss_factor,
                f"agent/{self.node_id}/consecutive_improvements": self.consecutive_improvements,
                f"agent/{self.node_id}/consecutive_regressions": self.consecutive_regressions,
                f"agent/{self.node_id}/panic_level": self.panic_level
            })

        elif self.focus == 'regularization':
            market_value = self.economy.market_value
            adjustment_factor = 0.1 if resource_granted else 0.02
            
            # Regularization increases during overfitting signs
            if self.consecutive_regressions >= 2:
                # Increase regularization
                direction = 1.0 * (0.8 + 0.2 * self.innovation_drive)
            elif self.consecutive_improvements >= 3:
                # Decrease regularization to allow more learning
                direction = -0.5 * (0.6 + 0.2 * self.innovation_drive)
            elif previous_loss is not None:
                if loss_improved:
                    direction = -0.3 * (0.5 + 0.2 * self.innovation_drive)
                else:
                    direction = 0.7 * (0.7 + 0.3 * self.innovation_drive)
            else:
                direction = -0.2 + 0.4 * np.random.random() * self.innovation_drive

            # Apply regularization changes
            reg_change = adjustment_factor * direction
            new_strategy['weight_decay'] *= (1.0 + reg_change)
            new_strategy['dropout'] *= (1.0 + reg_change * 0.5)
            new_strategy['lora_dropout'] *= (1.0 + reg_change * 0.3)
            
            # Clamp values
            new_strategy['weight_decay'] = max(0.001, min(0.1, new_strategy['weight_decay']))
            new_strategy['dropout'] = max(0.0, min(0.5, new_strategy['dropout']))
            new_strategy['lora_dropout'] = max(0.0, min(0.3, new_strategy['lora_dropout']))

        elif self.focus == 'architecture':
            market_value = self.economy.market_value
            adjustment_factor = 0.15 if resource_granted else 0.05
            
            # Architecture changes based on gradient behavior
            if gradient_norm is not None and gradient_norm > 2.0:
                # High gradients suggest need for more capacity
                direction = 1.0 * (0.8 + 0.2 * self.innovation_drive)
            elif self.consecutive_improvements >= 2:
                # Keep current architecture but fine-tune
                direction = 0.1 * self.innovation_drive
            elif self.consecutive_regressions >= 3:
                # Try different architecture
                direction = np.random.choice([-1, 1]) * (0.6 + 0.4 * self.innovation_drive)
            else:
                direction = -0.3 + 0.6 * np.random.random() * self.innovation_drive

            # Apply architecture changes
            arch_change = adjustment_factor * direction
            new_strategy['lora_rank'] = int(new_strategy['lora_rank'] * (1.0 + arch_change))
            new_strategy['lora_alpha'] *= (1.0 + arch_change * 0.5)
            
            # Clamp values
            new_strategy['lora_rank'] = max(4, min(256, new_strategy['lora_rank']))
            new_strategy['lora_alpha'] = max(4.0, min(64.0, new_strategy['lora_alpha']))

        elif self.focus == 'data_focus':
            market_value = self.economy.market_value
            adjustment_factor = 0.12 if resource_granted else 0.03
            
            # Data focus adjustments based on learning progress
            if self.consecutive_improvements >= 3:
                # Increase sequence length for more complex learning
                direction = 1.0 * (0.7 + 0.3 * self.innovation_drive)
            elif self.consecutive_regressions >= 2:
                # Reduce complexity, add augmentation
                direction = -0.8 * (0.6 + 0.4 * self.innovation_drive)
            elif previous_loss is not None:
                if loss_improved:
                    direction = 0.5 * (0.5 + 0.3 * self.innovation_drive)
                else:
                    direction = -0.4 * (0.6 + 0.2 * self.innovation_drive)
            else:
                direction = -0.2 + 0.4 * np.random.random() * self.innovation_drive

            # Apply data focus changes
            data_change = adjustment_factor * direction
            new_strategy['sequence_length'] = int(new_strategy['sequence_length'] * (1.0 + data_change * 0.1))
            new_strategy['augmentation'] += data_change * 0.1
            
            # Update batch priority based on performance
            if self.consecutive_improvements >= 2:
                new_strategy['batch_priority'] = 'hard_examples'
            elif self.consecutive_regressions >= 2:
                new_strategy['batch_priority'] = 'easy_examples'
            else:
                new_strategy['batch_priority'] = 'default'
            
            # Clamp values
            new_strategy['sequence_length'] = max(64, min(self.config.max_seq_length, new_strategy['sequence_length']))
            new_strategy['augmentation'] = max(0.0, min(0.3, new_strategy['augmentation']))

        self.last_loss = current_loss
        self.current_strategy = new_strategy

        return new_strategy

    def _initiate_trade_if_needed(self):
        token_balance = self.economy.tokens[self.node_id]
        avg_token = sum(self.economy.tokens.values()) / len(self.economy.tokens)

        if token_balance < avg_token * 0.7:
            amount_needed = int((avg_token - token_balance) * 0.5)
            self.economy._create_trade_offer(self.node_id, amount_needed, "need_tokens")
            logging.info(f"Agent {self.node_id}: 'need_tokens' offer for {amount_needed}")
            return True
        elif token_balance > avg_token * 1.3 and np.random.random() < self.cooperation_tendency:
            amount_to_offer = int((token_balance - avg_token) * 0.3)
            self.economy._create_trade_offer(self.node_id, amount_to_offer, "offer_tokens")
            logging.info(f"Agent {self.node_id}: 'offer_tokens' offer for {amount_to_offer}")
            return True
        return False

    def on_trade_completed(self, trade):
        self.trade_history.append(trade)
        if trade['buyer'] == self.node_id:
            self.cooperation_tendency = min(0.95, self.cooperation_tendency * 1.05)
        elif trade['seller'] == self.node_id:
            self.cooperation_tendency = min(0.95, self.cooperation_tendency * 1.03)

# =========================================================================================
#   Training Procedure
# =========================================================================================

def collate_fn(batch_texts, tokenizer, max_length=256):
    enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    enc["labels"] = enc["input_ids"].clone()
    return enc

def main():
    logging.basicConfig(level=logging.INFO)
    config = MACAOConfig()
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
    wandb.config.update(vars(config))

    if not os.path.isfile(config.training_data_file):
        raise FileNotFoundError(f"Training file {config.training_data_file} not found.")
    dataset = load_dataset("json", data_files=config.training_data_file)["train"]

    def format_data(ex):
        if "instruction" in ex and "response" in ex:
            return {"text": f"Instruction: {ex['instruction']}\n\nResponse: {ex['response']}"}
        if "text" in ex:
            return {"text": ex["text"]}
        return {"text": ""}

    dataset = dataset.map(format_data).remove_columns([c for c in dataset.column_names if c != "text"])
    data_texts = [r["text"] for r in dataset]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    logging.info("Loading base model + tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                 f"({trainable_params / total_params:.2%})")
    wandb.log({"trainable_params_ratio": trainable_params / total_params})

    train_dataloader = DataLoader(
        data_texts,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=config.max_seq_length)
    )

    economy = EnhancedQuantumEconomy(config)
    pheromone_sys = NeuroPheromoneSystem(config, dimensions=64)
    nodes = [EnhancedCognitiveNode(i, economy, config) for i in range(config.n_agents)]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_params, lr=config.learning_rate)
    num_training_steps = config.num_epochs * len(train_dataloader) // config.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    visualization_interval = config.visualization_interval
    global_step = 0
    model.train()

    loss_ema = None
    initial_loss = None

    for epoch in range(config.num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_val = loss.item()

            # Compute a quick token-level accuracy
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions = (predictions == batch["labels"]).float()
                mask = (batch["labels"] != -100).float()
                if mask.sum() > 0:
                    accuracy = (correct_predictions * mask).sum() / mask.sum()
                    wandb.log({"train/accuracy": accuracy.item()})

            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                # Gradient norm
                gradient_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        gradient_norm += grad_norm * grad_norm
                gradient_norm = gradient_norm ** 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Track EMA of loss
                if loss_ema is None:
                    loss_ema = loss_val
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss_val

                if initial_loss is None:
                    initial_loss = loss_val
                    economy.initial_loss = initial_loss

                if global_step % config.log_interval == 0:
                    logging.info(f"Step {global_step} | Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.3e}")
                    wandb.log({
                        "train/step": global_step,
                        "train/loss": loss_val,
                        "train/loss_ema": loss_ema,
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "train/gradient_norm": gradient_norm
                    })

                    # Optionally visualize the economy state
                    if config.enable_visualization and global_step % visualization_interval == 0:
                        try:
                            viz_path = os.path.join(config.output_dir, f"economy_viz_{global_step}.png")
                            economy.visualize_economy(save_path=viz_path)
                            wandb.log({"economy/visualization": wandb.Image(viz_path)})

                            current_state_fig = economy.visualize_current_state(current_step=global_step)
                            current_state_path = os.path.join(config.output_dir, f"economy_state_{global_step}.png")
                            current_state_fig.savefig(current_state_path)
                            plt.close(current_state_fig)
                            wandb.log({"economy/current_state": wandb.Image(current_state_path)})
                        except Exception as e:
                            logging.warning(f"Error creating economy visualization: {e}")

                    if config.export_economy_data and global_step % (visualization_interval * 2) == 0:
                        try:
                            economy_data = {
                                "tokens": economy.tokens,
                                "n_agents": config.n_agents,
                                "market_value": economy.market_value,
                                "resource_scarcity": economy.resource_scarcity,
                                "recent_trades": len(economy.completed_trades[-10:]) if economy.completed_trades else 0,
                                "specializations": economy.agent_specializations,
                                "history_timestamps": economy.history['timestamp'],
                                "history_market_value": economy.history['market_value'],
                                "history_resource_scarcity": economy.history['resource_scarcity'],
                                "history_trade_volume": economy.history['trade_volume'],
                                "history_token_distribution": economy.history['token_distribution']
                            }
                            data_path = os.path.join(config.output_dir, f"economy_data_{global_step}.json")
                            with open(data_path, 'w') as f:
                                json.dump(economy_data, f, indent=2,
                                          default=lambda x: str(x) if isinstance(x, np.ndarray) else x)
                        except Exception as e:
                            logging.warning(f"Error exporting economy data: {e}")

                # Update market dynamics periodically
                if global_step % (config.log_interval // 2) == 0:
                    economy.update_market_dynamics(global_step)

                # Process trades occasionally
                if global_step % 5 == 0 and config.enable_trading:
                    economy.process_trades()

                # Agents propose updates
                agent_paths = []
                performances = []
                for i, node in enumerate(nodes):
                    # Provide previous_loss=loss_ema, gradient_norm=calc
                    proposal = node.propose_update(loss_val, global_step, previous_loss=loss_ema, gradient_norm=gradient_norm)
                    proposal_used = False

                    # If node is focusing on LR, check for a meaningful (3% by default in the conversation)
                    if node.focus == 'learning_rate' and 'lr' in proposal:
                        new_lr = proposal['lr']
                        current_lr = scheduler.get_last_lr()[0]
                        lr_diff = abs(new_lr - current_lr)
                        # Must exceed 3% difference to be considered "influencing training"
                        if lr_diff / max(current_lr, 1e-12) > 0.03:
                            for pg in optimizer.param_groups:
                                pg['lr'] = new_lr
                            wandb.log({
                                "lr_change": new_lr - current_lr,
                                "lr_change_pct": (new_lr - current_lr) / max(current_lr, 1e-12),
                                "proposing_agent": i
                            })
                            proposal_used = True

                    # Reward agent (use updated EnhancedQuantumEconomy.reward_performance)
                    rew, perf = economy.reward_performance(
                        node.node_id,
                        loss_val,
                        influenced_training=proposal_used,
                        loss_improved=(loss_ema > loss_val) if loss_ema is not None else False,
                        previous_loss=loss_ema,
                        agent_type=node.focus,
                        gradient_norm=gradient_norm
                    )

                    # For pheromone deposit, just a simple example path
                    path = [node.node_id, (node.node_id + 4) % 64]
                    agent_paths.append(path)
                    performances.append(perf)

                # Pheromone deposit & updates
                pheromone_sys.deposit_pheromones(agent_paths, performances)
                pheromone_sys.update_anxiety(1.0 / (1.0 + loss_val))
                pheromone_sys.apply_neurochemical_effects()

                # Check NeuroPheromone triggers
                if pheromone_sys.check_stagnation(1.0 / (1.0 + loss_val)):
                    wandb.log({"events/partial_reset": global_step})
                if pheromone_sys.check_quantum_burst():
                    wandb.log({"events/quantum_burst": global_step})
                if economy.get_resource_pressure() > 0.9:
                    economy.trigger_quantum_burst()
                    wandb.log({"events/economy_burst": global_step})

        if config.enable_visualization:
            try:
                epoch_viz_path = os.path.join(config.output_dir, f"economy_viz_epoch_{epoch+1}.png")
                economy.visualize_economy(save_path=epoch_viz_path)
                wandb.log({
                    "economy/epoch_visualization": wandb.Image(epoch_viz_path),
                    "epoch": epoch + 1
                })
            except Exception as e:
                logging.warning(f"Error creating end-of-epoch visualization: {e}")

        logging.info(f"Completed epoch {epoch+1}")

    logging.info("Saving final model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.enable_visualization:
        final_viz_path = os.path.join(config.output_dir, "economy_viz_final.png")
        economy.visualize_economy(save_path=final_viz_path)
        wandb.log({"economy/final_visualization": wandb.Image(final_viz_path)})

    if config.export_economy_data:
        try:
            final_economy_data = {
                "tokens": economy.tokens,
                "n_agents": config.n_agents,
                "market_value": economy.market_value,
                "resource_scarcity": economy.resource_scarcity,
                "recent_trades": len(economy.completed_trades[-20:]) if economy.completed_trades else 0,
                "specializations": economy.agent_specializations,
                "history_timestamps": economy.history['timestamp'],
                "history_market_value": economy.history['market_value'],
                "history_resource_scarcity": economy.history['resource_scarcity'],
                "history_trade_volume": economy.history['trade_volume'],
                "history_token_distribution": economy.history['token_distribution'],
                "completed_trades": economy.completed_trades[-100:] if economy.completed_trades else []
            }
            final_data_path = os.path.join(config.output_dir, "economy_data_final.json")
            with open(final_data_path, 'w') as f:
                json.dump(final_economy_data, f, indent=2,
                          default=lambda x: str(x) if isinstance(x, np.ndarray) else x)
            logging.info(f"Final economy data exported to {final_data_path}")
        except Exception as e:
            logging.warning(f"Error exporting final economy data: {e}")

    logging.info(f"Model saved to {config.output_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
