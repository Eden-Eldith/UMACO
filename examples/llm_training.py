#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Training with MACO
======================

This example demonstrates how to use MACO-LLM to fine-tune a language model.
The script sets up the multi-agent system with the enhanced quantum economy
and demonstrates the training process with visualization.
"""

import os
import argparse
import json
import logging
import torch
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umaco.maco_direct_train16 import (
    MACAOConfig, EnhancedQuantumEconomy, 
    EnhancedCognitiveNode, NeuroPheromoneSystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("maco_llm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collate_fn(batch_texts, tokenizer, max_length=256):
    """Prepare batches for training."""
    enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    enc["labels"] = enc["input_ids"].clone()
    return enc

def main():
    parser = argparse.ArgumentParser(description="MACO-LLM Training Script")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Model to fine-tune")
    parser.add_argument("--data_file", type=str, required=True, help="Training data file (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./macao_output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--n_agents", type=int, default=8, help="Number of MACO agents")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = MACAOConfig(**config_dict)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = MACAOConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            training_data_file=args.data_file,
            n_agents=args.n_agents,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            wandb_run_name=f"maco_{os.path.basename(args.model_name)}",
        )
        logger.info("Using default configuration with command line overrides")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=config.wandb_project, name=config.wandb_run_name)
            wandb.config.update(vars(config))
            logger.info("Initialized Weights & Biases logging")
        except ImportError:
            logger.warning("Could not import wandb. Continuing without W&B logging.")
            args.use_wandb = False
    
    # Load and prepare dataset
    logger.info(f"Loading dataset from {config.training_data_file}")
    dataset = load_dataset("json", data_files=config.training_data_file)["train"]
    
    def format_data(ex):
        if "instruction" in ex and "response" in ex:
            return {"text": f"Instruction: {ex['instruction']}\n\nResponse: {ex['response']}"}
        if "text" in ex:
            return {"text": ex["text"]}
        return {"text": ""}
    
    dataset = dataset.map(format_data).remove_columns([c for c in dataset.column_names if c != "text"])
    data_texts = [r["text"] for r in dataset]
    logger.info(f"Dataset loaded with {len(data_texts)} samples")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for LoRA training
    try:
        from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
        
        logger.info("Preparing model for LoRA training")
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
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                    f"({trainable_params / total_params:.2%})")
        
        if args.use_wandb:
            wandb.log({"trainable_params_ratio": trainable_params / total_params})
    
    except ImportError:
        logger.error("PEFT library not found. Install with: pip install peft")
        return
    
    # Create dataloader
    train_dataloader = DataLoader(
        data_texts,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=config.max_seq_length)
    )
    
    # Initialize MACO components
    economy = EnhancedQuantumEconomy(config)
    pheromone_sys = NeuroPheromoneSystem(config, dimensions=64)
    nodes = [EnhancedCognitiveNode(i, economy, config) for i in range(config.n_agents)]
    
    # Set up optimizer and scheduler
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
    
    # Training loop
    logger.info("Starting training")
    global_step = 0
    loss_ema = None
    initial_loss = None
    
    visualization_interval = config.visualization_interval
    model.train()
    
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_val = loss.item()
            
            # Compute accuracy for logging
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions = (predictions == batch["labels"]).float()
                mask = (batch["labels"] != -100).float()
                if mask.sum() > 0:
                    accuracy = (correct_predictions * mask).sum() / mask.sum()
                    if args.use_wandb:
                        wandb.log({"train/accuracy": accuracy.item()})
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % config.grad_accum_steps == 0:
                # Calculate gradient norm
                gradient_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        gradient_norm += grad_norm * grad_norm
                gradient_norm = gradient_norm ** 0.5
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Track exponential moving average of loss
                if loss_ema is None:
                    loss_ema = loss_val
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss_val
                
                if initial_loss is None:
                    initial_loss = loss_val
                    economy.initial_loss = initial_loss
                
                # Log progress
                if global_step % config.log_interval == 0:
                    log_msg = (f"Step {global_step} | Loss: {loss_val:.4f} | "
                              f"LR: {scheduler.get_last_lr()[0]:.3e}")
                    logger.info(log_msg)
                    
                    if args.use_wandb:
                        wandb.log({
                            "train/step": global_step,
                            "train/loss": loss_val,
                            "train/loss_ema": loss_ema,
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch + 1,
                            "train/gradient_norm": gradient_norm
                        })
                
                # Update market dynamics
                if global_step % (config.log_interval // 2) == 0:
                    economy.update_market_dynamics(global_step)
                
                # Process trades
                if global_step % 5 == 0 and config.enable_trading:
                    economy.process_trades()
                
                # Agent proposals
                agent_paths = []
                performances = []
                for i, node in enumerate(nodes):
                    proposal = node.propose_update(
                        loss_val, 
                        global_step, 
                        previous_loss=loss_ema, 
                        gradient_norm=gradient_norm
                    )
                    proposal_used = False
                    
                    # Apply learning rate changes if significant
                    if node.focus == 'learning_rate' and 'lr' in proposal:
                        new_lr = proposal['lr']
                        current_lr = scheduler.get_last_lr()[0]
                        lr_diff = abs(new_lr - current_lr)
                        
                        # Must exceed 3% difference to be considered "influencing training"
                        if lr_diff / max(current_lr, 1e-12) > 0.03:
                            for pg in optimizer.param_groups:
                                pg['lr'] = new_lr
                            
                            if args.use_wandb:
                                wandb.log({
                                    "lr_change": new_lr - current_lr,
                                    "lr_change_pct": (new_lr - current_lr) / max(current_lr, 1e-12),
                                    "proposing_agent": i
                                })
                            
                            proposal_used = True
                    
                    # Reward agent
                    rew, perf = economy.reward_performance(
                        node.node_id,
                        loss_val,
                        influenced_training=proposal_used,
                        loss_improved=(loss_ema > loss_val) if loss_ema is not None else False,
                        previous_loss=loss_ema,
                        agent_type=node.focus,
                        gradient_norm=gradient_norm
                    )
                    
                    # For pheromone deposit
                    path = [node.node_id, (node.node_id + 4) % 64]
                    agent_paths.append(path)
                    performances.append(perf)
                
                # Pheromone updates
                pheromone_sys.deposit_pheromones(agent_paths, performances)
                pheromone_sys.update_anxiety(1.0 / (1.0 + loss_val))
                pheromone_sys.apply_neurochemical_effects()
                
                # Check for NeuroPheromone triggers
                if pheromone_sys.check_stagnation(1.0 / (1.0 + loss_val)):
                    if args.use_wandb:
                        wandb.log({"events/partial_reset": global_step})
                
                if pheromone_sys.check_quantum_burst():
                    if args.use_wandb:
                        wandb.log({"events/quantum_burst": global_step})
                
                if economy.get_resource_pressure() > 0.9:
                    economy.trigger_quantum_burst()
                    if args.use_wandb:
                        wandb.log({"events/economy_burst": global_step})
                
                # Visualization
                if config.enable_visualization and global_step % visualization_interval == 0:
                    try:
                        viz_path = os.path.join(config.output_dir, f"economy_viz_{global_step}.png")
                        economy.visualize_economy(save_path=viz_path)
                        
                        if args.use_wandb:
                            wandb.log({"economy/visualization": wandb.Image(viz_path)})
                        
                        current_state_fig = economy.visualize_current_state(current_step=global_step)
                        current_state_path = os.path.join(config.output_dir, f"economy_state_{global_step}.png")
                        current_state_fig.savefig(current_state_path)
                        
                        if args.use_wandb:
                            wandb.log({"economy/current_state": wandb.Image(current_state_path)})
                        
                        logger.info(f"Saved visualization to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Error creating economy visualization: {e}")
    
    # Save the final model
    logger.info("Saving final model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Final visualization
    if config.enable_visualization:
        try:
            final_viz_path = os.path.join(config.output_dir, "economy_viz_final.png")
            economy.visualize_economy(save_path=final_viz_path)
            
            if args.use_wandb:
                wandb.log({"economy/final_visualization": wandb.Image(final_viz_path)})
            
            logger.info(f"Saved final visualization to {final_viz_path}")
        except Exception as e:
            logger.warning(f"Error creating final visualization: {e}")
    
    logger.info(f"Training completed. Model saved to {config.output_dir}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
