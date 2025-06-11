#!/usr/bin/env python3
"""
Training script for sparsity predictors in LlamaSkipConnection models.

This script trains the LoRA-based sparsity predictors to predict which neurons
will be most important based on ground truth activations from standard LLaMA.

Usage:
    python train_predictors.py \
        --config configs/llama_skip_causal_3b_predictor_training.json \
        --output_dir ./trained_predictors \
        --num_samples 50000 \
        --batch_size 8 \
        --num_epochs 5 \
        --use_wandb
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset
import wandb
from tqdm import tqdm

from src.models.llama.modelling_llama_skip import LlamaSkipConnectionForCausalLM, FastLoRAProjection
from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(tokenizer: AutoTokenizer,
                       dataset_name: str = "allenai/c4", 
                      dataset_config: str = "realnewslike",
                      max_length: int = 2000
                      ) -> Tuple[List[str], List[str]]:
    """Load and prepare training data."""
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    if dataset_name == "allenai/c4":
        train_dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        train_dataset = train_dataset.with_format("torch")
        val_dataset = train_dataset.take(1000)
        train_dataset = train_dataset.skip(1000)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return train_dataset.map(tokenize_function, batched=True) , val_dataset.map(tokenize_function, batched=True) 


def evaluate_predictor_accuracy(model: LlamaSkipConnectionForCausalLM,
                               dataloader: DataLoader,
                               device: torch.device,
                               max_batches: int = 50) -> Dict[str, float]:
    """Evaluate predictor accuracy against ground truth."""
    model.eval()  # Set to evaluation mode
    
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass to collect predictor scores and ground truth
            outputs = model(input_ids=input_ids)
            
            # Calculate metrics for each layer
            layer_accuracies = []
            layer_precisions = []
            layer_recalls = []
            layer_f1s = []
            
            for layer in model.model.layers:
                if hasattr(layer, 'training_mode') and layer.training_mode:
                    # Get last hidden states for this evaluation
                    hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                    if hidden_states is not None:
                        # Get predictor scores
                        hidden_reshaped = hidden_states.view(-1, hidden_states.shape[-1])
                        pred_scores = layer.mlp_lora_proj(hidden_reshaped)
                        
                        # Get ground truth activations
                        gt_activations = layer.get_ground_truth_activations(hidden_reshaped)
                        
                        # Create binary masks
                        _, gt_indices = torch.topk(torch.abs(gt_activations), k, dim=-1)
                        _, pred_indices = torch.topk(pred_scores, k, dim=-1)
                        
                        # Calculate metrics
                        gt_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                        pred_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                        gt_mask.scatter_(1, gt_indices, True)
                        pred_mask.scatter_(1, pred_indices, True)
                        
                        # Accuracy
                        correct = (gt_mask == pred_mask).float().mean()
                        layer_accuracies.append(correct.item())
                        
                        # Precision, Recall, F1
                        tp = (gt_mask & pred_mask).sum().float()
                        fp = (~gt_mask & pred_mask).sum().float()
                        fn = (gt_mask & ~pred_mask).sum().float()
                        
                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)
                        
                        layer_precisions.append(precision.item())
                        layer_recalls.append(recall.item())
                        layer_f1s.append(f1.item())
            
            if layer_accuracies:
                total_accuracy += sum(layer_accuracies) / len(layer_accuracies)
                total_precision += sum(layer_precisions) / len(layer_precisions)
                total_recall += sum(layer_recalls) / len(layer_recalls)
                total_f1 += sum(layer_f1s) / len(layer_f1s)
                num_batches += 1
    
    model.train()  # Switch back to training mode
    
    if num_batches == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "accuracy": total_accuracy / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches
    }


def train_predictors(
    model: LlamaSkipConnectionForCausalLM,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    batch_size: int,
    save_dir: str,
    eval_steps: int = 500,
    save_steps: int = 1000,
    use_wandb: bool = False
) -> None:
    """Train the sparsity predictors using PyTorch's built-in training mode."""
    
    # Setup training mode using PyTorch's built-in methods
    model.train()  # Enable training mode
    model.freeze_non_predictor_parameters()
    
    # Setup optimizer
    predictor_params = model.get_predictor_parameters()
    optimizer = torch.optim.AdamW(predictor_params, lr=learning_rate, weight_decay=0.01)
    num_steps = 1380000//batch_size
    train_dataloader = DataLoader(
        train_dataset,
          batch_size=batch_size, 
          num_workers=16, 
          pin_memory=True,
          prefetch_factor=4)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2)
    # Setup scheduler
    total_steps =  num_steps * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.01 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    best_f1 = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Number of predictor parameters: {sum(p.numel() for p in predictor_params)}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_predictor_loss = 0.0
        num_batches = 0
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        progress_bar = tqdm(total=num_steps, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_predictor_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation
            if global_step % eval_steps == 0:
                logger.info(f"Evaluating at step {global_step}")
                try:
                    eval_metrics = evaluate_predictor_accuracy(model, val_dataloader, device)
                    logger.info(f"Evaluation metrics: {eval_metrics}")
                    
                    if use_wandb:
                        wandb.log({
                            "eval/accuracy": eval_metrics["accuracy"],
                            "eval/precision": eval_metrics["precision"],
                            "eval/recall": eval_metrics["recall"],
                            "eval/f1": eval_metrics["f1"],
                            "step": global_step
                        })
                    
                    # Save best model
                    if eval_metrics["f1"] > best_f1:
                        best_f1 = eval_metrics["f1"]
                        best_model_path = os.path.join(save_dir, "best_predictors")
                        try:
                            model.save_pretrained(best_model_path)
                            logger.info(f"Saved best model with F1: {best_f1:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to save best model: {e}")
                    
                    # Model is already back in training mode from evaluate_predictor_accuracy
                except Exception as e:
                    logger.warning(f"Evaluation failed at step {global_step}: {e}")
                    model.train()  # Ensure we're back in training mode
            
            # Save checkpoint
            # if global_step % save_steps == 0:
            #     checkpoint_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            #     model.save_pretrained(checkpoint_path)
            #     logger.info(f"Saved checkpoint at step {global_step}")
            
            # Log training metrics
            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "step": global_step,
                } | {f"gradients/layer_{i}": p.grad.mean().item() for i, p in enumerate(predictor_params)})
            progress_bar.update(1)

        # End of epoch logging
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
    
    # Final save
    final_model_path = os.path.join(save_dir, "final_predictors")
    model.save_pretrained(final_model_path)
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train sparsity predictors for LlamaSkipConnection")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--dataset", type=str, default="allenai/c4", help="Dataset name (default: allenai/c4)")
    parser.add_argument("--dataset_config", type=str, default="realnewslike", 
                       help="Dataset configuration (default: realnewslike for C4)")
    parser.add_argument("--num_samples", type=int, default=13800000, 
                       help="Number of training samples (default: 13800000)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=24, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=50000, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=50000, help="Save frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-skip-predictors", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="llama-skip-predictors", help="W&B entity name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"predictor-training-{int(time.time())}"
        )
    
    # Load model configuration
    config = LlamaSkipConnectionConfig.from_json_file(args.config)
    checkpoint = config._name_or_path
    
    # Register custom models
    AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
    AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config)
    
    if device.type == 'cuda':
        for module in model.modules():
            if any(hasattr(p, 'is_meta') and p.is_meta for p in module.parameters()) and isinstance(module, FastLoRAProjection):
                module = module.to_empty(device="cpu")
                with torch.no_grad():
                    torch.nn.init.xavier_normal_(module.down.weight)
                    torch.nn.init.zeros_(module.up.weight)  # Initialize up projection to zeros for stable training
                
        model.tie_weights()
        model = model.to(device)
    
    # Load training data
    train_dataset, val_dataset = load_training_data(
        tokenizer,
        args.dataset, 
        args.dataset_config,
        args.max_length
    )
    
    # Train predictors
    train_predictors(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb
    )
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 