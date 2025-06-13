#!/usr/bin/env python3
"""
Training script for sparsity predictors in LlamaSkipConnection models with DDP support.

This script trains the LoRA-based sparsity predictors to predict which neurons
will be most important based on ground truth activations from standard LLaMA.

Usage:
    # Single GPU
    python train_predictors.py \
        --config configs/llama_skip_causal_3b_predictor_training.json \
        --output_dir ./trained_predictors \
        --num_samples 50000 \
        --batch_size 8 \
        --num_epochs 5 \
        --use_wandb
        
    # Multi-GPU DDP
    python train_predictors.py \
        --config configs/llama_skip_causal_3b_predictor_training.json \
        --output_dir ./trained_predictors \
        --num_samples 50000 \
        --batch_size 8 \
        --num_epochs 5 \
        --use_wandb \
        --use_ddp
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size
import torch.multiprocessing as mp
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

from src.models.llama.modelling_llama_skip import LlamaSkipConnectionForCausalLM, FastLoRAProjection, LlamaSkipDecoderLayer
from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ddp_setup(rank: int, world_size: int):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Set additional environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(rank)
    init_process_group(rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed process group."""
    destroy_process_group()


def load_training_data(tokenizer: AutoTokenizer,
                       dataset_name: str = "allenai/c4", 
                      dataset_config: str = "realnewslike",
                      max_length: int = 2000,
                      num_samples: int = 13800000,
                      rank: int = 0,
                      world_size: int = 1
                      ) -> Tuple[Dataset, Dataset]:
    """Load and prepare training data."""
    if rank == 0:
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
        logger.info(f"Target number of samples: {num_samples}")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    if dataset_name == "allenai/c4":
        train_dataset = load_dataset(dataset_name, dataset_config, split="train")
        train_dataset = train_dataset.with_format("torch")
        val_dataset = train_dataset.take(1000)
        train_dataset = train_dataset.skip(1000).take(num_samples)
        
        # Apply tokenization
        if rank == 0:
            logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if rank == 0:
        logger.info(f"Final train dataset size: {len(train_dataset)}")
        logger.info(f"Final val dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def evaluate_predictor_accuracy(model: torch.nn.Module,
                               dataloader: DataLoader,
                               device: torch.device,
                               max_batches: int = 50,
                               rank: int = 0) -> Dict[str, float]:
    """Evaluate predictor accuracy against ground truth."""
    model.eval()  # Set to evaluation mode
    
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if isinstance(model, DDP) else model

    layer_accuracies = []
    layer_precisions = []
    layer_recalls = []
    layer_f1s = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            outputs = actual_model(input_ids=input_ids, capture_activations=True)
            
            # Calculate metrics for each layer
            for metrics in outputs.loss:
                pred_scores = metrics[0]
                pred_mask = pred_scores > 0.5
                gt_mask = metrics[1].bool()
                # Accuracy
                correct = (gt_mask == pred_mask).sum() / gt_mask.numel()
                # Precision, Recall, F1
                tp = (gt_mask & pred_mask).sum()
                fp = (~gt_mask & pred_mask).sum()
                fn = (gt_mask & ~pred_mask).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                layer_accuracies.append(correct.item())
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
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    
    return {
        "accuracy": total_accuracy / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches
    }


def train_predictors(
    model: torch.nn.Module,
    config: LlamaSkipConnectionConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int,
    num_samples: int,
    learning_rate: float,
    device: torch.device,
    batch_size: int,
    save_dir: str,
    eval_steps: int = 500,
    save_steps: int = 1000,
    use_wandb: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> None:
    """Train the sparsity predictors using PyTorch's built-in training mode with DDP support."""
    
    # Setup training mode using PyTorch's built-in methods
    model.train()  # Enable training mode
    
    # Get actual model (unwrap DDP if needed) for accessing custom methods
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.freeze_non_predictor_parameters()
    
    # Setup optimizer - only get parameters from the actual model
    predictor_params = actual_model.get_predictor_parameters()
    optimizer = torch.optim.AdamW(predictor_params, lr=learning_rate, weight_decay=0.01)
    
    # Calculate steps per epoch based on world size
    num_steps = num_samples // (batch_size * world_size)
    
    # Setup distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,  # Reduced for distributed training
        pin_memory=True,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Setup scheduler
    total_steps = num_steps * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    best_f1 = 0.0
    
    if rank == 0:
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Steps per epoch: {num_steps}")
        logger.info(f"Effective batch size: {batch_size * world_size}")
        logger.info(f"Number of predictor parameters: {sum(p.numel() for p in predictor_params)}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_predictor_loss = 0.0
        num_batches = 0
        
        # Set epoch for distributed samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Progress bar only on rank 0
        progress_bar = None
        if rank == 0:
            progress_bar = tqdm(total=num_steps, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in train_dataloader:
            if global_step > num_steps:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.stack(outputs.loss).mean()
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
            
            # Update progress bar only on rank 0
            if rank == 0 and progress_bar is not None:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Evaluation (only on rank 0)
            if rank == 0 and global_step % eval_steps == 0:
                logger.info(f"Evaluating at step {global_step}")
                eval_metrics = evaluate_predictor_accuracy(model, val_dataloader, device, rank=rank)
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
                        # Save the actual model state dict, not the DDP wrapper
                        actual_model.save_pretrained(best_model_path, config=config)
                        logger.info(f"Saved best model with F1: {best_f1:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to save best model: {e}")
            
            # Save checkpoint (only on rank 0)
            if rank == 0 and global_step % save_steps == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                actual_model.save_pretrained(checkpoint_path, config=config)
                logger.info(f"Saved checkpoint at step {global_step}")
            
            # Log training metrics (only on rank 0)
            if rank == 0 and use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "step": global_step,
                } | {f"gradients/layer_{i}": p.grad.norm().item() for i, p in enumerate(predictor_params)})
            
            if rank == 0 and progress_bar is not None:
                progress_bar.update(1)

        # End of epoch logging (only on rank 0)
        if rank == 0:
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "epoch": epoch + 1
                })
            
            if progress_bar is not None:
                progress_bar.close()
    
    # Final save (only on rank 0)
    if rank == 0:
        final_model_path = os.path.join(save_dir, "final_predictors")
        actual_model.save_pretrained(final_model_path, config=config)
        logger.info("Training completed!")


def main_worker(rank: int, world_size: int, args):
    """Main worker function for distributed training."""
    
    # Setup distributed training
    if args.use_ddp:
        ddp_setup(rank, world_size)
    
    # Set seed
    set_seed(args.seed + rank)  # Different seed per rank
    
    # Setup device
    if args.use_ddp:
        device = torch.device(f"cuda:{rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")
    
    # Setup output directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb (only on rank 0)
    if args.use_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"predictor-training-ddp-{int(time.time())}"
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
    if rank == 0:
        logger.info("Loading model...")
    model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config)
    
    # Initialize meta tensors
    for module in model.modules():
        if any(hasattr(p, 'is_meta') and p.is_meta for p in module.parameters()) and isinstance(module, FastLoRAProjection):
            module = module.to_empty(device="cpu")
            with torch.no_grad():
                torch.nn.init.xavier_normal_(module.down.weight)
                torch.nn.init.zeros_(module.up.weight)  # Initialize up projection to zeros for stable training
                
    model.tie_weights()
    model = model.to(device)
    
    # Wrap model with DDP
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Load training data
    train_dataset, val_dataset = load_training_data(
        tokenizer,
        args.dataset, 
        args.dataset_config,
        args.max_length,
        args.num_samples,
        rank=rank,
        world_size=world_size
    )
    
    # Train predictors
    train_predictors(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        learning_rate=args.learning_rate,
        device=device,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        rank=rank,
        world_size=world_size
    )
    
    # Cleanup
    if args.use_wandb and rank == 0:
        wandb.finish()
    
    if args.use_ddp:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train sparsity predictors for LlamaSkipConnection with DDP support")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--dataset", type=str, default="allenai/c4", help="Dataset name (default: allenai/c4)")
    parser.add_argument("--dataset_config", type=str, default="realnewslike", 
                       help="Dataset configuration (default: realnewslike for C4)")
    parser.add_argument("--num_samples", type=int, default=13800000, 
                       help="Number of training samples (default: 13800000)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=24, help="Per-GPU training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-skip-predictors", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="llama-skip-predictors", help="W&B entity name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--use_ddp", action="store_true", help="Use Distributed Data Parallel")
    
    args = parser.parse_args()
    
    if args.use_ddp:
        # Multi-GPU training with DDP
        world_size = torch.cuda.device_count()
        if world_size < 2:
            logger.warning("DDP requested but only 1 GPU available. Falling back to single GPU training.")
            args.use_ddp = False
            main_worker(0, 1, args)
        else:
            logger.info(f"Starting DDP training on {world_size} GPUs")
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU training
        main_worker(0, 1, args)


if __name__ == "__main__":
    main() 