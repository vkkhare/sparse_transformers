import logging
import os
import csv
import numpy as np
from typing import Dict, Optional, Any


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers.optimization import get_cosine_schedule_with_warmup


import wandb
from tqdm import tqdm

from src.modeling_skip import FastLoRAProjection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkCache:
    """Simple LRU cache for .npz chunk files."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Get cached chunk data."""
        if file_path in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
            return self.cache[file_path]
        return None
    
    def put(self, file_path: str, data: Dict[str, np.ndarray]):
        """Cache chunk data with LRU eviction."""
        if file_path in self.cache:
            # Update existing entry
            self.cache[file_path] = data
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[file_path] = data
            self.access_order.append(file_path)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global chunk cache instance
_chunk_cache = ChunkCache(max_size=50)


def load_array_from_chunk(arrays_dir: str, batch_filename: str, batch_index: int, array_name: str) -> np.ndarray:
    """Load a specific array from a batch .npz file with caching."""
    file_path = os.path.join(arrays_dir, batch_filename)
    
    # Check cache first
    cached_data = _chunk_cache.get(file_path)
    if cached_data is not None:
        return cached_data[array_name][batch_index]
    
    # Load from disk and cache
    with np.load(file_path, allow_pickle=False) as data:
        # Convert to dict for caching (since npz files can't be cached directly)
        cached_data = {key: data[key] for key in data.files}
        _chunk_cache.put(file_path, cached_data)
        return cached_data[array_name][batch_index]


def get_sample_by_index(output_dir: str, index: int) -> Optional[Dict]:
    """Load a specific sample by index without loading the full dataset."""
    try:
        csv_file = os.path.join(output_dir, "dataset.csv")
        if not os.path.exists(csv_file):
            return None
        
        arrays_dir = os.path.join(output_dir, "arrays")
        
        # Find the row at the specified index
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for current_idx, row in enumerate(reader):
                if current_idx == index:
                    batch_filename = row['batch_file']
                    batch_index = int(row['batch_index'])
                    
                    # Load the arrays for this sample
                    sample = {
                        'text': row['text'],
                        'input_ids': load_array_from_chunk(arrays_dir, batch_filename, batch_index, 'input_ids'),
                        'attention_mask': load_array_from_chunk(arrays_dir, batch_filename, batch_index, 'attention_mask'),
                    }
                    
                    # Load layer arrays
                    for col in row.keys():
                        if col.startswith('layer_') and col.endswith('_available') and row[col].lower() == 'true':
                            layer_idx = int(col.split('_')[1])
                            sample[f'hidden_states_layer_{layer_idx}'] = load_array_from_chunk(
                                arrays_dir, batch_filename, batch_index, f'hidden_states_layer_{layer_idx}'
                            )
                            sample[f'mlp_activations_layer_{layer_idx}'] = load_array_from_chunk(
                                arrays_dir, batch_filename, batch_index, f'mlp_activations_layer_{layer_idx}'
                            )
                    
                    return sample
        
        return None  # Index out of range
        
    except Exception as e:
        logger.error(f"Error loading sample {index}: {e}")
        return None


class StreamingSparsityDataset(TorchDataset):
    """Streaming dataset that loads data on-demand from CSV and .npz files with caching."""
    
    def __init__(self, output_dir: str, layer_idx: int, cache_size: int = 50, load_full_dataset: bool = False):
        """
        Args:
            output_dir: Directory containing dataset.csv and arrays/ folder
            layer_idx: Which layer to load data for
            cache_size: Maximum number of .npz files to cache in memory
            load_full_dataset: If True, load all data into memory at initialization
        """
        self.output_dir = output_dir
        self.layer_idx = layer_idx
        self.csv_file = os.path.join(output_dir, "dataset.csv")
        self.arrays_dir = os.path.join(output_dir, "arrays")
        self.load_full_dataset = load_full_dataset
        self.full_data = None  # Will store all data if load_full_dataset is True
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        if not os.path.exists(self.arrays_dir):
            raise FileNotFoundError(f"Arrays directory not found: {self.arrays_dir}")
        
        # Configure cache size
        global _chunk_cache
        _chunk_cache = ChunkCache(max_size=cache_size)
        
        # Read CSV metadata once and store sample info
        self.samples = []
        with open(self.csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include samples that have data for our layer
                layer_key = f"layer_{layer_idx}_available"
                if layer_key in row and row[layer_key].lower() == 'true':
                    self.samples.append({
                        'text': row['text'],
                        'batch_file': row['batch_file'],
                        'batch_index': int(row['batch_index'])
                    })
        
        logger.info(f"Streaming dataset loaded with {len(self.samples)} samples for layer {layer_idx}")
        
        if self.load_full_dataset:
            logger.info("Loading full dataset into memory...")
            self._load_full_data()
            logger.info("Full dataset loaded into memory")
        else:
            logger.info(f"Chunk cache configured with max size: {cache_size}")
    
    def _load_full_data(self):
        """Load all data into memory."""
        self.full_data = []
        
        logger.info(f"Loading {len(self.samples)} samples into memory...")
        for i, sample_info in enumerate(tqdm(self.samples, desc="Loading samples")):
            batch_filename = sample_info['batch_file']
            batch_index = sample_info['batch_index']
            
            # Load arrays
            hidden_states = load_array_from_chunk(
                self.arrays_dir, batch_filename, batch_index, f'hidden_states_layer_{self.layer_idx}'
            )
            mlp_activations = load_array_from_chunk(
                self.arrays_dir, batch_filename, batch_index, f'mlp_activations_layer_{self.layer_idx}'
            )
            
            # Store as tensors for faster access
            self.full_data.append({
                'text': sample_info['text'],
                'hidden_states': torch.from_numpy(hidden_states).float(),
                'mlp_activations': torch.from_numpy(mlp_activations).float()
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample, either from memory or on-demand."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        # Return from memory if full dataset is loaded
        if self.load_full_dataset and self.full_data is not None:
            return self.full_data[idx]
        
        # Otherwise, load on-demand (with caching)
        sample_info = self.samples[idx]
        batch_filename = sample_info['batch_file']
        batch_index = sample_info['batch_index']
        
        # Load arrays on-demand (with caching)
        hidden_states = load_array_from_chunk(
            self.arrays_dir, batch_filename, batch_index, f'hidden_states_layer_{self.layer_idx}'
        )
        mlp_activations = load_array_from_chunk(
            self.arrays_dir, batch_filename, batch_index, f'mlp_activations_layer_{self.layer_idx}'
        )
        
        return {
            'text': sample_info['text'],
            'hidden_states': torch.from_numpy(hidden_states).float(),
            'mlp_activations': torch.from_numpy(mlp_activations).float()
        }
    
    def clear_cache(self):
        """Clear the chunk cache or full dataset from memory."""
        if self.load_full_dataset and self.full_data is not None:
            self.full_data = None
            logger.info("Cleared full dataset from memory")
        else:
            global _chunk_cache
            _chunk_cache.clear()
            logger.info("Cleared chunk cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.load_full_dataset:
            return {
                "full_dataset_loaded": self.full_data is not None,
                "total_samples_in_memory": len(self.full_data) if self.full_data else 0,
                "cache_type": "full_dataset"
            }
        else:
            global _chunk_cache
            return {
                "cache_size": _chunk_cache.size(),
                "max_cache_size": _chunk_cache.max_size,
                "cache_type": "chunk_cache"
            }


class LayerwisePredictorTrainer:
    """Trainer for layer-wise predictors."""
    
    def __init__(self, layer_idx: int, hidden_size: int, intermediate_size: int, lora_size: int, device: torch.device):
        self.device = device
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Initialize predictors for each layer
        self.predictor = FastLoRAProjection(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                lora_size=lora_size
            ).to(device)
        # self.predictor._fix_unloaded_weights()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3,7]))
    
    def compute_loss(self, 
                    hidden_states: torch.Tensor,
                    mlp_activations: torch.Tensor) -> torch.Tensor:
        """Compute predictor loss."""
        # Get predictor scores
        pred_scores = self.predictor(hidden_states)  # [batch_size, intermediate_size]
        gt_mask = (mlp_activations > 0).float()
        loss = self.loss_fn(pred_scores, gt_mask)
        return loss
    
    def evaluate_predictor(self, 
                          dataloader: DataLoader,
                          max_batches: int = 50) -> Dict[str, float]:
        """Evaluate predictor performance."""
        self.predictor.eval()
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        num_batches = 0
        total_accuracy = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                hidden_states = batch["hidden_states"].to(self.device)
                mlp_activations = batch["mlp_activations"].to(self.device)
                
                # Get predictions
                pred_scores = self.predictor(hidden_states)
                pred_mask = (F.sigmoid(pred_scores) > 0.5)
                
                # Get ground truth
                gt_mask = (mlp_activations > 0)
                tp = (pred_mask * gt_mask).sum().item()
                fp = (pred_mask * (~gt_mask)).sum().item()
                fn = ((~pred_mask) * gt_mask).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_accuracy += (pred_mask == gt_mask).sum().item()/ pred_mask.numel()
                num_batches += hidden_states.shape[0]
        
        self.predictor.train()
        
        if num_batches == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        return {
            "accuracy": total_accuracy / num_batches,
            "precision": total_precision / num_batches,
            "recall": total_recall / num_batches,
            "f1": total_f1 / num_batches
        }
    
    def train_layer(self, train_dataset: TorchDataset,
                   val_dataset: TorchDataset, num_epochs: int,
                   batch_size: int, learning_rate: float,
                   use_wandb: bool = False, save_dir: Optional[str] = None,
                   save_interval: int = 1000, resume_from_checkpoint: bool = False,
                   checkpoint_path: Optional[str] = None) -> FastLoRAProjection:
        """Train a single layer's predictor.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            use_wandb: Whether to log to Weights & Biases
            save_dir: Directory to save checkpoints and best model (optional)
            save_interval: Save checkpoint every N steps
            resume_from_checkpoint: Whether to resume from latest checkpoint
            checkpoint_path: Specific checkpoint path to resume from (optional)
        """
        
        logger.info(f"Training predictor for layer {self.layer_idx}")
        
        predictor = self.predictor
        predictor.train()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
            num_cycles=10
        )
        
        best_f1 = 0.0
        global_step = 0
        start_epoch = 0
        
        # Handle checkpoint resuming
        if resume_from_checkpoint:
            resume_checkpoint_path = checkpoint_path
            if not resume_checkpoint_path and save_dir:
                resume_checkpoint_path = self.find_latest_checkpoint(save_dir)
            
            if resume_checkpoint_path:
                try:
                    checkpoint_data = self.load_checkpoint(resume_checkpoint_path, optimizer, scheduler)
                    global_step = checkpoint_data['global_step']
                    start_epoch = checkpoint_data['epoch']
                    best_f1 = checkpoint_data['best_f1']
                    logger.info(f"Resumed training from step {global_step}, epoch {start_epoch}, best_f1: {best_f1:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}. Starting fresh training.")
                    global_step = 0
                    start_epoch = 0
                    best_f1 = 0.0
            else:
                logger.info("No checkpoint found. Starting fresh training.")
        
        # Calculate total steps for progress bar
        total_steps = len(train_loader) * num_epochs
        progress_bar = tqdm(total=total_steps, desc=f"Training Layer {self.layer_idx}")
        progress_bar.update(global_step)  # Update progress bar to current position
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                hidden_states = batch["hidden_states"].to(self.device)
                mlp_activations = batch["mlp_activations"].to(self.device)
                
                # Compute loss
                loss = self.compute_loss(hidden_states, mlp_activations)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'epoch': f"{epoch+1}/{num_epochs}",
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'step': f"{global_step}/{total_steps}"
                })
                
                # Log to wandb
                if use_wandb and global_step % 50 == 0:
                    # Compute gradient norm safely
                    grad_norms = [p.grad.norm() for p in predictor.parameters() if p.grad is not None]
                    grad_norm = torch.norm(torch.stack(grad_norms)) if grad_norms else 0.0
                    
                    wandb.log({
                        f"layer_{self.layer_idx}/train_loss": loss.item(),
                        f"layer_{self.layer_idx}/learning_rate": scheduler.get_last_lr()[0],
                        f"layer_{self.layer_idx}/gradient_norm": grad_norm,
                        "step": global_step
                    })
                
                # Save checkpoint at regular intervals
                if save_dir and global_step % save_interval == 0:
                    self.save_checkpoint(save_dir, global_step, epoch, optimizer, scheduler, best_f1, loss.item())
            
            # Evaluation
            eval_metrics = self.evaluate_predictor(val_loader)
            
            if use_wandb:
                wandb.log({
                    f"layer_{self.layer_idx}/eval_accuracy": eval_metrics["accuracy"],
                    f"layer_{self.layer_idx}/eval_precision": eval_metrics["precision"],
                    f"layer_{self.layer_idx}/eval_recall": eval_metrics["recall"],
                    f"layer_{self.layer_idx}/eval_f1": eval_metrics["f1"],
                    "epoch": epoch + 1
                })
            
            # Save best model
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]                
                # Save the best model immediately
                if save_dir:
                    best_model_name = f"best_predictor_layer_{self.layer_idx}"
                    self.save_predictor(save_dir, name=best_model_name)
                    logger.info(f"Saved new best model: {best_model_name}")
                    
                    # Also log to wandb if enabled
                    if use_wandb:
                        wandb.log({
                            f"layer_{self.layer_idx}/best_f1": best_f1,
                            "epoch": epoch + 1
                        })
        
        # Close progress bar
        progress_bar.close()
        
        return self.predictor  # type: ignore
    
    def save_predictor(self, save_dir: str, name: str = "predictor"):
        """Save all trained predictors."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictor
        torch.save(self.predictor.state_dict(), 
                      os.path.join(save_dir, f"{name}.pt"))
        
        logger.info(f"Saved predictors to {save_dir}")

    def save_checkpoint(self, save_dir: str, global_step: int, epoch: int, 
                       optimizer: torch.optim.Optimizer, scheduler, 
                       best_f1: float, loss: float):
        """Save training checkpoint with full state."""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'global_step': global_step,
            'epoch': epoch,
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'loss': loss,
            'layer_idx': self.layer_idx,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
        }
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_layer_{self.layer_idx}_step_{global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(save_dir, f"latest_checkpoint_layer_{self.layer_idx}.pt")
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, optimizer: torch.optim.Optimizer, scheduler):
        """Load training checkpoint and restore full state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load predictor state
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        
        # Load optimizer and scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return checkpoint metadata
        return {
            'global_step': checkpoint['global_step'],
            'epoch': checkpoint['epoch'],
            'best_f1': checkpoint['best_f1'],
            'loss': checkpoint['loss']
        }

    def find_latest_checkpoint(self, save_dir: str) -> Optional[str]:
        """Find the latest checkpoint file in the save directory."""
        if not os.path.exists(save_dir):
            return None
            
        # First try the latest checkpoint file
        latest_path = os.path.join(save_dir, f"latest_checkpoint_layer_{self.layer_idx}.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        # Otherwise, find the checkpoint with the highest step number
        import glob
        pattern = os.path.join(save_dir, f"checkpoint_layer_{self.layer_idx}_step_*.pt")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
        
        # Extract step numbers and find the latest
        latest_step = -1
        latest_checkpoint = None
        
        for checkpoint_path in checkpoints:
            try:
                # Extract step number from filename
                filename = os.path.basename(checkpoint_path)
                step_str = filename.split('_step_')[1].split('.')[0]
                step = int(step_str)
                
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = checkpoint_path
                    
            except (IndexError, ValueError):
                continue
        
        return latest_checkpoint

    def load_predictor(self, save_dir: str, name: str = "predictor"):
        """Load a trained predictor from file."""
        predictor_path = os.path.join(save_dir, f"{name}.pt")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        
        logger.info(f"Loading predictor from {predictor_path}")
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        logger.info(f"Successfully loaded predictor from {predictor_path}")

