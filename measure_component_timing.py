import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.models.llama.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipConnectionConfig

def run_and_measure_components(model, tokenizer, prompt, num_tokens=50, device='cpu'):
    """Run model and extract component timing statistics."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    model.eval()
    
    # Clear any existing timing stats
    for layer in model.model.layers:
        if hasattr(layer, 'timing_stats'):
            layer.timing_stats = {
                'lora_projection': [],
                'quantile_computation': [],
                'mask_creation': [],
                'weight_cache_update': [],
                'total_sparsity': [],
                'mlp_forward': []
            }
    
    with torch.no_grad():
        # Generate tokens
        for _ in range(num_tokens):
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    # Collect timing stats from all layers
    all_stats = {
        'lora_projection': [],
        'quantile_computation': [],
        'mask_creation': [],
        'weight_cache_update': [],
        'total_sparsity': [],
        'mlp_forward': []
    }
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'timing_stats'):
            for key in all_stats:
                all_stats[key].extend(layer.timing_stats[key])
    
    return all_stats

def analyze_timing(stats):
    """Analyze and display timing statistics."""
    print("\n=== Exact Component Timing (ms) ===\n")
    
    # Sparsity selection components
    sparsity_components = ['lora_projection', 'quantile_computation', 'mask_creation', 'weight_cache_update']
    
    print("Sparsity Selection Components:")
    print("-" * 50)
    
    total_per_component = {}
    for comp in sparsity_components:
        if comp in stats and len(stats[comp]) > 0:
            times = stats[comp]
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            total_per_component[comp] = mean_time
            
            print(f"{comp:25s}: {mean_time:6.2f} ± {std_time:4.2f} ms")
            print(f"{'':25s}  (min: {min_time:6.2f}, max: {max_time:6.2f})")
    
    # Total sparsity overhead
    total_sparsity_overhead = sum(total_per_component.values())
    print(f"\n{'Total Sparsity Overhead':25s}: {total_sparsity_overhead:6.2f} ms\n")
    
    # MLP Forward timing
    if 'mlp_forward' in stats and len(stats['mlp_forward']) > 0:
        print("MLP Execution:")
        print("-" * 50)
        mlp_times = stats['mlp_forward']
        mlp_mean = np.mean(mlp_times)
        mlp_std = np.std(mlp_times)
        mlp_min = np.min(mlp_times)
        mlp_max = np.max(mlp_times)
        
        print(f"{'Sparse MLP Forward':25s}: {mlp_mean:6.2f} ± {mlp_std:4.2f} ms")
        print(f"{'':25s}  (min: {mlp_min:6.2f}, max: {mlp_max:6.2f})")
    
    # Total time per token
    print(f"\n{'Total Time Breakdown':25s}:")
    print("-" * 50)
    print(f"{'Sparsity Selection':25s}: {total_sparsity_overhead:6.2f} ms")
    if 'mlp_forward' in stats and len(stats['mlp_forward']) > 0:
        print(f"{'MLP Computation':25s}: {mlp_mean:6.2f} ms")
        print(f"{'Total MLP Time':25s}: {total_sparsity_overhead + mlp_mean:6.2f} ms")
    
    # Percentage breakdown
    print("\nPercentage Breakdown (Sparsity Components):")
    print("-" * 50)
    for comp, time in total_per_component.items():
        percentage = (time / total_sparsity_overhead) * 100
        print(f"{comp:25s}: {percentage:5.1f}%")
    
    # Compare with measured total
    if 'total_sparsity' in stats and len(stats['total_sparsity']) > 0:
        measured_total = np.mean(stats['total_sparsity'])
        print(f"\n{'Measured Sparsity Total':25s}: {measured_total:6.2f} ms")
        print(f"{'Sum of Components':25s}: {total_sparsity_overhead:6.2f} ms")
        print(f"{'Overhead (sync, etc)':25s}: {measured_total - total_sparsity_overhead:6.2f} ms")

def main():
    """Main function to measure component timing."""
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load config and model
    config = LlamaSkipConnectionConfig.from_json_file("configs/llama_skip_causal_3b.json")
    checkpoint = config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading SkipLLaMA model...")
    AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
    AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
    
    skip_model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config).to(device)
    skip_model.eval()
    
    prompt = "The future of artificial intelligence is"
    print(f"\nGenerating 50 tokens from prompt: '{prompt}'\n")
    
    # Warmup
    print("Warming up...")
    _ = run_and_measure_components(skip_model, tokenizer, prompt, num_tokens=5, device=device)
    
    # Measure
    print("Measuring component timing...")
    stats = run_and_measure_components(skip_model, tokenizer, prompt, num_tokens=50, device=device)
    
    # Analyze
    analyze_timing(stats)
    
    # Per-layer analysis
    print("\n=== Per-Layer Analysis ===")
    print(f"Model has {len(skip_model.model.layers)} layers")
    print(f"Each token goes through all {len(skip_model.model.layers)} layers")
    
    if stats['lora_projection']:
        total_calls = len(stats['lora_projection'])
        tokens_generated = total_calls // len(skip_model.model.layers)
        print(f"Tokens generated: {tokens_generated}")
        print(f"Total sparsity selections: {total_calls}")
        
        # Calculate per-layer average
        per_layer_avg = np.mean(stats['total_sparsity']) / len(skip_model.model.layers)
        print(f"\nAverage time per layer: {per_layer_avg:.2f} ms")
        print(f"Total time for all layers: {np.mean(stats['total_sparsity']):.2f} ms")

if __name__ == "__main__":
    main() 