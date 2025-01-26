# %%
import os
os.environ['MAX_JOBS'] = '16' 
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig
from transformers.models.llama import LlamaForCausalLM
import gc
import time

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Register the custom model and config
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

# Load base model and tokenizer
model_id = "vkkhare/llama-skip"
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

# Set padding token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token

# Create custom config and model
config = LlamaSkipConnectionConfig.from_pretrained(model_id)

# Load model without device_map
model = LlamaSkipConnectionForCausalLM.from_pretrained(
    checkpoint, 
    config=config,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

# Move all masks to the correct device
for module in model.modules():
    if hasattr(module, 'mask'):
        module.mask = module.mask.to(device)

model.eval()

# Generate text
sequence = "Give recipe of burrito including all the ingredients and their quantity."
inputs = tokenizer(
    sequence, 
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Explicitly move all input tensors to the same device as model
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Debug prints
print(f"Model device: {next(model.parameters()).device}")
print(f"Input IDs device: {input_ids.device}")
print(f"Attention Mask device: {attention_mask.device}")


llamaSkipPipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    device=device,
    eos_token_id=tokenizer.eos_token_id
)
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# with torch.no_grad():
#     outputs = llamaSkipPipe.model.generate(
#         input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=5,
#         temperature=0.7,
#         top_p=0.9,
#         num_return_sequences=1,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id
#     )
    
    
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("SkipLlama", generated_text)


llamaPipe = pipeline(
    "text-generation",
    device=device,
    model=checkpoint,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    eos_token_id=tokenizer.eos_token_id
)

llamaPipe.model.to(torch.float32)

# with torch.no_grad():
#     outputs = llamaPipe.model.generate(
#         input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=10,
#         temperature=0.7,
#         top_p=0.9,
#         num_return_sequences=1,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id
#     )
    
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("standard Llama ", generated_text)


def run_inference(model, input_ids, attention_mask, tokenizer, num_runs=5):
    model = model.cpu()
    base_input_ids = input_ids.cpu()
    base_attention_mask = attention_mask.cpu()
    
    print(f"\nModel type: {type(model)}")
    print(f"Model config: {model.config}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    times = []
    mlp_times = {i: [] for i in range(16)}
    
    # Store original forward methods
    original_forwards = {}
    
    def wrap_forward(module, layer_idx):
        original_forward = module.forward
        
        def timed_forward(*args, **kwargs):
            start = time.perf_counter()
            result = original_forward(*args, **kwargs)
            end = time.perf_counter()
            mlp_times[layer_idx].append(end - start)
            return result
            
        return timed_forward
    
    # Register wrapped forwards
    if isinstance(model, LlamaSkipConnectionForCausalLM):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp'):
                original_forwards[i] = layer.mlp.forward
                layer.mlp.forward = wrap_forward(layer.mlp, i)
    elif isinstance(model, LlamaForCausalLM):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp'):
                original_forwards[i] = layer.mlp.forward
                layer.mlp.forward = wrap_forward(layer.mlp, i)
    
    print(f"Wrapped {len(original_forwards)} MLP forwards")
    
    try:
        for i in range(num_runs):
            torch.cuda.empty_cache()
            gc.collect()
            
            # Randomize input for each run
            random_shift = torch.randint(-100, 100, base_input_ids.shape)
            input_ids = torch.clamp(base_input_ids + random_shift, min=0, max=tokenizer.vocab_size-1)
            attention_mask = base_attention_mask  # Keep attention mask same
            
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None
            model._past_length = 0
            model.config.use_cache = False
            
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                    return_dict_in_generate=False
                )
                
            end = time.perf_counter()
            times.append(end - start)
            
            print(f"Run {i} output length: {len(outputs[0])}")
    finally:
        # Restore original forwards
        if isinstance(model, LlamaSkipConnectionForCausalLM):
            for i, layer in enumerate(model.model.layers):
                if i in original_forwards:
                    layer.mlp.forward = original_forwards[i]
        elif isinstance(model, LlamaForCausalLM):
            for i, layer in enumerate(model.model.layers):
                if i in original_forwards:
                    layer.mlp.forward = original_forwards[i]
    
    # Print MLP timing statistics
    if mlp_times[0]:
        print("\nMLP timing statistics:")
        for layer_idx, timings in mlp_times.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                min_time = min(timings)
                max_time = max(timings)
                print(f"Layer {layer_idx} MLP: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
    
    return times

print("Running CPU inference benchmarks...")
print("-" * 50)

# Warm up runs
print("Warming up models...")
_ = run_inference(llamaSkipPipe.model, input_ids, attention_mask, tokenizer, num_runs=15)
_ = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer, num_runs=15)

# Actual benchmarks
skip_times = run_inference(llamaSkipPipe.model, input_ids, attention_mask, tokenizer)
std_times = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer)

print("\nSkipLLaMA CPU Results:")
print(f"Average time: {sum(skip_times)/len(skip_times):.3f}s")
print(f"Min time: {min(skip_times):.3f}s")
print(f"Max time: {max(skip_times):.3f}s")
print(f"Individual times: {[f'{t:.3f}s' for t in skip_times]}")

print("\nStandard LLaMA CPU Results:")
print(f"Average time: {sum(std_times)/len(std_times):.3f}s")
print(f"Min time: {min(std_times):.3f}s")
print(f"Max time: {max(std_times):.3f}s")
print(f"Individual times: {[f'{t:.3f}s' for t in std_times]}")

print("\nCPU Speedup:")
print(f"Average speedup: {(sum(std_times)/len(std_times))/(sum(skip_times)/len(skip_times)):.2f}x")