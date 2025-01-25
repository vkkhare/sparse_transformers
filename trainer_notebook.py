# %%
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig
from transformers.models.llama import LlamaForCausalLM

# Set device
device = torch.device("cpu")

# Register the custom model and config
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

try:
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

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
    

# # Update the pipeline to use the same device
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     max_new_tokens=1000,
#     eos_token_id=tokenizer.eos_token_id
# )


# Comment out unused code
# %%
# model.push_to_hub("vkkhare/llama-skip")
# config.push_to_hub("vkkhare/llama-skip")

# %%
# sequence = "In a hole in the ground there lived a hobbit."
# input= tokenizer(sequence, return_tensors='pt')
# print(model.eval())


# %%
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
# out = pipe.model.generate(input["input_ids"], max_length=20)
# print(tokenizer.decode(out[0]))