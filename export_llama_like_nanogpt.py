# export_nanogpt.py

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from executorch.exir import EdgeCompileConfig, to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.export import export, export_for_training
from executorch.runtime import Runtime, Verification



# TORCH_LOGS="+dynamic"
# checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
# # AutoConfig.register("llama2", LlamaConfig2)
# # AutoModel.register(LlamaConfig2, LlamaForCausalLM)
# # AutoModelForCausalLM.register(LlamaConfig2, LlamaForCausalLM)
# # llamaSkipConfig = LlamaConfig.from_json_file("./configs/llama_skip_causal.json")
# llamaSkipModel = LlamaForCausalLM.from_pretrained(checkpoint)

# print(llamaSkipModel.config)
# # Load the model.
# model = llamaSkipModel

# # Create example inputs. This is used in the export process to provide
# # hints on the expected shape of the model input.
example_inputs = (torch.randint(0, 100, (1, 10), dtype=torch.long), )

# # Set up dynamic shape configuration. This allows the sizes of the input tensors
# # to differ from the sizes of the tensors in `example_inputs` during runtime, as
# # long as they adhere to the rules specified in the dynamic shape configuration.
# # Here we set the range of 0th model input's 1st dimension as
# # [0, model.config.block_size].
# # See https://pytorch.org/executorch/main/concepts.html#dynamic-shapes
# # for details about creating dynamic shapes.
# dynamic_shape = (
#     {1: torch.export.Dim("token_dim", max=1024)},
# )

# # Trace the model, converting it to a portable intermediate representation.
# # The torch.no_grad() call tells PyTorch to exclude training-specific logic.
# with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
#     m = export_for_training(model, example_inputs, dynamic_shapes=dynamic_shape).module()
#     traced_model = export(model.eval(), example_inputs, dynamic_shapes=dynamic_shape)

# # Convert the model into a runnable ExecuTorch program.
# edge_config = EdgeCompileConfig(_check_ir_validity=False)
# edge_manager = to_edge(traced_model,  compile_config=edge_config)
# et_program = edge_manager.to_executorch()


pte_path= "llama3_like_nano.pte"
# Save the ExecuTorch program to a file.
# with open(pte_path, "wb") as file:
#     file.write(et_program.buffer)


runtime = Runtime.get()

program = runtime.load_program(pte_path)
print("load_program successfull")
print("Program methods:", program.method_names)

method = program.load_method("forward")
print("load_method done")

# Breaking here with 
# F 00:02:14.997224 executorch:pybindings.cpp:749] In function run_method(), assert failed (false): Execution should not reach this point. <class 'transformers.tokenization_utils_base.BatchEncoding'>
# Aborted (core dumped)
inputs = (torch.ones(2, 2), torch.ones(2, 2))
output = method.execute(example_inputs)
print("method.execute done")