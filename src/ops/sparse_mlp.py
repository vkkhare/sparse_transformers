import torch
import sparse_mlp_cuda

class SparseMlpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gate_weight, up_weight, down_weight, mask, activation_fn):
        output = sparse_mlp_cuda.sparse_mlp_forward(
            input, gate_weight, up_weight, down_weight, mask, activation_fn
        )
        ctx.save_for_backward(input, gate_weight, up_weight, down_weight, mask)
        ctx.activation_fn = activation_fn
        return output

    @staticmethod
    def backward(ctx):
        # Implement backward pass if needed
        raise NotImplementedError("Backward pass not implemented yet")

def sparse_mlp_forward(input, gate_weight, up_weight, down_weight, mask, activation_fn):
    return SparseMlpFunction.apply(input, gate_weight, up_weight, down_weight, mask, activation_fn) 