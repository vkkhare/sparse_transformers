# Import directly from the extension module
from torch.utils.cpp_extension import load

# Load the CUDA extension
_sparse_mlp = load(
    name="sparse_mlp",
    sources=["csrc/sparse_mlp_op.cpp", "csrc/sparse_mlp_cuda.cu"],
    extra_cflags=["-fopenmp"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-fopenmp"],
    verbose=True
)

# Export the function
sparse_mlp_forward = _sparse_mlp.sparse_mlp_forward

__all__ = ['sparse_mlp_forward'] 