// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For custom operator registration
#include <torch/library.h>

// For tensor options
#include <c10/core/TensorOptions.h>

// Standard C++ headers
#include <vector>

// OpenMP for parallelization
#include <omp.h>

// Forward declaration with proper types
torch::Tensor sparse_mlp_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor mask,
    std::string act_fn_name) {
    
    // Get dimensions
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
    auto down_proj = torch::empty({batch_size, hidden_size}, options);
    
    // Enable nested parallelism
    omp_set_nested(1);
    int max_threads = omp_get_max_threads();
    int outer_threads = std::min(max_threads / 2, static_cast<int>(batch_size));
    int inner_threads = max_threads / outer_threads;
    
    // Process batches in parallel
    #pragma omp parallel num_threads(outer_threads)
    {
        #pragma omp for
        for (int64_t i = 0; i < batch_size; i++) {
            auto x_batch = x[i].view({1, hidden_size});
            auto mask_row = mask[i];
            
            // Get indices where mask is True
            auto active_indices = mask_row.nonzero().squeeze();
            
            // Select active rows from weights
            auto active_gate_weight = gate_weight.index_select(0, active_indices);
            auto active_up_weight = up_weight.index_select(0, active_indices);
            auto active_down_weight = down_weight.index_select(1, active_indices).transpose(0, 1);
            
            // Pre-allocate tensors with correct shapes
            auto gate_proj = torch::empty({1, active_indices.size(0)}, options);
            auto up_proj = torch::empty({1, active_indices.size(0)}, options);
            auto out_proj = torch::empty({1, hidden_size}, options);
            
            // Compute projections in parallel
            #pragma omp parallel num_threads(inner_threads)
            {
                #pragma omp sections
                {
                    #pragma omp section
                    {
                        torch::matmul_out(gate_proj, x_batch, active_gate_weight.t());
                    }
                    #pragma omp section
                    {
                        torch::matmul_out(up_proj, x_batch, active_up_weight.t());
                    }
                }
            }
            // Compute activation and final projection (all operations keep batch dim)
            auto activated = gate_proj * torch::sigmoid(gate_proj);
            auto gate_act = activated * up_proj;

            torch::matmul_out(out_proj, gate_act, active_down_weight);
            
            #pragma omp critical
            {
                down_proj[i].copy_(out_proj[0].detach());  // Only one copy at the end
            }            
        }
    }
    omp_set_nested(0);
    std::cout << "out copy done for batch " << std::endl;

    return down_proj;
}

// Register the operator
TORCH_LIBRARY(sparse_mlp, m) {
    m.def("forward", sparse_mlp_forward);
}

// Make the function visible to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_mlp_forward, "Sparse MLP forward");
} 