// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's OpenMP wrapper
#include <ATen/ParallelOpenMP.h>

// For timing and logging
#include <chrono>
#include <iomanip>
#include <ctime>

// Helper function for timestamp
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M:%S") 
       << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Forward declaration with proper types
torch::Tensor sparse_mlp_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor mask,
    std::string act_fn_name) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
    auto down_proj = torch::empty({batch_size, hidden_size}, options);
    
    // Process batches in parallel
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            auto x_batch = x[i].view({1, hidden_size});
            auto mask_row = mask[i];
            
            // Get indices where mask is True
            auto active_indices = mask_row.nonzero().squeeze();
            
            // Select active rows from weights
            auto active_gate_weight = gate_weight.index_select(0, active_indices);
            auto active_up_weight = up_weight.index_select(0, active_indices);
            auto active_down_weight = down_weight.index_select(1, active_indices).transpose(0, 1);
            
            // Pre-allocate tensors with correct shapes and no grad
            auto gate_proj = torch::empty({1, active_indices.size(0)}, options).detach();
            auto up_proj = torch::empty({1, active_indices.size(0)}, options).detach();
            auto out_proj = torch::empty({1, hidden_size}, options).detach();
            

            // Use at::parallel_for for the two matmuls
            at::parallel_for(0, 2, 1, [&](int64_t start, int64_t end) {
                for (int64_t j = start; j < end; j++) {
                    if (j == 0) {
                        torch::matmul_out(gate_proj, x_batch.detach(), active_gate_weight.t().detach());
                    } else {
                        torch::matmul_out(up_proj, x_batch.detach(), active_up_weight.t().detach());
                    }
                }
            });
            
            auto activated = gate_proj * torch::sigmoid(gate_proj);
            auto gate_act = activated * up_proj;
            
            torch::matmul_out(out_proj, gate_act, active_down_weight);
            down_proj[i].copy_(out_proj[0].detach());
        }
    });
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