// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's OpenMP wrapper
#include <ATen/ParallelOpenMP.h>

// Add timing utilities
#include <chrono>
#include <unordered_map>

// Timing helper class
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void stop(const char* name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << name << ": " << duration / 1000.0 << "ms\n";
    }

    void restart() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Forward declaration with proper types
torch::Tensor sparse_mlp_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor mask,
    std::string act_fn_name) {
        
    Timer timer;
    
    // Get dimensions
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    // Output allocation
    timer.restart();
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x.device())
        .layout(torch::kStrided);
    auto down_proj = torch::empty({batch_size, hidden_size}, options);
    timer.stop("Output allocation");
    
    // Process batches in parallel
    timer.restart();
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        Timer batch_timer;
        for (int64_t i = start; i < end; i++) {
            // Input conversion
            batch_timer.restart();
            auto x_batch = x[i].view({1, hidden_size}).to(torch::kFloat32);
            auto mask_row = mask[i];
            batch_timer.stop("Input conversion");
            
            // Index computation
            batch_timer.restart();
            auto active_indices = mask_row.nonzero().squeeze();
            int64_t active_size = active_indices.size(0);
            auto gate_proj = torch::empty({1, active_size}, options).detach();
            auto up_proj = torch::empty({1, active_size}, options).detach();
            auto out_proj = torch::empty({1, hidden_size}, options).detach();
            batch_timer.stop("Index computation");
            
            // Weight selection
            batch_timer.restart();
            auto active_gate_weight = gate_weight.index_select(0, active_indices);
            auto active_up_weight = up_weight.index_select(0, active_indices);
            auto active_down_weight = down_weight.index_select(1, active_indices).transpose(0, 1);
            batch_timer.stop("Weight selection");
            
            // Matrix multiplications
            batch_timer.restart();
            at::parallel_for(0, 2, 1, [&](int64_t start, int64_t end) {
                for (int64_t j = start; j < end; j++) {
                    if (j == 0) {
                        torch::matmul_out(gate_proj, x_batch.detach(), active_gate_weight.t().detach());
                    } else {
                        torch::matmul_out(up_proj, x_batch.detach(), active_up_weight.t().detach());
                    }
                }
            });
            batch_timer.stop("Matrix multiplications");
            
            // Activation
            batch_timer.restart();
            auto activated = gate_proj * torch::sigmoid(gate_proj);
            auto gate_act = activated * up_proj;
            batch_timer.stop("Activation");
            
            // Final projection
            batch_timer.restart();
            torch::matmul_out(out_proj, gate_act, active_down_weight);
            down_proj[i].copy_(out_proj[0].detach());
            batch_timer.stop("Final projection");
        }
    });
    timer.stop("Total batch processing");
    
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