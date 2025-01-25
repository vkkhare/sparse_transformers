#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <cmath>

// Constants for GELU approximation
constexpr double SQRT_2_PI = 2.506628274631000502415765284811045253006986740609938316629923576;
constexpr double SQRT_1_2 = 0.707106781186547524400844362104849039284835937688474036588339869;

namespace {
template <typename scalar_t>
void sparse_mlp_forward_cpu_impl(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn,
    torch::Tensor& output) {
    
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto intermediate_size = gate_weight.size(0);
    
    // Pre-allocate buffers once
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device())
        .requires_grad(false);
        
    auto max_masked_size = int(intermediate_size * 0.2);  // Based on sparsity
    std::vector<torch::Tensor> gate_proj_buffers(omp_get_max_threads());
    std::vector<torch::Tensor> up_proj_buffers(omp_get_max_threads());
    std::vector<torch::Tensor> activated_buffers(omp_get_max_threads());
    
    // Pre-allocate all buffers with maximum possible size in parallel
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        gate_proj_buffers[i] = torch::empty({1, max_masked_size}, options);
        up_proj_buffers[i] = torch::empty({1, max_masked_size}, options);
        activated_buffers[i] = torch::empty({1, max_masked_size}, options);
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        try {
            int thread_id = omp_get_thread_num();
            auto batch_input = input.select(0, i);
            auto batch_mask = mask.select(0, i);
            auto mask_indices = batch_mask.nonzero().squeeze(-1);
            
            if (mask_indices.numel() > 0) {
                auto& gate_proj_buffer = gate_proj_buffers[thread_id];
                auto& up_proj_buffer = up_proj_buffers[thread_id];
                auto& activated_buffer = activated_buffers[thread_id];
                
                // Use narrow instead of new allocation
                auto gate_proj_view = gate_proj_buffer.narrow(1, 0, mask_indices.numel());
                auto up_proj_view = up_proj_buffer.narrow(1, 0, mask_indices.numel());
                auto activated_view = activated_buffer.narrow(1, 0, mask_indices.numel());
                
                // Compute projections directly
                auto batch_input_view = batch_input.view({1, -1}).detach();
                auto gate_weight_masked = gate_weight.index({mask_indices}).detach();
                auto up_weight_masked = up_weight.index({mask_indices}).detach();
                
                torch::matmul_out(gate_proj_view, batch_input_view, gate_weight_masked.t());
                torch::matmul_out(up_proj_view, batch_input_view, up_weight_masked.t());
                
                // Apply activation
                if (activation_fn == "silu") {
                    activated_view.copy_(up_proj_view * gate_proj_view * torch::sigmoid(gate_proj_view));
                } else {
                    const auto& x = gate_proj_view;
                    const double beta = 0.5 / SQRT_1_2;
                    const double kappa = 0.044715;
                    auto inner = beta * (x + kappa * x * x * x);
                    activated_view.copy_(x * 0.5 * (1.0 + torch::tanh(inner)) * up_proj_view);
                }
                
                // Final projection
                auto down_weight_masked = down_weight.index({torch::indexing::Slice(), mask_indices}).detach();
                auto output_row = output.select(0, i).detach().view({1, -1});
                torch::matmul_out(output_row, activated_view, down_weight_masked.t());
            } else {
                output.select(0, i).zero_();
            }
        } catch (const c10::Error& e) {
            fprintf(stderr, "Error processing batch %ld: %s\n", i, e.what());
        }
    }
}
}  // namespace

// CPU forward implementation
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn) {
    
    auto output = torch::zeros({input.size(0), input.size(1)}, 
                             input.options());  // Preserve requires_grad
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_forward_cpu", [&] {
        sparse_mlp_forward_cpu_impl<scalar_t>(
            input, gate_weight, up_weight, down_weight, mask, activation_fn, output);
    });
    
    return output;
}

// CUDA forward declaration
torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn);

// Forward function that dispatches to CPU/CUDA implementations
torch::Tensor sparse_mlp_forward(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn) {
    
    if (input.device().is_cuda()) {
        return sparse_mlp_forward_cuda(input, gate_weight, up_weight, down_weight, mask, activation_fn);
    }
    return sparse_mlp_forward_cpu(input, gate_weight, up_weight, down_weight, mask, activation_fn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mlp_forward", &sparse_mlp_forward, "Sparse MLP forward");
} 