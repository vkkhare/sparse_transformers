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
    
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(mask.dim() == 2, "Mask must be 2D tensor");
    TORCH_CHECK(gate_weight.dim() == 2, "Gate weight must be 2D tensor");
    TORCH_CHECK(up_weight.dim() == 2, "Up weight must be 2D tensor");
    TORCH_CHECK(down_weight.dim() == 2, "Down weight must be 2D tensor");
    
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    
    TORCH_CHECK(mask.size(0) == batch_size, "Mask batch size mismatch");
    TORCH_CHECK(gate_weight.size(1) == hidden_size, "Gate weight dimension mismatch");
    TORCH_CHECK(up_weight.size(1) == hidden_size, "Up weight dimension mismatch");
    TORCH_CHECK(down_weight.size(0) == hidden_size, "Down weight dimension mismatch");
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        try {
            auto batch_input = input.select(0, i);
            auto batch_mask = mask.select(0, i);
            
            // Get indices where mask is 1
            auto mask_indices = batch_mask.nonzero().squeeze(-1);
            
            if (mask_indices.numel() > 0) {
                // Select only the required rows from weights
                auto gate_weight_masked = gate_weight.index_select(0, mask_indices);
                auto up_weight_masked = up_weight.index_select(0, mask_indices);
                
                // Compute projections using batch_input
                auto batch_input_view = batch_input.view({1, -1});
                auto gate_proj = torch::matmul(batch_input_view, gate_weight_masked.t());
                auto up_proj = torch::matmul(batch_input_view, up_weight_masked.t());
                
                // Apply activation function
                torch::Tensor activated;
                if (activation_fn == "silu") {
                    activated = up_proj * gate_proj * torch::sigmoid(gate_proj);
                } else {  // gelu
                    const auto x = gate_proj;
                    const double beta = 0.5 / SQRT_1_2;
                    const double kappa = 0.044715;
                    auto x_cube = x * x * x;
                    auto inner = beta * (x + kappa * x_cube);
                    activated = x * 0.5 * (1.0 + torch::tanh(inner)) * up_proj;
                }
                
                // Select relevant columns from down_weight and compute final projection
                auto down_weight_masked = down_weight.index({torch::indexing::Slice(), mask_indices});
                auto batch_output = torch::matmul(activated, down_weight_masked.t());
                
                // Create a detached copy for the output
                auto output_row = output.select(0, i);
                auto result = batch_output.squeeze(0).detach();
                output_row.copy_(result);
            }
        } catch (const c10::Error& e) {
            // Log error but continue processing other batches
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
                             input.options().requires_grad(false));  // Create output without gradients
    
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