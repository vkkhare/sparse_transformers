#include <torch/extension.h>
#include <vector>
#include <omp.h>

// CPU forward implementation
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn) {
    
    // Get input dimensions
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto intermediate_size = gate_weight.size(0);
    
    // Initialize output tensor
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    // Parallelize the batch processing
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        // Get current batch input and mask
        auto batch_input = input[i].unsqueeze(0);  // [1, hidden_size]
        auto batch_mask = mask[i];  // [intermediate_size]
        
        // Get indices where mask is 1
        auto mask_indices = batch_mask.nonzero().squeeze();
        
        if (mask_indices.numel() > 0) {
            // Select only the required rows from gate_weight and up_weight
            auto gate_weight_masked = gate_weight.index_select(0, mask_indices);
            auto up_weight_masked = up_weight.index_select(0, mask_indices);
            
            // Compute gate and up projections for masked positions
            auto gate_proj = torch::mm(batch_input, gate_weight_masked.t());
            auto up_proj = torch::mm(batch_input, up_weight_masked.t());
            
            // Apply activation function and multiply
            torch::Tensor activated;
            if (activation_fn == "silu") {
                activated = torch::sigmoid(gate_proj) * up_proj;
            } else if (activation_fn == "gelu") {
                activated = torch::gelu(gate_proj) * up_proj;
            } else {
                throw std::runtime_error("Unsupported activation function");
            }
            
            // Select relevant columns from down_weight and compute final projection
            auto down_weight_masked = down_weight.index({torch::indexing::Slice(), mask_indices});
            auto batch_output = torch::mm(activated, down_weight_masked);
            
            // Store result for this batch
            output[i] = batch_output.squeeze(0);
        }
    }
    
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