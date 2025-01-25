#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for sparse MLP forward pass
template <typename scalar_t>
__global__ void sparse_mlp_forward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ up_weight,
    const scalar_t* __restrict__ down_weight,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size,
    const bool use_silu) {
    
    // Calculate indices for parallel processing
    const int batch_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int num_threads = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Get input offset for this batch
    const scalar_t* batch_input = input + batch_idx * hidden_size;
    const bool* batch_mask = mask + batch_idx * intermediate_size;
    scalar_t* batch_output = output + batch_idx * hidden_size;
    
    // Shared memory for partial sums
    extern __shared__ char shared_mem[];
    scalar_t* shared_output = (scalar_t*)shared_mem;
    
    // Initialize shared memory
    for (int i = thread_idx; i < hidden_size; i += num_threads) {
        shared_output[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes a subset of intermediate positions
    for (int j = thread_idx; j < intermediate_size; j += num_threads) {
        if (!batch_mask[j]) continue;
        
        // Compute gate and up projections
        scalar_t gate_val = 0.0f;
        scalar_t up_val = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < hidden_size; k++) {
            gate_val += batch_input[k] * gate_weight[j * hidden_size + k];
            up_val += batch_input[k] * up_weight[j * hidden_size + k];
        }
        
        // Apply activation
        scalar_t activated;
        if (use_silu) {
            activated = gate_val / (1.0f + expf(-up_val));  // SiLU/Swish
        } else {
            // GELU approximation
            const scalar_t x = gate_val;
            activated = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * 
                (x + 0.044715f * x * x * x))) * up_val;
        }
        
        // Down projection - atomic add to shared memory
        #pragma unroll
        for (int k = 0; k < hidden_size; k++) {
            atomicAdd(&shared_output[k], 
                     activated * down_weight[k * intermediate_size + j]);
        }
    }
    
    // Wait for all threads to complete
    __syncthreads();
    
    // Write final results to global memory
    for (int i = thread_idx; i < hidden_size; i += num_threads) {
        batch_output[i] = shared_output[i];
    }
}

// CUDA forward implementation
torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn) {
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    const auto intermediate_size = gate_weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    // Launch parameters optimized for modern GPUs
    const int threads_per_block = 256;  // Standard warp size * 8
    const int shared_mem_size = hidden_size * sizeof(float);
    
    // Ensure we're not exceeding device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    TORCH_CHECK(shared_mem_size <= prop.sharedMemPerBlock,
               "Required shared memory exceeds device limits");
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "sparse_mlp_forward_cuda", ([&] {
        sparse_mlp_forward_cuda_kernel<scalar_t><<<batch_size, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            gate_weight.data_ptr<scalar_t>(),
            up_weight.data_ptr<scalar_t>(),
            down_weight.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            intermediate_size,
            activation_fn == "silu"
        );
    }));
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, 
               "CUDA kernel execution failed: ", 
               cudaGetErrorString(error));
    
    return output;
}