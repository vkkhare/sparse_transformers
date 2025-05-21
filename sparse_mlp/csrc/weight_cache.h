#pragma once

#include <torch/extension.h>
#include <vector>

class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    // Store the raw data as vector blobs
    std::vector<float> gate_blob;
    std::vector<float> up_blob;
    std::vector<float> down_blob;
    std::vector<int64_t> gate_sizes;
    std::vector<int64_t> up_sizes;
    std::vector<int64_t> down_sizes;
    torch::ScalarType dtype;
    
    torch::Tensor active_weights; // Combined weights
    torch::Tensor active_downs;
    torch::Device current_device = torch::kCPU;
    
public:
    WeightCache(const torch::Tensor& mask, int64_t hidden_size, 
                const torch::Tensor& gate_weight, const torch::Tensor& up_weight, 
                const torch::Tensor& down_weight) {
        init(mask, hidden_size, gate_weight, up_weight, down_weight);
    }
    
    // Delete copy/move operations
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
    // Destructor to clean up resources
    ~WeightCache() {
        clear();
    }
    
    void clear() {
        gate_blob.clear();
        up_blob.clear();
        down_blob.clear();
        gate_sizes.clear();
        up_sizes.clear();
        down_sizes.clear();
        active_weights = torch::Tensor();
        active_downs = torch::Tensor();
        is_initialized = false;
    }
        
    // New method to select and store active weights based on non-zero indices from mask
    void update_active_weights(const torch::Tensor& mask) {
        // Get non-zero indices from the mask
        // torch::Tensor nonzero_indices = torch::nonzero(mask).select(1, 1);
        int64_t sparse_size = mask.size(1);

        // Get tensors from blobs
        torch::Tensor gate_tensor = get_gate_tensor();
        torch::Tensor up_tensor = get_up_tensor();
        torch::Tensor down_tensor = get_down_tensor();

        auto active_gate = gate_tensor.narrow(0, 0, sparse_size).detach();
        auto active_up = up_tensor.narrow(0, 0, sparse_size).detach();
        // Concatenate gate and up weights
        active_weights = torch::cat({active_gate, active_up}, 0);
        active_downs = down_tensor.narrow(1, 0, sparse_size).detach();


        // // Select weights at the non-zero indices using index_select
        // auto active_gate = torch::index_select(gate_tensor, 0, nonzero_indices).detach();
        // auto active_up = torch::index_select(up_tensor, 0, nonzero_indices).detach();
        
        // // Concatenate gate and up weights
        // active_weights = torch::cat({active_gate, active_up}, 0).to(current_device);
        
        // // For down_weight, we index the second dimension (dim=1)
        // active_downs = torch::index_select(down_tensor, 1, nonzero_indices).detach().to(current_device);
    }
    
    // Separate getters instead of structured bindings
    torch::Tensor get_concat_weight() const {
        return active_weights;
    }
    
    torch::Tensor get_active_down_weight() const {
        return active_downs;
    }
    
    torch::Device device() const {
        return current_device;
    }

    private:
        void init(const torch::Tensor& mask, int64_t hidden_size, 
                const torch::Tensor& gate_weight, const torch::Tensor& up_weight, const torch::Tensor& down_weight) {
            clear();
            current_device = mask.device();
            dtype = gate_weight.scalar_type();
            
            // Ensure sizes are aligned for half precision
            int64_t aligned_hidden_size = (hidden_size + 1) & ~1;
            int64_t aligned_sparse_size = (mask.size(1) + 1) & ~1;
            
            auto options = torch::TensorOptions()
                .device(current_device)
                .dtype(dtype);
            
            // Store tensor sizes
            gate_sizes = gate_weight.sizes().vec();
            up_sizes = up_weight.sizes().vec();
            down_sizes = down_weight.sizes().vec();
            
            // Copy data to CPU for storage
            auto gate_cpu = gate_weight.cpu().contiguous();
            auto up_cpu = up_weight.cpu().contiguous();
            auto down_cpu = down_weight.cpu().contiguous();
            
            // Get data size
            size_t gate_size = gate_weight.numel() * gate_weight.element_size();
            size_t up_size = up_weight.numel() * up_weight.element_size();
            size_t down_size = down_weight.numel() * down_weight.element_size();
            
            // Copy data to vectors
            gate_blob.resize(gate_size / sizeof(float));
            up_blob.resize(up_size / sizeof(float));
            down_blob.resize(down_size / sizeof(float));
            
            // Copy memory from tensors to vectors
            std::memcpy(gate_blob.data(), gate_cpu.data_ptr(), gate_size);
            std::memcpy(up_blob.data(), up_cpu.data_ptr(), up_size);
            std::memcpy(down_blob.data(), down_cpu.data_ptr(), down_size);
            
            // Allocate space for active weights
            active_weights = torch::empty({aligned_hidden_size, 2*aligned_sparse_size}, options);
            active_downs = torch::empty({aligned_sparse_size, aligned_hidden_size}, options);
            
            is_initialized = true;
        }
    
        // Get back tensors from blobs when needed
        torch::Tensor get_gate_tensor() const {
            if (!is_initialized || gate_blob.empty()) return torch::Tensor();
            torch::Tensor gate_tensor = torch::from_blob(
                const_cast<float*>(gate_blob.data()),
                gate_sizes,
                torch::TensorOptions().dtype(dtype)
            ).clone(); // Clone to make it safe
            return gate_tensor.to(current_device);
        }
        
        torch::Tensor get_up_tensor() const {
            if (!is_initialized || up_blob.empty()) return torch::Tensor();
            torch::Tensor up_tensor = torch::from_blob(
                const_cast<float*>(up_blob.data()),
                up_sizes,
                torch::TensorOptions().dtype(dtype)
            ).clone(); // Clone to make it safe
            return up_tensor.to(current_device);
        }
        
        torch::Tensor get_down_tensor() const {
            if (!is_initialized || down_blob.empty()) return torch::Tensor();
            torch::Tensor down_tensor = torch::from_blob(
                const_cast<float*>(down_blob.data()),
                down_sizes,
                torch::TensorOptions().dtype(dtype)
            ).clone(); // Clone to make it safe
            return down_tensor.to(current_device);
        }

}; 