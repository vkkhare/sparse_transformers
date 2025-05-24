#pragma once

#include <torch/extension.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <unordered_set>

class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    
    // Memory pools for all weight data
    std::unique_ptr<float[]> gate_memory_pool;
    std::unique_ptr<float[]> up_memory_pool;
    std::unique_ptr<float[]> down_memory_pool;
    
    // Matrix dimensions
    int64_t hidden_dim = 0;
    int64_t sparse_dim = 0;
    int64_t gate_row_size = 0;
    int64_t up_row_size = 0;
    int64_t down_row_size = 0;
    
    torch::ScalarType dtype;
    torch::Device current_device = torch::kCPU;
    
    // Currently active indices (sorted for efficient access)
    std::vector<int64_t> active_indices;
    
    // Vector of vectors for active data (points to memory pool data)
    std::vector<float*> active_gate_rows;
    std::vector<float*> active_up_rows;
    std::vector<float*> active_down_cols;
    
    // Current mask for differential updates
    torch::Tensor current_mask;
    
    // Cached active weight tensors
    torch::Tensor active_weights_cache;
    torch::Tensor active_downs_cache;
    bool cache_valid = false;
    
    // Helper function to differentially update vector-of-vectors
    void update_active_vectors(const std::unordered_set<int64_t>& indices_to_add,
                              const std::unordered_set<int64_t>& indices_to_remove) {
        
        // Remove indices (in reverse order to maintain vector integrity)
        for (auto it = active_indices.rbegin(); it != active_indices.rend(); ++it) {
            if (indices_to_remove.count(*it)) {
                size_t pos = std::distance(active_indices.begin(), 
                                         std::find(active_indices.begin(), active_indices.end(), *it));
                
                // Remove from vectors
                active_gate_rows.erase(active_gate_rows.begin() + pos);
                active_up_rows.erase(active_up_rows.begin() + pos);
                active_down_cols.erase(active_down_cols.begin() + pos);
            }
        }
        
        // Remove from active_indices
        active_indices.erase(
            std::remove_if(active_indices.begin(), active_indices.end(),
                          [&](int64_t idx) { return indices_to_remove.count(idx); }),
            active_indices.end());
        
        // Add new indices (maintain sorted order)
        for (int64_t idx : indices_to_add) {
            auto pos = std::lower_bound(active_indices.begin(), active_indices.end(), idx);
            size_t insert_pos = std::distance(active_indices.begin(), pos);
            
            // Insert into active_indices
            active_indices.insert(pos, idx);
            
            // Insert pointers to memory pool data
            active_gate_rows.insert(active_gate_rows.begin() + insert_pos,
                                   gate_memory_pool.get() + idx * gate_row_size);
            active_up_rows.insert(active_up_rows.begin() + insert_pos,
                                 up_memory_pool.get() + idx * up_row_size);
            
            // For down matrix, we need column access (transpose)
            float* down_col = new float[down_row_size];
            for (int64_t j = 0; j < down_row_size; ++j) {
                down_col[j] = down_memory_pool.get()[j * sparse_dim + idx];
            }
            active_down_cols.insert(active_down_cols.begin() + insert_pos, down_col);
        }
    }
    
    // Helper function to create tensors from vector-of-vectors using from_blob
    void create_tensors_from_vectors() {
        if (active_indices.empty()) {
            auto options = torch::TensorOptions().device(current_device).dtype(dtype);
            active_weights_cache = torch::empty({0, hidden_dim}, options);
            active_downs_cache = torch::empty({0, hidden_dim}, options);
            cache_valid = true;
            return;
        }
        
        const size_t num_active = active_indices.size();
        
        // Create contiguous memory blocks for tensor data
        static std::vector<float> gate_data_buffer;
        static std::vector<float> up_data_buffer;
        static std::vector<float> down_data_buffer;
        
        gate_data_buffer.resize(num_active * gate_row_size);
        up_data_buffer.resize(num_active * up_row_size);
        down_data_buffer.resize(down_row_size * num_active);
        
        // Copy row pointers to contiguous memory
        for (size_t i = 0; i < num_active; ++i) {
            std::memcpy(gate_data_buffer.data() + i * gate_row_size,
                       active_gate_rows[i], gate_row_size * sizeof(float));
            std::memcpy(up_data_buffer.data() + i * up_row_size,
                       active_up_rows[i], up_row_size * sizeof(float));
            std::memcpy(down_data_buffer.data() + i * down_row_size,
                       active_down_cols[i], down_row_size * sizeof(float));
        }
        
        // Create tensors from contiguous data using from_blob (zero-copy)
        auto gate_tensor = torch::from_blob(gate_data_buffer.data(),
                                          {static_cast<int64_t>(num_active), gate_row_size},
                                          torch::TensorOptions().dtype(dtype)).clone();
        
        auto up_tensor = torch::from_blob(up_data_buffer.data(),
                                        {static_cast<int64_t>(num_active), up_row_size},
                                        torch::TensorOptions().dtype(dtype)).clone();
        
        auto down_tensor = torch::from_blob(down_data_buffer.data(),
                                          {down_row_size, static_cast<int64_t>(num_active)},
                                          torch::TensorOptions().dtype(dtype)).clone();
        
        // Concatenate and move to target device
        active_weights_cache = torch::cat({gate_tensor, up_tensor}, 0).to(current_device);
        active_downs_cache = down_tensor.to(current_device);
        
        cache_valid = true;
    }

public:
    WeightCache(const torch::Tensor& mask, int64_t hidden_size, 
                const torch::Tensor& gate_weight, const torch::Tensor& up_weight, 
                const torch::Tensor& down_weight) {
        init(mask, hidden_size, gate_weight, up_weight, down_weight);
    }
    
    // Delete copy/move operations for safety
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
    // Resource cleanup
    void clear() {
        // Clean up down column allocations
        for (auto ptr : active_down_cols) {
            delete[] ptr;
        }
        
        gate_memory_pool.reset();
        up_memory_pool.reset();
        down_memory_pool.reset();
        
        active_indices.clear();
        active_gate_rows.clear();
        active_up_rows.clear();
        active_down_cols.clear();
        
        active_weights_cache = torch::Tensor();
        active_downs_cache = torch::Tensor();
        current_mask = torch::Tensor();
        
        hidden_dim = 0;
        sparse_dim = 0;
        gate_row_size = 0;
        up_row_size = 0;
        down_row_size = 0;
        is_initialized = false;
        cache_valid = false;
    }
    
    // Initialize with weight matrices
    void init(const torch::Tensor& mask, int64_t hidden_size, 
             const torch::Tensor& gate_weight, const torch::Tensor& up_weight, 
             const torch::Tensor& down_weight) {
        clear();
        
        // Validate inputs
        TORCH_CHECK(gate_weight.dim() == 2, "Gate weight must be 2D");
        TORCH_CHECK(up_weight.dim() == 2, "Up weight must be 2D");
        TORCH_CHECK(down_weight.dim() == 2, "Down weight must be 2D");
        TORCH_CHECK(gate_weight.size(0) == up_weight.size(0), "Gate and up weights must have same number of rows");
        TORCH_CHECK(gate_weight.size(0) == down_weight.size(1), "Gate rows must match down columns");
        
        current_device = mask.device();
        dtype = gate_weight.scalar_type();
        
        // Store dimensions
        hidden_dim = hidden_size;
        sparse_dim = gate_weight.size(0);
        gate_row_size = gate_weight.size(1);
        up_row_size = up_weight.size(1);
        down_row_size = down_weight.size(0);
        
        // Move weights to CPU for storage
        auto gate_cpu = gate_weight.to(torch::kCPU).contiguous();
        auto up_cpu = up_weight.to(torch::kCPU).contiguous();
        auto down_cpu = down_weight.to(torch::kCPU).contiguous();
        
        // Allocate memory pools
        const size_t gate_total_size = gate_cpu.numel();
        const size_t up_total_size = up_cpu.numel();
        const size_t down_total_size = down_cpu.numel();
        
        gate_memory_pool = std::make_unique<float[]>(gate_total_size);
        up_memory_pool = std::make_unique<float[]>(up_total_size);
        down_memory_pool = std::make_unique<float[]>(down_total_size);
        
        // Copy matrices to memory pools
        std::memcpy(gate_memory_pool.get(), gate_cpu.data_ptr<float>(), gate_total_size * sizeof(float));
        std::memcpy(up_memory_pool.get(), up_cpu.data_ptr<float>(), up_total_size * sizeof(float));
        std::memcpy(down_memory_pool.get(), down_cpu.data_ptr<float>(), down_total_size * sizeof(float));
        
        // Initialize with mask
        update_active_weights(mask);
        
        is_initialized = true;
    }
    
    // Update active weights based on mask using differential updates
    void update_active_weights(const torch::Tensor& mask) {
        if (!is_initialized) return;
        
        // Normalize mask to 1D boolean tensor
        torch::Tensor normalized_mask;
        if (mask.dim() > 1) {
            auto nonzero_positions = torch::nonzero(mask);
            if (nonzero_positions.size(0) > 0) {
                normalized_mask = torch::zeros({sparse_dim}, torch::TensorOptions().dtype(torch::kBool).device(mask.device()));
                auto col_indices = nonzero_positions.select(1, 1);
                normalized_mask.index_put_({col_indices}, true);
            } else {
                normalized_mask = torch::zeros({sparse_dim}, torch::TensorOptions().dtype(torch::kBool).device(mask.device()));
            }
        } else {
            normalized_mask = mask.to(torch::kBool);
        }
        
        auto mask_cpu = normalized_mask.to(torch::kCPU).contiguous();
        
        // If no current mask, do full initialization
        if (current_mask.numel() == 0) {
            auto indices_tensor = torch::nonzero(mask_cpu).squeeze(-1);
            
            std::unordered_set<int64_t> indices_to_add;
            if (indices_tensor.numel() > 0) {
                auto* indices_ptr = indices_tensor.data_ptr<int64_t>();
                for (int64_t i = 0; i < indices_tensor.numel(); ++i) {
                    indices_to_add.insert(indices_ptr[i]);
                }
            }
            
            std::unordered_set<int64_t> empty_remove_set;
            update_active_vectors(indices_to_add, empty_remove_set);
            create_tensors_from_vectors();
            
            current_mask = normalized_mask.clone();
            return;
        }
        
        // Differential update: compute changes
        auto current_mask_cpu = current_mask.to(torch::kCPU);
        auto diff_mask = mask_cpu.to(torch::kInt) - current_mask_cpu.to(torch::kInt);
        
        std::unordered_set<int64_t> indices_to_add;
        std::unordered_set<int64_t> indices_to_remove;
        
        // Process additions (diff = 1)
        auto additions = torch::nonzero(diff_mask == 1).squeeze(-1);
        if (additions.numel() > 0) {
            auto* add_ptr = additions.data_ptr<int64_t>();
            for (int64_t i = 0; i < additions.numel(); ++i) {
                int64_t idx = add_ptr[i];
                if (idx >= 0 && idx < sparse_dim) {
                    indices_to_add.insert(idx);
                }
            }
        }
        
        // Process removals (diff = -1)
        auto removals = torch::nonzero(diff_mask == -1).squeeze(-1);
        if (removals.numel() > 0) {
            auto* rem_ptr = removals.data_ptr<int64_t>();
            for (int64_t i = 0; i < removals.numel(); ++i) {
                int64_t idx = rem_ptr[i];
                indices_to_remove.insert(idx);
            }
        }
        
        // Apply differential updates only if there are changes
        if (!indices_to_add.empty() || !indices_to_remove.empty()) {
            update_active_vectors(indices_to_add, indices_to_remove);
            create_tensors_from_vectors();
        }
        
        current_mask = normalized_mask.clone();
    }
    
    // Get concatenated weights (gate + up)
    torch::Tensor get_concat_weight() const {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_weights_cache;
    }
    
    // Get active down weights  
    torch::Tensor get_active_down_weight() const {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_downs_cache;
    }
    
    // Get current active indices
    const std::vector<int64_t>& get_active_indices() const {
        return active_indices;
    }
    
    // Get number of active indices
    size_t get_num_active() const {
        return active_indices.size();
    }
    
    // Get device
    torch::Device device() const {
        return current_device;
    }
    
    // Check if initialized
    bool initialized() const {
        return is_initialized;
    }
    
    // Destructor
    ~WeightCache() {
        clear();
    }
};