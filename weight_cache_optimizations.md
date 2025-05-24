# Weight Cache Optimizations

## Overview
The optimized weight cache implementation (`WeightCacheOptimized`) reduces the overhead of dynamic weight selection from 21.37ms to potentially ~5-10ms through several key optimizations.

## Key Optimizations

### 1. **Parallelized Memory Operations**
- **Parallel memcpy**: Uses OpenMP to parallelize large memory copies in chunks
- **Parallel column gathering**: Parallelizes the transpose operation for down matrix columns
- **Parallel difference detection**: Uses OpenMP to find mask differences in parallel

### 2. **Memory Management Improvements**
- **Cache-aligned memory**: 64-byte aligned allocations for better cache performance
- **Pre-allocated buffers**: Eliminates allocation overhead during updates
- **Smart pointers with custom deleters**: Efficient memory management without overhead

### 3. **Algorithmic Optimizations**
- **Fast-path for minimal changes**: Skips update if mask changes are below threshold
- **Differential updates**: Only processes changed indices instead of full recomputation
- **Optimized tensor creation**: Uses `torch::from_blob` for zero-copy tensor creation

### 4. **OpenMP Parallelization Details**
```cpp
// Parallel memory copy
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < num_chunks; ++i) {
    // Process 16KB chunks for optimal cache usage
}

// Parallel column gathering with collapse
#pragma omp parallel for collapse(2) schedule(static)
for (size_t i = 0; i < num_active; ++i) {
    for (int64_t j = 0; j < num_rows; ++j) {
        // Efficient column-wise access
    }
}
```

### 5. **Expected Performance Gains**
- **Memory copy operations**: 3-4x faster with parallelization
- **Column transpose**: 4-6x faster with parallel gathering
- **Overall update time**: 2-4x faster depending on sparsity changes

## Usage

### In Python:
```python
from sparse_mlp import WeightCacheOptimized

# Use in LlamaSkipMLP
self.weight_cache = WeightCacheOptimized(
    mask, hidden_size, gate_weight, up_weight, down_weight
)
```

### Configuration:
```python
# In model config
config.use_optimized_weight_cache = True  # Default

# Or per-layer
mlp = LlamaSkipMLP(
    hidden_size, 
    intermediate_size, 
    sparsity,
    use_optimized_cache=True
)
```

## Testing
Run `test_optimized_cache.py` to verify correctness and measure performance improvements:
```bash
python test_optimized_cache.py
```

## Build Requirements
- OpenMP support (`-fopenmp` flag)
- C++14 or later
- PyTorch 1.8+
- POSIX-compliant system (for posix_memalign)

## Future Optimizations
1. **SIMD operations**: Use AVX2/AVX512 for vectorized memory operations
2. **GPU offloading**: Move weight selection to GPU for CUDA models
3. **Batched updates**: Process multiple mask updates together
4. **Memory pooling**: Reuse buffers across layers 