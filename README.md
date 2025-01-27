# LLaMA Skip Connection Optimization

A PyTorch C++ implementation of optimized sparse MLP operations for LLaMA models with skip connections.

## Overview

This project implements and optimizes sparse MLP operations in LLaMA models using:
- Custom C++ CUDA kernels
- Efficient weight selection
- Parallel processing optimizations
- Memory layout optimizations

## Performance Results

### Model Comparison
```
Standard LLaMA:
- Average time: 0.515s
- Min: 0.495s
- Max: 0.537s
- Variance: ~8%

SkipLLaMA Scripted:
- Average time: 0.404s
- Min: 0.392s
- Max: 0.421s
- Variance: ~7%

Speedup: ~1.27x [WIP] faster
```

### Custom Operation Breakdown
```
1. Weight Selection:
   - Original: 20-25ms
   - Optimized: 5-7ms
   - Improvement: ~4x

2. Matrix Multiplications:
   - Original: 4-6ms
   - Optimized: 2-3ms
   - Improvement: ~2x

3. Total Processing:
   - Original: 60-80ms
   - Optimized: 8-10ms
   - Overall Improvement: ~7x
```

## Implementation Details

### Core Components

1. Sparse MLP Operation (`sparse_mlp/csrc/sparse_mlp_op.cpp`):
```cpp
torch::Tensor sparse_mlp_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor mask,
    std::string act_fn_name)
```

2. Skip Connection Model (`src/models/modelling_llama_skip.py`):
```python
class LlamaSkipConnectionForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaSkipModel(config)
```

### Key Optimizations

1. Memory Management:
```cpp
auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .device(x.device())
    .layout(torch::kStrided);
```

2. Parallel Processing:
```cpp
at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
    // Batch processing
});
```

## Installation

1. Requirements:
```bash
torch>=1.10.0
transformers>=4.20.0
ninja
```

2. Build C++ extensions:
```bash
# Clone repository
git clone https://github.com/username/llama-skip
cd llama-skip

# Install in editable mode
pip install -e .
```

3. Verify installation:
```bash
python -c "import sparse_mlp; print('Installation successful!')"
```

## Usage

### Quick Start

```python
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig

# Load model
config = LlamaSkipConnectionConfig.from_pretrained("model_id")
model = LlamaSkipConnectionForCausalLM.from_pretrained(
    "checkpoint",
    config=config
)

# Enable optimizations
model.eval()
scripted_model = torch.jit.script(model)
```

## Project Structure

```
├── sparse_mlp/
│   └── csrc/
│       ├── sparse_mlp_op.cpp    # C++ implementation
│       └── sparse_mlp_op.h      # Header file
├── src/
│   └── models/
│       ├── modelling_llama_skip.py      # PyTorch model
│       └── configuration_llama_skip.py   # Model config
├── setup.py                     # Build script
└── README.md
```

## Benchmarking

### Running Benchmarks

1. Detailed profiling with logging:
```bash
# Run trainer notebook with timing output
python trainer_notebook.py 2>&1 | tee benchmark.log
```

### Sample Benchmark Output
```
Weight Selection: 5.136ms
Matrix Multiplications: 1.513ms
Activation: 0.098ms
Final Projection: 0.738ms
Total Processing: 7.645ms
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Open pull request

## Citation

If you use this code, please cite:
```bibtex:README.md
@software{llama_skip_connection,
  title = {LLaMA Skip Connection Optimization},
  year = {2024},
  author = {Author},
  url = {https://github.com/username/llama-skip}
}
```