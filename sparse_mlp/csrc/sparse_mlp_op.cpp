// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's OpenMP wrapper
#include <ATen/ParallelOpenMP.h>

// Add pybind11 and namespace
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Add weight cache class with layer and batch support TODO Make it Model instance dependent
class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    std::vector<torch::Tensor> active_gates;
    std::vector<torch::Tensor> active_ups;
    std::vector<torch::Tensor> active_downs;
    
    // Delete copy/move operations
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
protected:
    WeightCache() = default;
    friend c10::intrusive_ptr<WeightCache>;  // Allow make_intrusive to access constructor

public:
    static c10::intrusive_ptr<WeightCache> getInstance() {
        static c10::intrusive_ptr<WeightCache> instance = c10::make_intrusive<WeightCache>();
        return instance;
    }
    
    void clear() {
        active_gates = std::vector<torch::Tensor>();
        active_ups = std::vector<torch::Tensor>();
        active_downs = std::vector<torch::Tensor>();
        is_initialized = false;
    }
    
    void init(int64_t batch_size) {
        clear();
        active_gates = std::vector<torch::Tensor>(batch_size);
        active_ups = std::vector<torch::Tensor>(batch_size);
        active_downs = std::vector<torch::Tensor>(batch_size);
        is_initialized = true;
    }
    
    void store(int64_t batch_idx, 
               const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& down) {
        active_gates[batch_idx] = gate;
        active_ups[batch_idx] = up;
        active_downs[batch_idx] = down;
    }
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get(int64_t batch_idx) {
        auto gate = active_gates[batch_idx];
        auto up = active_ups[batch_idx];
        auto down = active_downs[batch_idx];
        
        if (!gate.defined() || !up.defined() || !down.defined()) {
            std::cout << "Error: Weights not initialized for batch " << batch_idx << std::endl;
            return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
        }
        
        return std::make_tuple(gate, up, down);
    }
};

// Modified function to use batch-aware cache
void compute_active_weights(
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask) {
    
    int64_t batch_size = mask.size(0);
    WeightCache::getInstance()->init(batch_size);
    
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {            
            auto batch_mask = mask[batch_idx];
            auto active_indices = batch_mask.nonzero().squeeze();
            int64_t num_active = active_indices.size(0);
            
            // Use slice operations with correct dimensions
            auto active_gate = gate_weight.narrow(0, 0, num_active).detach();
            auto active_up = up_weight.narrow(0, 0, num_active).detach();
            auto active_down = down_weight.narrow(1, 0, num_active).detach();
            
            WeightCache::getInstance()->store(batch_idx, active_gate, active_up, active_down);
        }
    });
}

// Modified sparse MLP forward to use batch-aware cache
torch::Tensor sparse_mlp_forward(torch::Tensor x, std::string act_fn_name) {
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x.device())
        .layout(torch::kStrided);
    
    auto down_proj = torch::empty({batch_size, hidden_size}, options);

    
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
            auto [active_gate_weight, active_up_weight, active_down_weight] = 
                WeightCache::getInstance()->get(batch_idx);

            auto x_batch = x[batch_idx].unsqueeze(0);
            auto gate_proj = torch::empty({1, active_gate_weight.size(0)}, options);
            auto up_proj = torch::empty({1, active_up_weight.size(0)}, options);
            // Gate projection with pre-allocated tensors
            torch::matmul_out(gate_proj, x_batch.detach(), active_gate_weight.t());
            gate_proj.mul_(torch::sigmoid(gate_proj));
            
            // Up projection with pre-allocated tensors
            torch::matmul_out(up_proj, x_batch.detach(), active_up_weight.t());
            
            // Final projection
            auto gate_act = gate_proj.mul(up_proj);
            down_proj[batch_idx] = torch::matmul(gate_act, active_down_weight.t())[0];
        }
    });
    
    return down_proj;
}

// Register operators and expose WeightCache to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_mlp_forward, "Sparse MLP forward");
    m.def("compute_active_weights", &compute_active_weights, "Compute active weights");
    
    // Expose WeightCache class
    py::class_<WeightCache, c10::intrusive_ptr<WeightCache>>(m, "WeightCache")
        .def_static("getInstance", &WeightCache::getInstance)
        .def("init", &WeightCache::init)
        .def("store", &WeightCache::store)
        .def("get", &WeightCache::get)
        .def("clear", &WeightCache::clear)
        .def("__repr__", [](const WeightCache&) {
            return "WeightCache(singleton)";
        });
}

// Register TorchScript operators
TORCH_LIBRARY(sparse_mlp, m) {
    m.def("forward", sparse_mlp_forward);
    m.def("compute_active_weights", compute_active_weights);
    m.class_<WeightCache>("WeightCache")
        .def_static("getInstance", &WeightCache::getInstance)
        .def("init", &WeightCache::init)
        .def("store", &WeightCache::store)
        .def("get", &WeightCache::get)
        .def("clear", &WeightCache::clear);
} 