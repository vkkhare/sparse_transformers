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

// Add required headers
#include <future>
#include <thread>
#include <mutex>

// Add timing utilities
#include <chrono>

// Add weight cache class with layer and batch support TODO Make it Model instance dependent
class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    torch::Tensor active_weights; // Combined weights
    torch::Tensor active_downs;
    
    // Delete copy/move operations
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
protected:
    WeightCache() = default;
    friend c10::intrusive_ptr<WeightCache>;

public:
    static c10::intrusive_ptr<WeightCache> getInstance() {
        static c10::intrusive_ptr<WeightCache> instance = c10::make_intrusive<WeightCache>();
        return instance;
    }
    
    void clear() {
        active_weights = torch::Tensor();
        active_downs = torch::Tensor();
        is_initialized = false;
    }
    
    void init(int64_t batch_size) {
        clear();
        active_weights = torch::empty({2048, 3276}); // Combined size for gate+up
        active_downs = torch::empty({1638, 2048});
        is_initialized = true;
    }
    
    void store(const torch::Tensor& concat_weights, const torch::Tensor& down) {
        active_weights = concat_weights;
        active_downs = down;
    }
    
    std::tuple<torch::Tensor, torch::Tensor> get() {
        return std::make_tuple(active_weights, active_downs);
    }
};

// Background task manager with proper synchronization
class BackgroundTaskManager : public torch::CustomClassHolder {
private:
    std::mutex mtx;
    std::future<void> current_task;
    bool task_running = false;
    
    BackgroundTaskManager() = default;
    friend c10::intrusive_ptr<BackgroundTaskManager>;

public:
    static c10::intrusive_ptr<BackgroundTaskManager> getInstance() {
        static c10::intrusive_ptr<BackgroundTaskManager> instance = 
            c10::make_intrusive<BackgroundTaskManager>();
        return instance;
    }
    
    void start_task(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_running && current_task.valid()) {
            current_task.wait();
        }
        current_task = std::async(std::launch::async, task);
        task_running = true;
    }
    
    void wait() {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_running && current_task.valid()) {
            current_task.wait();
            task_running = false;
        }
    }
};
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop(const std::string& timer_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << timer_name << ": " << duration/1000.0 << "ms" << std::endl;
    }
};

void compute_active_weights(
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask) {
    int64_t batch_size = mask.size(0);
    WeightCache::getInstance()->init(batch_size);
    auto active_gate = gate_weight.narrow(0, 0, 1638).detach();
    auto active_up = up_weight.narrow(0, 0, 1638).detach();
    
    // Concatenate gate and up weights
    auto concat_weights = torch::cat({active_gate, active_up}, 0);
    auto active_down = down_weight.narrow(1, 0, 1638).detach();
    
    WeightCache::getInstance()->store(concat_weights, active_down);
}

torch::Tensor sparse_mlp_forward(
    torch::Tensor x, 
    torch::Tensor down_proj_buffer,
    torch::Tensor combined_proj_buffer,
    std::string act_fn_name) {
    
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
            auto [concat_weight, active_down_weight] = WeightCache::getInstance()->get();
            int64_t gate_size = concat_weight.size(0) / 2;
            
            auto x_batch = x[batch_idx].unsqueeze(0).detach();
            
            // Single matmul for both gate and up projections
            auto proj_view = combined_proj_buffer[batch_idx].unsqueeze(0).narrow(1, 0, concat_weight.size(0));
            torch::matmul_out(proj_view, x_batch, concat_weight.t());
            
            // Split result into gate and up projections using computed size
            auto gate_proj = proj_view.narrow(1, 0, gate_size);
            auto up_proj = proj_view.narrow(1, gate_size, gate_size);
            
            // Apply activations
            gate_proj.mul_(torch::sigmoid(gate_proj));
            gate_proj.mul_(up_proj);
            
            // Final projection
            down_proj_buffer[batch_idx] = torch::matmul(gate_proj, active_down_weight.t())[0];
        }
    });
    return down_proj_buffer;
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
} 