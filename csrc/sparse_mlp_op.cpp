#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <chrono>

// Constants for GELU approximation
constexpr double SQRT_2_PI = 2.506628274631000502415765284811045253006986740609938316629923576;
constexpr double SQRT_1_2 = 0.707106781186547524400844362104849039284835937688474036588339869;

// Replace DBG macro with a proper variadic macro
#define DBG(...) do { \
    fprintf(stderr, "[DEBUG:%s:%d] ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    fflush(stderr); \
} while (0)

// Modify Timer class for manual control
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    TimePoint start_time;
    const char* name;
    bool running;
    
public:
    Timer(const char* n) : name(n), running(false) {}
    
    void start() {
        start_time = Clock::now();
        running = true;
    }
    
    void stop() {
        if (running) {
            auto end_time = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            fprintf(stderr, "Timer %s: %.3f ms\n", name, duration / 1000.0);
            running = false;
        }
    }
};

// Change from static to instance-based buffer pool
class BufferPool {
private:
    std::vector<std::vector<torch::Tensor>> gate_proj_buffers;
    std::vector<std::vector<torch::Tensor>> up_proj_buffers;
    std::vector<std::vector<torch::Tensor>> activated_buffers;
    int max_threads;
    int max_size;
    torch::TensorOptions options;
    std::mutex mtx;  // Add mutex for thread safety
    
public:
    BufferPool(int num_threads, int buffer_size, const torch::TensorOptions& opts) 
        : max_threads(num_threads), max_size(buffer_size), options(opts) {
        
        gate_proj_buffers.resize(num_threads);
        up_proj_buffers.resize(num_threads);
        activated_buffers.resize(num_threads);
        
        for (int i = 0; i < num_threads; i++) {
            try {
                gate_proj_buffers[i].push_back(torch::zeros({1, buffer_size}, options));
                up_proj_buffers[i].push_back(torch::zeros({1, buffer_size}, options));
                activated_buffers[i].push_back(torch::zeros({1, buffer_size}, options));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to initialize buffers: " + std::string(e.what()));
            }
        }
    }
    
    ~BufferPool() = default;
    
    torch::Tensor& get_gate_proj(int thread_id, int idx = 0) {
        std::lock_guard<std::mutex> lock(mtx);
        if (thread_id < 0 || thread_id >= max_threads || idx >= gate_proj_buffers[thread_id].size()) {
            throw std::runtime_error("Invalid thread ID or buffer index");
        }
        return gate_proj_buffers[thread_id][idx];
    }
    
    torch::Tensor& get_up_proj(int thread_id, int idx = 0) {
        std::lock_guard<std::mutex> lock(mtx);
        if (thread_id < 0 || thread_id >= max_threads || idx >= up_proj_buffers[thread_id].size()) {
            throw std::runtime_error("Invalid thread ID or buffer index");
        }
        return up_proj_buffers[thread_id][idx];
    }
    
    torch::Tensor& get_activated(int thread_id, int idx = 0) {
        std::lock_guard<std::mutex> lock(mtx);
        if (thread_id < 0 || thread_id >= max_threads || idx >= activated_buffers[thread_id].size()) {
            throw std::runtime_error("Invalid thread ID or buffer index");
        }
        return activated_buffers[thread_id][idx];
    }

    int get_max_size() const { return max_size; }
};

// Create a wrapper class to hold the buffer pool as a PyTorch custom class
class BufferPoolHandle : public torch::CustomClassHolder {
private:
    std::unique_ptr<BufferPool> pool;
    
public:
    BufferPoolHandle(int64_t hidden_size, int64_t intermediate_size) {
        auto max_masked_size = int(intermediate_size * 0.2);
        auto options = torch::TensorOptions()
            .dtype(torch::kHalf)
            .device(torch::kCPU)
            .requires_grad(false);
        
        try {
            pool = std::make_unique<BufferPool>(omp_get_max_threads(), max_masked_size, options);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create BufferPool: " + std::string(e.what()));
        }
    }
    
    ~BufferPoolHandle() = default;
    
    BufferPool* get() {
        if (!pool) {
            throw std::runtime_error("BufferPool is null");
        }
        return pool.get();
    }
};

namespace {
template <typename scalar_t>
void sparse_mlp_forward_cpu_impl(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn,
    torch::Tensor& output,
    BufferPool* buffer_pool) {
    
    Timer total_timer("total_forward");
    total_timer.start();

    auto batch_size = input.size(0);
    std::vector<double> thread_times(omp_get_max_threads(), 0.0);
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        int thread_id = omp_get_thread_num();
        auto thread_start = std::chrono::high_resolution_clock::now();
        
        try {
            auto mask_indices = mask.select(0, i).nonzero().squeeze(-1);
            
            if (mask_indices.numel() > 0) {
                Timer proj_timer(("proj_" + std::to_string(thread_id)).c_str());
                Timer matmul_timer(("matmul_" + std::to_string(thread_id)).c_str());
                Timer act_timer(("act_" + std::to_string(thread_id)).c_str());
                Timer down_timer(("down_" + std::to_string(thread_id)).c_str());
                Timer down_index_timer(("down_index_" + std::to_string(thread_id)).c_str());
                Timer down_matmul_timer(("down_matmul_" + std::to_string(thread_id)).c_str());

                proj_timer.start();
                auto batch_input = input.select(0, i);
                auto& gate_proj_view = buffer_pool->get_gate_proj(thread_id);
                auto& up_proj_view = buffer_pool->get_up_proj(thread_id);
                auto& activated_view = buffer_pool->get_activated(thread_id);
                
                auto batch_input_view = batch_input.view({1, -1}).contiguous().detach();
                auto gate_weight_masked = gate_weight.index({mask_indices}).contiguous().t().detach();
                auto up_weight_masked = up_weight.index({mask_indices}).contiguous().t().detach();
                
                auto gate_proj_view_narrow = gate_proj_view.narrow(0, 0, batch_input_view.size(0))
                                                      .narrow(1, 0, gate_weight_masked.size(1));
                auto up_proj_view_narrow = up_proj_view.narrow(0, 0, batch_input_view.size(0))
                                                  .narrow(1, 0, up_weight_masked.size(1));
                auto activated_view_narrow = activated_view.narrow(0, 0, batch_input_view.size(0))
                                                       .narrow(1, 0, gate_weight_masked.size(1));
                proj_timer.stop();
                
                matmul_timer.start();
                torch::matmul_out(gate_proj_view_narrow, batch_input_view, gate_weight_masked);
                torch::matmul_out(up_proj_view_narrow, batch_input_view, up_weight_masked);
                matmul_timer.stop();

                act_timer.start();
                if (activation_fn == "silu") {
                    activated_view_narrow.copy_(gate_proj_view_narrow);
                    activated_view_narrow.sigmoid_().mul_(gate_proj_view_narrow).mul_(up_proj_view_narrow);
                } else if (activation_fn == "gelu") {
                    activated_view_narrow.copy_(gate_proj_view_narrow);
                    activated_view_narrow.mul_(SQRT_1_2);
                    activated_view_narrow.erf_();
                    activated_view_narrow.add_(1);
                    activated_view_narrow.mul_(0.5);
                    activated_view_narrow.mul_(gate_proj_view_narrow);
                    activated_view_narrow.mul_(up_proj_view_narrow);
                }
                act_timer.stop();

                down_timer.start();
                down_index_timer.start();
                auto down_weight_masked = down_weight.index_select(1, mask_indices);
                down_index_timer.stop();

                down_weight_masked = down_weight_masked.contiguous().t().detach();

                down_matmul_timer.start();
                auto output_view = output.select(0, i).view({1, -1}).detach();
                torch::matmul_out(output_view, activated_view_narrow, down_weight_masked);
                down_matmul_timer.stop();
                down_timer.stop();
            } else {
                output.select(0, i).zero_();
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Error in batch " + std::to_string(i) + 
                                   ", thread " + std::to_string(thread_id) + 
                                   ": " + e.what());
        }
        
        auto thread_end = std::chrono::high_resolution_clock::now();
        thread_times[thread_id] += std::chrono::duration<double, std::milli>(
            thread_end - thread_start).count();
    }
    
    // Print thread statistics
    fprintf(stderr, "\nThread Statistics:\n");
    for (int i = 0; i < omp_get_max_threads(); i++) {
        fprintf(stderr, "Thread %d total time: %.3f ms\n", i, thread_times[i]);
    }
    
    total_timer.stop();
}
}  // namespace

// CPU forward implementation
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask,
    const std::string& activation_fn,
    const c10::intrusive_ptr<BufferPoolHandle>& buffer_pool) {
    
    auto output = torch::zeros({input.size(0), input.size(1)}, input.options());
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_forward_cpu", [&] {
        sparse_mlp_forward_cpu_impl<scalar_t>(
            input, gate_weight, up_weight, down_weight, mask, activation_fn, output, 
            buffer_pool->get());
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
    const std::string& activation_fn,
    const c10::intrusive_ptr<BufferPoolHandle>& buffer_pool) {
    
    if (input.device().is_cuda()) {
        return sparse_mlp_forward_cuda(input, gate_weight, up_weight, down_weight, mask, activation_fn);
    }    
    return sparse_mlp_forward_cpu(input, gate_weight, up_weight, down_weight, mask, activation_fn, buffer_pool);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BufferPoolHandle, c10::intrusive_ptr<BufferPoolHandle>>(m, "BufferPoolHandle")
        .def(py::init([](int64_t hidden_size, int64_t intermediate_size) {
            return c10::make_intrusive<BufferPoolHandle>(hidden_size, intermediate_size);
        }))
        .def("__repr__", [](const BufferPoolHandle& self) {
            return "<BufferPoolHandle>";
        });
        
    m.def("sparse_mlp_forward", 
          [](const torch::Tensor& input,
             const torch::Tensor& gate_weight,
             const torch::Tensor& up_weight,
             const torch::Tensor& down_weight,
             const torch::Tensor& mask,
             const std::string& activation_fn,
             const c10::intrusive_ptr<BufferPoolHandle>& buffer_pool) {
              if (!buffer_pool) {
                  throw std::runtime_error("buffer_pool is null");
              }
              return sparse_mlp_forward(
                  input, gate_weight, up_weight, down_weight, 
                  mask, activation_fn, buffer_pool);
          },
          "Sparse MLP forward",
          py::arg("input"), 
          py::arg("gate_weight"),
          py::arg("up_weight"),
          py::arg("down_weight"),
          py::arg("mask"),
          py::arg("activation_fn"),
          py::arg("buffer_pool"));
} 