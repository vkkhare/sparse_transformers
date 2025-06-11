import torch
import socket
import logging
import numpy as np
import time
import threading
from typing import Dict, List, Optional


class GPUMonitor:
    """Monitor GPU usage during inference."""
    
    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self._gpu_memory_usage = []
        self._gpu_utilization = []
        self._is_monitoring = False
        self._monitor_thread = None
        
    def _monitor_gpu(self):
        """Background monitoring of GPU metrics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self._is_monitoring:
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory_info.used / 1024**2
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                
                self._gpu_memory_usage.append(memory_used_mb)
                self._gpu_utilization.append(gpu_util)
                
                time.sleep(self.monitoring_interval)
                
        except ImportError:
            # Fallback to torch methods if pynvml not available
            while self._is_monitoring:
                if torch.cuda.is_available():
                    memory_used_mb = torch.cuda.memory_allocated() / 1024**2
                    self._gpu_memory_usage.append(memory_used_mb)
                time.sleep(self.monitoring_interval)
    
    def start(self):
        """Start GPU monitoring."""
        self._is_monitoring = True
        self._gpu_memory_usage.clear()
        self._gpu_utilization.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_gpu)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop GPU monitoring."""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def get_peak_usage(self) -> Dict:
        """Get peak GPU usage metrics."""
        if not self._gpu_memory_usage:
            return {"peak_gpu_memory_mb": 0, "p90_gpu_utilization": 0}
        
        return {
            "peak_gpu_memory_mb": max(self._gpu_memory_usage),
            "p90_gpu_memory_mb": np.percentile(self._gpu_memory_usage, 90),
            "max_gpu_utilization": max(self._gpu_utilization) if self._gpu_utilization else 0,
            "p90_gpu_utilization": np.percentile(self._gpu_utilization, 90) if self._gpu_utilization else 0
        }


def map_to_cuda(args, device=None, **kwargs):
    if isinstance(args, (list, tuple)):
        return [map_to_cuda(arg, device, **kwargs) for arg in args]
    elif isinstance(args, dict):
        return {k: map_to_cuda(v, device, **kwargs) for k, v in args.items()}
    elif isinstance(args, torch.Tensor):
        return args.cuda(device, **kwargs)
    else:
        raise TypeError("unsupported type for cuda migration")


def map_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def mapping_processes_to_gpus(gpu_config, process_id, worker_number):
    if gpu_config == None:
        device = torch.device("cpu")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        logging.info(gpu_config)
        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_config.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (host, gpu_j)
                    i += 1
        logging.info("Process: %d" % (process_id))
        logging.info("host: %s" % (gpu_util_map[process_id][0]))
        logging.info("gethostname: %s" % (socket.gethostname()))
        logging.info("gpu: %d" % (gpu_util_map[process_id][1]))
        assert i == worker_number

        device = torch.device(
            "cuda:" + str(gpu_util_map[process_id][1])
            if torch.cuda.is_available() else "cpu")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    

def initialize_cuda_safely() -> bool:
    """Initialize CUDA context safely, handling common issues."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to initialize CUDA context
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # Test basic CUDA operations
        device_count = torch.cuda.device_count()
        print(f"CUDA initialized successfully. Found {device_count} GPU(s).")
        
        # Test memory allocation on each device
        for i in range(device_count):
            try:
                torch.cuda.set_device(i)
                # Try a small memory allocation
                test_tensor = torch.randn(10, device=f'cuda:{i}')
                del test_tensor
                torch.cuda.empty_cache()
                print(f"GPU {i} is accessible and functional.")
            except Exception as e:
                print(f"Warning: GPU {i} has issues: {e}")
        
        return True
        
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        print("Falling back to CPU mode.")
        return False



def get_gpu_info() -> Optional[List[Dict]]:
    """Get GPU information if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    try:
        # Clear any existing CUDA context issues
        torch.cuda.empty_cache()
        
        for i in range(torch.cuda.device_count()):
            try:
                # Set the device to ensure proper context
                torch.cuda.set_device(i)
                props = torch.cuda.get_device_properties(i)
                
                # Try to get memory info with retries
                mem_info = None
                for retry in range(3):
                    try:
                        mem_info = torch.cuda.mem_get_info(i)
                        break
                    except RuntimeError as e:
                        if retry == 2:  # Last retry
                            print(f"Warning: Could not get memory info for GPU {i}: {e}")
                            # Use default values if we can't get memory info
                            mem_info = (0, props.total_memory)
                        else:
                            # Wait a bit and clear cache before retry
                            time.sleep(0.1)
                            torch.cuda.empty_cache()
                
                if mem_info:
                    free_memory = mem_info[0] / 1024**3  # Convert to GB
                    total_memory = mem_info[1] / 1024**3  # Convert to GB
                else:
                    # Fallback to device properties
                    free_memory = props.total_memory / 1024**3
                    total_memory = props.total_memory / 1024**3
                
                gpu_info.append({
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory': f"{total_memory:.2f}GB",
                    'free_memory': f"{free_memory:.2f}GB",
                    'multi_processor_count': props.multi_processor_count
                })
                
            except RuntimeError as e:
                print(f"Warning: Could not get info for GPU {i}: {e}")
                # Add a placeholder entry
                gpu_info.append({
                    'name': f"GPU {i} (Error accessing device)",
                    'compute_capability': "Unknown",
                    'total_memory': "Unknown",
                    'free_memory': "Unknown", 
                    'multi_processor_count': "Unknown"
                })
                
    except Exception as e:
        print(f"Warning: Could not enumerate GPUs: {e}")
        return None
    
    return gpu_info if gpu_info else None

def setup_cuda_debugging(verbose: bool = False):
    """Setup CUDA debugging flags."""
    try:
        # Always set CUDA_LAUNCH_BLOCKING for better error reporting
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Clear any existing CUDA context issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Try to set primary device safely
            try:
                torch.cuda.set_device(0)
            except RuntimeError as e:
                print(f"Warning: Could not set CUDA device 0: {e}")
            
            if verbose:
                try:
                    # Enable CUDA memory stats with error handling
                    torch.cuda.memory.set_per_process_memory_fraction(0.9)
                    torch.cuda.memory._record_memory_history(max_entries=10000)
                except Exception as e:
                    print(f"Warning: Could not setup CUDA memory debugging: {e}")
    except Exception as e:
        print(f"Warning: CUDA debugging setup failed: {e}")