import platform
import psutil
import argparse
import torch
from typing import Dict
from .cuda_utils import get_gpu_info

def get_system_info() -> Dict:
    """Get system information including CPU and RAM details."""
    cpu_info = {
        'processor': platform.processor(),
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'max_frequency': f"{psutil.cpu_freq().max:.0f}MHz" if psutil.cpu_freq() else "Unknown",
        'current_frequency': f"{psutil.cpu_freq().current:.0f}MHz" if psutil.cpu_freq() else "Unknown"
    }
    
    memory = psutil.virtual_memory()
    ram_info = {
        'total': f"{memory.total / (1024**3):.2f}GB",
        'available': f"{memory.available / (1024**3):.2f}GB",
        'used_percent': f"{memory.percent}%"
    }
    
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'cpu': cpu_info,
        'ram': ram_info
    }


def print_system_info(args: argparse.Namespace) -> None:
    """Print system configuration information."""
    print("\nSystem Configuration:")
    print("-" * 50)
    system_info = get_system_info()
    print(f"OS: {system_info['system']} {system_info['release']}")
    print(f"CPU: {system_info['cpu']['processor']}")
    print(f"Physical cores: {system_info['cpu']['physical_cores']}")
    print(f"Total cores: {system_info['cpu']['total_cores']}")
    print(f"Max CPU frequency: {system_info['cpu']['max_frequency']}")
    print(f"Current CPU frequency: {system_info['cpu']['current_frequency']}")
    print(f"RAM: Total={system_info['ram']['total']}, Available={system_info['ram']['available']} ({system_info['ram']['used_percent']} used)")

    if args.device == 'cuda':
        print("\nGPU Configuration:")
        print("-" * 50)
        gpu_info = get_gpu_info()
        for i, gpu in enumerate(gpu_info or []):
            print(f"\nGPU {i}: {gpu['name']}")
            print(f"Compute capability: {gpu['compute_capability']}")
            print(f"Total memory: {gpu['total_memory']}")
            print(f"Free memory: {gpu['free_memory']}")
            print(f"Multi processors: {gpu['multi_processor_count']}")

    print("\nPyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
    print("-" * 50)
