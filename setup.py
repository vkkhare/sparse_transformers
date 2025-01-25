from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA arch list from current GPU
def get_cuda_arch_flags():
    # Get compute capability of current device
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return [f'-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}']
    return []

# Common C++ optimization flags
CXX_FLAGS = [
    '-O3',  # Maximum optimization
    '-march=native',  # Optimize for current CPU architecture
    '-fopenmp',  # Enable OpenMP
    '-ffast-math',  # Enable fast math operations
    '-fno-finite-math-only',  # Allow non-finite math (needed for some PyTorch ops)
    '-funsafe-math-optimizations',  # Enable unsafe math optimizations
    '-fno-trapping-math',  # Disable floating point traps
    '-Wall',  # Enable all warnings
    '-Wno-unknown-pragmas',  # Disable unknown pragma warnings
    '-std=c++17'  # Use C++17
]

# CUDA optimization flags
CUDA_FLAGS = [
    '-O3',  # Maximum optimization
    '--use_fast_math',  # Enable fast math
    '-Xcompiler=-O3,-march=native,-fopenmp',  # Pass through compiler flags
    '--compiler-options=-ffast-math',  # Enable fast math in host code
    '--ptxas-options=-v',  # Verbose PTX assembly
    '--generate-line-info',  # Generate line number information
] + get_cuda_arch_flags()  # Add architecture-specific flags

setup(
    name='sparse_mlp',
    ext_modules=[
        CUDAExtension('sparse_mlp', 
            sources=['csrc/sparse_mlp_op.cpp', 'csrc/sparse_mlp_cuda.cu'],
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': CUDA_FLAGS
            },
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)  # Use ninja for faster builds
    }
) 