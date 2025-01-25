from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

# Optimize for speed with architecture-specific flags
extra_compile_args = {
    'cxx': [
        '-O3',  # Maximum optimization
        '-ffast-math',  # Enable fast math operations
        '-march=native',  # Optimize for current CPU architecture
        '-fopenmp',  # Enable OpenMP
        '-std=c++17'
    ],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__'
    ]
}

# Add CUDA arch flags if CUDA is available
if torch.cuda.is_available():
    # Get compute capability of the current GPU
    arch_list = [f'sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}']
    for arch in arch_list:
        extra_compile_args['nvcc'].extend([
            f'-gencode=arch=compute_{arch[3:]},code={arch}',
        ])

setup(
    name='sparse_mlp',
    ext_modules=[
        CUDAExtension(
            name='sparse_mlp',
            sources=[
                'csrc/sparse_mlp_op.cpp',
                'csrc/sparse_mlp_cuda.cu'
            ],
            extra_compile_args=extra_compile_args
        ) if torch.cuda.is_available() else CppExtension(
            name='sparse_mlp',
            sources=['csrc/sparse_mlp_op.cpp'],
            extra_compile_args=extra_compile_args['cxx']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 