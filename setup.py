from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension('sparse_mlp', 
            sources=['csrc/sparse_mlp_op.cpp', 'csrc/sparse_mlp_cuda.cu'],
            extra_compile_args={
                'cxx': ['-fopenmp'],
                'nvcc': ['-O3']
            },
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 