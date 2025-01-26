from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch
from pathlib import Path
import shutil
import sys

# Create build directory if it doesn't exist
build_dir = Path(__file__).parent / 'build'
if build_dir.exists():
    shutil.rmtree(build_dir)
build_dir.mkdir(parents=True)
(build_dir / 'lib').mkdir(exist_ok=True)

# Set environment variables to control build output
os.environ['TORCH_BUILD_DIR'] = str(build_dir)
os.environ['BUILD_LIB'] = str(build_dir / 'lib')
os.environ['BUILD_TEMP'] = str(build_dir / 'temp')

class CustomBuildExtension(BuildExtension):
    def get_ext_filename(self, ext_name):
        # Force output to build directory
        filename = super().get_ext_filename(ext_name)
        return str(build_dir / 'lib' / os.path.basename(filename))

    def get_ext_fullpath(self, ext_name):
        # Override to ensure extension is built in our build directory
        filename = self.get_ext_filename(ext_name)
        return str(build_dir / 'lib' / filename)

setup(
    name='sparse_mlp',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='sparse_mlp',
            sources=['sparse_mlp/csrc/sparse_mlp_op.cpp'],
            extra_compile_args={
                'cxx': ['-O3', '-ffast-math', '-march=native', '-fopenmp'],
            },
            libraries=['gomp'],
            include_dirs=[
                # Main PyTorch include directory
                os.path.dirname(torch.__file__) + '/include',
                # TorchScript includes
                os.path.dirname(torch.__file__) + '/include/torch/csrc/api/include',
                # ATen includes
                os.path.dirname(torch.__file__) + '/include/ATen',
                # C10 includes
                os.path.dirname(torch.__file__) + '/include/c10',
            ],
            library_dirs=[str(build_dir / 'lib')],  # Specify library output directory
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension.with_options(no_python_abi_suffix=True),
    }
) 