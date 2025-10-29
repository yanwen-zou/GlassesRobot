from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import torch, os

def get_extensions():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    abi_flag = torch._C._GLIBCXX_USE_CXX11_ABI

    c_flags = [
        '-O3', '-std=c++17',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(abi_flag)}'
    ]
    nvcc_flags = [
        '-O3', '-std=c++17',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(abi_flag)}',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__'
    ]

    ext_modules = [
        CUDAExtension(
            'common',
            ['bindings.cpp', 'common.cu'],
            extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}
        ),
        CUDAExtension(
            'gridencoder',
            [
                f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
                f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
            ],
            extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}
        ),
    ]
    return ext_modules

setup(
    name='common',
    ext_modules=get_extensions(),
    include_dirs=[
        # ✅ 优先使用 conda eigen
        os.path.join(os.environ['CONDA_PREFIX'], 'include', 'eigen3'),
        # ✅ 然后 torch 自带头文件
        *include_paths(),
        # ✅ 最后系统 eigen（仅备用）
        '/usr/include/eigen3',
    ],
    cmdclass={'build_ext': BuildExtension},
)
