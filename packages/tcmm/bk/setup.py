from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='tcmm_cpp',
    ext_modules=[CUDAExtension(
                   name='tcmm', 
                   sources=['tcmm.cpp', 'tcmm_kernel.cu'],
                   libraries=['cusolver'],
                   library_dirs=['/usr/local/cuda-10.1/lib64'],
                   include_dirs=['/usr/local/cuda-10.1/samples/common/inc/']
                )],
    cmdclass={'build_ext': BuildExtension},
    author='Shaohuai Shi',
    author_email='shaohuais@cse.ust.hk',
    description='Efficient PyTorch Extension for K-FAC',
)

