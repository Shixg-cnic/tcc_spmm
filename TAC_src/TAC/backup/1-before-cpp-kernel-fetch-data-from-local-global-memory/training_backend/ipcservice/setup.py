import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
setup(
    name='ipcservice',
    ext_modules=[
        CUDAExtension(
            name='ipc_service', 
            sources= [
                'src/ipc_service.cpp',
                'src/helper_multiprocess.cpp',
                'src/ipc_cuda_kernel.cu',
            ],
            include_dirs=[os.path.abspath('./include')] # 需要使用绝对路径
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })