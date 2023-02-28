from setuptools import setup, Extension
from torch.utils import cpp_extension
ext = cpp_extension.CppExtension('sparse_tensor_ops', ['sparse_tensor_ops.cpp'],
                                 extra_compile_args=['-fopenmp'],
                                 extra_link_args=['-lgomp'])
setup(name='sparse_tensor_ops',
      ext_modules=[ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
