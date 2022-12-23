from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='perm_utils',
      ext_modules=[cpp_extension.CppExtension('perm_utils', ['permutation.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
