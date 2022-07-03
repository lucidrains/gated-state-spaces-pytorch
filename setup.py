from setuptools import setup, find_packages

setup(
  name = 'gated-state-spaces-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.10',
  license='MIT',
  description = 'Gated State Spaces - GSS - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/gated-state-spaces-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'state spaces',
    'long context'
  ],
  install_requires=[
    'einops>=0.4',
    'scipy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
