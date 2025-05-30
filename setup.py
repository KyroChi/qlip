# install QLIP
# Download LLaVA locally as well as downloading the LLaVA models
# Apply patch to VLMEvalKit?
# Download V* Benchmark and set VSTAR_BENCHMARK_FOLDER environment variable
from datasets import load_dataset
from setuptools import setup

setup(
    name='qlip',
    packages=['qlip'],
    version='0.1.0',
    description='Code for enabling QLIP vision encoder',
    author='Kyle R. Chickering, Bangzheng Li, and Muhao Chen',
)