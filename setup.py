#!/usr/bin/env python
from setuptools import setup, find_packages

project_name = "neurophox"

requirements = [
    "numpy",
    "scipy",
    "torch>=1.1",
    "tensorflow>=2.0.0a"
]

setup(
    name=project_name,
    version="0.1.0-alpha.1",
    packages=find_packages(),
    description='A simulation framework for unitary neural networks and photonic devices',
    author='Sunil Pai',
    author_email='sunilpai@stanford.edu',
    license='MIT',
    url="https://github.com/solgaardlab/neurophox",
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ),
    install_requires=requirements
)
