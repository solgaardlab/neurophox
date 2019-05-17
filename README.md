# ![neurophox](media/logo.png)

The `neurophox` module is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in [arxiv/1808.00458](https://arxiv.org/pdf/1808.00458.pdf) and [arxiv/1903.04579](https://arxiv.org/pdf/1903.04579.pdf).

## Motivation

Orthogonal and unitary neural networks have interesting properties and have been studied for synthetic natural language processing tasks (see [unitary mesh-based RNN](http://proceedings.mlr.press/v70/jing17a/jing17a.pdf), [unitary evolution RNN](https://arxiv.org/pdf/1511.06464.pdf), and [orthogonal evolution RNN](https://arxiv.org/pdf/1602.06662.pdf)). Furthermore, new energy-efficient photonic technologies are being built to realize such neural networks using light as the computing medium as opposed to conventional analog electronics.

## Introduction

`neurophox` provides a robust and general framework for mesh network implementations of orthogonal and unitary neural networks. We use an efficient definition for any feedforward mesh architecture, `neurophox.meshmodel.MeshModel`, to develop mesh layer architectures in Numpy (`neurophox.numpy.layers`), Tensorflow 2 (`neurophox.tensorflow.layers`), and (soon) PyTorch.

Scattering matrix models used in unitary mesh networks for photonics simulations are provided in `neurophox.components`. The models for all layers are fully defined in `neurophox.meshmodel`, which provides a general framework for efficient implementation of any unitary mesh network.

## Getting started

### Installation

There are three options to install `neurophox`:
1. Installation via `conda` (must be in a dedicated conda environment!) for Linux and Mac OS targets.

    GPU version:
    ```bash
    conda install -c sunilpai neurophox-gpu
    ```
    CPU version:
    ```bash
    conda install -c sunilpai neurophox
    ```
2. Installation via `pip` (all other dependencies installed manually).
    ```bash
    pip install neurophox
    ```
3. Installation from source via `pip`.
    ```bash
    git clone https://github.com/solgaardlab/neurophox
    pip install -e .
    pip install -r requirements.txt
    ```
    
If not installing via `conda`, you'll need to install [PyTorch](https://pytorch.org/) (for future version compatibility) and [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf).

If using the `conda` package installation, it is much easier to install GPU dependencies using CUDA 10.0 using the following commands:
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu==2.0.0-alpha0
```

### Examples

#### Imports

```python
import numpy as np
from neurophox.numpy import RMNumpy
from neurophox.tensorflow import RM

N = 16

tf_layer = RM(N)
np_layer = RMNumpy(N, phases=tf_layer.phases)

np.allclose(tf_layer.matrix, np_layer.matrix)  # True
```

#### Inspection

We can inspect the parameters for each layer using `neurophox.control.MeshPhases` which can be accessed via `tf_layer.phases` and `np_layer.phases`.


We can inspect the matrix elements implemented by each layer as follows via `tf_layer.matrix` and `np_layer.matrix`.


#### Visualize

Matrix visualization works by writing `tf_layer.plot(plt)` or `np_layer.plot(plt)` in a Jupyter notebook.


More visualization examples are provided in [].

## Contributions

The module is under development and is not yet stable. We welcome pull requests and contributions from the broader community.

If you would like to contribute, please submit a pull request. If you find a bug, please submit an issue on Github.

## Dependencies and requirements

Some important requirements for `neurophox` are:
1. Python >=3.6
2. Tensorflow 2.0
3. PyTorch 1.1

The dependencies for `neurophox` (specified in `requirements.txt`) are:
```text
numpy
scipy
matplotlib
tensorflow==2.0
torch==1.1
tensorboard
```

## Authors
`neurophox` was written by Sunil Pai (email: sunilpai@stanford.edu).