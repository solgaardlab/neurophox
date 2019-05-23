<div style="text-align:center"><img src="https://user-images.githubusercontent.com/7623867/57964848-ae1b0000-78f0-11e9-8cc8-a9ba628c3a89.png" width="50%" alt="Logo"></div>

# 
![Build Status](https://img.shields.io/travis/solgaardlab/neurophox/master.svg?style=for-the-badge)
![Docs](https://readthedocs.com/projects/solgaard-lab-neurophox/badge/?version=latest&style=for-the-badge)
![PiPy](https://img.shields.io/pypi/v/neurophox.svg?style=for-the-badge)
![CodeCov](https://img.shields.io/codecov/c/github/solgaardlab/neurophox/master.svg?style=for-the-badge)


The [Neurophox module](https://solgaard-lab-neurophox.readthedocs-hosted.com/en/latest/) is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in [arxiv/1808.00458](https://arxiv.org/pdf/1808.00458.pdf) and [arxiv/1903.04579](https://arxiv.org/pdf/1903.04579.pdf).

![neurophox](https://user-images.githubusercontent.com/7623867/57964658-87a79580-78ed-11e9-8f1e-c4af30c32e65.gif)


![neurophox](https://user-images.githubusercontent.com/7623867/57976056-57fb9a80-798c-11e9-9aef-8d1f07af7ca7.gif)


## Motivation

Orthogonal and unitary neural networks have interesting properties and have been studied for synthetic natural language processing tasks (see [unitary mesh-based RNN](http://proceedings.mlr.press/v70/jing17a/jing17a.pdf), [unitary evolution RNN](https://arxiv.org/pdf/1511.06464.pdf), and [orthogonal evolution RNN](https://arxiv.org/pdf/1602.06662.pdf)). Furthermore, new energy-efficient photonic technologies are being built to realize unitary mesh-based neural networks using light as the computing medium as opposed to conventional analog electronics.

## Introduction

Neurophox provides a robust and general framework for mesh network layers in orthogonal and unitary neural networks. We use an efficient definition for any feedforward mesh architecture, `neurophox.meshmodel.MeshModel`, to develop mesh layer architectures in Numpy (`neurophox.numpy.layers`), Tensorflow 2 (`neurophox.tensorflow.layers`), and (soon) PyTorch.

Scattering matrix models used in unitary mesh networks for photonics simulations are provided in `neurophox.components`. The models for all layers are fully defined in `neurophox.meshmodel`, which provides a general framework for efficient implementation of any feedforward unitary mesh network.

## Dependencies and requirements

Some important requirements for Neurophox are:
1. Python >=3.6
2. Tensorflow >=2.0.0a
3. PyTorch

The dependencies for Neurophox (specified in `requirements.txt`) are:
```text
numpy
scipy
matplotlib
tensorflow>=2.0.0a
torch>=1.1
```

## Getting started

### Installation

There are currently two options to install Neurophox:
1. Installation via `pip`:
    ```bash
    pip install neurophox
    ```
2. Installation from source via `pip`:
    ```bash
    git clone https://github.com/solgaardlab/neurophox
    pip install -e neurophox
    pip install -r requirements.txt
    ```
    
If installing other dependencies manually, ensure you install [PyTorch](https://pytorch.org/) (since PyTorch mesh layers are currently in development) and [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf).

#### Using the GPU

If using a GPU, we recommend using a `conda` environement to install GPU dependencies using CUDA 10.0 with the following commands:
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu==2.0.0-alpha0
```

### Imports

```python
import numpy as np
from neurophox.numpy import RMNumpy
from neurophox.tensorflow import RM

N = 16

tf_layer = RM(N)
np_layer = RMNumpy(N, phases=tf_layer.phases)

np.allclose(tf_layer.matrix, np_layer.matrix)  # True
np.allclose(tf_layer(np.eye(N)), np_layer.matrix)  # True
```

### Inspection

We can inspect the parameters for each layer using `neurophox.control.MeshPhases` which can be accessed via `tf_layer.phases` and `np_layer.phases`.


We can inspect the matrix elements implemented by each layer as follows via `tf_layer.matrix` and `np_layer.matrix`.

### Phase shift settings visualization
The phase shift patterns used to generate the above propagation patterns can also be visualized by plotting `np_layer.phases`:

Rectangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964850-aeb39680-78f0-11e9-8785-e6e46c705b34.png)
Triangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964852-aeb39680-78f0-11e9-8a5c-d08e9f6dce89.png)


### Light propagation visualization

For the phase shift settings above, we can visualize the propagation of light (field magnitude), as the data "flows" through the mesh.

Rectangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964851-aeb39680-78f0-11e9-9ff3-41e8cebd25a6.png)
Triangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964853-aeb39680-78f0-11e9-8cd4-1364d2cec339.png)


The code to generate these visualization examples are provided in [`neurophox-notebooks`](https://github.com/solgaardlab/neurophox-notebooks).


### Small machine learning example

It is possible to compose Neurophox Tensorflow layers into unitary neural networks using `tf.keras.Sequential`
to solve machine learning problems. Here we use absolute value nonlinearities and categorical cross entropy.

```python
import tensorflow as tf
from neurophox.tensorflow import RM
from neurophox.ml.nonlinearities import cnorm, cnormsq

ring_model = tf.keras.Sequential([
    RM(3, activation=tf.keras.layers.Activation(cnorm)),
    RM(3, activation=tf.keras.layers.Activation(cnorm)),
    RM(3, activation=tf.keras.layers.Activation(cnorm)),
    RM(3, activation=tf.keras.layers.Activation(cnorm)),
    tf.keras.layers.Activation(cnormsq),
    tf.keras.layers.Lambda(lambda x: tf.math.real(x[:, :2])), # get first 2 output ports (we already know it is real from the activation),
    tf.keras.layers.Activation('softmax')
])

ring_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.0025)
)
```

Below is a visualization for many planar classification problems:

![neurophox](https://user-images.githubusercontent.com/7623867/58128090-b2cf0500-7bcb-11e9-8986-25450bfd68a9.png)
![neurophox](https://user-images.githubusercontent.com/7623867/58132218-95069d80-7bd5-11e9-9d08-20e1de5c3727.png)
The code to generate the above example is provided in [`neurophox-notebooks`](https://github.com/solgaardlab/neurophox-notebooks).


## Authors and citing this repository
Neurophox was written by Sunil Pai (email: sunilpai@stanford.edu).

If you find this repository useful, please cite at least one of the following papers depending on your application:
1. Optimization of unitary mesh networks:
    ```text
    @article{pai2018matrix,
      title={Matrix optimization on universal unitary photonic devices},
      author={Pai, Sunil and Bartlett, Ben and Solgaard, Olav and Miller, David AB},
      journal={arXiv preprint arXiv:1808.00458},
      year={2018}
    }
    ```
2. Optical neural network nonlinearities:
    ```text
    @article{williamson2019reprogrammable,
      title={Reprogrammable Electro-Optic Nonlinear Activation Functions for Optical Neural Networks},
      author={Williamson, Ian A.D. and Hughes, Tyler W. and Minkov, Momchil and Bartlett, Ben and Pai, Sunil and Fan, Shanhui},
      journal={arXiv preprint arXiv:1903.04579},
      year={2019}
    }
    ```

## Future Work and Contributions

Neurophox is under development and is not yet stable. 

If you find a bug, have a question, or would like to recommend a feature, please submit an issue on Github.

We welcome pull requests and contributions from the broader community. If you would like to contribute, please submit a pull request and title your branch `bug/bug-fix-title` or `feature/feature-title`.

Some future feature ideas include:
1. Implement multi-wavelength and multi-mode operators for photonics simulation using batch and broadcasting operations.
2. Add nonlinearities and a unitary mesh-based RNN cell for use in deep learning architectures.



