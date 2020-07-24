<div style="text-align:center"><img src="https://user-images.githubusercontent.com/7623867/57964848-ae1b0000-78f0-11e9-8cc8-a9ba628c3a89.png" width="50%" alt="Logo"></div>

# 
![Build Status](https://img.shields.io/travis/solgaardlab/neurophox/master.svg?style=for-the-badge)
![Docs](https://readthedocs.org/projects/neurophox/badge/?style=for-the-badge)
![PiPy](https://img.shields.io/pypi/v/neurophox.svg?style=for-the-badge)
![CodeCov](https://img.shields.io/codecov/c/github/solgaardlab/neurophox/master.svg?style=for-the-badge)


The [Neurophox module](https://neurophox.readthedocs.io) is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in [arxiv/1909.06179](https://arxiv.org/pdf/1909.06179.pdf), [arxiv/1808.00458](https://arxiv.org/pdf/1808.00458.pdf), and [arxiv/1903.04579](https://arxiv.org/pdf/1903.04579.pdf).

![neurophox](https://user-images.githubusercontent.com/7623867/57964658-87a79580-78ed-11e9-8f1e-c4af30c32e65.gif)


![neurophox](https://user-images.githubusercontent.com/7623867/57976056-57fb9a80-798c-11e9-9aef-8d1f07af7ca7.gif)


## Motivation

Neurophox provides a robust and efficient framework for simulating **optical neural networks** (ONNs) that promise fast and energy-efficient machine learning. Scalable ONNs are made possible by integrated [**reconfigurable nanophotonic processors**](https://www.osapublishing.org/optica/abstract.cfm?uri=optica-5-12-1623), feedforward networks of 2 x 2 nodes that compute matrix multiplication by simply allowing light to flow through them (thus, a time- and energy-efficient way to compute).
 
 Reconfigurable photonic networks are currently being [developed to scale](https://medium.com/lightmatter/matrix-processing-with-nanophotonics-998e294dabc1) in the optical domain using Mach-Zehnder interferometers (MZIs). Such matrix multiplier processors behave differently from conventional matrix multipliers:
1. They act as **unitary operators rather than general linear operators**, preserving the norm of the data flowing through the network (i.e., intensity of the light is preserved due to energy conservation).
2. The matrix elements are not directly trained during backpropagation. Instead, the node parameters are trained, and the **matrix elements are nonlinear functions of those node parameters**.

The goal of Neurophox is to provide utilities for calibrating and training such processors for machine learning and other signal processing tasks. These simulated models might furthermore be useful for a relatively unexplored regime of machine learning: "unitary machine learning" (see e.g., [arxiv/1612.05231](https://arxiv.org/pdf/1612.05231.pdf)).

## Dependencies and requirements

Some important requirements for Neurophox are:
1. Python >=3.6
2. Tensorflow >=2.0.0a
3. PyTorch

The dependencies for Neurophox (specified in `requirements.txt`) are:
```text
numpy>=1.16
scipy
matplotlib
tensorflow>=2.0.0a
```

Optionally, the user may install `torch>=1.7` to run the Pytorch layers.

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
    pip install -r neurophox/requirements.txt
    ```
    
If installing other dependencies manually, ensure you install [PyTorch](https://pytorch.org/) (since PyTorch mesh layers are currently in development) and [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf).

#### Using the GPU

If using a GPU, we recommend using a `conda` environement to install GPU dependencies using CUDA 10.0 with the following commands:
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly
pip install tensorflow-gpu==2.2.0
```

### Features

Using Neurophox, we can simulate light physically propagating through such networks (using `layer.propagate`), and observe its equivalence to a _matrix operation_ (using `layer.matrix`).

We demonstrate the **R**ectangular **M**esh (RM layer) using either `numpy` or `tensorflow`:

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

### Visualizations

We provide code to visualize network phase shift parameters and light propagation in [`neurophox-notebooks`](https://github.com/solgaardlab/neurophox-notebooks) for rectangular and triangular meshes.

Rectangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964850-aeb39680-78f0-11e9-8785-e6e46c705b34.png)
Triangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964852-aeb39680-78f0-11e9-8a5c-d08e9f6dce89.png)

For the **phase shift settings** above, we visualize the **propagation of light** (scaled by light intensity), equivalently representing data "flowing" through the mesh.

Rectangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964851-aeb39680-78f0-11e9-9ff3-41e8cebd25a6.png)
Triangular mesh:
![neurophox](https://user-images.githubusercontent.com/7623867/57964853-aeb39680-78f0-11e9-8cd4-1364d2cec339.png)

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
1. Definition and calibration of feedforward photonic networks:
    ```text
    @article{pai2019parallel,
      title={Parallel fault-tolerant programming of an arbitrary feedforward photonic network},
      author={Pai, Sunil and Williamson, Ian AD and Hughes, Tyler W and Minkov, Momchil and Solgaard, Olav and Fan, Shanhui and Miller, David AB},
      journal={arXiv preprint arXiv:1909.06179},
      year={2019}
    }
    ```
2. Optimization of unitary mesh networks:
    ```text
    @article{pai_matrix_2019,
      author = {Pai, Sunil and Bartlett, Ben and Solgaard, Olav and Miller, David A. B.},
      doi = {10.1103/PhysRevApplied.11.064044},
      journal = {Physical Review Applied},
      month = jun,
      number = {6},
      pages = {064044},
      title = {Matrix Optimization on Universal Unitary Photonic Devices},
      volume = {11},
      year = {2019}
    }
    ```
3. Optical neural network nonlinearities:
    ```text
    @article{williamson_reprogrammable_2020,
      author = {Williamson, I. A. D. and Hughes, T. W. and Minkov, M. and Bartlett, B. and Pai, S. and Fan, S.},
      doi = {10.1109/JSTQE.2019.2930455},
      issn = {1077-260X},
      journal = {IEEE Journal of Selected Topics in Quantum Electronics},
      month = jan,
      number = {1},
      pages = {1-12},
      title = {Reprogrammable Electro-Optic Nonlinear Activation Functions for Optical Neural Networks},
      volume = {26},
      year = {2020}
    }
    ```

## Future Work and Contributions

Neurophox is under development and is not yet stable. 

If you find a bug, have a question, or would like to recommend a feature, please submit an issue on Github.

We welcome pull requests and contributions from the broader community. If you would like to contribute, please submit a pull request and title your branch `bug/bug-fix-title` or `feature/feature-title`.


