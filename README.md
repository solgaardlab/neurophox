# ![neurophox](media/logo.png)

The `neurophox` module is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in [arxiv/1808.00458](https://arxiv.org/pdf/1808.00458.pdf) and [arxiv/1903.04579](https://arxiv.org/abs/1903.04579).

## Motivation

Orthogonal and unitary neural networks have interesting properties and have been studied for synthetic natural language processing tasks (see [EUNN](http://proceedings.mlr.press/v70/jing17a/jing17a.pdf), [uRNN](https://arxiv.org/pdf/1511.06464.pdf), and [oRNN](https://arxiv.org/pdf/1602.06662.pdf)). Furthermore, new photonic technologies are being built to realize such neural networks using light as the computing medium as opposed to conventional electronics.

## Introduction

`neurophox` provides a robust and general framework for mesh network implementations of orthogonal and unitary neural networks. We use an efficient and general definition for feedforward mesh architectures in `neurophox.meshmodel` to allow for a plug-and-play interface for defining mesh layer architectures in Numpy (`neurophox.numpy.layers`), Tensorflow 2 (`neurophox.tensorflow.layers`), and (soon) PyTorch.

Scattering matrix models used in unitary mesh networks for photonics simulations are provided in `neurophox.components`. The models for all layers are fully defined in `neurophox.meshmodel`, which provides a general framework for efficient implementation of any unitary mesh network.

## Getting started

There are three options to install `neurophox`:
1. Installation via `conda` (recommended, must be in a dedicated conda environment!).
    
    GPU version:
    ```angular2html
    conda install -c sunilpai neurophox-gpu
    ```
    CPU version:
    ```angular2html
    conda install -c sunilpai neurophox
    ```
2. Installation via `pip` (all other dependencies installed manually).
    ```angular2html
    pip install neurophox
    ```
3. Installation from source via `pip`.
    ```angular2html
    git clone https://github.com/solgaardlab/neurophox
    pip install -e .
    pip install
    ```


## Contributions

The module is under development and is not yet stable. We welcome pull requests and contributions from the broader community.

If you would like to contribute, please submit a pull request. If you find a bug, please submit an issue on Github.

## Authors
`neurophox` was written by Sunil Pai.