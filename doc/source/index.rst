.. neurophox documentation master file, created by
   sphinx-quickstart on Sun May 19 10:27:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Neurophox
=========

Neurophox is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in `arxiv/1808.00458 <https://arxiv.org/pdf/1808.00458.pdf>`_ and `arxiv/1903.04579 <https://arxiv.org/pdf/1903.04579.pdf>`_.

`Note`: ``neurophox`` is under active development and is not yet stable, so expect to see the documentation change frequently.

.. image:: https://user-images.githubusercontent.com/7623867/57964658-87a79580-78ed-11e9-8f1e-c4af30c32e65.gif
   :target: https://user-images.githubusercontent.com/7623867/57964658-87a79580-78ed-11e9-8f1e-c4af30c32e65.gif
   :alt: neurophox
   :width: 100%



.. image:: https://user-images.githubusercontent.com/7623867/57976056-57fb9a80-798c-11e9-9aef-8d1f07af7ca7.gif
   :target: https://user-images.githubusercontent.com/7623867/57976056-57fb9a80-798c-11e9-9aef-8d1f07af7ca7.gif
   :alt: neurophox
   :width: 100%


Motivation
----------

Orthogonal and unitary neural networks have interesting properties and have been studied for synthetic natural language processing tasks (see `unitary mesh-based RNN <http://proceedings.mlr.press/v70/jing17a/jing17a.pdf>`_\ , `unitary evolution RNN <https://arxiv.org/pdf/1511.06464.pdf>`_\ , and `orthogonal evolution RNN <https://arxiv.org/pdf/1602.06662.pdf>`_\ ). Furthermore, new energy-efficient photonic technologies are being built to realize unitary mesh-based neural networks using light as the computing medium as opposed to conventional analog electronics.

Introduction
------------

Neurophox provides a general framework for mesh network layers in orthogonal and unitary neural networks. We use an efficient definition for any feedforward mesh architecture, ``neurophox.meshmodel.MeshModel``\ , to develop mesh layer architectures in Numpy (\ ``neurophox.numpy.layers``\ ), `Tensorflow 2.0 <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf>`_ (\ ``neurophox.tensorflow.layers``\ ), and (soon) `PyTorch <https://pytorch.org/>`_.

Scattering matrix models used in unitary mesh networks for photonics simulations are provided in ``neurophox.components``. The models for all layers are fully defined in ``neurophox.meshmodel``\ , which provides a general framework for efficient implementation of any unitary mesh network.


Dependencies and requirements
-----------------------------

Some important requirements for ``neurophox`` are:


#. Python >=3.6
#. Tensorflow >=2.0
#. PyTorch >=1.3

The dependencies for ``neurophox`` (specified in ``requirements.txt``\ ) are:

.. code-block:: text

   numpy>=1.16
   scipy
   matplotlib
   tensorflow>=2.0.0a

The user may also optionally install ``torch>=1.3`` to run the ``neurophox.torch`` module.

Getting started
---------------

Installation
^^^^^^^^^^^^

There are currently two options to install ``neurophox``\ :


#. Installation via ``pip``\ :

    .. code-block:: bash

        pip install neurophox

#. Installation from source via ``pip``\ :

    .. code-block:: bash

        git clone https://github.com/solgaardlab/neurophox
        pip install -e neurophox
        pip install -r requirements.txt

If installing other dependencies manually, ensure you install `PyTorch <https://pytorch.org/>`_ (since PyTorch mesh layers are currently in development) and `Tensorflow 2.0 <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf>`_.

Using the GPU
~~~~~~~~~~~~~

If using a GPU, we recommend using a ``conda`` environement to install GPU dependencies using CUDA 10.0 with the following commands:

.. code-block:: bash

   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   pip install tensorflow-gpu==2.0.0-alpha0

Imports
^^^^^^^

.. code-block:: python

   import numpy as np
   from neurophox.numpy import RMNumpy
   from neurophox.tensorflow import RM

   N = 16

   tf_layer = RM(N)
   np_layer = RMNumpy(N, phases=tf_layer.phases)

   np.allclose(tf_layer.matrix, np_layer.matrix)  # True
   np.allclose(tf_layer(np.eye(N)), np_layer.matrix)  # True


Inspection
^^^^^^^^^^

We can inspect the parameters for each layer using ``neurophox.control.MeshPhases`` which can be accessed via ``tf_layer.phases`` and ``np_layer.phases``.

We can inspect the matrix elements implemented by each layer as follows via ``tf_layer.matrix`` and ``np_layer.matrix``.



Bottleneck
^^^^^^^^^^
The bottleneck in time for a ``neurophox`` mesh layer is `not` dominated by the number of inputs/outputs (referred to as ``units`` or :math:`N`), but rather the number of "vertical layers" in the mesh layer (referred to as ``num_layers`` or :math:`L`). This is to say that fairly large unitary matrices can be implemented by  ``neurophox`` as long as the ``num_layers`` parameter is kept small (e.g., empirically, :math:`\log N` layers seems to be enough for butterfly mesh architectures used in `unitary mesh-based RNNs <http://proceedings.mlr.press/v70/jing17a/jing17a.pdf>`_\ ).


Visualizations
--------------


Phase shift settings visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The phase shift patterns used to generate the above propagation patterns can also be visualized by plotting ``np_layer.phases``\ :

Rectangular mesh:

.. image:: https://user-images.githubusercontent.com/7623867/57964850-aeb39680-78f0-11e9-8785-e6e46c705b34.png
   :target: https://user-images.githubusercontent.com/7623867/57964850-aeb39680-78f0-11e9-8785-e6e46c705b34.png
   :alt: neurophox
   :width: 100%

Triangular mesh:

.. image:: https://user-images.githubusercontent.com/7623867/57964852-aeb39680-78f0-11e9-8a5c-d08e9f6dce89.png
   :target: https://user-images.githubusercontent.com/7623867/57964852-aeb39680-78f0-11e9-8a5c-d08e9f6dce89.png
   :alt: neurophox
   :width: 100%


Light propagation visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the phase shift settings above, we can visualize the propagation of light (field magnitude), as the data "flows" through the mesh.

Rectangular mesh:

.. image:: https://user-images.githubusercontent.com/7623867/57964851-aeb39680-78f0-11e9-9ff3-41e8cebd25a6.png
   :target: https://user-images.githubusercontent.com/7623867/57964851-aeb39680-78f0-11e9-9ff3-41e8cebd25a6.png
   :alt: neurophox
   :width: 100%

Triangular mesh:

.. image:: https://user-images.githubusercontent.com/7623867/57964853-aeb39680-78f0-11e9-8cd4-1364d2cec339.png
   :target: https://user-images.githubusercontent.com/7623867/57964853-aeb39680-78f0-11e9-8cd4-1364d2cec339.png
   :alt: neurophox
   :width: 100%


The code to generate these visualization examples are provided in `neurophox-notebooks <https://github.com/solgaardlab/neurophox-notebooks>`_\.

Small machine learning example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is simple to compose ``neurophox`` Tensorflow layers into unitary neural networks using ``Sequential`` to solve machine learning problems.


.. code-block:: python

   import tensorflow as tf
   from neurophox.tensorflow import RM
   from neurophox.ml.nonlinearities import cnorm, cnormsq

   ring_model = tf.keras.Sequential([
       RM(3, activation=tf.keras.layers.Activation(cnorm)),
       RM(3, activation=tf.keras.layers.Activation(cnorm)),
       RM(3, activation=tf.keras.layers.Activation(cnorm)),
       RM(3, activation=tf.keras.layers.Activation(cnorm)),
       tf.keras.layers.Activation(cnormsq),
       tf.keras.layers.Lambda(lambda x: tf.math.real(x[:, :2])), # get first 2 output ports,
       tf.keras.layers.Activation('softmax')
   ])

   ring_model.compile(
       loss='categorical_crossentropy',
       optimizer=tf.keras.optimizers.Adam(lr=0.0025)
   )

Below is a visualization for many planar classification problems (including the ring model defined above):

.. image:: https://user-images.githubusercontent.com/7623867/58128090-b2cf0500-7bcb-11e9-8986-25450bfd68a9.png
   :target: https://user-images.githubusercontent.com/7623867/58128090-b2cf0500-7bcb-11e9-8986-25450bfd68a9.png
   :alt: neurophox
   :width: 100%

.. image:: https://user-images.githubusercontent.com/7623867/58132218-95069d80-7bd5-11e9-9d08-20e1de5c3727.png
   :target: https://user-images.githubusercontent.com/7623867/58132218-95069d80-7bd5-11e9-9d08-20e1de5c3727.png
   :alt: neurophox
   :width: 100%



The code to generate the above example is provided in `neurophox-notebooks <https://github.com/solgaardlab/neurophox-notebooks>`_\.


Authors and citing this repository
----------------------------------

``neurophox`` was written by Sunil Pai (email: sunilpai@stanford.edu).

If you find this repository useful, please cite at least one of the following papers depending on your application:

#. Calibration of optical neural networks

    .. code-block:: text

        @article{pai2019parallel,
          title={Parallel fault-tolerant programming of an arbitrary feedforward photonic network},
          author={Pai, Sunil and Williamson, Ian AD and Hughes, Tyler W and Minkov, Momchil and Solgaard, Olav and Fan, Shanhui and Miller, David AB},
          journal={arXiv preprint arXiv:1909.06179},
          year={2019}
        }


#. Optimization of unitary mesh networks:

   .. code-block:: text

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

#. Optical neural network nonlinearities:

   .. code-block:: text

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

Contributions
-------------

``neurophox`` is under active development and is not yet stable.

If you find a bug or would like to recommend a feature, please submit an issue on Github.

We welcome pull requests and contributions from the broader community. If you would like to contribute to the codebase, please submit a pull request and title your branch ``bug/bug-fix-title`` or ``feature/feature-title``.

Documentation
=============

``neurophox`` is under active development and is not yet stable, so expect to see the documentation change frequently.

Summary
-------

The layer APIs of ``neurophox`` are provided via the subpackages:

* ``neurophox.tensorflow``
* ``neurophox.numpy``
* ``neurophox.torch``

Some machine learning experiment starter code with ``neurophox`` can be found in `neurophox-notebooks <https://github.com/solgaardlab/neurophox-notebooks>`_\ and/or ``neurophox.ml``.

All other subpackages have code relevant to the inner workings of ``neurophox``.


Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
--------

.. toctree::
   :hidden:

   index

.. toctree::
   neurophox
