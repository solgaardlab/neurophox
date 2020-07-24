from typing import Optional, List, Union

from .generic import MeshTorchLayer, PermutationLayer
from ..meshmodel import RectangularMeshModel, TriangularMeshModel, PermutingRectangularMeshModel, ButterflyMeshModel
from ..helpers import rectangular_permutation, butterfly_layer_permutation
from ..config import DEFAULT_BASIS

import numpy as np


class RMTorch(MeshTorchLayer):
    """Rectangular mesh network layer for unitary operators implemented in tensorflow

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        num_layers: The number of layers (:math:`L`) of the mesh
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init: Initializer for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(theta_init, theta_fn)`.
        phi_init: Initializer for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`):
                  a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(phi_init, phi_fn)`.
        gamma_init: Initializer for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`):
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(gamma_init, gamma_fn)`.
    """

    def __init__(self, units: int, num_layers: int = None, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init: Union[str, tuple, np.ndarray] = "haar_rect",
                 phi_init: Union[str, tuple, np.ndarray] = "random_phi",
                 gamma_init: Union[str, tuple, np.ndarray] = "random_gamma"):
        super(RMTorch, self).__init__(
            RectangularMeshModel(units, num_layers, hadamard, bs_error, basis,
                                 theta_init, phi_init, gamma_init))


class TMTorch(MeshTorchLayer):
    """Triangular mesh network layer for unitary operators implemented in tensorflow

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init: Initializer for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(theta_init, theta_fn)`.
        phi_init: Initializer for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`):
                  a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(phi_init, phi_fn)`.
        gamma_init: Initializer for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`):
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(gamma_init, gamma_fn)`.
    """

    def __init__(self, units: int, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init: Union[str, tuple, np.ndarray] = "haar_rect",
                 phi_init: Union[str, tuple, np.ndarray] = "random_phi",
                 gamma_init: Union[str, tuple, np.ndarray] = "random_gamma"):
        super(TMTorch, self).__init__(
            TriangularMeshModel(units, hadamard, bs_error, basis,
                                theta_init, phi_init, gamma_init)
        )


class PRMTorch(MeshTorchLayer):
    """Permuting rectangular mesh unitary layer

    Args:
        units: The dimension of the unitary matrix (:math:`N`) to be modeled by this transformer
        tunable_layers_per_block: The number of tunable layers per block (overrides :code:`num_tunable_layers_list`, :code:`sampling_frequencies`)
        num_tunable_layers_list: Number of tunable layers in each block in order from left to right
        sampling_frequencies: Frequencies of sampling frequencies between the tunable layers
        bs_error: Photonic error in the beamsplitter
        theta_init: Initializer for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(theta_init, theta_fn)`.
        phi_init: Initializer for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`):
                  a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(phi_init, phi_fn)`.
        gamma_init: Initializer for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`):
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(gamma_init, gamma_fn)`.
    """

    def __init__(self, units: int, tunable_layers_per_block: int = None,
                 num_tunable_layers_list: Optional[List[int]] = None, sampling_frequencies: Optional[List[int]] = None,
                 bs_error: float = 0.0, hadamard: bool = False,
                 theta_init: Union[str, tuple, np.ndarray] = "haar_rect",
                 phi_init: Union[str, tuple, np.ndarray] = "random_phi",
                 gamma_init: Union[str, tuple, np.ndarray] = "random_gamma"):
        if theta_init == 'haar_prm' and tunable_layers_per_block is not None:
            raise NotImplementedError('haar_prm initializer is incompatible with setting tunable_layers_per_block.')
        super(PRMTorch, self).__init__(
            PermutingRectangularMeshModel(units, tunable_layers_per_block, num_tunable_layers_list,
                                          sampling_frequencies, bs_error, hadamard,
                                          theta_init, phi_init, gamma_init)
        )


class BMTorch(MeshTorchLayer):
    """Butterfly mesh unitary layer

    Args:
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init: Initializer for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(theta_init, theta_fn)`.
        phi_init: Initializer for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`):
                  a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(phi_init, phi_fn)`.
        gamma_init: Initializer for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`):
                    a :code:`str`, :code:`ndarray`, or tuple of the form :code:`(gamma_init, gamma_fn)`.
    """

    def __init__(self, num_layers: int, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init: Union[str, tuple, np.ndarray] = "haar_rect",
                 phi_init: Union[str, tuple, np.ndarray] = "random_phi",
                 gamma_init: Union[str, tuple, np.ndarray] = "random_gamma"):
        super(BMTorch, self).__init__(
            ButterflyMeshModel(num_layers, hadamard, bs_error, basis, theta_init, phi_init, gamma_init)
        )


class RectangularPerm(PermutationLayer):
    """Rectangular permutation layer

    The rectangular permutation layer for a frequency :math:`f` corresponds effectively is equivalent to adding
    :math:`f` layers of cross state MZIs in a grid configuration to the existing mesh.

    Args:
        units: Dimension of the input (number of input waveguide ports), :math:`N`
        frequency: Frequency of interacting mesh wires (waveguides)
    """

    def __init__(self, units: int, frequency: int):
        self.frequency = frequency
        super(RectangularPerm, self).__init__(permuted_indices=rectangular_permutation(units, frequency))


class ButterflyPerm(PermutationLayer):
    """Butterfly (FFT) permutation layer

    The butterfly or FFT permutation for a frequency :math:`f` corresponds to switching all inputs
    that are :math:`f` inputs apart. This works most cleanly in a butterfly mesh architecture where
    the number of inputs, :math:`N`, and the frequencies, :math:`f` are powers of two.

    Args:
        units: Dimension of the input (number of input waveguide ports), :math:`N`
        frequency: Frequency of interacting mesh wires (waveguides)
    """

    def __init__(self, units: int, frequency: int):
        self.frequency = frequency
        super(ButterflyPerm, self).__init__(permuted_indices=butterfly_layer_permutation(units, frequency))
