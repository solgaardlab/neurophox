from typing import Tuple, Union, Optional

import tensorflow as tf
import torch
from torch.nn.parameter import Parameter

import numpy as np

from .config import TF_FLOAT, NP_FLOAT, TEST_SEED
from .helpers import get_alpha_checkerboard_general, get_default_coarse_grain_block_sizes,\
    get_efficient_coarse_grain_block_sizes
from scipy.stats import rv_discrete


class MeshPhaseInitializer:
    def __init__(self, units: int, num_layers: int):
        """

        Args:
            units: Input dimension, :math:`N`
            num_layers: Number of layers :math:`L`
        """
        self.units, self.num_layers = units, num_layers

    def to_np(self) -> np.ndarray:
        """

        Returns:
            Initialized Numpy array
        """
        raise NotImplementedError('Need to implement numpy initialization')

    def to_tf(self, phase_varname: str) -> tf.Variable:
        """

        Returns:
            Initialized Tensorflow Variable
        """
        phase_np = self.to_np()
        return tf.Variable(
            name=phase_varname,
            initial_value=phase_np,
            dtype=TF_FLOAT
        )

    def to_torch(self, is_trainable: bool=True) -> Parameter:
        """

        Returns:
            Initialized torch Parameter
        """
        phase_initializer = self.to_np()
        phase = Parameter(torch.tensor(phase_initializer, requires_grad=is_trainable))
        return phase


class HaarRandomPhaseInitializer(MeshPhaseInitializer):
    """
    Haar-random initialization of rectangular and triangular mesh architectures.

    Args:
        units: Input dimension, :math:`N`
        num_layers: Number of layers, :math:`L`
        hadamard: Whether to use Hadamard convention
        tri: Initializer for the triangular mesh architecture
    """
    def __init__(self, units: int, num_layers: int=None, hadamard: bool=False, tri: bool=False):
        self.tri = tri
        if self.tri:
            self.num_layers = 2 * units - 3
        else:
            self.num_layers = units if not num_layers else num_layers
        self.hadamard = hadamard
        super(HaarRandomPhaseInitializer, self).__init__(units, self.num_layers)

    def to_np(self) -> np.ndarray:
        theta_0, theta_1 = get_haar_theta(self.units, self.num_layers, hadamard=self.hadamard, tri=self.tri)
        theta = np.zeros((self.num_layers, self.units // 2))
        theta[::2, :] = theta_0
        if self.units % 2:
            theta[1::2, :] = theta_1
        else:
            theta[1::2, :-1] = theta_1
        return theta.astype(NP_FLOAT)


class PRMPhaseInitializer(MeshPhaseInitializer):
    def __init__(self, units: int, hadamard: bool, tunable_layers_per_block: Optional[int]=None):
        """
        A useful initialization of permuting mesh architectures based on the Haar random initialization above.
        This currently only works if using default permuting mesh architecture or setting :math:`tunable_layers_per_block`.

        Args:
            units: Input dimension, :math:`N`
            hadamard: Whether to use Hadamard convention
            tunable_layers_per_block: Number of tunable layers per block (same behavior as :code:`PermutingRectangularMeshModel`).
        """
        self.tunable_block_sizes, _ = get_default_coarse_grain_block_sizes(units) if tunable_layers_per_block is None \
            else get_efficient_coarse_grain_block_sizes(units, tunable_layers_per_block)
        self.hadamard = hadamard
        self.num_layers = int(np.sum(self.tunable_block_sizes))
        super(PRMPhaseInitializer, self).__init__(units, self.num_layers)

    def to_np(self) -> np.ndarray:
        thetas = []
        for block_size in self.tunable_block_sizes:
            theta_0, theta_1 = get_haar_theta(self.units, block_size, hadamard=self.hadamard)
            theta = np.zeros((block_size, self.units // 2))
            theta[::2, :] = theta_0
            if self.units % 2:
                theta[1::2, :] = theta_1
            else:
                theta[1::2, :-1] = theta_1
            thetas.append(theta.astype(NP_FLOAT))
        return np.vstack(thetas)


class UniformRandomPhaseInitializer(MeshPhaseInitializer):
    def __init__(self, units: int, num_layers: int, max_phase, min_phase: float=0):
        """
        Defines a uniform random initializer up to some maximum phase,
        e.g. :math:`\\theta \in [0, \pi]` or :math:`\phi \in [0, 2\pi)`.

        Args:
            units: Input dimension, :math:`N`.
            num_layers: Number of layers, :math:`L`.
            max_phase: Maximum phase
            min_phase: Minimum phase (usually 0)
        """
        self.units = units
        self.num_layers = units
        self.max_phase = max_phase
        self.min_phase = min_phase
        super(UniformRandomPhaseInitializer, self).__init__(units, num_layers)

    def to_np(self) -> np.ndarray:
        phase = (self.max_phase - self.min_phase) * np.random.rand(self.num_layers, self.units // 2) + self.min_phase
        return phase.astype(NP_FLOAT)


class ConstantPhaseInitializer(MeshPhaseInitializer):
    def __init__(self, units: int, num_layers: int, constant_phase: float):
        """

        Args:
            units: Input dimension, :math:`N`
            num_layers: Number of layers, :math:`L`
            constant_phase: The constant phase to set all array elements
        """
        self.constant_phase = constant_phase
        super(ConstantPhaseInitializer, self).__init__(units, num_layers)

    def to_np(self) -> np.ndarray:
        return self.constant_phase * np.ones((self.units, self.num_layers))


def get_haar_theta(units: int, num_layers: int, hadamard: bool,
                   tri: bool=False) -> Union[Tuple[np.ndarray, np.ndarray],
                                             Tuple[tf.Variable, tf.Variable],
                                             tf.Variable]:
    if tri:
        alpha_rows = np.repeat(np.linspace(1, units - 1, units - 1)[:, np.newaxis], units * 2 - 3, axis=1).T
        theta_0_root = 2 * alpha_rows[::2, ::2]
        theta_1_root = 2 * alpha_rows[1::2, 1::2]
    else:
        alpha_checkerboard = get_alpha_checkerboard_general(units, num_layers)
        theta_0_root = 2 * alpha_checkerboard.T[::2, ::2]
        theta_1_root = 2 * alpha_checkerboard.T[1::2, 1::2]
    theta_0_init = 2 * np.arcsin(np.random.rand(*theta_0_root.shape) ** (1 / theta_0_root))
    theta_1_init = 2 * np.arcsin(np.random.rand(*theta_1_root.shape) ** (1 / theta_1_root))
    if not hadamard:
        theta_0_init = np.pi - theta_0_init
        theta_1_init = np.pi - theta_1_init
    return theta_0_init.astype(dtype=NP_FLOAT), theta_1_init.astype(dtype=NP_FLOAT)


def get_ortho_haar_theta(units: int, num_layers: int,
                         hadamard: bool, num_samples: int=10000) -> Union[Tuple[np.ndarray, np.ndarray],
                                                                          Tuple[tf.Variable, tf.Variable],
                                                                          tf.Variable]:
    alpha_checkerboard = get_alpha_checkerboard_general(units, num_layers)
    theta_0_root = alpha_checkerboard.T[::2, ::2] - 1
    theta_1_root = alpha_checkerboard.T[1::2, 1::2] - 1
    theta_0_init = np.zeros_like(theta_0_root)
    theta_1_init = np.zeros_like(theta_1_root)
    for i in range(units):
        t = np.linspace(0, np.pi, num_samples)
        len_0 = np.sum(theta_0_root == i)
        len_1 = np.sum(theta_1_root == i)
        if i > 0:
            haar_dist = rv_discrete(values=(np.arange(num_samples), np.sin(t) ** i / np.sum(np.sin(t) ** i)))
            theta_0_init[theta_0_root == i] = 2 * t[haar_dist.rvs(size=len_0)]
            theta_1_init[theta_1_root == i] = 2 * t[haar_dist.rvs(size=len_1)]
        else:
            theta_0_init[theta_0_root == i] = 2 * np.pi * np.random.rand(len_0)
            theta_1_init[theta_1_root == i] = 2 * np.pi * np.random.rand(len_1)
    if not hadamard:
        theta_0_init = np.pi - theta_0_init
        theta_1_init = np.pi - theta_1_init
    return theta_0_init.astype(dtype=NP_FLOAT), theta_1_init.astype(dtype=NP_FLOAT)


def get_initializer(units: int, num_layers: int, initializer_name: str,
                    hadamard: bool=False, testing: bool=False) -> MeshPhaseInitializer:
    if testing:
        np.random.seed(TEST_SEED)
    initializer_name_to_initializer = {
        'haar_rect': HaarRandomPhaseInitializer(units, num_layers, hadamard),
        'haar_tri': HaarRandomPhaseInitializer(units, num_layers, hadamard, tri=True),
        'haar_prm': PRMPhaseInitializer(units, hadamard=hadamard),
        'random_phi': UniformRandomPhaseInitializer(units, num_layers, 2 * np.pi),
        'random_gamma': UniformRandomPhaseInitializer(2 * units, 1, 2 * np.pi),
        'constant_gamma': UniformRandomPhaseInitializer(2 * units, 1, 0.0),
        'constant_max_gamma': UniformRandomPhaseInitializer(2 * units, 1, 2 * np.pi),
        'random_constant': ConstantPhaseInitializer(units, num_layers, np.pi * np.random.rand()),
        'random_theta': UniformRandomPhaseInitializer(units, num_layers, np.pi),
        'constant_phi': ConstantPhaseInitializer(units, num_layers, 0.0),
        'constant_max_phi': ConstantPhaseInitializer(units, num_layers, 2 * np.pi),
        'bar': ConstantPhaseInitializer(units, num_layers, 0.0 if hadamard else np.pi),
        'cross': ConstantPhaseInitializer(units, num_layers, np.pi if hadamard else 0),
        'transmissive': UniformRandomPhaseInitializer(units, num_layers,
                                                      min_phase=np.pi / 2,
                                                      max_phase=np.pi)
    }
    return initializer_name_to_initializer[initializer_name]
