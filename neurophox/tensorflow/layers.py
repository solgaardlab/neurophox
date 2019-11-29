from typing import Optional, List, Dict

import tensorflow as tf
from tensorflow.keras.layers import Activation
import numpy as np

from .generic import TransformerLayer, MeshLayer, CompoundTransformerLayer, PermutationLayer
from ..meshmodel import RectangularMeshModel, TriangularMeshModel, PermutingRectangularMeshModel, ButterflyMeshModel
from ..helpers import rectangular_permutation, butterfly_layer_permutation
from ..config import DEFAULT_BASIS, TF_FLOAT, TF_COMPLEX


class RM(MeshLayer):
    """Rectangular mesh network layer for unitary operators implemented in tensorflow

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        num_layers: The number of layers (:math:`L`) of the mesh
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, num_layers: int = None, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init_name: Optional[str] = "haar_rect",
                 phi_init_name: Optional[str] = "random_phi", gamma_init_name: Optional[str] = "random_gamma",
                 include_diagonal_phases=True, activation: Activation = None, **kwargs):
        super(RM, self).__init__(
            RectangularMeshModel(units, num_layers, hadamard, bs_error, basis,
                                 theta_init_name, phi_init_name, gamma_init_name),
            activation=activation, include_diagonal_phases=include_diagonal_phases, **kwargs
        )


class TM(MeshLayer):
    """Triangular mesh network layer for unitary operators implemented in tensorflow

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init_name: Optional[str] = "haar_tri",
                 phi_init_name: Optional[str] = "random_phi", gamma_init_name: Optional[str] = "random_gamma",
                 activation: Activation = None, **kwargs):
        super(TM, self).__init__(
            TriangularMeshModel(units, hadamard, bs_error, basis,
                                theta_init_name, phi_init_name, gamma_init_name),
            activation, **kwargs
        )


class PRM(MeshLayer):
    """Permuting rectangular mesh unitary layer

    Args:
        units: The dimension of the unitary matrix (:math:`N`) to be modeled by this transformer
        tunable_layers_per_block: The number of tunable layers per block (overrides :code:`num_tunable_layers_list`, :code:`sampling_frequencies`)
        num_tunable_layers_list: Number of tunable layers in each block in order from left to right
        sampling_frequencies: Frequencies of sampling frequencies between the tunable layers
        is_trainable: Whether the parameters are trainable
        bs_error: Photonic error in the beamsplitter
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, tunable_layers_per_block: int = None,
                 num_tunable_layers_list: Optional[List[int]] = None, sampling_frequencies: Optional[List[int]] = None,
                 bs_error: float = 0.0, hadamard: bool = False,
                 theta_init_name: Optional[str] = "haar_prm", phi_init_name: Optional[str] = "random_phi",
                 gamma_init_name: Optional[str] = "random_gamma",
                 activation: Activation = None, **kwargs):
        if theta_init_name == 'haar_prm' and tunable_layers_per_block is not None:
            raise NotImplementedError('haar_prm initializer is incompatible with setting tunable_layers_per_block.')
        super(PRM, self).__init__(
            PermutingRectangularMeshModel(units, tunable_layers_per_block, num_tunable_layers_list,
                                          sampling_frequencies, bs_error, hadamard,
                                          theta_init_name, phi_init_name, gamma_init_name),
            activation=activation, **kwargs
        )


class BM(MeshLayer):
    """Butterfly mesh unitary layer

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, num_layers: int, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, theta_init_name: Optional[str] = "haar_tri",
                 phi_init_name: Optional[str] = "random_phi",
                 activation: Activation = None, **kwargs):
        super(BM, self).__init__(
            ButterflyMeshModel(num_layers, hadamard, bs_error, basis, theta_init_name, phi_init_name),
            activation=activation, **kwargs
        )


class SVD(CompoundTransformerLayer):
    """Singular value decomposition transformer for implementing a matrix.

    Notes:
        SVD requires you specify the unitary transformers used to implement the SVD in `unitary_transformer_dict`,
        specifying transformer name and arguments for that transformer.

    Args:
        units: The number of inputs (:math:`M`) of the :math:`M \\times N` matrix to be modelled by the SVD
        mesh_dict: The name and properties of the mesh layer used for the SVD
        output_units: The dimension of the output (:math:`N`) of the :math:`M \\times N` matrix to be modelled by the SVD
        pos_singular_values: Whether to allow only positive singular values
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, mesh_dict: Dict, output_units: Optional[int] = None,
                 pos_singular_values: bool = False,
                 activation: Activation = None):
        self.units = units
        self.output_units = output_units if output_units is not None else units
        if output_units != units and output_units is not None:
            raise NotImplementedError("Still working out a clean implementation of non-square linear operators.")
        self.mesh_name = mesh_dict['name']
        self.mesh_properties = mesh_dict.get('properties', {})
        self.pos = pos_singular_values

        mesh_name2layer = {
            'rm': RM,
            'prm': PRM,
            'tm': TM
        }

        self.v = mesh_name2layer[self.mesh_name](units=units, name="v", **self.mesh_properties)
        self.diag = Diagonal(units, output_units=output_units, pos=self.pos)
        self.u = mesh_name2layer[self.mesh_name](units=units, name="u", **self.mesh_properties)

        self.activation = activation

        super(SVD, self).__init__(
            units=self.units,
            transformer_list=[self.v, self.diag, self.u]
        )


class DiagonalPhaseLayer(TransformerLayer):
    """Diagonal matrix of phase shifts

    Args:
        units: Dimension of the input (number of input waveguide ports), :math:`N`
    """

    def __init__(self, units: int, **kwargs):
        super(DiagonalPhaseLayer, self).__init__(units=units)
        self.gamma = tf.Variable(
            name="gamma",
            initial_value=tf.constant(2 * np.pi * np.random.rand(units), dtype=TF_FLOAT),
            dtype=TF_FLOAT,
            **kwargs
        )
        self.diag_vec = tf.complex(tf.cos(self.gamma), tf.sin(self.gamma))
        self.inv_diag_vec = tf.complex(tf.cos(-self.gamma), tf.sin(-self.gamma))
        self.variables.append(self.gamma)

    @tf.function
    def transform(self, inputs: tf.Tensor):
        return self.diag_vec * inputs

    @tf.function
    def inverse_transform(self, outputs: tf.Tensor):
        return self.inv_diag_vec * outputs


class Diagonal(TransformerLayer):
    """Diagonal matrix of gains and losses (not necessarily real)

    Args:
        units: Dimension of the input (number of input waveguide ports), :math:`N`
        is_complex: Whether to use complex values or not
        output_units: Dimension of the output (number of output waveguide ports), :math:`M`. If :math:`M < N`, remove last :math:`N - M` elements. If :math:`M > N`, pad with :math:`M - N` zeros.
        pos: Enforce positive definite matrix (only positive singular values)

    """

    def __init__(self, units: int, is_complex: bool = True, output_units: Optional[int] = None,
                 pos: bool = False, **kwargs):
        super(Diagonal, self).__init__(units=units, is_complex=is_complex, **kwargs)
        self.output_dim = output_units if output_units is not None else units
        self.pos = pos
        singular_value_dim = min(self.units, self.output_dim)
        self.sigma = tf.Variable(
            name="sigma",
            initial_value=tf.constant(2 * np.pi * np.random.randn(singular_value_dim), dtype=TF_FLOAT),
            dtype=TF_FLOAT
        )

    @tf.function
    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        sigma = tf.abs(self.sigma) if self.pos else self.sigma
        diag_vec = tf.cast(sigma, TF_COMPLEX) if self.is_complex else sigma
        if self.output_dim == self.units:
            return diag_vec * inputs
        elif self.output_dim < self.units:
            return diag_vec * inputs[:self.output_dim]
        else:
            return tf.pad(diag_vec * inputs, tf.constant([[0, 0], [0, self.output_dim - self.units]]))

    @tf.function
    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        sigma = tf.abs(self.sigma) if self.pos else self.sigma
        inv_diag_vec = tf.cast(1 / sigma, TF_COMPLEX) if self.is_complex else 1 / sigma
        if self.output_dim == self.units:
            return inv_diag_vec * outputs
        elif self.output_dim > self.units:
            return inv_diag_vec * outputs[:self.units]
        else:
            return tf.pad(inv_diag_vec * outputs, tf.constant([[0, 0], [0, self.units - self.output_dim]]))


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
        super(RectangularPerm, self).__init__(
            permuted_indices=rectangular_permutation(units, frequency))


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
