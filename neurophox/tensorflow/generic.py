from typing import List

import tensorflow as tf
import numpy as np

from ..control import MeshPhases, MeshPhasesTensorflow
from ..meshmodel import MeshModel
from ..helpers import pairwise_off_diag_permutation, roll_tensor, plot_complex_matrix, inverse_permutation
from ..config import TFKERAS


class TransformerLayer(tf.keras.layers.Layer):
    """Base transformer class for transformer layers (invertible functions, usually linear)

    Args:
        units: Dimension of the input to be transformed by the transformer
        is_complex: Whether the input to be transformed is complex or not
        activation: Nonlinear activation function (None if there's no nonlinearity)
    """

    def __init__(self, units: int, is_complex: bool = True, activation: tf.keras.layers.Activation = None, **kwargs):
        self.units = units
        self.is_complex = is_complex
        self.activation = activation
        super(TransformerLayer, self).__init__(**kwargs)

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs

    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        return outputs

    def call(self, inputs, training=None, mask=None):
        outputs = self.transform(inputs)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    @property
    def matrix(self):
        identity_matrix = np.eye(self.units, dtype=np.complex64)
        return self.transform(identity_matrix).numpy()

    @property
    def inverse_matrix(self):
        identity_matrix = np.eye(self.units, dtype=np.complex64)
        return self.inverse_transform(identity_matrix).numpy()

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix)


class CompoundTransformerLayer(TransformerLayer):
    """Compound transformer class for unitary matrices

    Args:
        units: Dimension of the input to be transformed by the transformer
        transformer_list: List of :class:`Transformer` objects to apply to the inputs
        is_complex: Whether the input to be transformed is complex
    """

    def __init__(self, units: int, transformer_list: List[TransformerLayer],
                 is_complex: bool = True):
        self.transformer_list = transformer_list
        super(CompoundTransformerLayer, self).__init__(units=units,
                                                       is_complex=is_complex)

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs
        for transformer in self.transformer_list:
            outputs = transformer.transform(outputs)
        return outputs

    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        inputs = outputs
        for transformer in self.transformer_list[::-1]:
            inputs = transformer.inverse_transform(inputs)
        return inputs


class PermutationLayer(TransformerLayer):
    """Permutation layer

    Args:
        permuted_indices: order of indices for the permutation matrix (efficient permutation representation)
    """

    def __init__(self, permuted_indices: np.ndarray):
        super(PermutationLayer, self).__init__(units=permuted_indices.shape[0])
        self.permuted_indices = np.asarray(permuted_indices, dtype=np.int32)
        self.inv_permuted_indices = inverse_permutation(self.permuted_indices)

    def transform(self, inputs: tf.Tensor):
        return tf.gather(inputs, self.permuted_indices, axis=-1)

    def inverse_transform(self, outputs: tf.Tensor):
        return tf.gather(outputs, self.inv_permuted_indices, axis=-1)


class MeshVerticalLayer(TransformerLayer):
    def __init__(self, pairwise_perm_idx: np.ndarray, diag: tf.Tensor, off_diag: tf.Tensor,
                 perm: PermutationLayer = None, right_perm: PermutationLayer = None):
        """
        Args:
            diag: the diagonal terms to multiply
            off_diag: the off-diagonal terms to multiply
            perm: the permutation for the mesh vertical layer (prior to the coupling operation)
            right_perm: the right permutation for the mesh vertical layer
                (usually for the final layer and after the coupling operation)
        """
        self.diag = diag
        self.off_diag = off_diag
        self.perm = perm
        self.right_perm = right_perm
        self.pairwise_perm_idx = pairwise_perm_idx
        super(MeshVerticalLayer, self).__init__(pairwise_perm_idx.shape[0])

    def transform(self, inputs: tf.Tensor):
        outputs = inputs if self.perm is None else self.perm.transform(inputs)
        outputs = outputs * self.diag + tf.gather(outputs * self.off_diag, self.pairwise_perm_idx, axis=-1)
        return outputs if self.right_perm is None else self.right_perm.transform(outputs)

    def inverse_transform(self, outputs: tf.Tensor):
        inputs = outputs if self.right_perm is None else self.right_perm.inverse_transform(outputs)
        diag = tf.math.conj(self.diag)
        off_diag = tf.gather(tf.math.conj(self.off_diag), self.pairwise_perm_idx, axis=-1)
        inputs = inputs * diag + tf.gather(inputs * off_diag, self.pairwise_perm_idx, axis=-1)
        return inputs if self.perm is None else self.perm.inverse_transform(inputs)


class Mesh:
    def __init__(self, model: MeshModel):
        """
        General mesh network layer defined by `neurophox.meshmodel.MeshModel`

        Args:
            model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers
        self.pairwise_perm_idx = pairwise_off_diag_permutation(self.units)
        self.enn, self.enp, self.epn, self.epp = self.model.mzi_error_tensors
        self.perm_layers = [PermutationLayer(self.model.perm_idx[layer]) for layer in range(self.num_layers + 1)]

    def mesh_layers(self, phases: MeshPhasesTensorflow) -> List[MeshVerticalLayer]:
        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers
        # smooth trick to efficiently perform the layerwise coupling computation

        if self.model.hadamard:
            s11 = (self.epp * internal_psl + self.enn * roll_tensor(internal_psl, up=True))
            s22 = roll_tensor(self.enn * internal_psl + self.epp * roll_tensor(internal_psl, up=True))
            s12 = roll_tensor(self.enp * internal_psl - self.epn * roll_tensor(internal_psl, up=True))
            s21 = (self.epn * internal_psl - self.enp * roll_tensor(internal_psl, up=True))
        else:
            s11 = (self.epp * internal_psl - self.enn * roll_tensor(internal_psl, up=True))
            s22 = roll_tensor(-self.enn * internal_psl + self.epp * roll_tensor(internal_psl, up=True))
            s12 = 1j * roll_tensor(self.enp * internal_psl + self.epn * roll_tensor(internal_psl, up=True))
            s21 = 1j * (self.epn * internal_psl + self.enp * roll_tensor(internal_psl, up=True))

        diag_layers = external_psl * (s11 + s22) / 2
        off_diag_layers = roll_tensor(external_psl) * (s21 + s12) / 2

        if self.units % 2:
            diag_layers = tf.concat((diag_layers[:-1], tf.ones_like(diag_layers[-1:])), axis=0)

        diag_layers, off_diag_layers = tf.transpose(diag_layers), tf.transpose(off_diag_layers)

        mesh_layers = []
        for layer in range(self.num_layers - 1):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[layer], off_diag_layers[layer],
                                                 self.perm_layers[layer]))
        mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[-1], off_diag_layers[-1],
                                             self.perm_layers[-2], self.perm_layers[-1]))
        return mesh_layers


class MeshLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        activation: Nonlinear activation function (None if there's no nonlinearity)
    """

    def __init__(self, mesh_model: MeshModel,  activation: tf.keras.layers.Activation = None,
                 include_diagonal_phases: bool = True, **kwargs):
        self.mesh = Mesh(mesh_model)
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.include_diagonal_phases = include_diagonal_phases
        super(MeshLayer, self).__init__(self.units, activation=activation, **kwargs)
        self.theta, self.phi, self.gamma = self.mesh.model.init(backend=TFKERAS)

    @tf.function
    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        mesh_phases, mesh_layers = self.phases_and_layers
        outputs = inputs * mesh_phases.input_phase_shift_layer if self.include_diagonal_phases else inputs
        for layer in range(self.num_layers):
            outputs = mesh_layers[layer].transform(outputs)
        return outputs

    @tf.function
    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        mesh_phases, mesh_layers = self.phases_and_layers
        inputs = outputs
        for layer in reversed(range(self.num_layers)):
            inputs = mesh_layers[layer].inverse_transform(inputs)
        if self.include_diagonal_phases:
            inputs = inputs * tf.math.conj(mesh_phases.input_phase_shift_layer)
        return inputs

    @property
    def phases_and_layers(self):
        mesh_phases = MeshPhasesTensorflow(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            hadamard=self.mesh.model.hadamard,
            units=self.units,
            basis=self.mesh.model.basis
        )
        mesh_layers = self.mesh.mesh_layers(mesh_phases)
        return mesh_phases, mesh_layers

    def adjoint_transform(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.inverse_transform(inputs)

    @property
    def phases(self) -> MeshPhases:
        return MeshPhases(
            theta=self.theta.numpy() * self.mesh.model.mask,
            phi=self.phi.numpy() * self.mesh.model.mask,
            mask=self.mesh.model.mask,
            gamma=self.gamma.numpy()
        )
