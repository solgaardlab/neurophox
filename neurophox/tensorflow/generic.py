from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation
import numpy as np

from ..numpy.generic import MeshPhases
from ..meshmodel import MeshModel
from ..helpers import pairwise_off_diag_permutation, plot_complex_matrix, inverse_permutation
from ..config import TF_COMPLEX, BLOCH, SINGLEMODE


class TransformerLayer(Layer):
    """Base transformer class for transformer layers (invertible functions, usually linear)

    Args:
        units: Dimension of the input to be transformed by the transformer
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """
    def __init__(self, units: int, activation: Activation = None, **kwargs):
        self.units = units
        self.activation = activation
        super(TransformerLayer, self).__init__(**kwargs)

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Transform inputs using layer (needs to be overwritten by child classes)

        Args:
            inputs: Inputs to be transformed by layer

        Returns:
            Transformed inputs
        """
        raise NotImplementedError("Needs to be overwritten by child class.")

    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        """
        Transform outputs using layer

        Args:
            outputs: Outputs to be inverse-transformed by layer

        Returns:
            Transformed outputs
        """
        raise NotImplementedError("Needs to be overwritten by child class.")

    def call(self, inputs, training=None, mask=None):
        outputs = self.transform(inputs)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    @property
    def matrix(self):
        """
        Shortcut of :code:`transformer.transform(np.eye(self.units))`

        Returns:
            Matrix implemented by layer
        """
        identity_matrix = np.eye(self.units, dtype=np.complex64)
        return self.transform(identity_matrix).numpy()

    @property
    def inverse_matrix(self):
        """
        Shortcut of :code:`transformer.inverse_transform(np.eye(self.units))`

        Returns:
            Inverse matrix implemented by layer
        """
        identity_matrix = np.eye(self.units, dtype=np.complex64)
        return self.inverse_transform(identity_matrix).numpy()

    def plot(self, plt):
        """
        Plot :code:`transformer.matrix`.

        Args:
            plt: :code:`matplotlib.pyplot` for plotting
        """
        plot_complex_matrix(plt, self.matrix)


class CompoundTransformerLayer(TransformerLayer):
    """Compound transformer class for unitary matrices

    Args:
        units: Dimension of the input to be transformed by the transformer
        transformer_list: List of :class:`Transformer` objects to apply to the inputs
        is_complex: Whether the input to be transformed is complex
    """
    def __init__(self, units: int, transformer_list: List[TransformerLayer]):
        self.transformer_list = transformer_list
        super(CompoundTransformerLayer, self).__init__(units=units)

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        """Inputs are transformed by :math:`L` transformer layers :math:`T^{(\ell)} \in \mathbb{C}^{N \\times N}` as follows:

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} \prod_{\ell=1}^L T_\ell,

        where :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: Input batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Transformed :code:`inputs`, :math:`V_{\mathrm{out}}`
        """
        outputs = inputs
        for transformer in self.transformer_list:
            outputs = transformer.transform(outputs)
        return outputs

    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        """Outputs are inverse-transformed by :math:`L` transformer layers :math:`T^{(\ell)} \in \mathbb{C}^{N \\times N}` as follows:

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} \prod_{\ell=L}^1 T_\ell^{-1},

        where :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: Output batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Transformed :code:`outputs`, :math:`V_{\mathrm{in}}`
        """
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
        """
        Performs the permutation for this layer represented by :math:`P` defined by `permuted_indices`:

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} P,

        where :math:`P` is any :math:`N`-dimensional permutation and
        :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: Input batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Permuted :code:`inputs`, :math:`V_{\mathrm{out}}`
        """
        return tf.gather(inputs, self.permuted_indices, axis=-1)

    def inverse_transform(self, outputs: tf.Tensor):
        """
        Performs the inverse permutation for this layer represented by :math:`P^{-1}` defined by `inv_permuted_indices`:

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} P^{-1},

        where :math:`P` is any :math:`N`-dimensional permutation and
        :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Permuted :code:`outputs`, :math:`V_{\mathrm{in}}`
        """
        return tf.gather(outputs, self.inv_permuted_indices, axis=-1)


class MeshVerticalLayer(TransformerLayer):
    """
    Args:
        diag: the diagonal terms to multiply
        off_diag: the off-diagonal terms to multiply
        left_perm: the permutation for the mesh vertical layer (prior to the coupling operation)
        right_perm: the right permutation for the mesh vertical layer
            (usually for the final layer and after the coupling operation)
    """
    def __init__(self, pairwise_perm_idx: np.ndarray, diag: tf.Tensor, off_diag: tf.Tensor,
                 right_perm: PermutationLayer = None, left_perm: PermutationLayer = None):
        self.diag = diag
        self.off_diag = off_diag
        self.left_perm = left_perm
        self.right_perm = right_perm
        self.pairwise_perm_idx = pairwise_perm_idx
        super(MeshVerticalLayer, self).__init__(pairwise_perm_idx.shape[0])

    def transform(self, inputs: tf.Tensor):
        """
        Propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U^{(\ell')},

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Propaged :code:`inputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`.
        """
        outputs = inputs if self.left_perm is None else self.left_perm.transform(inputs)
        outputs = outputs * self.diag + tf.gather(outputs * self.off_diag, self.pairwise_perm_idx, axis=-1)
        return outputs if self.right_perm is None else self.right_perm.transform(outputs)

    def inverse_transform(self, outputs: tf.Tensor):
        """
        Inverse-propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} (U^{(\ell')})^\dagger,

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse propaged :code:`outputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.
        """
        inputs = outputs if self.right_perm is None else self.right_perm.inverse_transform(outputs)
        diag = tf.math.conj(self.diag)
        off_diag = tf.gather(tf.math.conj(self.off_diag), self.pairwise_perm_idx, axis=-1)
        inputs = inputs * diag + tf.gather(inputs * off_diag, self.pairwise_perm_idx, axis=-1)
        return inputs if self.left_perm is None else self.left_perm.inverse_transform(inputs)


class MeshParamTensorflow:
    """A class that cleanly arranges parameters into a specific arrangement that can be used to simulate any mesh

    Args:
        param: parameter to arrange in mesh
        units: number of inputs/outputs of the mesh
    """
    def __init__(self, param: tf.Tensor, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self):
        """
        The single-mode arrangement based on the :math:`L(\\theta)` transfer matrix for :code:`PhaseShiftUpper`
        is one where elements of `param` are on the even rows and all odd rows are zero.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_m` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the single-mode arrangement has the stripe array form
        :math:`\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{0}, \\boldsymbol{\\theta}_2, \\boldsymbol{0}, \ldots \\boldsymbol{\\theta}_N, \\boldsymbol{0}]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh
        and :math:`\\boldsymbol{0}` represents an array of zeros of the same size as :math:`\\boldsymbol{\\theta}_n`.

        Returns:
            Single-mode arrangement array of phases

        """
        tensor_t = tf.transpose(self.param)
        stripe_tensor = tf.reshape(tf.concat((tensor_t, tf.zeros_like(tensor_t)), 1),
                                   shape=(tensor_t.shape[0] * 2, tensor_t.shape[1]))
        if self.units % 2:
            return tf.concat([stripe_tensor, tf.zeros(shape=(1, tensor_t.shape[1]))], axis=0)
        else:
            return stripe_tensor

    @property
    def common_mode_arrangement(self) -> tf.Tensor:
        """
        The common-mode arrangement based on the :math:`C(\\theta)` transfer matrix for :code:`PhaseShiftCommonMode`
        is one where elements of `param` are on the even rows and repeated on respective odd rows.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the common-mode arrangement has the stripe array form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_N, \\boldsymbol{\\theta}_N]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh.


        Returns:
            Common-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases + roll_tensor(phases)

    @property
    def differential_mode_arrangement(self) -> tf.Tensor:
        """
        The differential-mode arrangement is based on the :math:`D(\\theta)` transfer matrix
        for :code:`PhaseShiftDifferentialMode`.

        Given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\cdots \\boldsymbol{\\theta}_m \cdots]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the differential-mode arrangement has the form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = \\left[\cdots \\frac{\\boldsymbol{\\theta}_m}{2}, -\\frac{\\boldsymbol{\\theta}_m}{2} \cdots \\right]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh.

        Returns:
            Differential-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases / 2 - roll_tensor(phases / 2)

    def __add__(self, other):
        return MeshParamTensorflow(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTensorflow(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTensorflow(self.param * other.param, self.units)


class MeshPhasesTensorflow:
    """Organizes the phases in the mesh into appropriate arrangements

    Args:
        theta: Array to be converted to :math:`\\boldsymbol{\\theta}`
        phi: Array to be converted to :math:`\\boldsymbol{\\phi}`
        gamma: Array to be converted to :math:`\\boldsymbol{\gamma}`
        mask: Mask over values of :code:`theta` and :code:`phi` that are not in bar state
        basis: Phase basis to use
        hadamard: Whether to use Hadamard convention
    """
    def __init__(self, theta: tf.Variable, phi: tf.Variable, mask: np.ndarray, gamma: tf.Variable, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParamTensorflow(theta * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.phi = MeshParamTensorflow(phi * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.gamma = gamma
        self.basis = basis
        self.input_phase_shift_layer = tf.complex(tf.cos(gamma), tf.sin(gamma))
        if self.theta.param.shape != self.phi.param.shape:
            raise ValueError("Internal phases (theta) and external phases (phi) need to have the same shape.")

    @property
    def internal_phase_shifts(self):
        """

        The internal phase shift matrix of the mesh corresponds to an `L \\times N` array of phase shifts
        (in between beamsplitters, thus internal) where :math:`L` is number of layers and :math:`N` is number of inputs/outputs

        Returns:
            Internal phase shift matrix corresponding to :math:`\\boldsymbol{\\theta}`
        """
        if self.basis == BLOCH:
            return self.theta.differential_mode_arrangement
        elif self.basis == SINGLEMODE:
            return self.theta.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def external_phase_shifts(self):
        """The external phase shift matrix of the mesh corresponds to an `L \\times N` array of phase shifts
        (outside of beamsplitters, thus external) where :math:`L` is number of layers and :math:`N` is number of inputs/outputs

        Returns:
            External phase shift matrix corresponding to :math:`\\boldsymbol{\\phi}`
        """
        if self.basis == BLOCH or self.basis == SINGLEMODE:
            return self.phi.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def internal_phase_shift_layers(self):
        """Elementwise applying complex exponential to :code:`internal_phase_shifts`.

        Returns:
            Internal phase shift layers corresponding to :math:`\\boldsymbol{\\theta}`
        """
        internal_ps = self.internal_phase_shifts
        return tf.complex(tf.cos(internal_ps), tf.sin(internal_ps))

    @property
    def external_phase_shift_layers(self):
        """Elementwise applying complex exponential to :code:`external_phase_shifts`.

        Returns:
            External phase shift layers corresponding to :math:`\\boldsymbol{\\phi}`
        """
        external_ps = self.external_phase_shifts
        return tf.complex(tf.cos(external_ps), tf.sin(external_ps))


class Mesh:
    def __init__(self, model: MeshModel):
        """General mesh network layer defined by `neurophox.meshmodel.MeshModel`

        Args:
            model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers
        self.pairwise_perm_idx = pairwise_off_diag_permutation(self.units)
        ss, cs, sc, cc = self.model.mzi_error_tensors
        self.ss, self.cs, self.sc, self.cc = tf.constant(ss, dtype=TF_COMPLEX), tf.constant(cs, dtype=TF_COMPLEX), \
                                               tf.constant(sc, dtype=TF_COMPLEX), tf.constant(cc, dtype=TF_COMPLEX)
        self.perm_layers = [PermutationLayer(self.model.perm_idx[layer]) for layer in range(self.num_layers + 1)]

    def mesh_layers(self, phases: MeshPhasesTensorflow) -> List[MeshVerticalLayer]:
        """

        Args:
            phases:  The :code:`MeshPhasesTensorflow` object containing :math:`\\boldsymbol{\\theta}, \\boldsymbol{\\phi}, \\boldsymbol{\\gamma}`

        Returns:
            List of mesh layers to be used by any instance of :code:`MeshLayer`
        """
        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers
        # smooth trick to efficiently perform the layerwise coupling computation

        if self.model.hadamard:
            s11 = (self.cc * internal_psl + self.ss * roll_tensor(internal_psl, up=True))
            s22 = roll_tensor(self.ss * internal_psl + self.cc * roll_tensor(internal_psl, up=True))
            s12 = roll_tensor(self.cs * internal_psl - self.sc * roll_tensor(internal_psl, up=True))
            s21 = (self.sc * internal_psl - self.cs * roll_tensor(internal_psl, up=True))
        else:
            s11 = (self.cc * internal_psl - self.ss * roll_tensor(internal_psl, up=True))
            s22 = roll_tensor(-self.ss * internal_psl + self.cc * roll_tensor(internal_psl, up=True))
            s12 = 1j * roll_tensor(self.cs * internal_psl + self.sc * roll_tensor(internal_psl, up=True))
            s21 = 1j * (self.sc * internal_psl + self.cs * roll_tensor(internal_psl, up=True))

        diag_layers = external_psl * (s11 + s22) / 2
        off_diag_layers = roll_tensor(external_psl) * (s21 + s12) / 2

        if self.units % 2:
            diag_layers = tf.concat((diag_layers[:-1], tf.ones_like(diag_layers[-1:])), axis=0)

        diag_layers, off_diag_layers = tf.transpose(diag_layers), tf.transpose(off_diag_layers)

        mesh_layers = [MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[0], off_diag_layers[0],
                                         self.perm_layers[1], self.perm_layers[0])]
        for layer in range(1, self.num_layers):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[layer], off_diag_layers[layer],
                                                 self.perm_layers[layer + 1]))
        return mesh_layers


class MeshLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, mesh_model: MeshModel, activation: Activation = None,
                 include_diagonal_phases: bool = True, incoherent: bool = False, **kwargs):
        self.mesh = Mesh(mesh_model)
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.include_diagonal_phases = include_diagonal_phases
        self.incoherent = incoherent
        super(MeshLayer, self).__init__(self.units, activation=activation, **kwargs)
        theta_init, phi_init, gamma_init = self.mesh.model.init()
        self.theta, self.phi, self.gamma = theta_init.to_tf("theta"), phi_init.to_tf("phi"), gamma_init.to_tf("gamma")

    @tf.function
    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Transformed :code:`inputs`, :math:`V_{\mathrm{out}}`
        """
        _inputs = np.eye(self.units, dtype=np.complex64) if self.incoherent else inputs
        mesh_phases, mesh_layers = self.phases_and_layers
        outputs = _inputs * mesh_phases.input_phase_shift_layer if self.include_diagonal_phases else _inputs
        for layer in range(self.num_layers):
            outputs = mesh_layers[layer].transform(outputs)
        if self.incoherent:
            power_matrix = tf.math.real(outputs) ** 2 + tf.math.imag(outputs) ** 2
            power_inputs = tf.math.real(inputs) ** 2 + tf.math.imag(inputs) ** 2
            outputs = power_inputs @ power_matrix
            return tf.complex(tf.sqrt(outputs), tf.zeros_like(outputs))
        return outputs

    @tf.function
    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} U^\dagger,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse transformed :code:`outputs`, :math:`V_{\mathrm{in}}`
        """
        mesh_phases, mesh_layers = self.phases_and_layers
        inputs = outputs
        for layer in reversed(range(self.num_layers)):
            inputs = mesh_layers[layer].inverse_transform(inputs)
        if self.include_diagonal_phases:
            inputs = inputs * tf.math.conj(mesh_phases.input_phase_shift_layer)
        return inputs

    @property
    def phases_and_layers(self) -> Tuple[MeshPhasesTensorflow, List[MeshVerticalLayer]]:
        """

        Returns:
            Phases and layers for this mesh layer
        """
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

    @property
    def phases(self) -> MeshPhases:
        """

        Returns:
            The :code:`MeshPhases` object for this layer
        """
        return MeshPhases(
            theta=self.theta.numpy() * self.mesh.model.mask,
            phi=self.phi.numpy() * self.mesh.model.mask,
            mask=self.mesh.model.mask,
            gamma=self.gamma.numpy()
        )


def roll_tensor(tensor: tf.Tensor, up=False):
    # a complex number-friendly roll that works on gpu
    if up:
        return tf.concat([tensor[1:], tensor[tf.newaxis, 0]], axis=0)
    return tf.concat([tensor[tf.newaxis, -1], tensor[:-1]], axis=0)