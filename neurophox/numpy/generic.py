from typing import List, Optional, Tuple, Callable, Union

import numpy as np
from ..helpers import plot_complex_matrix, inverse_permutation, ordered_viz_permutation, to_stripe_array
from ..components import MZI, Beamsplitter
from ..config import NP_COMPLEX, BLOCH, SINGLEMODE
from ..meshmodel import MeshModel


class TransformerNumpyLayer:
    """Base transformer layer class for transformers in numpy (invertible functions, usually linear)

    Args:
        units: Dimension of the input, :math:`N`.
    """

    def __init__(self, units: int):
        self.units = units

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        """
        Transform inputs using layer (needs to be overwritten by child classes)

        Args:
            inputs: Inputs to be transformed by layer

        Returns:
            Transformed inputs
        """
        raise NotImplementedError("Needs to be overwritten by child class.")

    def inverse_transform(self, outputs: np.ndarray) -> np.ndarray:
        """
        Transform outputs using layer

        Args:
            outputs: Outputs to be inverse-transformed by layer

        Returns:
            Transformed outputs
        """
        raise NotImplementedError("Needs to be overwritten by child class.")

    @property
    def matrix(self):
        """
        Shortcut of :code:`transformer.transform(np.eye(self.units))`

        Returns:
            Matrix implemented by layer
        """
        return self.transform(np.eye(self.units))

    @property
    def inverse_matrix(self):
        """
        Shortcut of :code:`transformer.inverse_transform(np.eye(self.units))`

        Returns:
            Inverse matrix implemented by layer
        """
        return self.inverse_transform(np.eye(self.units))

    def plot(self, plt):
        """
        Plot :code:`transformer.matrix`.

        Args:
            plt: :code:`matplotlib.pyplot` for plotting
        """
        plot_complex_matrix(plt, self.matrix)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.transform(inputs)


class CompoundTransformerNumpyLayer(TransformerNumpyLayer):
    def __init__(self, units: int, transformer_list: List[TransformerNumpyLayer]):
        """Compound transformer class for unitary matrices

        Args:
            units: Dimension acted on by the layer
            transformer_list: List of :class:`Transformer` objects to apply to the inputs
        """
        self.transformer_list = transformer_list
        super(CompoundTransformerNumpyLayer, self).__init__(units=units)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for transformer in self.transformer_list:
            outputs = transformer.transform(outputs)
        return outputs

    def inverse_transform(self, outputs: np.ndarray) -> np.ndarray:
        inputs = outputs
        for transformer in self.transformer_list[::-1]:
            inputs = transformer.inverse_transform(inputs)
        return inputs


class MeshVerticalNumpyLayer(TransformerNumpyLayer):
    """
    Args:
        tunable_layer: tunable layer
        left_perm_idx: the permutation for the mesh vertical layer (prior to the coupling operation)
        right_perm_idx: the right permutation for the mesh vertical layer
            (usually for the final layer and after the coupling operation)
    """

    def __init__(self, tunable_layer: np.ndarray,
                 right_perm_idx: Optional[np.ndarray] = None, left_perm_idx: Optional[np.ndarray] = None):
        self.tunable_layer = tunable_layer
        self.left_perm_idx = left_perm_idx
        self.right_perm_idx = right_perm_idx
        super(MeshVerticalNumpyLayer, self).__init__(self.tunable_layer.shape[0])

    def transform(self, inputs: np.ndarray):
        """
        Propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U^{(\ell')},

        where :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Propagation of :code:`inputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`.
        """
        if self.left_perm_idx is None:
            outputs = inputs @ self.tunable_layer
        else:
            outputs = inputs.take(self.left_perm_idx, axis=-1) @ self.tunable_layer
        if self.right_perm_idx is None:
            return outputs
        else:
            return outputs.take(self.right_perm_idx, axis=-1)

    def inverse_transform(self, outputs: np.ndarray):
        """
        Inverse-propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} (U^{(\ell')})^\dagger,

        where :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse propagation of :code:`outputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.
        """
        if self.right_perm_idx is None:
            inputs = outputs @ self.tunable_layer.conj().T
        else:
            inputs = outputs.take(inverse_permutation(self.right_perm_idx), axis=-1) @ self.tunable_layer.conj().T
        if self.left_perm_idx is None:
            return inputs
        else:
            return inputs.take(inverse_permutation(self.left_perm_idx), axis=-1)


class MeshParam:
    """A class that arranges parameters to simulate any feedforward mesh

    Args:
        param: parameter to arrange in mesh
        units: number of inputs/outputs of the mesh
    """

    def __init__(self, param: np.ndarray, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self) -> np.ndarray:
        """
        The single-mode arrangement based on the :math:`L(\\theta)` transfer matrix for :code:`PhaseShiftUpper`
        is one where elements of `param` are on the even rows and all odd rows are zero.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_m` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the single-mode arrangement has the stripe array form
        :math:`\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{0}, \\boldsymbol{\\theta}_2, \\boldsymbol{0}, \ldots \\boldsymbol{\\theta}_M, \\boldsymbol{0}]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}}` defines the spatial arrangement of mesh phases
        and :math:`\\boldsymbol{0}` represents an array of zeros of the same size as :math:`\\boldsymbol{\\theta}_m`.

        Returns:
            Single-mode arrangement array of phases

        """
        return to_stripe_array(self.param, self.units)

    @property
    def checkerboard_arrangement(self) -> np.ndarray:
        """

        Returns:
            Checkerboard arrangement of phases useful for grid mesh structures (e.g. rectangular and triangular meshes)
        """
        checkerboard = np.zeros((self.units, self.param.shape[0]), dtype=self.param.dtype)
        if self.units % 2:
            checkerboard[:-1][::2, ::2] = self.param[::2].T
        else:
            checkerboard[::2, ::2] = self.param[::2].T
        checkerboard[1::2, 1::2] = self.param[1::2].T
        return checkerboard

    @property
    def common_mode_arrangement(self) -> np.ndarray:
        """
        The common-mode arrangement based on the :math:`C(\\theta)` transfer matrix for :code:`PhaseShiftCommonMode`
        is one where elements of `param` are on the even rows and repeated on respective odd rows.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the common-mode arrangement has the stripe array form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M, \\boldsymbol{\\theta}_M]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}}` defines the spatial arrangement of mesh phases.


        Returns:
            Common-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases + np.roll(phases, 1, axis=0)

    @property
    def differential_mode_arrangement(self) -> np.ndarray:
        """
        The differential-mode arrangement is based on the :math:`D(\\theta)` transfer matrix
        for :code:`PhaseShiftDifferentialMode`.

        Given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\cdots \\boldsymbol{\\theta}_m \cdots]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the differential-mode arrangement has the form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = \\left[\cdots \\frac{\\boldsymbol{\\theta}_m}{2}, -\\frac{\\boldsymbol{\\theta}_m}{2} \cdots \\right]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the spatial arrangement of mesh phases.

        Returns:
            Differential-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases / 2 - np.roll(phases / 2, 1, axis=0)

    def param_list(self, mask: np.ndarray) -> np.ndarray:
        """

        Args:
            mask: Mask to ignore params in output

        Returns:
            A flattened array of unmasked params in :code:`param`
        """
        return self.param[mask.astype(np.bool)]

    def __add__(self, other):
        return MeshParam(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParam(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParam(self.param * other.param, self.units)


class MeshPhases:
    """Arranges the phases in the mesh appropriately depending on :code:`basis` using the :code:`MeshParam` class.

    Args:
        theta: Array to be converted to :math:`\\boldsymbol{\\theta}`
        phi: Array to be converted to :math:`\\boldsymbol{\\phi}`
        gamma: Array to be converted to :math:`\\boldsymbol{\gamma}`
        mask: Mask over values of :code:`theta` and :code:`phi` that are not in bar state
        basis: Phase basis to use
        hadamard: Whether to use Hadamard convention
        phase_loss_fn: Incorporate phase shift-dependent loss into the model
    """

    def __init__(self, theta: np.ndarray, phi: np.ndarray, gamma: np.ndarray, mask: np.ndarray = None,
                 basis: str = BLOCH, hadamard: bool = False,
                 phase_loss_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParam(theta * self.mask + (1 - self.mask) * (1 - hadamard) * np.pi, gamma.size)
        self.phi = MeshParam(phi * self.mask + (1 - self.mask) * (1 - hadamard) * np.pi, gamma.size)
        self.gamma = gamma
        self.hadamard = hadamard
        self.basis = basis
        self.phase_loss_fn = (lambda x: 0) if phase_loss_fn is None else phase_loss_fn
        self.input_phase_shift_layer = np.exp(1j * gamma) * (1 - self.phase_loss_fn(gamma))
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
        """

        The external phase shift matrix of the mesh corresponds to an `L \\times N` array of phase shifts
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
        """

        Elementwise applying complex exponential to :code:`internal_phase_shifts`.

        Returns:
            Internal phase shift layers corresponding to :math:`\\boldsymbol{\\theta}`
        """
        return np.exp(1j * self.internal_phase_shifts) * (1 - self.phase_loss_fn(self.internal_phase_shifts))

    @property
    def external_phase_shift_layers(self):
        """

        Elementwise applying complex exponential to :code:`external_phase_shifts`.

        Returns:
            External phase shift layers corresponding to :math:`\\boldsymbol{\\phi}`
        """
        return np.exp(1j * self.external_phase_shifts) * (1 - self.phase_loss_fn(self.external_phase_shifts))

    def __add__(self, other_rm_mesh_phases):
        return MeshPhases(self.theta.param + other_rm_mesh_phases.theta.param,
                          self.phi.param + other_rm_mesh_phases.phi.param,
                          self.mask,
                          self.gamma + other_rm_mesh_phases.gamma)

    def __sub__(self, other_rm_mesh_phases):
        return MeshPhases(self.theta.param - other_rm_mesh_phases.theta.param,
                          self.phi.param - other_rm_mesh_phases.phi.param,
                          self.mask,
                          self.gamma - other_rm_mesh_phases.gamma)

    @property
    def params(self):
        return self.theta.param, self.phi.param, self.gamma


class MeshNumpy:
    """
    Args:
        model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
    """

    def __init__(self, model: MeshModel):
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers

    def mesh_layers(self, phases: MeshPhases, use_different_errors=False) -> List[MeshVerticalNumpyLayer]:
        """

        Args:
            phases: The :code:`MeshPhases` object containing :math:`\\boldsymbol{\\theta}, \\boldsymbol{\\phi}, \\boldsymbol{\\gamma}`
            use_different_errors: use different errors for the left and right beamsplitters

        Returns:
            List of mesh layers to be used by any instance of :code:`MeshNumpyLayer`
        """
        mesh_layers = []
        internal_phases = phases.internal_phase_shifts
        external_phases = phases.external_phase_shifts
        e_l, e_r = self.model.mzi_error_matrices

        for layer in range(self.num_layers):
            tunable_layer = np.diag(np.exp(1j * external_phases[:, layer]))
            errors_l = e_l[layer]
            if use_different_errors:
                errors_r = e_r[layer]
            else:
                errors_r = errors_l
            for idx in range(self.units // 2):
                wvg_idx = idx * 2
                tunable_layer[wvg_idx:wvg_idx + 2, wvg_idx:wvg_idx + 2] = MZI(
                    internal_upper=internal_phases[wvg_idx, layer],
                    internal_lower=internal_phases[wvg_idx + 1, layer],
                    external_upper=external_phases[wvg_idx, layer],
                    external_lower=external_phases[wvg_idx + 1, layer],
                    hadamard=self.model.hadamard,
                    epsilon=(errors_l[idx], errors_r[idx])
                ).matrix
            mesh_layers.append(MeshVerticalNumpyLayer(
                tunable_layer=tunable_layer,
                left_perm_idx=None if layer > 0 else self.model.perm_idx[0],
                right_perm_idx=self.model.perm_idx[layer + 1])
            )
        return mesh_layers

    def beamsplitter_layers(self):
        """

        Returns:
            List of beamsplitter layers to be used by any instance of :code:`MeshNumpyLayer`
        """
        beamsplitter_layers_l = []
        beamsplitter_layers_r = []
        e_l, e_r = self.model.mzi_error_matrices

        for layer in range(self.num_layers):
            errors_l = e_l[layer]
            errors_r = e_r[layer]
            beamsplitter_layer_l = np.eye(self.units, dtype=NP_COMPLEX)
            beamsplitter_layer_r = np.eye(self.units, dtype=NP_COMPLEX)
            for idx in range(self.units // 2):
                wvg_idx = idx * 2
                beamsplitter_layer_l[wvg_idx:wvg_idx + 2, wvg_idx:wvg_idx + 2] = Beamsplitter(
                    hadamard=self.model.hadamard,
                    epsilon=errors_l[idx]
                ).matrix
                beamsplitter_layer_r[wvg_idx:wvg_idx + 2, wvg_idx:wvg_idx + 2] = Beamsplitter(
                    hadamard=self.model.hadamard,
                    epsilon=errors_r[idx]
                ).matrix
            beamsplitter_layers_l.append(MeshVerticalNumpyLayer(beamsplitter_layer_l,
                                                                left_perm_idx=self.model.perm_idx[layer]))
            beamsplitter_layers_r.append(MeshVerticalNumpyLayer(beamsplitter_layer_r,
                                                                right_perm_idx=None if layer < self.num_layers - 1 else
                                                                self.model.perm_idx[-1]))
        return beamsplitter_layers_l, beamsplitter_layers_r


class MeshNumpyLayer(TransformerNumpyLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
    """

    def __init__(self, mesh_model: MeshModel, phase_loss_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.mesh = MeshNumpy(mesh_model)
        self.phase_loss_fn = phase_loss_fn
        self._setup()
        super(MeshNumpyLayer, self).__init__(self.units)

    def _setup(self, testing: bool = False):
        self.mesh.model.testing = testing
        theta_init, phi_init, gamma_init = self.mesh.model.init
        self.theta, self.phi, self.gamma = theta_init.to_np(), phi_init.to_np(), gamma_init.to_np()
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.internal_phase_shift_layers = self.phases.internal_phase_shift_layers.T
        self.external_phase_shift_layers = self.phases.external_phase_shift_layers.T
        self.mesh_layers = self.mesh.mesh_layers(self.phases)
        self.beamsplitter_layers_l, self.beamsplitter_layers_r = self.mesh.beamsplitter_layers()

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Forward transformation of :code:`inputs`
        """
        outputs = inputs * self.phases.input_phase_shift_layer
        if self.phase_loss_fn:
            for layer in range(self.num_layers):
                # first coupling event
                outputs = self.beamsplitter_layers_l[layer].transform(outputs)
                # phase shift event
                outputs = outputs * self.internal_phase_shift_layers[layer]
                # second coupling event
                outputs = self.beamsplitter_layers_r[layer].transform(outputs)
                # phase shift event
                if layer == self.num_layers - 1:
                    outputs = outputs * self.external_phase_shift_layers[layer].take(
                        self.beamsplitter_layers_r[layer].right_perm_idx)
                else:
                    outputs = outputs * self.external_phase_shift_layers[layer]
        else:
            for layer in range(self.num_layers):
                outputs = self.mesh_layers[layer].transform(outputs)
        return outputs

    def inverse_transform(self, outputs: np.ndarray) -> np.ndarray:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} U^\dagger,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse transformation of :code:`outputs`
        """
        inputs = outputs
        for layer in reversed(range(self.num_layers)):
            inputs = self.mesh_layers[layer].inverse_transform(inputs)
        inputs = inputs * np.conj(self.phases.input_phase_shift_layer)
        return inputs

    def propagate(self, inputs: np.ndarray, explicit: bool = False, viz_perm_idx: np.ndarray = None) -> np.ndarray:
        """
        Propagate :code:`inputs` for each :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\ell} = V_{\mathrm{in}} \prod_{\ell' = 1}^{\ell} U^{(\ell')},

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\ell}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`
            explicit: explicitly show field propagation through the MZIs (useful for photonic simulations)
            viz_perm_idx: permutation of fields to visualize the propagation (:code:`None` means do not permute fields),
            this is useful for grid meshes, e.g. rectangular and triangular meshes.
            phase_loss_fn: a function converting phase shift to loss

        Returns:
            Propagation of :code:`inputs` over all :math:`L` layers to form an array
            :math:`V_{\mathrm{prop}} \in \mathbb{C}^{L \\times M \\times N}`,
            which is a concatenation of the :math:`V_{\ell}`.
        """
        viz_perm_idx = viz_perm_idx if viz_perm_idx is not None else ordered_viz_permutation(self.units,
                                                                                             self.num_layers)
        outputs = inputs * self.phases.input_phase_shift_layer
        if explicit or self.phase_loss_fn is not None:
            fields = np.zeros((4 * self.num_layers + 1, *outputs.shape), dtype=NP_COMPLEX)
            fields[0] = outputs
            for layer in range(self.num_layers):
                # first coupling event
                outputs = self.beamsplitter_layers_l[layer].transform(outputs)
                fields[4 * layer + 1] = outputs.take(viz_perm_idx[layer + 1], axis=-1)
                # phase shift event
                outputs = outputs * self.internal_phase_shift_layers[layer]
                fields[4 * layer + 2] = outputs.take(viz_perm_idx[layer + 1], axis=-1)
                # second coupling event
                outputs = self.beamsplitter_layers_r[layer].transform(outputs)
                fields[4 * layer + 3] = outputs.take(viz_perm_idx[layer + 1], axis=-1)
                # phase shift event
                # outputs = outputs * self.external_phase_shift_layers[layer]
                if layer == self.num_layers - 1:
                    outputs = outputs * self.external_phase_shift_layers[layer].take(
                        self.beamsplitter_layers_r[layer].right_perm_idx)
                else:
                    outputs = outputs * self.external_phase_shift_layers[layer]
                fields[4 * layer + 4] = outputs.take(viz_perm_idx[layer + 1], axis=-1)
        else:
            fields = np.zeros((self.num_layers + 1, *outputs.shape), dtype=NP_COMPLEX)
            fields[0] = outputs
            for layer in range(self.num_layers):
                outputs = self.mesh_layers[layer].transform(outputs)
                fields[layer + 1] = outputs.take(viz_perm_idx[layer + 1], axis=-1)
        return fields

    def inverse_propagate(self, outputs: np.ndarray, explicit: bool = False,
                          viz_perm_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Inverse propagate :code:`outputs` for each :math:`\ell < L` (where :math:`U_\ell`
        represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\ell} = V_{\mathrm{out}} \prod_{\ell' = L}^{\ell} (U^{(\ell')})^\dagger,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\ell}, V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`
            explicit: explicitly show field propagation through the MZIs (useful for photonic simulations)
            viz_perm_idx: permutation of fields to visualize the propagation (:code:`None` means do not permute fields),
            this is useful for grid meshes, e.g. rectangular and triangular meshes.

        Returns:
            Propagation of :code:`outputs` over all :math:`L` layers to form an array
            :math:`V_{\mathrm{prop}} \in \mathbb{C}^{L \\times M \\times N}`,
            which is a concatenation of the :math:`V_{\ell}`.
        """
        inputs = outputs
        if explicit:
            fields = np.zeros((self.num_layers * 4 + 1, *inputs.shape), dtype=NP_COMPLEX)
            for layer in reversed(range(self.num_layers)):
                # measure phi fields
                fields[4 * layer + 4] = inputs.take(viz_perm_idx[layer + 1],
                                                    axis=-1) if viz_perm_idx is not None else inputs
                # inputs = inputs * np.conj(self.external_phase_shift_layers[layer])
                if layer == self.num_layers - 1:
                    inputs = inputs * self.external_phase_shift_layers[layer].take(
                        self.beamsplitter_layers_r[layer].right_perm_idx).conj()
                else:
                    inputs = inputs * self.external_phase_shift_layers[layer].conj()
                # first coupling event
                fields[4 * layer + 3] = inputs.take(viz_perm_idx[layer + 1],
                                                    axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.beamsplitter_layers_r[layer].inverse_transform(inputs)
                # measure theta fields, phase shift event
                fields[4 * layer + 2] = inputs.take(viz_perm_idx[layer + 1],
                                                    axis=-1) if viz_perm_idx is not None else inputs
                inputs = inputs * np.conj(self.internal_phase_shift_layers[layer])
                # second coupling event
                fields[4 * layer + 1] = inputs.take(viz_perm_idx[layer + 1],
                                                    axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.beamsplitter_layers_l[layer].inverse_transform(inputs)
            fields[0] = inputs
        else:
            fields = np.zeros((self.num_layers + 1, *inputs.shape), dtype=NP_COMPLEX)
            for layer in reversed(range(self.num_layers)):
                fields[layer + 1] = inputs.take(viz_perm_idx[layer + 1],
                                                axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.mesh_layers[layer].inverse_transform(inputs)
            fields[0] = inputs
        return fields

    @property
    def nullification_set(self) -> np.ndarray:
        """

        The nullification set is calculated to program layers in parallel from *final layer
        towards the first layer* since he architecture assumed for this calculation is currently the *inverse* of
        our feedforward mesh definition. Therefore, we find vectors that can be shined backwards
        (using :code:`inverse_propagate`) starting from the outputs to program this device from final layer
        towards the first layer.

        Returns:
            The :math:`N \times L` nullification set array for the inverse of this layer specified by `model`
        """
        propagated_unitary = self.inverse_propagate(
            np.eye(self.units),
            viz_perm_idx=ordered_viz_permutation(self.units, self.num_layers)
        )
        nullification_set = np.zeros((self.num_layers, self.units), dtype=NP_COMPLEX)
        desired_vector = np.zeros((self.units,))
        desired_vector[::2] = 1
        desired_vector /= np.linalg.norm(desired_vector)
        for layer in range(self.num_layers):
            nullification_set[layer] = propagated_unitary[layer].conj() @ desired_vector
        return nullification_set

    def adjoint_variable_fields(self, inputs: np.ndarray, adjoint_inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # shine inputs forward
        input_fields = self.propagate(inputs, True,
                                      ordered_viz_permutation(units=self.units,
                                                              num_layers=self.num_layers))
        # shine adjoint inputs (backpropagated error) backward
        adjoint_input_fields = self.inverse_propagate(adjoint_inputs, True,
                                                      ordered_viz_permutation(units=self.units,
                                                                              num_layers=self.num_layers))
        # get interference term for input and adjoint input fields
        interference_meas = 2 * (input_fields * adjoint_input_fields.conj()).imag

        return input_fields, adjoint_input_fields, interference_meas

    def adjoint_variable_gradient(self, inputs: np.ndarray, adjoint_inputs: np.ndarray) -> MeshPhases:
        # get measurements
        input_fields, adjoint_input_fields, interference_meas = self.adjoint_variable_fields(inputs, adjoint_inputs)

        input_meas = interference_meas[0]
        # layer 1 mod 4 is after internal phase shifters
        internal_meas = interference_meas[1:][1::4]
        internal_meas = internal_meas[:, :-1] if self.units % 2 else internal_meas
        # layer 3 mod 4 is after external phase shifters
        external_meas = interference_meas[1:][3::4]
        external_meas = external_meas[:, :-1] if self.units % 2 else external_meas

        # use interference fields to get gradient information for RD mesh
        if self.mesh.model.basis == BLOCH:
            return MeshPhases(np.sum(internal_meas[:, ::2] / 2 - internal_meas[:, 1::2] / 2, axis=-1),
                              np.sum(external_meas[:, ::2], axis=-1),
                              self.mesh.model.mask,
                              np.sum(input_meas, axis=-1))
        else:
            return MeshPhases(np.sum(internal_meas[:, ::2], axis=-1),
                              np.sum(external_meas[:, ::2], axis=-1),
                              self.mesh.model.mask,
                              np.sum(input_meas, axis=-1))

    @property
    def phases(self):
        """

        Returns:
            The :code:`MeshPhases` object for this layer
        """
        return MeshPhases(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            basis=self.mesh.model.basis,
            hadamard=self.mesh.model.hadamard,
            phase_loss_fn=self.phase_loss_fn
        )
