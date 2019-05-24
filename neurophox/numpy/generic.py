from typing import List, Optional

import numpy as np
from ..helpers import plot_complex_matrix, inverse_permutation, ordered_viz_permutation
from ..components import MZI, Beamsplitter
from ..control import MeshPhases
from ..config import NP_COMPLEX
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
        perm_idx: the permutation for the mesh vertical layer (prior to the coupling operation)
        right_perm_idx: the right permutation for the mesh vertical layer
            (usually for the final layer and after the coupling operation)
    """
    def __init__(self, tunable_layer: np.ndarray, perm_idx: Optional[np.ndarray] = None,
                 right_perm_idx: Optional[np.ndarray]=None):
        self.tunable_layer = tunable_layer
        self.perm_idx = perm_idx
        self.right_perm_idx = right_perm_idx
        self.inv_right_perm_idx = inverse_permutation(right_perm_idx) if self.right_perm_idx is not None else None
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
        if self.perm_idx is None:
            outputs = inputs @ self.tunable_layer
        else:
            outputs = inputs.take(self.perm_idx, axis=-1) @ self.tunable_layer
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
        if self.perm_idx is None:
            return inputs
        else:
            return inputs.take(inverse_permutation(self.perm_idx), axis=-1)


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
                perm_idx=self.model.perm_idx[layer],
                right_perm_idx=None if layer < self.num_layers - 1 else self.model.perm_idx[-1])
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
                                                                perm_idx=self.model.perm_idx[layer]))
            beamsplitter_layers_r.append(MeshVerticalNumpyLayer(beamsplitter_layer_r,
                                                                right_perm_idx=None if layer < self.num_layers - 1 else self.model.perm_idx[-1]))
        return beamsplitter_layers_l, beamsplitter_layers_r


class MeshNumpyLayer(TransformerNumpyLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        phases: The MeshPhases control parameters for the mesh
    """
    def __init__(self, mesh_model: MeshModel, phases: Optional[MeshPhases]=None):
        self.mesh = MeshNumpy(mesh_model)
        self._setup(phases)
        super(MeshNumpyLayer, self).__init__(self.units)

    def _setup(self, phases, testing: bool=False):
        self.mesh.model.testing = testing
        if phases is None:
            self.theta, self.phi, self.gamma = self.mesh.model.init()
        else:
            self.theta, self.phi, self.gamma = phases.theta.param, phases.phi.param, phases.gamma
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

    def propagate(self, inputs: np.ndarray, explicit: bool=False, viz_perm_idx: np.ndarray=None) -> np.ndarray:
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

        Returns:
            Propagation of :code:`inputs` over all :math:`L` layers to form an array
            :math:`V_{\mathrm{prop}} \in \mathbb{C}^{L \\times M \\times N}`,
            which is a concatenation of the :math:`V_{\ell}`.
        """
        viz_perm_idx = viz_perm_idx if viz_perm_idx is not None else ordered_viz_permutation(self.units, self.num_layers)
        outputs = inputs * self.phases.input_phase_shift_layer
        if explicit:
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

    def inverse_propagate(self, outputs: np.ndarray, explicit: bool=False,
                          viz_perm_idx: Optional[np.ndarray]=None) -> np.ndarray:
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
                fields[4 * layer + 4] = inputs.take(viz_perm_idx[layer + 1], axis=-1) if viz_perm_idx is not None else inputs
                # inputs = inputs * np.conj(self.external_phase_shift_layers[layer])
                if layer == self.num_layers - 1:
                    inputs = inputs * self.external_phase_shift_layers[layer].take(
                        self.beamsplitter_layers_r[layer].right_perm_idx).conj()
                else:
                    inputs = inputs * self.external_phase_shift_layers[layer].conj()
                # first coupling event
                fields[4 * layer + 3] = inputs.take(viz_perm_idx[layer + 1], axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.beamsplitter_layers_r[layer].inverse_transform(inputs)
                # measure theta fields, phase shift event
                fields[4 * layer + 2] = inputs.take(viz_perm_idx[layer + 1], axis=-1) if viz_perm_idx is not None else inputs
                inputs = inputs * np.conj(self.internal_phase_shift_layers[layer])
                # second coupling event
                fields[4 * layer + 1] = inputs.take(viz_perm_idx[layer + 1], axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.beamsplitter_layers_l[layer].inverse_transform(inputs)
            fields[0] = inputs
        else:
            fields = np.zeros((self.num_layers + 1, *inputs.shape), dtype=NP_COMPLEX)
            for layer in reversed(range(self.num_layers)):
                fields[layer + 1] = inputs.take(viz_perm_idx[layer + 1], axis=-1) if viz_perm_idx is not None else inputs
                inputs = self.mesh_layers[layer].inverse_transform(inputs)
            fields[0] = inputs
        return fields

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
            hadamard=self.mesh.model.hadamard
        )
