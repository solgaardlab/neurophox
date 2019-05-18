from typing import List, Optional

import numpy as np
from ..helpers import plot_complex_matrix, inverse_permutation
from ..components import MZI, Beamsplitter
from ..control import MeshPhases
from ..config import NP_COMPLEX
from ..meshmodel import MeshModel


class TransformerNumpyLayer:
    """Base transformer layer class for transformers in numpy (invertible functions, usually linear)

    Args:
        units: Dimension of the input to be transformed by the transformer
    """
    def __init__(self, units: int):
        self.units = units

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def inverse_transform(self, outputs: np.ndarray) -> np.ndarray:
        return outputs

    @property
    def matrix(self):
        return self.transform(np.eye(self.units))

    @property
    def inverse_matrix(self):
        return self.inverse_transform(np.eye(self.units))

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.transform(inputs)


class CompoundTransformerNumpyLayer(TransformerNumpyLayer):
    """Compound transformer class for unitary matrices

    Args:
        units: Dimension acted on by the layer
        transformer_list: List of :class:`Transformer` objects to apply to the inputs
    """
    def __init__(self, units: int, transformer_list: List[TransformerNumpyLayer]):
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
    def __init__(self, tunable_layer: np.ndarray, perm_idx: Optional[np.ndarray] = None,
                 right_perm_idx: Optional[np.ndarray]=None):
        """
        Args:
            tunable_layer: tunable layer
            perm_idx: the permutation for the mesh vertical layer (prior to the coupling operation)
            right_perm_idx: the right permutation for the mesh vertical layer
                (usually for the final layer and after the coupling operation)
        """
        self.tunable_layer = tunable_layer
        self.perm_idx = perm_idx
        self.right_perm_idx = right_perm_idx
        self.inv_right_perm_idx = inverse_permutation(perm_idx) if self.perm_idx is not None else None
        super(MeshVerticalNumpyLayer, self).__init__(self.tunable_layer.shape[0])

    def transform(self, inputs: np.ndarray):
        if self.perm_idx is None:
            outputs = inputs @ self.tunable_layer
        else:
            outputs = inputs.take(self.perm_idx, axis=-1) @ self.tunable_layer
        if self.right_perm_idx is None:
            return outputs
        else:
            return outputs.take(self.right_perm_idx, axis=-1)

    def inverse_transform(self, outputs: np.ndarray):
        if self.right_perm_idx is None:
            inputs = outputs @ self.tunable_layer.conj().T
        else:
            inputs = outputs.take(inverse_permutation(self.right_perm_idx), axis=-1) @ self.tunable_layer.conj().T
        if self.perm_idx is None:
            return inputs
        else:
            return inputs.take(inverse_permutation(self.perm_idx), axis=-1)


class MeshNumpy:
    def __init__(self, model: MeshModel):
        """
        Args:
            model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers

    def mesh_layers(self, phases: MeshPhases, use_different_errors=False) -> List[MeshVerticalNumpyLayer]:
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
        beamsplitter_layers_l = []
        beamsplitter_layers_r = []
        e_l, e_r = self.model.mzi_error_matrices

        for layer in range(self.num_layers):
            num_beamsplitters = self.units // 2 - (layer % 2 and not self.units % 2)
            errors_l = e_l[layer]
            errors_r = e_r[layer]
            beamsplitter_layer_l = np.eye(self.units, dtype=NP_COMPLEX)
            beamsplitter_layer_r = np.eye(self.units, dtype=NP_COMPLEX)
            for idx in range(num_beamsplitters):
                wvg_idx = idx * 2
                beamsplitter_layer_l[wvg_idx:wvg_idx + 2, wvg_idx:wvg_idx + 2] = Beamsplitter(
                    hadamard=self.model.hadamard,
                    epsilon=errors_l[idx]
                ).matrix
                beamsplitter_layer_r[wvg_idx:wvg_idx + 2, wvg_idx:wvg_idx + 2] = Beamsplitter(
                    hadamard=self.model.hadamard,
                    epsilon=errors_r[idx]
                ).matrix
            beamsplitter_layers_l.append(MeshVerticalNumpyLayer(beamsplitter_layer_l))
            beamsplitter_layers_r.append(MeshVerticalNumpyLayer(beamsplitter_layer_r))
        return beamsplitter_layers_l, beamsplitter_layers_r


class MeshNumpyLayer(TransformerNumpyLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        phases: The MeshPhases control parameters for the mesh
    """
    def __init__(self, mesh_model: MeshModel, phases: Optional[MeshPhases]=None):
        self.mesh = MeshNumpy(mesh_model)
        if phases is None:
            self.theta, self.phi, self.gamma = mesh_model.init()
        else:
            self.theta, self.phi, self.gamma = phases.theta.param, phases.phi.param, phases.gamma
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.internal_phase_shift_layers = self.phases.internal_phase_shift_layers.T
        self.external_phase_shift_layers = self.phases.external_phase_shift_layers.T
        self.mesh_layers = self.mesh.mesh_layers(self.phases)
        self.beamsplitter_layers_l, self.beamsplitter_layers_r = self.mesh.beamsplitter_layers()
        super(MeshNumpyLayer, self).__init__(self.units)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs * self.phases.input_phase_shift_layer
        for layer in range(self.num_layers):
            outputs = self.mesh_layers[layer].transform(outputs)
        return outputs

    def inverse_transform(self, outputs: np.ndarray) -> np.ndarray:
        inputs = outputs
        for layer in reversed(range(self.num_layers)):
            inputs = self.mesh_layers[layer].inverse_transform(inputs)
        inputs = inputs * np.conj(self.phases.input_phase_shift_layer)
        return inputs

    def adjoint_transform(self, inputs: np.ndarray) -> np.ndarray:
        return self.inverse_transform(inputs)

    def propagate(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs * self.phases.input_phase_shift_layer
        fields = np.zeros((self.num_layers + 1, *outputs.shape), dtype=NP_COMPLEX)
        fields[0] = outputs
        for layer in range(self.num_layers):
            outputs = self.mesh_layers[layer].transform(outputs)
            fields[layer + 1] = outputs
        return fields

    def inverse_propagate(self, outputs: np.ndarray) -> np.ndarray:
        inputs = outputs
        fields = np.zeros((self.num_layers + 1, *inputs.shape), dtype=NP_COMPLEX)
        for layer in reversed(range(self.num_layers)):
            fields[layer + 1] = inputs
            inputs = self.mesh_layers[layer].inverse_transform(inputs)
        fields[0] = inputs
        return fields

    def adjoint_propagate(self, inputs: np.ndarray) -> np.ndarray:
        return self.inverse_propagate(inputs)

    def adjoint_variable_gradient(self, inputs: np.ndarray, adjoint_inputs: np.ndarray):
        raise NotImplementedError("Propagate methods do not yet allow for adjoint variable gradient computation.")

    @property
    def phases(self):
        return MeshPhases(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            basis=self.mesh.model.basis,
            hadamard=self.mesh.model.hadamard
        )
