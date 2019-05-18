import numpy as np
from typing import Optional, List

from .generic import MeshNumpyLayer
from ..control import MeshPhases
from ..config import BLOCH, NP_COMPLEX
from ..meshmodel import RectangularMeshModel, TriangularMeshModel, PermutingRectangularMeshModel


class RMNumpy(MeshNumpyLayer):
    """Rectangular mesh network layer for unitary operators implemented in numpy

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        num_layers: The number of layers (:math:`L`) of the mesh
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        phases: The MeshPhases control parameters for the mesh
    """

    def __init__(self, units: int, num_layers: int = None, hadamard: bool = False, basis: str = BLOCH,
                 bs_error: float = 0.0, phases: Optional[MeshPhases] = None, theta_init_name="haar_rect",
                 phi_init_name="random_phi", gamma_init_name="random_gamma"):
        super(RMNumpy, self).__init__(
            RectangularMeshModel(units, num_layers, hadamard, bs_error, basis,
                                 theta_init_name, phi_init_name, gamma_init_name), phases
        )

    def propagate(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs * self.phases.input_phase_shift_layer
        fields = np.zeros((self.num_layers + 1, *outputs.shape), dtype=NP_COMPLEX)
        fields[0] = outputs
        for layer in range(self.num_layers):
            outputs = self.mesh_layers[layer].transform(outputs)
            fields[layer + 1] = np.roll(outputs, 1, 1) if layer % 2 else outputs
        return fields

    def inverse_propagate(self, outputs: np.ndarray) -> np.ndarray:
        inputs = outputs
        fields = np.zeros((self.num_layers + 1, *inputs.shape), dtype=NP_COMPLEX)
        for layer in reversed(range(self.num_layers)):
            fields[layer + 1] = np.roll(inputs, 1, 1) if layer % 2 else inputs
            inputs = self.mesh_layers[layer].inverse_transform(inputs)
        fields[0] = inputs
        return fields


class TMNumpy(MeshNumpyLayer):
    """Triangular mesh network layer for unitary operators implemented in numpy

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        phases: The MeshPhases control parameters for the mesh
    """

    def __init__(self, units: int, hadamard: bool = False, basis: str = BLOCH,
                 bs_error: float = 0.0, phases: Optional[MeshPhases] = None,
                 theta_init_name="haar_tri", phi_init_name="random_phi", gamma_init_name="random_gamma"):
        super(TMNumpy, self).__init__(
            TriangularMeshModel(units, hadamard, bs_error, basis,
                                theta_init_name, phi_init_name, gamma_init_name), phases
        )

    def propagate(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs * self.phases.input_phase_shift_layer
        fields = np.zeros((self.num_layers + 1, *outputs.shape), dtype=NP_COMPLEX)
        fields[0] = outputs
        for layer in range(self.num_layers):
            outputs = self.mesh_layers[layer].transform(outputs)
            fields[layer + 1] = np.roll(outputs, 1, 1) if layer % 2 else outputs
        return fields

    def inverse_propagate(self, outputs: np.ndarray) -> np.ndarray:
        inputs = outputs
        fields = np.zeros((self.num_layers + 1, *inputs.shape), dtype=NP_COMPLEX)
        for layer in reversed(range(self.num_layers)):
            fields[layer + 1] = np.roll(inputs, 1, 1) if layer % 2 else inputs
            inputs = self.mesh_layers[layer].inverse_transform(inputs)
        fields[0] = inputs
        return fields


class PRMNumpy(MeshNumpyLayer):
    """Permuting rectangular mesh unitary layer

    Args:
        units: The dimension of the unitary matrix (:math:`N`) to be modeled by this transformer
        tunable_layers_per_block: The number of tunable layers per block (overrides `num_tunable_layers_list`, `sampling_frequencies`)
        num_tunable_layers_list: Number of tunable layers in each block in order from left to right
        sampling_frequencies: Frequencies of sampling frequencies between the tunable layers
        bs_error: Photonic error in the beamsplitter
        theta_init_name: Initializer name for theta
        phi_init_name: Initializer name for phi
    """

    def __init__(self, units: int, phases: Optional[MeshPhases] = None, tunable_layers_per_block: int = None,
                 num_tunable_layers_list: Optional[List[int]] = None, sampling_frequencies: Optional[List[int]] = None,
                 bs_error: float = 0.0, hadamard: bool = False, theta_init_name: Optional[str] = 'haar_prm',
                 phi_init_name: Optional[str] = 'random_phi'):
        if theta_init_name == 'haar_prm' and tunable_layers_per_block is not None:
            raise NotImplementedError('haar_prm initializer is incompatible with setting tunable_layers_per_block.')
        super(PRMNumpy, self).__init__(
            PermutingRectangularMeshModel(units, tunable_layers_per_block, num_tunable_layers_list,
                                          sampling_frequencies, bs_error, hadamard,
                                          theta_init_name, phi_init_name), phases
        )
