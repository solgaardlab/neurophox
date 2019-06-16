import numpy as np
from typing import Optional, List


from .generic import MeshNumpyLayer
from ..control import MeshPhases
from ..config import DEFAULT_BASIS
from ..meshmodel import RectangularMeshModel, TriangularMeshModel, PermutingRectangularMeshModel
from ..helpers import grid_viz_permutation, grid_permutation


class RMNumpy(MeshNumpyLayer):
    def __init__(self, units: int, num_layers: int = None, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, phases: Optional[MeshPhases] = None, theta_init_name="haar_rect",
                 phi_init_name="random_phi", gamma_init_name="random_gamma"):
        """Rectangular mesh network layer for unitary operators implemented in numpy

        Args:
            units: The dimension of the unitary matrix (:math:`N`)
            num_layers: The number of layers (:math:`L`) of the mesh
            hadamard: Hadamard convention for the beamsplitters
            basis: Phase basis to use
            bs_error: Beamsplitter split ratio error
            phases: The MeshPhases control parameters for the mesh
            theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
            phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
            gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        """
        super(RMNumpy, self).__init__(
            RectangularMeshModel(units, num_layers, hadamard, bs_error, basis,
                                 theta_init_name, phi_init_name, gamma_init_name), phases
        )

    def propagate(self, inputs: np.ndarray, explicit: bool=False, viz_perm_idx: Optional[np.ndarray]=None) -> np.ndarray:
        viz_perm_idx = grid_viz_permutation(self.units, self.num_layers, flip=explicit) if viz_perm_idx is None else viz_perm_idx
        # viz_perm_idx = None
        return super(RMNumpy, self).propagate(inputs, explicit, viz_perm_idx)

    def inverse_propagate(self, inputs: np.ndarray, explicit: bool=False, viz_perm_idx: Optional[np.ndarray]=None) -> np.ndarray:
        viz_perm_idx = grid_viz_permutation(self.units, self.num_layers, flip=explicit) if viz_perm_idx is None else viz_perm_idx
        # viz_perm_idx = None
        return super(RMNumpy, self).inverse_propagate(inputs, explicit, viz_perm_idx)


class TMNumpy(MeshNumpyLayer):
    def __init__(self, units: int, hadamard: bool = False, basis: str = DEFAULT_BASIS,
                 bs_error: float = 0.0, phases: Optional[MeshPhases] = None,
                 theta_init_name="haar_tri", phi_init_name="random_phi", gamma_init_name="random_gamma"):
        """Triangular mesh network layer for unitary operators implemented in numpy

        Args:
            units: The dimension of the unitary matrix (:math:`N`)
            hadamard: Hadamard convention for the beamsplitters
            basis: Phase basis to use
            bs_error: Beamsplitter split ratio error
            phases: The MeshPhases control parameters for the mesh
        """
        super(TMNumpy, self).__init__(
            TriangularMeshModel(units, hadamard, bs_error, basis,
                                theta_init_name, phi_init_name, gamma_init_name), phases
        )

    def propagate(self, inputs: np.ndarray, explicit: bool=False, viz_perm_idx: Optional[np.ndarray]=None) -> np.ndarray:
        viz_perm_idx = grid_viz_permutation(self.units, self.num_layers) if viz_perm_idx is None else viz_perm_idx
        return super(TMNumpy, self).propagate(inputs, explicit, viz_perm_idx)

    def inverse_propagate(self, inputs: np.ndarray, explicit: bool=False, viz_perm_idx: Optional[np.ndarray]=None) -> np.ndarray:
        viz_perm_idx = grid_viz_permutation(self.units, self.num_layers) if viz_perm_idx is None else viz_perm_idx
        return super(TMNumpy, self).inverse_propagate(inputs, explicit, viz_perm_idx)


class PRMNumpy(MeshNumpyLayer):
    def __init__(self, units: int, phases: Optional[MeshPhases] = None, tunable_layers_per_block: int = None,
                 num_tunable_layers_list: Optional[List[int]] = None, sampling_frequencies: Optional[List[int]] = None,
                 bs_error: float = 0.0, hadamard: bool = False, theta_init_name: Optional[str] = 'haar_prm',
                 phi_init_name: Optional[str] = 'random_phi'):
        """Permuting rectangular mesh unitary layer

        Args:
            units: The dimension of the unitary matrix (:math:`N`) to be modeled by this transformer
            tunable_layers_per_block: The number of tunable layers per block (overrides `num_tunable_layers_list`, `sampling_frequencies`)
            num_tunable_layers_list: Number of tunable layers in each block in order from left to right
            sampling_frequencies: Frequencies of sampling frequencies between the tunable layers
            bs_error: Photonic error in the beamsplitter
            theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
            phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        """
        if theta_init_name == 'haar_prm' and tunable_layers_per_block is not None:
            raise NotImplementedError('haar_prm initializer is incompatible with setting tunable_layers_per_block.')
        super(PRMNumpy, self).__init__(
            PermutingRectangularMeshModel(units, tunable_layers_per_block, num_tunable_layers_list,
                                          sampling_frequencies, bs_error, hadamard,
                                          theta_init_name, phi_init_name), phases
        )
