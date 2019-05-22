from typing import Optional, Union, Tuple, List
import numpy as np
import tensorflow as tf
from .helpers import butterfly_permutation, grid_permutation, to_stripe_array, prm_permutation, \
    get_efficient_coarse_grain_block_sizes, get_default_coarse_grain_block_sizes
from .initializers import get_initializer
from .config import BLOCH, TF_COMPLEX, NUMPY, TFKERAS, TEST_SEED


class MeshModel:
    """Any feedforward mesh model of :math:`N` inputs/outputs and `L` layers.

    Args:
        perm_idx: A numpy array of :math:`N \\times L` permutation indices for all layers of the mesh
        hadamard: Whether to use Hadamard convention
        num_mzis: A numpy array of :math:`L` integers, where for layer :math:`\ell`, :math:`M_\ell \leq \\lfloor N / 2\\rfloor`, used to defined the phase shift mask.
        bs_error: Beamsplitter error (ignore for pure machine learning applications)
        testing: Use a seed for randomizing error (ignore for pure machine learning applications)
        use_different_errors: Use different errors for the left and right beamsplitter errors
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        basis: Phase basis to use for controlling each pairwise unitary (simulated interferometer) in the mesh
    """
    def __init__(self, perm_idx: np.ndarray, hadamard: bool = False, num_mzis: Optional[np.ndarray] = None,
                 bs_error: float = 0.0, testing: bool = False, use_different_errors: bool = False,
                 theta_init_name: str = "random_theta", phi_init_name: str = "random_phi",
                 gamma_init_name: str = "random_gamma", basis: str = BLOCH):

        self.units = perm_idx.shape[1]
        self.num_layers = perm_idx.shape[0] - 1
        self.perm_idx = perm_idx
        self.inv_perm_idx = np.zeros_like(self.perm_idx)
        for idx, perm_idx in enumerate(self.perm_idx):
            self.inv_perm_idx[perm_idx] = idx
        self.num_mzis = num_mzis
        self.hadamard = hadamard
        self.bs_error = bs_error
        self.testing = testing
        self.use_different_errors = use_different_errors
        self.mask = np.zeros((self.num_layers, self.units // 2))
        self.theta_init_name = theta_init_name
        self.phi_init_name = phi_init_name
        self.gamma_init_name = gamma_init_name
        self.basis = basis
        for layer in range(self.num_layers):
            self.mask[layer][:int(self.num_mzis[layer])] = 1
        if self.num_mzis.shape[0] != self.num_layers:
            raise ValueError("num_mzis, perm_idx num_layers mismatch.")
        if self.units < 2:
            raise ValueError("units must be at least 2.")

    def init(self, backend: str = NUMPY) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                  Tuple[tf.Variable, tf.Variable, tf.Variable]]:
        """

        Args:
            backend: Whether to use Numpy, Tensorflow, or PyTorch (not yet supported, will throw error).

        Returns:
            Numpy arrays or Tensorflow variables corresponding to :math:`\\boldsymbol{\\theta}, \\boldsymbol{\\phi}, \gamma_n`.
        """
        theta_init = get_initializer(self.units, self.num_layers, self.theta_init_name, self.hadamard, self.testing)
        phi_init = get_initializer(self.units, self.num_layers, self.phi_init_name, self.hadamard, self.testing)
        gamma_init = get_initializer(self.units, self.num_layers, self.gamma_init_name, self.hadamard, self.testing)
        if backend == NUMPY:
            return theta_init.to_np(), phi_init.to_np(), gamma_init.to_np()
        elif backend == TFKERAS:
            return theta_init.to_tf("theta"), phi_init.to_tf("phi"), gamma_init.to_tf("gamma")
        else:
            raise NotImplementedError(f"Backend {backend} not supported.")

    def get_bs_error_matrix(self, right: bool):
        """

        Args:
            right: whether to set errors for second set of MZI beamsplitters, useful for photonic simulations.

        Returns:
            Error numpy arrays for "beamsplitter layers."
        """
        if self.testing:
            np.random.seed(right + TEST_SEED)
        mask = self.mask if self.mask is not None else np.ones((self.num_layers, self.units // 2))
        return np.random.randn(self.num_layers, self.units // 2) * self.bs_error * mask

    @property
    def mzi_error_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:
            Error numpy arrays for Numpy :code:`MeshNumpyLayer`
        """
        if self.testing:
            np.random.seed(TEST_SEED)
        mask = self.mask if self.mask is not None else np.ones((self.num_layers, self.units // 2))
        e_l = np.random.randn(self.num_layers, self.units // 2) * self.bs_error * mask
        if self.use_different_errors:
            if self.testing:
                np.random.seed(TEST_SEED + 1)
            e_r = np.random.randn(self.num_layers, self.units // 2) * self.bs_error * mask
        else:
            e_r = e_l
        return e_l, e_r

    @property
    def mzi_error_tensors(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """

        Returns:
            Error tensors for Tensorflow :code:`MeshLayer`
        """
        e_l, e_r = self.mzi_error_matrices

        enn = to_stripe_array(np.sqrt(1 - e_l) * np.sqrt(1 - e_r), self.units)
        epn = to_stripe_array(np.sqrt(1 + e_l) * np.sqrt(1 - e_r), self.units)
        enp = to_stripe_array(np.sqrt(1 - e_l) * np.sqrt(1 + e_r), self.units)
        epp = to_stripe_array(np.sqrt(1 + e_l) * np.sqrt(1 + e_r), self.units)

        return tf.constant(enn, dtype=TF_COMPLEX), tf.constant(enp, dtype=TF_COMPLEX), \
               tf.constant(epn, dtype=TF_COMPLEX), tf.constant(epp, dtype=TF_COMPLEX)


class RectangularMeshModel(MeshModel):
    """Rectangular mesh

    The rectangular mesh contains :math:`N` inputs/outputs and :math:`L` layers in rectangular grid arrangement
    of pairwise unitary operators to implement :math:`U \in \mathrm{U}(N)`.

    Args:
        units: Input dimension, :math:`N`
        num_layers: Number of layers, :math:`L`
        hadamard: Hadamard convention
        bs_error: Beamsplitter layer
        basis: Phase basis to use for controlling each pairwise unitary (simulated interferometer) in the mesh
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
    """
    def __init__(self, units: int, num_layers: int = None, hadamard: bool = False, bs_error: float = 0.0,
                 basis: str = BLOCH, theta_init_name: str = "haar_rect", phi_init_name: str = "random_phi",
                 gamma_init_name: str = "random_gamma"):
        self.num_layers = num_layers if num_layers else units
        perm_idx = grid_permutation(units, self.num_layers).astype(np.int32)
        num_mzis = (np.ones((self.num_layers,)) * units // 2).astype(np.int32)
        num_mzis[1::2] = (units - 1) // 2
        super(RectangularMeshModel, self).__init__(perm_idx,
                                                   hadamard=hadamard,
                                                   bs_error=bs_error,
                                                   num_mzis=num_mzis,
                                                   theta_init_name=theta_init_name,
                                                   phi_init_name=phi_init_name,
                                                   gamma_init_name=gamma_init_name,
                                                   basis=basis)


class TriangularMeshModel(MeshModel):
    """Triangular mesh

    The triangular mesh contains :math:`N` inputs/outputs and :math:`L = 2N - 3` layers in triangular grid arrangement
    of pairwise unitary operators to implement any :math:`U \in \mathrm{U}(N)`.

    Args:
        units: Input dimension, :math:`N`
        hadamard: Hadamard convention
        bs_error: Beamsplitter layer
        basis: Phase basis to use for controlling each pairwise unitary (simulated interferometer) in the mesh
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
    """
    def __init__(self, units: int, hadamard: bool = False, bs_error: float = 0.0, basis: str = BLOCH,
                 theta_init_name: str = "haar_tri", phi_init_name: str = "random_phi",
                 gamma_init_name: str = "random_gamma"):
        perm_idx = grid_permutation(units, 2 * units - 3).astype(np.int32)
        num_mzis = ((np.hstack([np.arange(1, units), np.arange(units - 2, 0, -1)]) + 1) // 2).astype(np.int32)
        super(TriangularMeshModel, self).__init__(perm_idx,
                                                  hadamard=hadamard,
                                                  bs_error=bs_error,
                                                  num_mzis=num_mzis,
                                                  theta_init_name=theta_init_name,
                                                  phi_init_name=phi_init_name,
                                                  gamma_init_name=gamma_init_name,
                                                  basis=basis
                                                  )


class ButterflyMeshModel(MeshModel):
    """Butterfly mesh

    The butterfly mesh contains :math:`L` layers and :math:`N = 2^L` inputs/outputs to implement :math:`U \in \mathrm{U}(N)`.
    Unlike the triangular and full (:math:`L = N`) rectangular mesh, the butterfly mesh is not universal. However,
    it has attractive properties for efficient machine learning and compact photonic implementations of unitary mesh models.

    Args:
        num_layers: Number of layers, :math:`L`
        hadamard: Hadamard convention
        bs_error: Beamsplitter layer
    """
    def __init__(self, num_layers: int, hadamard: bool = False, bs_error: float = 0.0):
        super(ButterflyMeshModel, self).__init__(np.vstack(
            [butterfly_permutation(2 ** num_layers, 2 ** layer) for layer in range(num_layers)]).astype(np.int32),
                                                 hadamard=hadamard,
                                                 bs_error=bs_error
                                                 )


class PermutingRectangularMeshModel(MeshModel):
    """Permuting rectangular mesh model

    Args:
        units: Input dimension, :math:`N`
        tunable_layers_per_block: The number of tunable layers per block (overrides `num_tunable_layers_list`, `sampling_frequencies`)
        num_tunable_layers_list: Number of tunable layers in each block in order from left to right
        sampling_frequencies: Frequencies of sampling frequencies between the tunable layers
        bs_error: Photonic error in the beamsplitter
        hadamard: Whether to use hadamard convention (otherwise use beamsplitter convention)
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
    """

    def __init__(self, units: int, tunable_layers_per_block: int = None,
                 num_tunable_layers_list: Optional[List[int]] = None, sampling_frequencies: Optional[List[int]] = None,
                 bs_error: float = 0.0, hadamard: bool = False, theta_init_name: Optional[str] = 'haar_prm',
                 phi_init_name: Optional[str] = 'random_phi', gamma_init_name: str = 'random_gamma'):

        if tunable_layers_per_block is not None:
            self.block_sizes, self.sampling_frequencies = get_efficient_coarse_grain_block_sizes(
                units=units,
                tunable_layers_per_block=tunable_layers_per_block
            )
        elif sampling_frequencies is None or num_tunable_layers_list is None:
            self.block_sizes, self.sampling_frequencies = get_default_coarse_grain_block_sizes(units)
        else:
            self.block_sizes, self.sampling_frequencies = num_tunable_layers_list, sampling_frequencies

        num_mzis_list = []
        for block_size in self.block_sizes:
            num_mzis_list.append((np.ones((block_size,)) * units // 2).astype(np.int32))
            num_mzis_list[-1][1::2] = (units - 1) // 2
        num_mzis = np.hstack(num_mzis_list)

        super(PermutingRectangularMeshModel, self).__init__(
            perm_idx=prm_permutation(units=units, tunable_block_sizes=self.block_sizes,
                                     sampling_frequencies=self.sampling_frequencies,
                                     butterfly=False),
            num_mzis=num_mzis,
            hadamard=hadamard,
            bs_error=bs_error,
            theta_init_name=theta_init_name,
            phi_init_name=phi_init_name,
            gamma_init_name=gamma_init_name
        )
