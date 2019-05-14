from collections import Callable

import numpy as np
from .control import MeshPhases, MeshParam
from .config import NP_FLOAT
from .components import BlochMZI
from .helpers import to_absolute_theta


class RectangularDecomposition:
    """Rectangular decomposition of a unitary matrix using a progressive nullifying algorithm by Clements et al.

    Args:
        unitary: Unitary matrix to decompose
    """

    def __init__(self, unitary: np.ndarray, hadamard: bool=False):
        self.unitary = unitary
        self.hadamard = hadamard
        self.unitary_hat = np.copy(unitary)
        self.units = unitary.shape[0]
        self.num_layers = self.units
        # odd and even layer dimensions
        self.layer_dim = (self.units // 2, (self.units - 1) // 2)
        self.theta_checkerboard = np.zeros_like(unitary, dtype=NP_FLOAT)
        self.phi_checkerboard = np.zeros_like(unitary, dtype=NP_FLOAT)
        self.mzi_checkerboard = {}

    def decompose(self, pbar_handle: Callable = None):
        """
        Returns: Clements decomposition
        """
        iterator = pbar_handle(range(self.units - 1)) if pbar_handle else range(self.units - 1)
        for i in iterator:
            if i % 2 == 0:
                for j in range(i + 1):
                    pairwise_index = i - j
                    target_row, target_col = self.units - j - 1, i - j
                    m, n = self.units - 1 - target_row, self.units - 1 - target_col
                    theta = np.arctan(np.abs(
                        self.unitary_hat[target_row, target_col] / self.unitary_hat[target_row, target_col + 1])) * 2 + np.pi
                    phi = np.angle(
                        self.unitary_hat[target_row, target_col] / self.unitary_hat[target_row, target_col + 1])
                    mzi = BlochMZI(theta, phi, hadamard=self.hadamard, dtype=np.complex128)
                    right_multiplier = mzi.givens_rotation(units=self.units, m=pairwise_index, i_factor=True)
                    self.unitary_hat = self.unitary_hat @ right_multiplier.conj().T
                    self.theta_checkerboard[m, n - 1] = theta
                    self.phi_checkerboard[m, n - 1] = phi
            else:
                for j in range(i + 1):
                    pairwise_index = self.units + j - i - 2
                    target_row, target_col = self.units + j - i - 1, j
                    m, n = target_row, target_col
                    theta = np.arctan(np.abs(self.unitary_hat[target_row, target_col] / self.unitary_hat[target_row - 1, target_col])) * 2 + np.pi
                    phi = np.angle(-self.unitary_hat[target_row, target_col] / self.unitary_hat[target_row - 1, target_col])
                    mzi = BlochMZI(theta, phi, hadamard=self.hadamard, dtype=np.complex128)
                    left_multiplier = mzi.givens_rotation(units=self.units, m=pairwise_index, i_factor=True)
                    self.unitary_hat = left_multiplier @ self.unitary_hat
                    self.theta_checkerboard[m, n] = theta
                    self.phi_checkerboard[m, n] = phi
        self.theta_checkerboard = to_absolute_theta(self.theta_checkerboard).T
        self.phi_checkerboard = np.mod(self.phi_checkerboard, 2 * np.pi)
        self.diagonal_phases = np.diag(self.unitary_hat)
        self.phases = MeshPhases(theta=MeshParam(self.theta_checkerboard[:-1][::2, ::2], self.theta_checkerboard[:-1][1::2, 1::2]),
                               phi=MeshParam(self.phi_checkerboard[:-1][::2, ::2], self.phi_checkerboard[:-1][1::2, 1::2]),
                               gamma=self.diagonal_phases)
