from typing import Optional

import numpy as np

from ..config import NP_COMPLEX


class LinearOpticalComponent:
    """Two-port unitary linear optical component (unitary scattering matrix for waveguide modal interaction)

    Notes:
        By default, this class uses the identity matrix as the unitary
        and implements the transform and givens rotation on an input
        given whatever unitary definition is specified in the constructor

    """
    def __init__(self, dtype=NP_COMPLEX):
        self.dtype = dtype

    @property
    def reflection_coefficient(self):
        return 1

    @property
    def transmission_coefficient(self):
        return 0

    @property
    def matrix(self):
        return np.eye(2, dtype=self.dtype)

    @property
    def inverse_matrix(self):
        return self.matrix.conj().T

    def givens_rotation(self, units: int, m: int, n: Optional[int]=None, i_factor=False) -> np.ndarray:
        """Givens rotation matrix

        Notes:
            If trying to transform an input vector with a Givens rotation, you can use the `transform` method.
            This method just returns the Givens rotation matrix.

        Args:
            units: Input dimension to rotate
            m: Lower index to rotate
            n: Upper index to rotate, requires :math:`n > m` (defaults to `None`, which implies :math:`n = m + 1`)
            i_factor: multiply by i (useful for Clements and Reck decompositions)

        Returns:
            np.ndarray: the Givens rotation matrix

        """
        if n is None:
            n = m + 1
        if not m < n:
            raise Exception('Require m < n')
        unitary = self.matrix
        givens_rotation = np.eye(units, dtype=self.dtype)
        givens_rotation[m][m] = unitary[0, 0] * (-i_factor * 1j + (1 - i_factor))
        givens_rotation[m][n] = unitary[0, 1] * (-i_factor * 1j + (1 - i_factor))
        givens_rotation[n][m] = unitary[1, 0] * (i_factor * 1j + (1 - i_factor))
        givens_rotation[n][n] = unitary[1, 1] * (i_factor * 1j + (1 - i_factor))
        return givens_rotation

    def transform(self, input_vector: np.ndarray, m: int, n: Optional[int]=None) -> np.ndarray:
        """Transform an input vector

        Notes:
            If trying to transform an input vector with a Givens rotation, you can use the `transform` method.
            This method just returns the Givens rotation matrix.

        Args:
            input_vector: Input vector to transform
            m: Lower index to rotate
            n: Upper index to rotate, requires :math:`n > m` (defaults to `None`, which implies :math:`n = m + 1`)

        Returns:
            np.ndarray: the Givens rotation matrix

        """
        if n is None:
            n = m + 1
        if not m < n:
            raise Exception('Require m < n')
        transformed_vector = input_vector
        transformed_coordinates = self.matrix @ np.asarray([input_vector[m], input_vector[n]], dtype=self.dtype)
        transformed_vector[m] = transformed_coordinates[0]
        transformed_vector[n] = transformed_coordinates[1]
        return transformed_vector


class Beamsplitter(LinearOpticalComponent):
    """Ideal 50/50 beamsplitter (L)

    Implements the Hadamard or beamsplitter operator.

    Args:
        hadamard: Amplitude-modulating phase shift

    """
    def __init__(self, hadamard: bool, epsilon: float=0, dtype=NP_COMPLEX):
        super(Beamsplitter, self).__init__(dtype=dtype)
        self.hadamard = hadamard
        self.epsilon = epsilon

    @property
    def reflection_coefficient(self):
        return 0.5 + self.epsilon / 2

    @property
    def transmission_coefficient(self):
        return 0.5 - self.epsilon / 2

    @property
    def matrix(self):
        r = np.sqrt(self.reflection_coefficient)
        t = np.sqrt(self.transmission_coefficient)
        if self.hadamard:
            return np.array([
                [r, t],
                [t, -r]
            ], dtype=self.dtype)
        else:
            return np.array([
                [r, 1j * t],
                [1j * t, r]
            ], dtype=self.dtype)


class PhaseShiftUpper(LinearOpticalComponent):
    """Ideal 50/50 beamsplitter

    Implements the phase shift gate.

    Args:
        phase_shift: Phase shift

    """
    def __init__(self, phase_shift: float, dtype=NP_COMPLEX):
        super(PhaseShiftUpper, self).__init__(dtype=dtype)
        self.phase_shift = phase_shift

    @property
    def matrix(self):
        return np.array([
            [np.exp(1j * self.phase_shift), 0],
            [0, 1]
        ], dtype=self.dtype)


class PhaseShiftLower(LinearOpticalComponent):
    """Ideal 50/50 beamsplitter

    Implements the phase shift gate.

    Args:
        phase_shift: Phase shift

    """
    def __init__(self, phase_shift: float, dtype=NP_COMPLEX):
        super(PhaseShiftLower, self).__init__(dtype=dtype)
        self.phase_shift = phase_shift

    @property
    def matrix(self):
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.phase_shift)]
        ], dtype=self.dtype)


class PhaseShiftDifferentialMode(LinearOpticalComponent):
    """Ideal 50/50 beamsplitter

    Implements the phase shift gate.

    Args:
        phase_shift: Phase shift

    """
    def __init__(self, phase_shift: float, dtype=NP_COMPLEX):
        super(PhaseShiftDifferentialMode, self).__init__(dtype=dtype)
        self.phase_shift = phase_shift

    @property
    def matrix(self):
        return np.array([
            [np.exp(1j * self.phase_shift / 2), 0],
            [0, np.exp(-1j * self.phase_shift / 2)]
        ], dtype=self.dtype)


class PhaseShiftCommonMode(LinearOpticalComponent):
    """Ideal 50/50 beamsplitter

    Implements the phase shift gate.

    Args:
        phase_shift: Phase shift

    """
    def __init__(self, phase_shift: float, dtype=NP_COMPLEX):
        super(PhaseShiftCommonMode, self).__init__(dtype=dtype)
        self.phase_shift = phase_shift

    @property
    def matrix(self):
        return np.array([
            [np.exp(1j * self.phase_shift), 0],
            [0, np.exp(1j * self.phase_shift)]
        ], dtype=self.dtype)
