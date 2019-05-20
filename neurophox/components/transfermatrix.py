from typing import Optional

import numpy as np

from ..config import NP_COMPLEX


class PairwiseUnitary:
    """Pairwise unitary

    This can be considered also a two-port unitary linear optical component
    (e.g. unitary scattering matrix for waveguide modal interaction).

    Notes:
        By default, this class uses the identity matrix as the unitary
        and implements the transform and givens rotation on an input
        given whatever unitary definition is specified in the constructor

    Args:
        dtype: Cast values as `dtype` for this pairwise unitary operator

    """
    def __init__(self, dtype=NP_COMPLEX):
        self.dtype = dtype

    @property
    def reflection_coefficient(self):
        """

        Returns:
            Reflection coefficient, :math:`r`, or reflectivity of :math:`2 \\times 2` transfer matrix
        """
        raise NotImplementedError("Need to override this method in child class.")

    @property
    def transmission_coefficient(self):
        """

        Returns:
            Transmission coefficient, :math:`t`, or transmissivity of :math:`2 \\times 2` transfer matrix
        """
        raise NotImplementedError("Need to override this method in child class.")

    @property
    def matrix(self):
        """

        Returns:
            :math:`U_2`, a :math:`2 \\times 2` unitary matrix implemented by this component
        """
        raise NotImplementedError("Need to override this method in child class.")

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
            raise ValueError('Require m < n')
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
            raise ValueError('Require m < n')
        transformed_vector = input_vector
        transformed_coordinates = self.matrix @ np.asarray([input_vector[m], input_vector[n]], dtype=self.dtype)
        transformed_vector[m] = transformed_coordinates[0]
        transformed_vector[n] = transformed_coordinates[1]
        return transformed_vector


class Beamsplitter(PairwiseUnitary):
    """Ideal 50/50 beamsplitter

    Implements the Hadamard or beamsplitter operator.

    Hadamard transformation, :math:`H`:

    .. math::
        H = \\frac{1}{\sqrt{2}}\\begin{bmatrix} 1 & 1\\\ 1 & -1 \\end{bmatrix}

    Beamsplitter transformation :math:`B`:

    .. math::
        B = \\frac{1}{\sqrt{2}}\\begin{bmatrix} 1 & i\\\ i & 1 \\end{bmatrix}

    Hadamard transformation, :math:`H_\epsilon` (with error :math:`\epsilon`):

    .. math::
        H_\epsilon = \\frac{1}{\sqrt{2}}\\begin{bmatrix} \sqrt{1 + \epsilon} & \sqrt{1 - \epsilon}\\\ \sqrt{1 - \epsilon} & -\sqrt{1 + \epsilon} \\end{bmatrix}

    Beamsplitter transformation :math:`B_\epsilon` (with error :math:`\epsilon`):

    .. math::
        B_\epsilon = \\frac{1}{\sqrt{2}}\\begin{bmatrix} \sqrt{1 + \epsilon} & i\sqrt{1 - \epsilon}\\\ i\sqrt{1 - \epsilon} & \sqrt{1 + \epsilon} \\end{bmatrix}

    Args:
        hadamard: Amplitude-modulating phase shift
        epsilon: Errors for beamsplitter operator
        dtype: Cast values as `dtype` for this pairwise unitary operator

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


class PhaseShiftUpper(PairwiseUnitary):
    """Upper phase shift operator

    Implements the upper phase shift operator :math:`L(\\theta)` given phase :math:`\\theta`:

    .. math::
        L(\\theta) = \\begin{bmatrix} e^{i\\theta} & 0\\\ 0 & 1 \\end{bmatrix}

    Args:
        phase_shift: Phase shift :math:`\\theta`
        dtype: Cast values as `dtype` for this pairwise unitary operator

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


class PhaseShiftLower(PairwiseUnitary):
    """Lower phase shift operator

    Implements the upper phase shift operator :math:`L(\\theta)` given phase :math:`\\theta`:

    .. math::
        R(\\theta) = \\begin{bmatrix} 1 & 0\\\ 0 & e^{i\\theta} \\end{bmatrix}

    Args:
        phase_shift: Phase shift :math:`\\theta`
        dtype: Cast values as `dtype` for this pairwise unitary operator

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


class PhaseShiftDifferentialMode(PairwiseUnitary):
    """Differential phase shift operator

    Implements the upper phase shift operator :math:`L(\\theta)` given phase :math:`\\theta`:

    .. math::
        D(\\theta) = \\begin{bmatrix} e^{i\\theta / 2} & 0\\\ 0 & e^{-i\\theta/2} \\end{bmatrix}

    Args:
        phase_shift: Phase shift :math:`\\theta`
        dtype: Cast values as `dtype` for this pairwise unitary operator

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


class PhaseShiftCommonMode(PairwiseUnitary):
    """Common mode phase shift operator

    Implements the upper phase shift operator :math:`L(\\theta)` given phase :math:`\\theta`:

    .. math::
        C(\\theta) = \\begin{bmatrix} e^{i\\theta} & 0\\\ 0 & e^{i\\theta} \\end{bmatrix}

    Args:
        phase_shift: Phase shift :math:`\\theta`
        dtype: Cast values as `dtype` for this pairwise unitary operator

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
