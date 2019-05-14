import numpy as np
from typing import Union, Tuple

from ..config import NP_COMPLEX
from .transfermatrix import LinearOpticalComponent


class OTB(LinearOpticalComponent):
    """Ideal orthogonal tunable beamsplitter

        Class simulating a phase-shifting orthogonal tunable beamsplitter (Bloch MZI with :math:`\\phi = 0`)
        As usual in our simulation environment, we have :math:`\\theta \in [0, \pi]`.

        Args:
            theta: Amplitude-modulating phase shift
            epsilon: Beamsplitter error
            dtype: Type-casting to use for the matrix elements
    """

    def __init__(self, theta: float, epsilon: Union[float, Tuple[float, float]] = 0.0, dtype=NP_COMPLEX):
        super(OTB, self).__init__(dtype=dtype)
        self.theta = theta  # reflectivity/transmittivity (internal phase shifter)
        self.epsilon = (epsilon, epsilon) if isinstance(epsilon, float) else epsilon

    @property
    def reflection_coefficient(self):
        return np.cos(self.theta / 2) ** 2

    @property
    def transmission_coefficient(self):
        return np.sin(self.theta / 2) ** 2

    @property
    def matrix(self):
        return get_mzi_transfer_matrix(
            internal_upper=self.theta / 2,
            internal_lower=-self.theta / 2,
            external_upper=np.pi / 2,
            external_lower=np.pi / 2,
            epsilon=self.epsilon,
            hadamard=False,
            dtype=self.dtype
        )


class MZI(LinearOpticalComponent):
    """Ideal Beamsplitter Mach-Zehnder Interferometer

    Class simulating an ideal phase-shifting Mach-Zehnder interferometer.
    As usual in our simulation environment, we have :math:`\\theta \in [0, \pi]` and :math:`\phi \in [0, 2\pi]`.

    Args:
        theta: Amplitude-modulating phase shift
        phi: Setpoint phase shift
        hadamard: Whether to use Hadamard convention
        epsilon: Beamsplitter error
        dtype: Type-casting to use for the matrix elements
    """

    def __init__(self, internal_upper: float, internal_lower: float, external_upper: float, external_lower: float,
                 hadamard: bool, epsilon: Union[float, Tuple[float, float]] = 0.0, dtype=NP_COMPLEX):
        super(MZI, self).__init__(dtype=dtype)
        self.internal_upper = internal_upper
        self.internal_lower = internal_lower
        self.external_upper = external_upper
        self.external_lower = external_lower
        self.hadamard = hadamard
        self.epsilon = (epsilon, epsilon) if isinstance(epsilon, float) or isinstance(epsilon, int) else epsilon

    @property
    def reflection_coefficient(self):
        return np.cos(self.internal_upper - self.internal_lower) ** 2

    @property
    def transmission_coefficient(self):
        return np.sin(self.internal_upper - self.internal_lower) ** 2

    @property
    def matrix(self):
        return get_mzi_transfer_matrix(
            internal_upper=self.internal_upper,
            internal_lower=self.internal_lower,
            external_upper=self.external_upper,
            external_lower=self.external_lower,
            epsilon=self.epsilon,
            hadamard=self.hadamard,
            dtype=self.dtype
        )


class SMMZI(MZI):
    """Ideal Beamsplitter Mach-Zehnder Interferometer (single-mode basis)

    Class simulating an ideal phase-shifting Mach-Zehnder interferometer.
    As usual in our simulation environment, we have :math:`\\theta \in [0, \pi]` and :math:`\phi \in [0, 2\pi)`.

    Args:
        theta: Amplitude-modulating phase shift
        phi: Setpoint phase shift,
        hadamard: Whether to use Hadamard convention
        epsilon: Beamsplitter error
        dtype: Type-casting to use for the matrix elements
    """

    def __init__(self, theta: float, phi: float, hadamard: bool,
                 epsilon: Union[float, Tuple[float, float]] = 0.0, dtype=NP_COMPLEX):
        self.theta = theta
        self.phi = phi
        super(SMMZI, self).__init__(
            internal_upper=theta,
            internal_lower=0,
            external_upper=phi,
            external_lower=0,
            epsilon=epsilon,
            hadamard=hadamard,
            dtype=dtype
        )


class BlochMZI(MZI):
    """Ideal Beamsplitter Mach-Zehnder Interferometer (Bloch basis)

    Class simulating an ideal phase-shifting Mach-Zehnder interferometer.
    As usual in our simulation environment, we have :math:`\\theta \in [0, \pi]` and :math:`\phi \in [0, 2\pi)`.

    Args:
        theta: Amplitude-modulating phase shift
        phi: Setpoint phase shift,
        hadamard: Whether to use Hadamard convention
        epsilon: Beamsplitter error
        dtype: Type-casting to use for the matrix elements
    """

    def __init__(self, theta: float, phi: float, hadamard: bool,
                 epsilon: Union[float, Tuple[float, float]] = 0.0, dtype=NP_COMPLEX):
        self.theta = theta
        self.phi = phi
        super(BlochMZI, self).__init__(
            internal_upper=theta / 2,
            internal_lower=-theta / 2,
            external_upper=phi,
            external_lower=0,
            epsilon=epsilon,
            hadamard=hadamard,
            dtype=dtype
        )


def get_mzi_transfer_matrix(internal_upper: float, internal_lower: float, external_upper: float, external_lower: float,
                            epsilon: Tuple[float, float], hadamard: float, dtype):
    epp = np.sqrt(1 + epsilon[0]) * np.sqrt(1 + epsilon[1])
    epn = np.sqrt(1 + epsilon[0]) * np.sqrt(1 - epsilon[1])
    enp = np.sqrt(1 - epsilon[0]) * np.sqrt(1 + epsilon[1])
    enn = np.sqrt(1 - epsilon[0]) * np.sqrt(1 - epsilon[1])
    iu, il, eu, el = internal_upper, internal_lower, external_upper, external_lower
    if hadamard:
        return 0.5 * np.array([
            [(epp * np.exp(1j * iu) + enn * np.exp(1j * il)) * np.exp(1j * eu),
             (epn * np.exp(1j * iu) - enp * np.exp(1j * il)) * np.exp(1j * el)],
            [(enp * np.exp(1j * iu) - epn * np.exp(1j * il)) * np.exp(1j * eu),
             (enn * np.exp(1j * iu) + epp * np.exp(1j * il)) * np.exp(1j * el)]
        ], dtype=dtype)
    else:
        return 0.5 * np.array([
            [(epp * np.exp(1j * iu) - enn * np.exp(1j * il)) * np.exp(1j * eu),
             1j * (epn * np.exp(1j * iu) + enp * np.exp(1j * il)) * np.exp(1j * el)],
            [1j * (enp * np.exp(1j * iu) + epn * np.exp(1j * il)) * np.exp(1j * eu),
             -(enn * np.exp(1j * iu) - epp * np.exp(1j * il)) * np.exp(1j * el)]
        ], dtype=dtype)
