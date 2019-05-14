import numpy as np
import tensorflow as tf
import torch
from torch.nn import Parameter

from ..config import BLOCH, SINGLEMODE
from .param import MeshParam, MeshParamTensorflow, MeshParamTorch


class MeshPhases:
    def __init__(self, theta: np.ndarray, phi: np.ndarray, gamma: np.ndarray, mask: np.ndarray=None,
                 basis: str = BLOCH, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParam(theta * self.mask + (1 - self.mask) * (1 - hadamard) * np.pi, gamma.size)
        self.phi = MeshParam(phi * self.mask + (1 - self.mask) * (1 - hadamard) * np.pi, gamma.size)
        self.gamma = gamma
        self.hadamard = hadamard
        self.basis = basis
        self.input_phase_shift_layer = np.exp(1j * gamma)
        if self.theta.param.shape != self.phi.param.shape:
            raise ValueError("Internal phases (theta) and external phases (phi) need to have the same shape.")

    @property
    def internal_phase_shifts(self):
        if self.basis == BLOCH:
            return self.theta.differential_mode_arrangement
        elif self.basis == SINGLEMODE:
            return self.theta.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def external_phase_shifts(self):
        if self.basis == BLOCH or self.basis == SINGLEMODE:
            return self.phi.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def internal_phase_shift_layers(self):
        return np.exp(1j * self.internal_phase_shifts)

    @property
    def external_phase_shift_layers(self):
        return np.exp(1j * self.external_phase_shifts)

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


class MeshPhasesTensorflow:
    def __init__(self, theta: tf.Variable, phi: tf.Variable, mask: np.ndarray, gamma: tf.Variable, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParamTensorflow(theta * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.phi = MeshParamTensorflow(phi * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.gamma = gamma
        self.basis = basis
        self.input_phase_shift_layer = tf.complex(tf.cos(gamma), tf.sin(gamma))
        if self.theta.param.shape != self.phi.param.shape:
            raise ValueError("Internal phases (theta) and external phases (phi) need to have the same shape.")

    @property
    def internal_phase_shifts(self):
        if self.basis == BLOCH:
            return self.theta.differential_mode_arrangement
        elif self.basis == SINGLEMODE:
            return self.theta.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def external_phase_shifts(self):
        if self.basis == BLOCH or self.basis == SINGLEMODE:
            return self.phi.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def internal_phase_shift_layers(self):
        internal_ps = self.internal_phase_shifts
        return tf.complex(tf.cos(internal_ps), tf.sin(internal_ps))

    @property
    def external_phase_shift_layers(self):
        external_ps = self.external_phase_shifts
        return tf.complex(tf.cos(external_ps), tf.sin(external_ps))


class MeshPhasesTorch:
    def __init__(self, theta: Parameter, phi: Parameter, mask: np.ndarray, gamma: Parameter, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParamTorch(theta * mask + (1 - mask) * (1 - hadamard) * np.pi, units)
        self.phi = MeshParamTorch(phi * mask, units=units)
        self.gamma = gamma
        self.basis = basis
        self.input_phase_shift_layer = tf.complex(tf.cos(gamma), tf.sin(gamma))
        if self.theta.param.shape != self.phi.param.shape:
            raise ValueError("Internal phases (theta) and external phases (phi) need to have the same shape.")

    @property
    def internal_phase_shifts(self):
        if self.basis == BLOCH:
            return self.theta.differential_mode_arrangement
        elif self.basis == SINGLEMODE:
            return self.theta.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def external_phase_shifts(self):
        if self.basis == BLOCH or self.basis == SINGLEMODE:
            return self.phi.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def internal_phase_shift_layers(self):
        internal_ps = self.internal_phase_shifts
        return torch.stack((internal_ps.cos(), internal_ps.sin()))

    @property
    def external_phase_shift_layers(self):
        external_ps = self.external_phase_shifts
        return torch.stack((external_ps.cos(), external_ps.sin()))
