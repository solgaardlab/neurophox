from typing import List

try:
    import torch
    from torch.nn import Module
except ImportError:
    pass

import numpy as np

from ..config import PYTORCH
from .helpers import MeshPhasesTorch
from ..meshmodel import MeshModel
from ..helpers import pairwise_off_diag_permutation, plot_complex_matrix


class TransformerLayer(Module):
    def __init__(self, units: int, is_complex: bool = True, is_trainable: bool = False):
        super(TransformerLayer, self).__init__()
        self.units = units
        self.is_trainable = is_trainable
        self.is_complex = is_complex

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

    @property
    def matrix(self) -> np.ndarray:
        torch_matrix = self.transform(np.eye(self.units, dtype=np.complex64)).detach().numpy()
        return torch_matrix[0] + 1j * torch_matrix[1]

    @property
    def inverse_matrix(self):
        torch_matrix = self.inverse_transform(np.eye(self.units, dtype=np.complex64)).detach().numpy()
        return torch_matrix[0] + 1j * torch_matrix[1]

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix)

    def forward(self, x):
        return self.transform(x)


class CompoundTransformerLayer(TransformerLayer):
    def __init__(self, units: int, transformer_list: List[TransformerLayer], is_complex: bool = True,
                 is_trainable: bool = False):
        self.transformer_list = transformer_list
        super(CompoundTransformerLayer, self).__init__(units=units, is_complex=is_complex, is_trainable=is_trainable)

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for transformer in self.transformer_list:
            outputs = transformer.transform(outputs)
        return outputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        inputs = outputs
        for transformer in self.transformer_list[::-1]:
            inputs = transformer.inverse_transform(inputs)
        return inputs


class PermutationLayer(TransformerLayer):
    def __init__(self, permuted_indices: np.ndarray):
        super(PermutationLayer, self).__init__(units=permuted_indices.shape[0])
        self.units = permuted_indices.shape[0]
        self.permuted_indices = torch.as_tensor(np.asarray(permuted_indices, dtype=np.long))
        self.inv_permuted_indices = torch.as_tensor(np.zeros_like(self.permuted_indices))
        for idx, perm_idx in enumerate(self.permuted_indices):
            self.inv_permuted_indices[perm_idx] = idx

    def transform(self, inputs: torch.Tensor):
        return inputs[..., self.permuted_indices]

    def inverse_transform(self, outputs: torch.Tensor):
        return outputs[..., self.inv_permuted_indices]


class MeshVerticalLayer(TransformerLayer):
    def __init__(self, pairwise_perm_idx: np.ndarray, diag: torch.Tensor, off_diag: torch.Tensor,
                 right_perm: PermutationLayer = None, left_perm: PermutationLayer = None):
        self.diag = diag
        self.off_diag = off_diag
        self.pairwise_perm_idx = pairwise_perm_idx
        super(MeshVerticalLayer, self).__init__(pairwise_perm_idx.shape[0])
        self.left_perm = left_perm
        self.right_perm = right_perm

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, np.ndarray):
            inputs = to_complex_t(inputs)
        outputs = inputs if self.left_perm is None else self.left_perm(inputs)
        diag_out = cc_mul(outputs, self.diag)
        off_diag_out = cc_mul(outputs, self.off_diag)
        outputs = diag_out + off_diag_out[..., self.pairwise_perm_idx]
        outputs = outputs if self.right_perm is None else self.right_perm(outputs)
        return outputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, np.ndarray):
            outputs = to_complex_t(outputs)
        inputs = outputs if self.right_perm is None else self.right_perm.inverse_transform(outputs)
        diag = conj_t(self.diag)
        off_diag = conj_t(self.off_diag[..., self.pairwise_perm_idx])
        diag_out = cc_mul(inputs, diag)
        off_diag_out = cc_mul(inputs, off_diag)
        inputs = diag_out + off_diag_out[..., self.pairwise_perm_idx]
        inputs = inputs if self.left_perm is None else self.left_perm.inverse_transform(inputs)
        return inputs


class MeshTorch:
    def __init__(self, model: MeshModel):
        """
        General mesh network layer defined by MeshModel

        Args:
            model: MeshModel to define the overall mesh structure and errors
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers
        self.pairwise_perm_idx = pairwise_off_diag_permutation(self.units)
        enn, enp, epn, epp = self.model.mzi_error_tensors
        self.enn, self.enp, self.epn, self.epp = torch.as_tensor(enn), torch.as_tensor(enp),\
                                                 torch.as_tensor(epn), torch.as_tensor(epp)
        self.perm_layers = [PermutationLayer(self.model.perm_idx[layer]) for layer in range(self.num_layers + 1)]

    def mesh_layers(self, phases: MeshPhasesTorch) -> List[MeshVerticalLayer]:
        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers

        # smooth trick to efficiently perform the layerwise coupling computation

        if self.model.hadamard:
            s11 = rc_mul(self.epp, internal_psl) + rc_mul(self.enn, internal_psl.roll(-1, 1))
            s22 = (rc_mul(self.enn, internal_psl) + rc_mul(self.epp, internal_psl.roll(-1, 1))).roll(1, 1)
            s12 = (rc_mul(self.enp, internal_psl) - rc_mul(self.epn, internal_psl.roll(-1, 1))).roll(1, 1)
            s21 = rc_mul(self.epn, internal_psl) - rc_mul(self.enp, internal_psl.roll(-1, 1))
        else:
            s11 = rc_mul(self.epp, internal_psl) - rc_mul(self.enn, internal_psl.roll(-1, 1))
            s22 = (-rc_mul(self.enn, internal_psl) + rc_mul(self.epp, internal_psl.roll(-1, 1))).roll(1, 1)
            s12 = s_mul(1j, (rc_mul(self.enp, internal_psl) + rc_mul(self.epn, internal_psl.roll(-1, 1))).roll(1, 1))
            s21 = s_mul(1j, (rc_mul(self.epn, internal_psl) + rc_mul(self.enp, internal_psl.roll(-1, 1))))

        diag_layers = cc_mul(external_psl, s11 + s22) / 2
        off_diag_layers = cc_mul(external_psl.roll(1, 1), s21 + s12) / 2

        if self.units % 2:
            diag_layers = torch.cat((diag_layers[:, :-1], to_complex_t(np.ones((1, diag_layers.size()[-1])))), dim=1)

        diag_layers, off_diag_layers = diag_layers.transpose(1, 2), off_diag_layers.transpose(1, 2)

        mesh_layers = [MeshVerticalLayer(
            self.pairwise_perm_idx, diag_layers[:, 0], off_diag_layers[:, 0], self.perm_layers[1], self.perm_layers[0])]
        for layer in range(1, self.num_layers):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[:, layer],
                                                 off_diag_layers[:, layer], self.perm_layers[layer + 1]))

        return mesh_layers


class MeshTorchLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The model of the mesh network (e.g., rectangular, triangular, butterfly)
    """

    def __init__(self, mesh_model: MeshModel, **kwargs):
        self.mesh = MeshTorch(mesh_model)
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        super(MeshTorchLayer, self).__init__(self.units, **kwargs)
        theta_init, phi_init, gamma_init = self.mesh.model.init()
        self.theta, self.phi, self.gamma = theta_init.to_torch(), phi_init.to_torch(), gamma_init.to_torch()

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        mesh_phases = MeshPhasesTorch(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            hadamard=self.mesh.model.hadamard,
            units=self.units,
            basis=self.mesh.model.basis
        )
        mesh_layers = self.mesh.mesh_layers(mesh_phases)
        if isinstance(inputs, np.ndarray):
            inputs = to_complex_t(inputs)
        outputs = cc_mul(inputs, mesh_phases.input_phase_shift_layer)
        for layer in range(self.num_layers):
            outputs = mesh_layers[layer].transform(outputs)
        return outputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        mesh_phases = MeshPhasesTorch(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            hadamard=self.mesh.model.hadamard,
            units=self.units,
            basis=self.mesh.model.basis
        )
        mesh_layers = self.mesh.mesh_layers(mesh_phases)
        inputs = to_complex_t(outputs) if isinstance(outputs, np.ndarray) else outputs
        for layer in reversed(range(self.num_layers)):
            inputs = mesh_layers[layer].inverse_transform(inputs)
        inputs = cc_mul(inputs, conj_t(mesh_phases.input_phase_shift_layer))
        return inputs

    def adjoint_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform(inputs)

    @property
    def phases(self) -> MeshPhasesTorch:
        return MeshPhasesTorch(
            theta=self.theta.numpy() * self.mesh.model.mask,
            phi=self.phi.numpy() * self.mesh.model.mask,
            mask=self.mesh.model.mask,
            gamma=self.gamma.numpy(),
            units=self.units
        )

# temporary helpers until pytorch supports complex numbers...which should be soon!
# rc_mul is "real * complex" op
# cc_mul is "complex * complex" op
# s_mul is "complex scalar * complex" op


def rc_mul(real: torch.Tensor, comp: torch.Tensor):
    return real.unsqueeze(dim=0) * comp


def cc_mul(comp1: torch.Tensor, comp2: torch.Tensor) -> torch.Tensor:
    real = comp1[0] * comp2[0] - comp1[1] * comp2[1]
    comp = comp1[0] * comp2[1] + comp1[1] * comp2[0]
    return torch.stack((real, comp), dim=0)


def s_mul(s: np.complex, comp: torch.Tensor):
    return s.real * comp + torch.stack((-s.imag * comp[1], s.imag * comp[0]))


def conj_t(comp: torch.Tensor):
    return torch.stack((comp[0], -comp[1]), dim=0)


def to_complex_t(nparray: np.ndarray):
    return torch.stack((torch.as_tensor(nparray.real), torch.as_tensor(nparray.imag)), dim=0)
