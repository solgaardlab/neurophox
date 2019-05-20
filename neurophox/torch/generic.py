from typing import List

import torch
import torch.nn as nn
import numpy as np

from ..config import PYTORCH
from ..control import MeshPhases
from ..meshmodel import MeshModel
from ..helpers import pairwise_off_diag_permutation, roll_torch, plot_complex_matrix


class TransformerLayer(nn.Module):
    def __init__(self, units: int, is_complex: bool=True, is_trainable: bool=False):
        super(TransformerLayer, self).__init__()
        self.units = units
        self.is_trainable = is_trainable
        self.is_complex = is_complex

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

    @property
    def matrix(self):
        return self.transform(np.eye(self.units, dtype=np.complex64)).numpy()

    @property
    def inverse_matrix(self):
        return self.inverse_transform(np.eye(self.units, dtype=np.complex64)).numpy()

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix)

    def forward(self, x):
        return self.transform(x)


class CompoundTransformerLayer(TransformerLayer):
    def __init__(self, units: int, transformer_list: List[TransformerLayer], is_complex: bool=True,
                 is_trainable: bool=False):
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
        super(PermutationLayer, self).__init__(units=len(permuted_indices))
        self.units = len(permuted_indices)
        self.permuted_indices = tuple(np.asarray(permuted_indices, dtype=np.int32))
        self.inv_permuted_indices = np.zeros_like(self.permuted_indices)
        for idx, perm_idx in enumerate(self.permuted_indices):
            self.inv_permuted_indices[perm_idx] = idx

    def transform(self, inputs: torch.Tensor):
        return inputs.gather(dim=-1, index=self.permuted_indices)

    def inverse_transform(self, outputs: torch.Tensor):
        return outputs.gather(dim=-1, index=self.inv_permuted_indices)


class MeshVerticalLayer(TransformerLayer):
    def __init__(self, units: int, diag: torch.Tensor, off_diag: torch.Tensor,
                 perm: PermutationLayer=None, right_perm: PermutationLayer=None):
        self.diag = diag
        self.off_diag = off_diag
        self.perm = perm
        self.right_perm = right_perm
        self.pairwise_perm_idx = pairwise_off_diag_permutation(units)
        super(MeshVerticalLayer, self).__init__(units)

    def transform(self, inputs: torch.Tensor):
        outputs = inputs.transpose(0, 1) if self.perm is None else self.perm.transform(inputs)
        diag_out = torch.stack(
            (outputs[0] * self.diag[0] - outputs[1] * self.diag[1],
             outputs[1] * self.diag[0] + outputs[0] * self.diag[1])
        )
        off_diag_out = torch.stack(
            (outputs[0] * self.off_diag[0] + outputs[1] * self.off_diag[1],
             outputs[1] * self.off_diag[0] + outputs[0] * self.off_diag[1])
        )
        outputs = diag_out + off_diag_out.gather(dim=-1, index=self.pairwise_perm_idx)
        outputs = outputs if self.right_perm is None else self.right_perm.transform(outputs)
        return outputs.transpose(0, 1)

    def inverse_transform(self, outputs: torch.Tensor):
        inputs = outputs.transpose(0, 1) if self.right_perm is None else self.right_perm.transform(outputs)
        diag = self.diag
        off_diag = self.off_diag.gather(dim=1, index=self.pairwise_perm_idx)
        diag_out = torch.stack(
            (inputs[0] * diag[0] + inputs[1] * diag[1],
             -inputs[1] * diag[0] + inputs[0] * diag[1])
        )
        off_diag_out = torch.stack(
            (inputs[0] * off_diag[0] + inputs[1] * off_diag[1],
             -inputs[1] * off_diag[0] + inputs[0] * off_diag[1])
        )
        inputs = diag_out + off_diag_out.gather(dim=-1, index=self.pairwise_perm_idx)
        inputs = inputs if self.perm is None else self.perm.transform(inputs)
        return inputs.transpose(0, 1)


class Mesh:
    def __init__(self, model: MeshModel):
        """
        General mesh network layer defined by MeshModel

        Args:
            model: MeshModel to define the overall mesh structure and errors
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers
        self.e_l = self.model.get_bs_error_matrix(right=False)
        if self.model.use_different_errors:
            self.e_r = self.model.get_bs_error_matrix(right=True)
        else:
            self.e_r = self.e_l
        self.enn, self.enp, self.epn, self.epp = self.model.mzi_error_tensors

    def mesh_layers(self, phases: MeshPhases):
        raise NotImplementedError("Pytorch is not yet supported.")

        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers

        # smooth trick to efficiently perform the layerwise coupling computation

        if self.model.hadamard:
            s11 = (self.epp * internal_psl + self.enn * roll_torch(internal_psl, up=True))
            s22 = roll_torch(self.enn * internal_psl + self.epp * roll_torch(internal_psl, up=True))
            s12 = roll_torch(self.enp * internal_psl - self.epn * roll_torch(internal_psl, up=True))
            s21 = (self.epn * internal_psl - self.enp * roll_torch(internal_psl, up=True))
        else:
            s11 = (self.epp * internal_psl - self.enn * roll_torch(internal_psl, up=True))
            s22 = roll_torch(-self.enn * internal_psl + self.epp * roll_torch(internal_psl, up=True))
            s12 = 1j * roll_torch(self.enp * internal_psl + self.epn * roll_torch(internal_psl, up=True))
            s21 = 1j * (self.epn * internal_psl + self.enp * roll_torch(internal_psl, up=True))

        diag_layers = external_psl * (s11 + s22) / 2
        off_diag_layers = roll_torch(external_psl) * (s21 + s12) / 2

        mesh_layers = []
        for layer in range(self.num_layers - 1):
            mesh_layers.append(MeshVerticalLayer(self.units, diag_layers[:, layer], off_diag_layers[:, layer],
                                                 PermutationLayer(self.model.perm_idx[layer])))
        mesh_layers.append(MeshVerticalLayer(self.units, diag_layers[-1], off_diag_layers[-1],
                                             PermutationLayer(self.model.perm_idx[-2]),
                                             PermutationLayer(self.model.perm_idx[-1])))
        return mesh_layers


class MeshLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The model of the mesh network (e.g., rectangular, triangular, butterfly)
        is_trainable: Whether variables in this layer are trainable
    """
    def __init__(self, mesh_model: MeshModel):
        self.mesh = Mesh(mesh_model)
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.theta, self.phi, self.gamma = self.mesh.model.init(backend=PYTORCH)
        super(MeshLayer, self).__init__(self.units)

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Pytorch is not yet supported.")

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Pytorch is not yet supported.")

    def adjoint_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform(inputs)

    @property
    def phases(self) -> MeshPhases:
        return MeshPhases(
            theta=self.theta.numpy() * self.mesh.model.mask,
            phi=self.phi.numpy() * self.mesh.model.mask,
            mask=self.mesh.model.mask,
            gamma=self.gamma.numpy()
        )
