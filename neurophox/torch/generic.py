from typing import List

import torch
from torch.nn import Module, Parameter
import numpy as np

from ..numpy.generic import MeshPhases
from ..config import BLOCH, SINGLEMODE
from ..meshmodel import MeshModel
from ..helpers import pairwise_off_diag_permutation, plot_complex_matrix


class TransformerLayer(Module):
    """Base transformer class for transformer layers (invertible functions, usually linear)

    Args:
        units: Dimension of the input to be transformed by the transformer
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, is_trainable: bool = False):
        super(TransformerLayer, self).__init__()
        self.units = units
        self.is_trainable = is_trainable

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

    @property
    def matrix(self) -> np.ndarray:
        torch_matrix = self.transform(np.eye(self.units, dtype=np.complex64)).cpu().detach().numpy()
        return torch_matrix[0] + 1j * torch_matrix[1]

    @property
    def inverse_matrix(self):
        torch_matrix = self.inverse_transform(np.eye(self.units, dtype=np.complex64)).cpu().detach().numpy()
        return torch_matrix[0] + 1j * torch_matrix[1]

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix)

    def forward(self, x):
        return self.transform(x)


class CompoundTransformerLayer(TransformerLayer):
    def __init__(self, units: int, transformer_list: List[TransformerLayer], is_trainable: bool = False):
        self.transformer_list = transformer_list
        super(CompoundTransformerLayer, self).__init__(units=units, is_trainable=is_trainable)

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
        self.permuted_indices = np.asarray(permuted_indices, dtype=np.long)
        self.inv_permuted_indices = np.zeros_like(self.permuted_indices)
        for idx, perm_idx in enumerate(self.permuted_indices):
            self.inv_permuted_indices[perm_idx] = idx

    def transform(self, inputs: torch.Tensor):
        return inputs[..., self.permuted_indices]

    def inverse_transform(self, outputs: torch.Tensor):
        return outputs[..., self.inv_permuted_indices]


class MeshVerticalLayer(TransformerLayer):
    """
    Args:
        diag: the diagonal terms to multiply
        off_diag: the off-diagonal terms to multiply
        left_perm: the permutation for the mesh vertical layer (prior to the coupling operation)
        right_perm: the right permutation for the mesh vertical layer
            (usually for the final layer and after the coupling operation)
    """

    def __init__(self, pairwise_perm_idx: np.ndarray, diag: torch.Tensor, off_diag: torch.Tensor,
                 right_perm: PermutationLayer = None, left_perm: PermutationLayer = None):
        self.diag = diag
        self.off_diag = off_diag
        self.pairwise_perm_idx = pairwise_perm_idx
        super(MeshVerticalLayer, self).__init__(pairwise_perm_idx.shape[0])
        self.left_perm = left_perm
        self.right_perm = right_perm

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U^{(\ell')},

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Propaged :code:`inputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`.
        """
        if isinstance(inputs, np.ndarray):
            inputs = to_complex_t(inputs, self.device)
        outputs = inputs if self.left_perm is None else self.left_perm(inputs)
        diag_out = cc_mul(outputs, self.diag)
        off_diag_out = cc_mul(outputs, self.off_diag)
        outputs = diag_out + off_diag_out[..., self.pairwise_perm_idx]
        outputs = outputs if self.right_perm is None else self.right_perm(outputs)
        return outputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Inverse-propagate :code:`inputs` through single layer :math:`\ell < L`
        (where :math:`U_\ell` represents the matrix for layer :math:`\ell`):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} (U^{(\ell')})^\dagger,

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse propaged :code:`outputs` through single layer :math:`\ell` to form an array
            :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.
        """
        if isinstance(outputs, np.ndarray):
            outputs = to_complex_t(outputs, self.device)
        inputs = outputs if self.right_perm is None else self.right_perm.inverse_transform(outputs)
        diag = conj_t(self.diag)
        off_diag = conj_t(self.off_diag[..., self.pairwise_perm_idx])
        diag_out = cc_mul(inputs, diag)
        off_diag_out = cc_mul(inputs, off_diag)
        inputs = diag_out + off_diag_out[..., self.pairwise_perm_idx]
        inputs = inputs if self.left_perm is None else self.left_perm.inverse_transform(inputs)
        return inputs


class MeshParamTorch:
    """A class that cleanly arranges parameters into a specific arrangement that can be used to simulate any mesh

    Args:
        param: parameter to arrange in mesh
        units: number of inputs/outputs of the mesh
    """

    def __init__(self, param: torch.Tensor, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self) -> torch.Tensor:
        """
        The single-mode arrangement based on the :math:`L(\\theta)` transfer matrix for :code:`PhaseShiftUpper`
        is one where elements of `param` are on the even rows and all odd rows are zero.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_m` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the single-mode arrangement has the stripe array form
        :math:`\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{0}, \\boldsymbol{\\theta}_2, \\boldsymbol{0}, \ldots \\boldsymbol{\\theta}_N, \\boldsymbol{0}]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh
        and :math:`\\boldsymbol{0}` represents an array of zeros of the same size as :math:`\\boldsymbol{\\theta}_n`.

        Returns:
            Single-mode arrangement array of phases

        """
        num_layers = self.param.shape[0]
        tensor_t = self.param.t()
        stripe_tensor = torch.zeros(self.units, num_layers, dtype=torch.float32, device=self.param.device)
        if self.units % 2:
            stripe_tensor[:-1][::2] = tensor_t
        else:
            stripe_tensor[::2] = tensor_t
        return stripe_tensor

    @property
    def common_mode_arrangement(self) -> torch.Tensor:
        """
        The common-mode arrangement based on the :math:`C(\\theta)` transfer matrix for :code:`PhaseShiftCommonMode`
        is one where elements of `param` are on the even rows and repeated on respective odd rows.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the common-mode arrangement has the stripe array form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_N, \\boldsymbol{\\theta}_N]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh.

        Returns:
            Common-mode arrangement array of phases
        """
        phases = self.single_mode_arrangement
        return phases + phases.roll(1, 0)

    @property
    def differential_mode_arrangement(self) -> torch.Tensor:
        """
        The differential-mode arrangement is based on the :math:`D(\\theta)` transfer matrix
        for :code:`PhaseShiftDifferentialMode`.

        Given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\cdots \\boldsymbol{\\theta}_m \cdots]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the differential-mode arrangement has the form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = \\left[\cdots \\frac{\\boldsymbol{\\theta}_m}{2}, -\\frac{\\boldsymbol{\\theta}_m}{2} \cdots \\right]^T`.
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the :math:`\\boldsymbol{\\theta}` of the final mesh.

        Returns:
            Differential-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases / 2 - phases.roll(1, 0) / 2

    def __add__(self, other):
        return MeshParamTorch(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTorch(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTorch(self.param * other.param, self.units)


class MeshPhasesTorch:
    """Organizes the phases in the mesh into appropriate arrangements

    Args:
        theta: Array to be converted to :math:`\\boldsymbol{\\theta}`
        phi: Array to be converted to :math:`\\boldsymbol{\\phi}`
        gamma: Array to be converted to :math:`\\boldsymbol{\gamma}`
        mask: Mask over values of :code:`theta` and :code:`phi` that are not in bar state
        basis: Phase basis to use
        hadamard: Whether to use Hadamard convention
    """

    def __init__(self, theta: Parameter, phi: Parameter, mask: np.ndarray, gamma: Parameter, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        torch_mask = torch.as_tensor(mask, dtype=theta.dtype, device=theta.device)
        torch_inv_mask = torch.as_tensor(1 - mask, dtype=theta.dtype, device=theta.device)
        self.theta = MeshParamTorch(theta * torch_mask + torch_inv_mask * (1 - hadamard) * np.pi, units=units)
        self.phi = MeshParamTorch(phi * torch_mask + torch_inv_mask * (1 - hadamard) * np.pi, units=units)
        self.gamma = gamma
        self.basis = basis
        self.input_phase_shift_layer = phasor(gamma)
        if self.theta.param.shape != self.phi.param.shape:
            raise ValueError("Internal phases (theta) and external phases (phi) need to have the same shape.")

    @property
    def internal_phase_shifts(self):
        """The internal phase shift matrix of the mesh corresponds to an `L \\times N` array of phase shifts
        (in between beamsplitters, thus internal) where :math:`L` is number of layers and :math:`N` is number of inputs/outputs

        Returns:
            Internal phase shift matrix corresponding to :math:`\\boldsymbol{\\theta}`
        """
        if self.basis == BLOCH:
            return self.theta.differential_mode_arrangement
        elif self.basis == SINGLEMODE:
            return self.theta.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")

    @property
    def external_phase_shifts(self):
        """The external phase shift matrix of the mesh corresponds to an `L \\times N` array of phase shifts
        (outside of beamsplitters, thus external) where :math:`L` is number of layers and :math:`N` is number of inputs/outputs

        Returns:
            External phase shift matrix corresponding to :math:`\\boldsymbol{\\phi}`
        """
        if self.basis == BLOCH or self.basis == SINGLEMODE:
            return self.phi.single_mode_arrangement
        else:
            raise NotImplementedError(f"{self.basis} is not yet supported or invalid.")


class MeshTorchLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The model of the mesh network (e.g., rectangular, triangular, butterfly)
    """

    def __init__(self, mesh_model: MeshModel):
        super(MeshTorchLayer, self).__init__(mesh_model.units)
        self.mesh_model = mesh_model
        enn, enp, epn, epp = self.mesh_model.mzi_error_tensors
        enn, enp, epn, epp = torch.as_tensor(enn, dtype=torch.float32), torch.as_tensor(enp, dtype=torch.float32), \
                             torch.as_tensor(epn, dtype=torch.float32), torch.as_tensor(epp, dtype=torch.float32)
        self.register_buffer("enn", enn)
        self.register_buffer("enp", enp)
        self.register_buffer("epn", epn)
        self.register_buffer("epp", epp)
        theta_init, phi_init, gamma_init = self.mesh_model.init()
        self.units, self.num_layers = self.mesh_model.units, self.mesh_model.num_layers
        self.theta, self.phi, self.gamma = theta_init.to_torch(), phi_init.to_torch(), gamma_init.to_torch()
        self.pairwise_perm_idx = pairwise_off_diag_permutation(self.units)
        self.perm_layers = [PermutationLayer(self.mesh_model.perm_idx[layer]) for layer in range(self.num_layers + 1)]

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Transformed :code:`inputs`, :math:`V_{\mathrm{out}}`
        """
        mesh_phases = MeshPhasesTorch(
            theta=self.theta, phi=self.phi, gamma=self.gamma,
            mask=self.mesh_model.mask, hadamard=self.mesh_model.hadamard,
            units=self.units, basis=self.mesh_model.basis
        )
        mesh_layers = self.mesh_layers(mesh_phases)
        if isinstance(inputs, np.ndarray):
            inputs = to_complex_t(inputs, self.theta.device)
        outputs = cc_mul(inputs, mesh_phases.input_phase_shift_layer)
        for mesh_layer in mesh_layers:
            outputs = mesh_layer(outputs)
        return outputs

    def inverse_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} U^\dagger,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse transformed :code:`outputs`, :math:`V_{\mathrm{in}}`
        """
        mesh_phases = MeshPhasesTorch(
            theta=self.theta, phi=self.phi, gamma=self.gamma,
            mask=self.mesh_model.mask, hadamard=self.mesh_model.hadamard,
            units=self.units, basis=self.mesh_model.basis
        )
        mesh_layers = self.mesh_layers(mesh_phases)
        inputs = to_complex_t(outputs, self.theta.device) if isinstance(outputs, np.ndarray) else outputs
        for layer in reversed(range(self.num_layers)):
            inputs = mesh_layers[layer].inverse_transform(inputs)
        inputs = cc_mul(inputs, conj_t(mesh_phases.input_phase_shift_layer))
        return inputs

    def adjoint_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform(inputs)

    @property
    def phases(self) -> MeshPhases:
        """

        Returns:
            The :code:`MeshPhases` object for this layer
        """
        return MeshPhases(
            theta=self.theta.detach().numpy() * self.mesh_model.mask,
            phi=self.phi.detach().numpy() * self.mesh_model.mask,
            mask=self.mesh_model.mask,
            gamma=self.gamma.detach().numpy()
        )

    def mesh_layers(self, phases: MeshPhasesTorch) -> List[MeshVerticalLayer]:
        """

        Args:
            phases:  The :code:`MeshPhasesTensorflow` object containing :math:`\\boldsymbol{\\theta}, \\boldsymbol{\\phi}, \\boldsymbol{\\gamma}`

        Returns:
            List of mesh layers to be used by any instance of :code:`MeshLayer`
        """
        internal_psl = phasor(phases.internal_phase_shifts)
        external_psl = phasor(phases.external_phase_shifts)

        # smooth trick to efficiently perform the layerwise coupling computation

        if self.mesh_model.hadamard:
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
            diag_layers = torch.cat((diag_layers[:, :-1], to_complex_t(np.ones((1, diag_layers.size()[-1])),
                                                                       diag_layers.device)), dim=1)

        diag_layers, off_diag_layers = diag_layers.transpose(1, 2), off_diag_layers.transpose(1, 2)

        mesh_layers = [MeshVerticalLayer(
            self.pairwise_perm_idx, diag_layers[:, 0], off_diag_layers[:, 0], self.perm_layers[1], self.perm_layers[0])]
        for layer in range(1, self.num_layers):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[:, layer],
                                                 off_diag_layers[:, layer], self.perm_layers[layer + 1]))

        return mesh_layers


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


def to_complex_t(nparray: np.ndarray, device: torch.device):
    return torch.stack((torch.as_tensor(nparray.real, device=device, dtype=torch.float32),
                        torch.as_tensor(nparray.imag, device=device, dtype=torch.float32)), dim=0)


def phasor(phase: torch.Tensor):
    return torch.stack((phase.cos(), phase.sin()), dim=0)