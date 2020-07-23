from typing import List, Optional, Callable

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

    def matrix(self) -> np.ndarray:
        torch_matrix = self.transform(np.eye(self.units, dtype=np.complex64)).cpu().detach().numpy()
        return torch_matrix

    def inverse_matrix(self) -> np.ndarray:
        torch_matrix = self.inverse_transform(np.eye(self.units, dtype=np.complex64)).cpu().detach().numpy()
        return torch_matrix

    def plot(self, plt):
        plot_complex_matrix(plt, self.matrix())

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
            inputs = torch.tensor(inputs, dtype=torch.cfloat, device=self.diag.device)
        outputs = inputs if self.left_perm is None else self.left_perm(inputs)
        diag_out = outputs * self.diag
        off_diag_out = outputs * self.off_diag
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
            outputs = torch.tensor(outputs, dtype=torch.cfloat, device=self.diag.device)
        inputs = outputs if self.right_perm is None else self.right_perm.inverse_transform(outputs)
        diag = self.diag.conj()
        off_diag = self.off_diag[..., self.pairwise_perm_idx].conj()
        inputs = inputs * diag + (inputs * off_diag)[..., self.pairwise_perm_idx]
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
        theta_fn: Pytorch-friendly phi function call to reparametrize phi (example use cases:
                can use a mask to keep some values of theta fixed or always bound theta between 0 and pi).
                By default, use identity function.
        phi_fn: Pytorch-friendly phi function call to reparametrize phi (example use cases:
                can use a mask to keep some values of phi fixed or always bound phi between 0 and 2 * pi).
                By default, use identity function.
        gamma_fn: Pytorch-friendly gamma function call to reparametrize gamma (example use cases:
                  can use a mask to keep some values of gamma fixed or always bound gamma between 0 and 2 * pi).
                  By default, use identity function.
        phase_loss_fn: Incorporate phase shift-dependent loss into the model.
                        The function is of the form phase_loss_fn(phases),
                        which returns the loss
    """

    def __init__(self, theta: Parameter, phi: Parameter, mask: np.ndarray, gamma: Parameter, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False, theta_fn: Optional[Callable] = None,
                 phi_fn: Optional[Callable] = None, gamma_fn: Optional[Callable] = None,
                 phase_loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.mask = mask if mask is not None else np.ones_like(theta)
        mask = torch.as_tensor(mask, dtype=theta.dtype, device=theta.device)
        self.theta_fn = (lambda x: x) if theta_fn is None else theta_fn
        self.phi_fn = (lambda x: x) if phi_fn is None else phi_fn
        self.gamma_fn = (lambda x: x) if gamma_fn is None else gamma_fn
        self.theta = MeshParamTorch(self.theta_fn(theta) * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.phi = MeshParamTorch(self.phi_fn(phi) * mask + (1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.gamma = self.gamma_fn(gamma)
        self.basis = basis
        self.phase_fn = lambda phase: torch.as_tensor(1 - phase_loss_fn(phase)) * phasor(phase) if phase_loss_fn is not None else phasor(phase)
        self.input_phase_shift_layer = self.phase_fn(gamma)
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

    @property
    def internal_phase_shift_layers(self):
        """Elementwise applying complex exponential to :code:`internal_phase_shifts`.

        Returns:
            Internal phase shift layers corresponding to :math:`\\boldsymbol{\\theta}`
        """
        return self.phase_fn(self.internal_phase_shifts)

    @property
    def external_phase_shift_layers(self):
        """Elementwise applying complex exponential to :code:`external_phase_shifts`.

        Returns:
            External phase shift layers corresponding to :math:`\\boldsymbol{\\phi}`
        """
        return self.phase_fn(self.external_phase_shifts)


class MeshTorchLayer(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The model of the mesh network (e.g., rectangular, triangular, butterfly)
    """

    def __init__(self, mesh_model: MeshModel):
        super(MeshTorchLayer, self).__init__(mesh_model.units)
        self.mesh_model = mesh_model
        ss, cs, sc, cc = self.mesh_model.mzi_error_tensors
        ss, cs, sc, cc = torch.as_tensor(ss, dtype=torch.float32), torch.as_tensor(cs, dtype=torch.float32), \
                         torch.as_tensor(sc, dtype=torch.float32), torch.as_tensor(cc, dtype=torch.float32)
        self.register_buffer("ss", ss)
        self.register_buffer("cs", cs)
        self.register_buffer("sc", sc)
        self.register_buffer("cc", cc)
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
            inputs = torch.tensor(inputs, dtype=torch.cfloat, device=self.theta.device)
        outputs = inputs * mesh_phases.input_phase_shift_layer
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
        inputs = torch.tensor(outputs, dtype=torch.cfloat, device=self.theta.device) if isinstance(outputs, np.ndarray) else outputs
        for layer in reversed(range(self.num_layers)):
            inputs = mesh_layers[layer].inverse_transform(inputs)
        inputs = inputs * mesh_phases.input_phase_shift_layer.conj()
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
        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers

        # smooth trick to efficiently perform the layerwise coupling computation

        if self.mesh_model.hadamard:
            s11 = self.cc * internal_psl + self.ss * internal_psl.roll(-1, 0)
            s22 = (self.ss * internal_psl + self.cc * internal_psl.roll(-1, 0)).roll(1, 0)
            s12 = (self.cs * internal_psl - self.sc * internal_psl.roll(-1, 0)).roll(1, 0)
            s21 = self.sc * internal_psl - self.cs * internal_psl.roll(-1, 0)
        else:
            s11 = self.cc * internal_psl - self.ss * internal_psl.roll(-1, 0)
            s22 = (-self.ss * internal_psl + self.cc * internal_psl.roll(-1, 0)).roll(1, 0)
            s12 = 1j * (self.cs * internal_psl + self.sc * internal_psl.roll(-1, 0)).roll(1, 0)
            s21 = 1j * (self.sc * internal_psl + self.cs * internal_psl.roll(-1, 0))

        diag_layers = external_psl * (s11 + s22) / 2
        off_diag_layers = external_psl.roll(1, 0) * (s21 + s12) / 2

        if self.units % 2:
            diag_layers = torch.cat((diag_layers[:-1], torch.ones_like(diag_layers[-1:])), dim=0)

        diag_layers, off_diag_layers = diag_layers.t(), off_diag_layers.t()

        mesh_layers = [MeshVerticalLayer(
            self.pairwise_perm_idx, diag_layers[0], off_diag_layers[0], self.perm_layers[1], self.perm_layers[0])]
        for layer in range(1, self.num_layers):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, diag_layers[layer],
                                                 off_diag_layers[layer], self.perm_layers[layer + 1]))

        return mesh_layers


def phasor(phase: torch.Tensor):
    return torch.cos(phase) + 1j * torch.sin(phase)

