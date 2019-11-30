import numpy as np
from ..config import BLOCH, SINGLEMODE

try:
    import torch
    from torch.nn import Parameter
except ImportError:
    pass


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
        return to_stripe_torch(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> torch.Tensor:
        phases = self.single_mode_arrangement
        return phases + phases.roll(1, 0)

    @property
    def differential_mode_arrangement(self) -> torch.Tensor:
        phases = self.single_mode_arrangement
        return phases / 2 - phases.roll(1, 0) / 2

    def __add__(self, other):
        return MeshParamTorch(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTorch(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTorch(self.param * other.param, self.units)


class MeshPhasesTorch:
    def __init__(self, theta: Parameter, phi: Parameter, mask: np.ndarray, gamma: Parameter, units: int,
                 basis: str = SINGLEMODE, hadamard: bool = False):
        self.mask = mask if mask is not None else np.ones_like(theta)
        self.theta = MeshParamTorch(theta * torch.as_tensor(mask) + torch.as_tensor(1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.phi = MeshParamTorch(phi * torch.as_tensor(mask) + torch.as_tensor(1 - mask) * (1 - hadamard) * np.pi, units=units)
        self.gamma = gamma
        self.basis = basis
        self.input_phase_shift_layer = torch.stack((torch.cos(gamma), torch.sin(gamma)), dim=0)
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
        return torch.stack((internal_ps.cos(), internal_ps.sin()), dim=0)

    @property
    def external_phase_shift_layers(self):
        external_ps = self.external_phase_shifts
        return torch.stack((external_ps.cos(), external_ps.sin()), dim=0)


def to_stripe_torch(tensor: torch.Tensor, units: int):
    """
    Convert a tensor of phase shifts of size (`num_layers`, `units`) into striped array
    for use in general feedforward mesh architectures.

    Args:
        tensor: tensor
        units: dimension the tensor acts on (depends on parity)

    Returns:
        A general mesh stripe tensor arrangement that is of size (`units`, `num_layers`)
    """
    num_layers = tensor.shape[0]
    tensor_t = tensor.t()
    stripe_tensor = torch.zeros(units, num_layers)
    if units % 2:
        stripe_tensor[:-1][::2] = tensor_t
    else:
        stripe_tensor[::2] = tensor_t
    return stripe_tensor


def to_rm_checkerboard_torch(tensor_0: torch.Tensor, tensor_1: torch.Tensor):
    """A general method to convert even/odd (0/1 parity) arrays of values into checkerboard tensors.
    This is a critical component of rectangular mesh simulations. Note that this method is much simpler than
    the tensorflow version.

    Args:
        tensor_0: Nonzero values in the even columns
        tensor_1: Nonzero values in the odd columns

    Returns:
        An RD checkerboard tensor arrangement that is of size (`units`, `num_layers`) for an RD mesh
    """
    if len(tensor_0.size()) == 2:
        num_layers = tensor_0.size()[0] + tensor_1.size()[0]
        units = tensor_0.size()[1] + tensor_1.size()[1] + 1
        checkerboard = torch.zeros(units - 1, num_layers)
        checkerboard[::2, ::2] = tensor_0.t()
        checkerboard[1::2, 1::2] = tensor_1.t()
        checkerboard = torch.cat([checkerboard, torch.zeros(1, num_layers)], dim=0)
    else:
        batch_size = tensor_0.size()[2]
        num_layers = tensor_0.size()[0] + tensor_1.size()[0]
        units = tensor_0.size()[1] + tensor_1.size()[1] + 1
        checkerboard = torch.zeros((units - 1, num_layers, batch_size))
        checkerboard[::2, ::2] = tensor_0.transpose(0, 1)
        checkerboard[1::2, 1::2] = tensor_1.transpose(0, 1)
        checkerboard = torch.cat([checkerboard, torch.zeros(1, num_layers, batch_size)], dim=0)
    return checkerboard
