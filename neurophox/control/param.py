import numpy as np
import torch
import tensorflow as tf

from ..helpers import roll_tensor, roll_torch, to_stripe_array, to_stripe_tensor,\
    to_stripe_torch, to_rm_checkerboard


class MeshParam:
    def __init__(self, param: np.ndarray, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self) -> np.ndarray:
        return to_stripe_array(self.param, self.units)

    @property
    def checkerboard_arrangement(self) -> np.ndarray:
        return to_rm_checkerboard(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> np.ndarray:
        phases = self.single_mode_arrangement
        return phases + np.roll(phases, 1, axis=0)

    @property
    def differential_mode_arrangement(self) -> np.ndarray:
        phases = self.single_mode_arrangement
        return phases / 2 - np.roll(phases / 2, 1, axis=0)

    def param_list(self, mask: np.ndarray) -> np.ndarray:
        return self.param[mask.astype(np.bool)]

    def __add__(self, other):
        return MeshParam(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParam(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParam(self.param * other.param, self.units)

    @property
    def batch_sum(self):
        if len(self.param.shape) > 2:
            return MeshParam(np.sum(self.param, axis=-1), self.units)
        else:
            return self


class MeshParamTensorflow:
    def __init__(self, param: tf.Tensor, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self):
        return to_stripe_tensor(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases + roll_tensor(phases)

    @property
    def differential_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases / 2 - roll_tensor(phases / 2)

    def __add__(self, other):
        return MeshParamTensorflow(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTensorflow(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTensorflow(self.param * other.param, self.units)


class MeshParamTorch:
    def __init__(self, param: torch.Tensor, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self):
        return to_stripe_torch(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases + roll_torch(phases)

    @property
    def differential_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases / 2 - roll_torch(phases / 2)

    def __add__(self, other):
        return MeshParamTorch(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTorch(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTorch(self.param * other.param, self.units)
