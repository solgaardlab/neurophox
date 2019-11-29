import numpy as np

try:
    import torch
except ImportError:
    pass

import tensorflow as tf

from ..helpers import roll_tensor, to_stripe_array, to_stripe_tensor,\
    to_stripe_torch, to_rm_checkerboard


class MeshParam:
    """A class that arranges parameters to simulate any feedforward mesh

    Args:
        param: parameter to arrange in mesh
        units: number of inputs/outputs of the mesh
    """
    def __init__(self, param: np.ndarray, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self) -> np.ndarray:
        """
        The single-mode arrangement based on the :math:`L(\\theta)` transfer matrix for :code:`PhaseShiftUpper`
        is one where elements of `param` are on the even rows and all odd rows are zero.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_m` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the single-mode arrangement has the stripe array form
        :math:`\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{0}, \\boldsymbol{\\theta}_2, \\boldsymbol{0}, \ldots \\boldsymbol{\\theta}_M, \\boldsymbol{0}]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}}` defines the spatial arrangement of mesh phases
        and :math:`\\boldsymbol{0}` represents an array of zeros of the same size as :math:`\\boldsymbol{\\theta}_m`.

        Returns:
            Single-mode arrangement array of phases

        """
        return to_stripe_array(self.param, self.units)

    @property
    def checkerboard_arrangement(self) -> np.ndarray:
        """

        Returns:
            Checkerboard arrangement of phases useful for grid mesh structures (e.g. rectangular and triangular meshes)
        """
        return to_rm_checkerboard(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> np.ndarray:
        """
        The common-mode arrangement based on the :math:`C(\\theta)` transfer matrix for :code:`PhaseShiftCommonMode`
        is one where elements of `param` are on the even rows and repeated on respective odd rows.

        In particular, given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the common-mode arrangement has the stripe array form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = [\\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_1, \\boldsymbol{\\theta}_2, \\boldsymbol{\\theta}_2, \ldots \\boldsymbol{\\theta}_M, \\boldsymbol{\\theta}_M]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}}` defines the spatial arrangement of mesh phases.


        Returns:
            Common-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases + np.roll(phases, 1, axis=0)

    @property
    def differential_mode_arrangement(self) -> np.ndarray:
        """
        The differential-mode arrangement is based on the :math:`D(\\theta)` transfer matrix
        for :code:`PhaseShiftDifferentialMode`.

        Given the :code:`param` array
        :math:`\\boldsymbol{\\theta} = [\cdots \\boldsymbol{\\theta}_m \cdots]^T`,
        where :math:`\\boldsymbol{\\theta}_n` represent row vectors and :math:`M = \\lfloor\\frac{N}{2}\\rfloor`, the differential-mode arrangement has the form
        :math:`\\widetilde{\\boldsymbol{\\theta}} = \\left[\cdots \\frac{\\boldsymbol{\\theta}_m}{2}, -\\frac{\\boldsymbol{\\theta}_m}{2} \cdots \\right]^T`
        where :math:`\widetilde{\\boldsymbol{\\theta}} \in \mathbb{R}^{N \\times L}` defines the spatial arrangement of mesh phases.

        Returns:
            Differential-mode arrangement array of phases

        """
        phases = self.single_mode_arrangement
        return phases / 2 - np.roll(phases / 2, 1, axis=0)

    def param_list(self, mask: np.ndarray) -> np.ndarray:
        """

        Args:
            mask: Mask to ignore params in output

        Returns:
            A flattened array of unmasked params in :code:`param`
        """
        return self.param[mask.astype(np.bool)]

    def __add__(self, other):
        return MeshParam(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParam(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParam(self.param * other.param, self.units)


class MeshParamTensorflow:
    """A class that cleanly arranges parameters into a specific arrangement that can be used to simulate any mesh

    Args:
        param: parameter to arrange in mesh
        units: number of inputs/outputs of the mesh
    """
    def __init__(self, param: tf.Tensor, units: int):
        self.param = param
        self.units = units

    @property
    def single_mode_arrangement(self):
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
        return to_stripe_tensor(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> tf.Tensor:
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
        return phases + roll_tensor(phases)

    @property
    def differential_mode_arrangement(self) -> tf.Tensor:
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
        return phases / 2 - roll_tensor(phases / 2)

    def __add__(self, other):
        return MeshParamTensorflow(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTensorflow(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTensorflow(self.param * other.param, self.units)


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
    def single_mode_arrangement(self):
        return to_stripe_torch(self.param, self.units)

    @property
    def common_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases + phases.roll(1, 0)

    @property
    def differential_mode_arrangement(self) -> tf.Tensor:
        phases = self.single_mode_arrangement
        return phases / 2 - phases.roll(1, 0) / 2

    def __add__(self, other):
        return MeshParamTorch(self.param + other.param, self.units)

    def __sub__(self, other):
        return MeshParamTorch(self.param - other.param, self.units)

    def __mul__(self, other):
        return MeshParamTorch(self.param * other.param, self.units)
