from typing import Optional, Callable, Tuple

import numpy as np
import tensorflow as tf
import torch
from .components import BlochMZI
from scipy.stats import multivariate_normal

from .config import NP_FLOAT, TF_FLOAT


def clements_decomposition(unitary: np.ndarray, hadamard: bool=False,
                           pbar_handle: Callable = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        unitary: unitary matrix :math:`U` to be decomposed into pairwise operators.
        hadamard: Whether to use Hadamard convention
        pbar_handle: Useful for larger matrices

    Returns:
        Clements decomposition of unitary matrix :math:`U` in terms of
        :math:`\\boldsymbol{\\theta}`, :math:`\\boldsymbol{\\phi}`, :math:`\\boldsymbol{\\gamma}`.

    """
    hadamard = hadamard
    unitary_hat = np.copy(unitary)
    units = unitary.shape[0]
    # odd and even layer dimensions
    theta_checkerboard = np.zeros_like(unitary, dtype=NP_FLOAT)
    phi_checkerboard = np.zeros_like(unitary, dtype=NP_FLOAT)
    iterator = pbar_handle(range(units - 1)) if pbar_handle else range(units - 1)
    for i in iterator:
        if i % 2 == 0:
            for j in range(i + 1):
                pairwise_index = i - j
                target_row, target_col = units - j - 1, i - j
                m, n = units - 1 - target_row, units - 1 - target_col
                theta = np.arctan(np.abs(
                    unitary_hat[target_row, target_col] / unitary_hat[target_row, target_col + 1])) * 2 + np.pi
                phi = np.angle(
                    unitary_hat[target_row, target_col] / unitary_hat[target_row, target_col + 1])
                mzi = BlochMZI(theta, phi, hadamard=hadamard, dtype=np.complex128)
                right_multiplier = mzi.givens_rotation(units=units, m=pairwise_index, i_factor=True)
                unitary_hat = unitary_hat @ right_multiplier.conj().T
                theta_checkerboard[m, n - 1] = theta
                phi_checkerboard[m, n - 1] = phi
        else:
            for j in range(i + 1):
                pairwise_index = units + j - i - 2
                target_row, target_col = units + j - i - 1, j
                m, n = target_row, target_col
                theta = np.arctan(np.abs(unitary_hat[target_row, target_col] / unitary_hat[target_row - 1, target_col])) * 2 + np.pi
                phi = np.angle(-unitary_hat[target_row, target_col] / unitary_hat[target_row - 1, target_col])
                mzi = BlochMZI(theta, phi, hadamard=hadamard, dtype=np.complex128)
                left_multiplier = mzi.givens_rotation(units=units, m=pairwise_index, i_factor=True)
                unitary_hat = left_multiplier @ unitary_hat
                theta_checkerboard[m, n] = theta
                phi_checkerboard[m, n] = phi
    theta_checkerboard = to_absolute_theta(theta_checkerboard).T
    phi_checkerboard = np.mod(phi_checkerboard, 2 * np.pi)
    return theta_checkerboard, phi_checkerboard, np.diag(unitary_hat)  # not quite the same as the model in MeshPhases


def to_rm_checkerboard(nparray: np.ndarray, units: int):
    """A general method to convert mesh phase params into a checkerboard (useful for plotting).
    This is a critical component for visualizing rectangular mesh simulations.

    Args:
        nparray: array of to arrange in checkerboard patterned array
        units: Dimension of the vector

    Returns:
        An RM checkerboard tensor arrangement that is of size (`units`, `num_layers`) for an RD mesh
    """
    num_layers = nparray.shape[0]
    checkerboard = np.zeros((units, num_layers), dtype=nparray.dtype)
    checkerboard[::2, ::2] = nparray[::2].T
    checkerboard[1::2, 1::2] = nparray[1::2].T
    return np.vstack([checkerboard, np.zeros(shape=(1, num_layers))])


def to_stripe_array(nparray: np.ndarray, units: int):
    """
    Convert a numpy array of phase shifts of size (`num_layers`, `units`)
    or (`batch_size`, `num_layers`, `units`) into striped array for use in
    general feedforward mesh architectures.

    Args:
        nparray: phase shift values for all columns
        units: dimension the stripe array acts on (depends on parity)

    Returns:
        A general mesh stripe array arrangement that is of size (`units`, `num_layers`)
        or (`batch_size`, `units`, `num_layers`)
    """
    if len(nparray.shape) == 2:
        num_layers, _ = nparray.shape
        stripe_array = np.zeros((units - 1, num_layers), dtype=nparray.dtype)
        stripe_array[::2] = nparray.T
        stripe_array = np.vstack([stripe_array, np.zeros(shape=(1, num_layers))])
    else:
        num_layers, _, batch_size = nparray.shape
        stripe_array = np.zeros((units - 1, num_layers, batch_size), dtype=nparray.dtype)
        stripe_array[::2] = nparray.transpose((1, 0, 2))
        stripe_array = np.vstack([stripe_array, np.zeros(shape=(1, num_layers, batch_size))])
    return stripe_array


def to_stripe_tensor(tensor: tf.Tensor, units: int):
    """
    Convert a tensor of phase shifts of size (`num_layers`, `units`) into striped array
    for use in general feedforward mesh architectures.

    Args:
        tensor: tensor
        units: dimension the tensor acts on (depends on parity)

    Returns:
        A general mesh stripe tensor arrangement that is of size (`units`, `num_layers`)
    """
    tensor_t = tf.transpose(tensor)
    stripe_tensor = tf.reshape(tf.concat((tensor_t, tf.zeros_like(tensor_t)), 1),
                               shape=(tensor_t.shape[0] * 2, tensor_t.shape[1]))
    if units % 2:
        return tf.concat([stripe_tensor, tf.zeros(shape=(1, tensor_t.shape[1]))], axis=0)
    else:
        return stripe_tensor


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
    num_layers = tensor.size()[0]
    tensor_t = tensor.t()
    stripe_tensor = torch.zeros(units, num_layers)
    if units % 2:
        stripe_tensor[:-1][::2] = tensor_t
    else:
        stripe_tensor[::2] = tensor_t
    return tensor_t


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


def to_absolute_theta_tf(theta: tf.Tensor) -> tf.Tensor:
    theta = tf.math.mod(theta, 2 * np.pi)
    theta = tf.where(tf.greater(theta, np.pi), 2 * np.pi * tf.ones_like(theta) - theta, theta)
    return theta


def to_absolute_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.mod(theta, 2 * np.pi)
    theta[theta > np.pi] = 2 * np.pi - theta[theta > np.pi]
    return theta


def roll_tensor(tensor: tf.Tensor, up=False):
    # a complex number-friendly roll that works on gpu
    if up:
        return tf.concat([tensor[1:], tensor[tf.newaxis, 0]], axis=0)
    return tf.concat([tensor[tf.newaxis, -1], tensor[:-1]], axis=0)


def roll_torch(x: torch.Tensor, up=False):
    if up:
        return torch.cat([x[1:], x[:1]], dim=0)
    return torch.cat((x[-1:], x[:-1]), dim=0)


def get_mesh_boundary_correction(units: int, num_layers: int, use_np: bool=False):
    correction = np.zeros((units, num_layers))
    correction[0, 1::2] = 1
    correction[-1, (1 - units % 2)::2] = 1
    if use_np:
        return correction.astype(NP_FLOAT)
    else:
        return tf.constant(correction, dtype=TF_FLOAT)


def get_haar_diagonal_sequence(diagonal_length, parity_odd: bool=False):
    odd_nums = list(diagonal_length + 1 - np.flip(np.arange(1, diagonal_length + 1, 2), axis=0))
    even_nums = list(diagonal_length + 1 - np.arange(2, 2 * (diagonal_length - len(odd_nums)) + 1, 2))
    nums = np.asarray(odd_nums + even_nums)
    if parity_odd:
        nums = nums[::-1]
    return nums


def get_smn_rank(row, col, units, num_layers):
    if row < num_layers and col > row:
        diagonal_length = num_layers - np.abs(row - col)
    elif row > units - num_layers and col < row - units + num_layers:
        diagonal_length = num_layers - np.abs(row - col - units + num_layers) - 1 * (units == num_layers)
    else:
        diagonal_length = num_layers - 1 * (units == num_layers)
    return units - diagonal_length


def get_alpha_rank(row, col, units, num_layers, diagonal_length_to_sequence):
    if row < num_layers and col > row:
        diagonal_length = num_layers - np.abs(row - col)
    elif row > units - num_layers and col < row - units + num_layers:
        diagonal_length = num_layers - np.abs(row - col - units + num_layers) - 1 * (units == num_layers)
    else:
        diagonal_length = num_layers - 1 * (units == num_layers)
    if diagonal_length == 1:
        return 1
    return diagonal_length_to_sequence[int(diagonal_length) - 1][min(row, col)]


def get_alpha_checkerboard(units: int, num_layers: int, include_off_mesh: bool=False, flipud=False):
    alpha_checkerboard = np.zeros((units - 1, num_layers))
    diagonal_length_to_sequence = [get_haar_diagonal_sequence(i, bool(num_layers % 2)) for i in range(1, num_layers + 1)]
    for i in range(units - 1):
        for j in range(num_layers):
            if (i + j) % 2 == 0:
                if units >= num_layers:
                    alpha_checkerboard[i, j] = get_alpha_rank(i, j, units, num_layers, diagonal_length_to_sequence)
                else:
                    alpha_checkerboard[i, j] = units  # TODO: need to fix this! raise NotImplementedError or actual error?
            else:
                if include_off_mesh:
                    alpha_checkerboard[i, j] = 1
    # symmetrize the checkerboard
    if units != num_layers:
        alpha_checkerboard = (alpha_checkerboard + np.flipud(alpha_checkerboard)) / 2
    return alpha_checkerboard if not flipud else np.flipud(alpha_checkerboard)


def get_alpha_checkerboard_general(units: int, num_layers: int):
    alpha_checkerboards = [get_alpha_checkerboard(units, units, flipud=bool(n % 2 and units % 2))
                           for n in range(num_layers // units)]
    extra_layers = num_layers - num_layers // units * units
    if extra_layers < units:
        # partial checkerboard
        alpha_checkerboards.append(
            get_alpha_checkerboard(units, extra_layers, flipud=not num_layers // units % 2 and units % 2))
    return np.hstack(alpha_checkerboards)


def get_efficient_coarse_grain_block_sizes(units: int, tunable_layers_per_block: int=2, use_cg_sequence: bool=True):
    num_blocks = int(np.rint(np.log2(units)))
    sampling_frequencies = [2 ** (block_num + 1) for block_num in range(num_blocks - 1)]
    if use_cg_sequence:
        sampling_frequencies = 2 ** get_haar_diagonal_sequence(num_blocks - 1)
    tunable_block_sizes = [tunable_layers_per_block for _ in range(num_blocks - 1)]
    return np.asarray(tunable_block_sizes, dtype=np.int32), np.asarray(sampling_frequencies, dtype=np.int32)


def get_default_coarse_grain_block_sizes(units: int, use_cg_sequence: bool=True):
    num_blocks = int(np.rint(np.log2(units)))
    sampling_frequencies = [2 ** (block_num + 1) for block_num in range(num_blocks - 1)]
    if use_cg_sequence:
        sampling_frequencies = 2 ** get_haar_diagonal_sequence(num_blocks - 1)
    tunable_layer_rank = int(np.floor(units / num_blocks))
    tunable_layer_rank = tunable_layer_rank + 1 if tunable_layer_rank % 2 else tunable_layer_rank
    tunable_block_sizes = [tunable_layer_rank for _ in range(num_blocks - 1)]
    tunable_block_sizes.append(units - tunable_layer_rank * (num_blocks - 1))
    return np.asarray(tunable_block_sizes, dtype=np.int32), np.asarray(sampling_frequencies, dtype=np.int32)


def prm_permutation(units: int, tunable_block_sizes: np.ndarray,
                    sampling_frequencies: np.ndarray, butterfly: bool=False):
    grid_perms = [grid_permutation(units, tunable_block_size) for tunable_block_size in tunable_block_sizes]
    perms_to_concatenate = [grid_perms[0][0]]
    for idx, frequency in enumerate(sampling_frequencies):
        perm_prev = grid_perms[idx][-1]
        perm_next = grid_perms[idx + 1][0]
        perm = butterfly_permutation(units, frequency) if butterfly else rectangular_permutation(units, frequency)
        glued_perm = glue_permutations(perm_prev, perm)
        glued_perm = glue_permutations(glued_perm, perm_next)
        perms_to_concatenate += [grid_perms[idx][1:-1], glued_perm]
    perms_to_concatenate.append(grid_perms[-1][1:])
    return np.vstack(perms_to_concatenate)


def butterfly_permutation(units: int, frequency: int):
    if units % 2:
        raise NotImplementedError('Odd input dimension case not yet implemented.')
    frequency = frequency
    unpermuted_indices = np.arange(units)
    num_splits = units // frequency
    total_num_indices = num_splits * frequency
    unpermuted_indices_remainder = unpermuted_indices[total_num_indices:]
    permuted_indices = np.hstack(
        [np.hstack([i, i + frequency] for i in range(frequency)) + 2 * frequency * split_num
         for split_num in range(num_splits // 2)] + [unpermuted_indices_remainder]
    )
    return permuted_indices.astype(np.int32)


def rectangular_permutation(units: int, frequency: int):
    unpermuted_indices = np.arange(units)
    frequency_offset = np.empty((units,))
    frequency_offset[::2] = -frequency
    frequency_offset[1::2] = frequency
    permuted_indices = unpermuted_indices + frequency_offset
    for idx in range(units):
        if permuted_indices[idx] < 0:
            permuted_indices[idx] = -1 - permuted_indices[idx]
        if permuted_indices[idx] > units - 1:
            permuted_indices[idx] = 2 * units - 1 - permuted_indices[idx]
    return permuted_indices.astype(np.int32)


def grid_permutation(units: int, num_layers: int):
    ordered_idx = np.arange(units)
    split_num_layers = (num_layers - num_layers // 2, num_layers // 2)
    left_shift = np.roll(ordered_idx, -1, axis=0)
    right_shift = np.roll(ordered_idx, 1, axis=0)
    permuted_indices = np.zeros((num_layers, units))
    permuted_indices[::2] = np.ones((split_num_layers[0], 1)) @ left_shift[np.newaxis, :]
    permuted_indices[1::2] = np.ones((split_num_layers[1], 1)) @ right_shift[np.newaxis, :]
    if num_layers % 2:
        return np.vstack((ordered_idx.astype(np.int32),
                          permuted_indices[:-1].astype(np.int32),
                          ordered_idx.astype(np.int32)))
    return np.vstack((ordered_idx.astype(np.int32),
                      permuted_indices.astype(np.int32)))


def grid_viz_permutation(units: int, num_layers: int):
    ordered_idx = np.arange(units)
    split_num_layers = (num_layers - num_layers // 2, num_layers // 2)
    right_shift = np.roll(ordered_idx, 1, axis=0)
    permuted_indices = np.zeros((num_layers, units))
    permuted_indices[::2] = np.ones((split_num_layers[0], 1)) @ ordered_idx[np.newaxis, :]
    permuted_indices[1::2] = np.ones((split_num_layers[1], 1)) @ right_shift[np.newaxis, :]
    return np.vstack((ordered_idx.astype(np.int32),
                      permuted_indices[:-1].astype(np.int32),
                      ordered_idx.astype(np.int32)))


def plot_complex_matrix(plt, matrix: np.ndarray):
    plt.figure(figsize=(15, 5), dpi=200)
    plt.subplot(131)
    plt.title('Absolute')
    plt.imshow(np.abs(matrix), cmap='hot')
    plt.colorbar(shrink=0.7)
    plt.subplot(132)
    plt.title('Real')
    plt.imshow(np.real(matrix), cmap='hot')
    plt.colorbar(shrink=0.7)
    plt.subplot(133)
    plt.title('Imag')
    plt.imshow(np.imag(matrix), cmap='hot')
    plt.colorbar(shrink=0.7)


def random_gaussian_batch(batch_size: int, units: int, covariance_matrix: Optional[np.ndarray]=None,
                          seed: Optional[int]=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    input_matrix = multivariate_normal.rvs(
        mean=np.zeros(units),
        cov=1 if not covariance_matrix else covariance_matrix,
        size=batch_size
    )
    random_phase = np.random.rand(batch_size, units).astype(dtype=NP_FLOAT) * 2 * np.pi
    return input_matrix * np.exp(1j * random_phase)


def glue_permutations(perm_idx_1: np.ndarray, perm_idx_2: np.ndarray):
    perm_idx = np.zeros_like(perm_idx_1)
    perm_idx[perm_idx_2] = perm_idx_1
    return perm_idx.astype(np.int32)


def inverse_permutation(permuted_indices: np.ndarray):
    inv_permuted_indices = np.zeros_like(permuted_indices)
    for idx, perm_idx in enumerate(permuted_indices):
        inv_permuted_indices[perm_idx] = idx
    return inv_permuted_indices


def pairwise_off_diag_permutation(units: int):
    ordered_idx = np.arange(units)
    perm_idx = np.zeros_like(ordered_idx)
    if units % 2:
        perm_idx[:-1][::2] = ordered_idx[1::2]
        perm_idx[1::2] = ordered_idx[:-1][::2]
        perm_idx[-1] = ordered_idx[-1]
    else:
        perm_idx[::2] = ordered_idx[1::2]
        perm_idx[1::2] = ordered_idx[::2]
    return perm_idx.astype(np.int32)
