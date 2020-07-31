from typing import Optional, Callable, Tuple

import numpy as np
import tensorflow as tf
try:
    import torch
except ImportError:
    # if the user did not install pytorch, just do tensorflow stuff
    pass

from scipy.stats import multivariate_normal

from .config import NP_FLOAT


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


def to_absolute_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.mod(theta, 2 * np.pi)
    theta[theta > np.pi] = 2 * np.pi - theta[theta > np.pi]
    return theta


def get_haar_diagonal_sequence(diagonal_length, parity_odd: bool = False):
    odd_nums = list(diagonal_length + 1 - np.flip(np.arange(1, diagonal_length + 1, 2), axis=0))
    even_nums = list(diagonal_length + 1 - np.arange(2, 2 * (diagonal_length - len(odd_nums)) + 1, 2))
    nums = np.asarray(odd_nums + even_nums)
    if parity_odd:
        nums = nums[::-1]
    return nums


def get_alpha_checkerboard(units: int, num_layers: int, include_off_mesh: bool = False, flipud=False):
    if units < num_layers:
        raise ValueError("Require units >= num_layers!")
    alpha_checkerboard = np.zeros((units - 1, num_layers))
    diagonal_length_to_sequence = [get_haar_diagonal_sequence(i, bool(num_layers % 2)) for i in
                                   range(1, num_layers + 1)]
    for i in range(units - 1):
        for j in range(num_layers):
            if (i + j) % 2 == 0:
                if i < num_layers and j > i:
                    diagonal_length = num_layers - np.abs(i - j)
                elif i > units - num_layers and j < i - units + num_layers:
                    diagonal_length = num_layers - np.abs(i - j - units + num_layers) - 1 * (units == num_layers)
                else:
                    diagonal_length = num_layers - 1 * (units == num_layers)
                alpha_checkerboard[i, j] = 1 if diagonal_length == 1 else \
                    diagonal_length_to_sequence[int(diagonal_length) - 1][min(i, j)]
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


def get_efficient_coarse_grain_block_sizes(units: int, tunable_layers_per_block: int = 2, use_cg_sequence: bool = True):
    num_blocks = int(np.rint(np.log2(units)))
    sampling_frequencies = [2 ** (block_num + 1) for block_num in range(num_blocks - 1)]
    if use_cg_sequence:
        sampling_frequencies = 2 ** get_haar_diagonal_sequence(num_blocks - 1)
    tunable_block_sizes = [tunable_layers_per_block for _ in range(num_blocks - 1)]
    return np.asarray(tunable_block_sizes, dtype=np.int32), np.asarray(sampling_frequencies, dtype=np.int32)


def get_default_coarse_grain_block_sizes(units: int, use_cg_sequence: bool = True):
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
                    sampling_frequencies: np.ndarray, butterfly: bool = False):
    grid_perms = [grid_permutation(units, tunable_block_size) for tunable_block_size in tunable_block_sizes]
    perms_to_concatenate = [grid_perms[0][0]]
    for idx, frequency in enumerate(sampling_frequencies):
        perm_prev = grid_perms[idx][-1]
        perm_next = grid_perms[idx + 1][0]
        perm = butterfly_layer_permutation(units, frequency) if butterfly else rectangular_permutation(units, frequency)
        glued_perm = glue_permutations(perm_prev, perm)
        glued_perm = glue_permutations(glued_perm, perm_next)
        perms_to_concatenate += [grid_perms[idx][1:-1], glued_perm]
    perms_to_concatenate.append(grid_perms[-1][1:])
    return np.vstack(perms_to_concatenate)


def butterfly_layer_permutation(units: int, frequency: int):
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


def grid_viz_permutation(units: int, num_layers: int, flip: bool = False):
    ordered_idx = np.arange(units)
    split_num_layers = (num_layers - num_layers // 2, num_layers // 2)
    right_shift = np.roll(ordered_idx, 1, axis=0)
    permuted_indices = np.zeros((num_layers, units))
    if flip:
        permuted_indices[::2] = np.ones((split_num_layers[0], 1)) @ ordered_idx[np.newaxis, :]
        permuted_indices[1::2] = np.ones((split_num_layers[1], 1)) @ right_shift[np.newaxis, :]
    else:
        permuted_indices[::2] = np.ones((split_num_layers[0], 1)) @ right_shift[np.newaxis, :]
        permuted_indices[1::2] = np.ones((split_num_layers[1], 1)) @ ordered_idx[np.newaxis, :]
    return np.vstack((ordered_idx.astype(np.int32),
                      permuted_indices[:-1].astype(np.int32),
                      ordered_idx.astype(np.int32)))


def ordered_viz_permutation(units: int, num_layers: int):
    ordered_idx = np.arange(units)
    permuted_indices = np.ones((num_layers + 1, 1)) @ ordered_idx[np.newaxis, :]
    return permuted_indices.astype(np.int32)


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


def random_gaussian_batch(batch_size: int, units: int, covariance_matrix: Optional[np.ndarray] = None,
                          seed: Optional[int] = None) -> np.ndarray:
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


def butterfly_permutation(num_layers: int):
    ordered_idx = np.arange(2 ** num_layers)
    permuted_idx = np.vstack(
        [butterfly_layer_permutation(2 ** num_layers, 2 ** layer) for layer in range(num_layers)]
    ).astype(np.int32)
    return np.vstack((ordered_idx.astype(np.int32),
                      permuted_idx[1:].astype(np.int32),
                      ordered_idx.astype(np.int32)))


def neurophox_matplotlib_setup(plt):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'family': 'serif', 'serif': ['Charter']})
    # plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams.update({'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']})


# Phase functions


def fix_phase_tf(fixed, mask):
    return lambda tensor: mask * tensor + (1 - mask) * fixed


def fix_phase_torch(fixed: np.ndarray, mask: np.ndarray,
                    device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.cfloat):
    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    fixed = torch.as_tensor(fixed, dtype=dtype, device=device)
    return lambda tensor: tensor * mask + (1 - mask) * fixed


def tri_phase_tf(phase_range: float):
    def pcf(phase):
        phase = tf.math.mod(phase, 2 * phase_range)
        phase = tf.where(tf.greater(phase, phase_range),
                         2 * phase_range * tf.ones_like(phase) - phase, phase)
        return phase

    return pcf


def tri_phase_torch(phase_range: float):
    def pcf(phase):
        phase = torch.fmod(phase, 2 * phase_range)
        phase[phase > phase_range] = 2 * phase_range - phase[phase > phase_range]
        return phase

    return pcf
