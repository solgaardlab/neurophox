import numpy as np
from .components import BlochMZI
from .config import NP_FLOAT
from typing import Callable
from .numpy import MeshNumpyLayer, RMNumpy
from .meshmodel import MeshModel
from .helpers import inverse_permutation


def clements_decomposition(u: np.ndarray, pbar_handle: Callable = None) -> RMNumpy:
    """Clements decomposition of unitary matrix :math:`U` to output a NumPy rectangular mesh layer

    Args:
        u: unitary matrix :math:`U` to be decomposed into pairwise operators.
        pbar_handle: Useful for larger matrices

    Returns:
        The :code:`RMNumpy` layer that outputs the unitary :math:`U`

    """
    u_hat = u.T.copy()
    n = u.shape[0]
    # odd and even layer dimensions
    theta_checkerboard = np.zeros_like(u, dtype=NP_FLOAT)
    phi_checkerboard = np.zeros_like(u, dtype=NP_FLOAT)
    phi_checkerboard = np.hstack((np.zeros((n, 1)), phi_checkerboard))
    iterator = pbar_handle(range(n - 1)) if pbar_handle else range(n - 1)

    for i in iterator:
        if i % 2:
            for j in range(i + 1):
                pairwise_index = n + j - i - 2
                target_row, target_col = n + j - i - 1, j
                theta = np.arctan(np.abs(u_hat[target_row - 1, target_col] / u_hat[target_row, target_col])) * 2
                phi = np.angle(u_hat[target_row, target_col] / u_hat[target_row - 1, target_col])
                mzi = BlochMZI(theta, phi, hadamard=False, dtype=np.complex128)
                left_multiplier = mzi.givens_rotation(units=n, m=pairwise_index)
                u_hat = left_multiplier @ u_hat
                theta_checkerboard[pairwise_index, j] = theta
                phi_checkerboard[pairwise_index, j] = -phi + np.pi
                phi_checkerboard[pairwise_index + 1, j] = np.pi
        else:
            for j in range(i + 1):
                pairwise_index = i - j
                target_row, target_col = n - j - 1, i - j
                theta = np.arctan(np.abs(u_hat[target_row, target_col + 1] / u_hat[target_row, target_col])) * 2
                phi = np.angle(-u_hat[target_row, target_col] / u_hat[target_row, target_col + 1])
                mzi = BlochMZI(theta, phi, hadamard=False, dtype=np.complex128)
                right_multiplier = mzi.givens_rotation(units=n, m=pairwise_index)
                u_hat = u_hat @ right_multiplier.conj().T
                theta_checkerboard[pairwise_index, -j - 1] = theta
                phi_checkerboard[pairwise_index, -j - 1] = phi + np.pi

    diag_phases = np.angle(np.diag(u_hat))
    theta = checkerboard_to_param(np.fliplr(theta_checkerboard), n)
    phi_checkerboard = np.fliplr(phi_checkerboard)
    if n % 2:
        phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    else:
        phi_checkerboard[:, 1:] += np.fliplr(np.diag(diag_phases))
        phi_checkerboard[-1, 2::2] += np.pi / 2  # neurophox layers assume pi / 2 phase shift in even layer "bounces"
        phi_checkerboard[0, 2::2] += np.pi / 2

    gamma = phi_checkerboard[:, 0]
    external_phases = phi_checkerboard[:, 1:]
    phi, gamma = grid_common_mode_flow(external_phases, gamma=gamma)
    phi = checkerboard_to_param(phi, n)

    # for some reason, we need to adjust gamma at the end in this strange way (found via trial and error...):
    gamma_adj = np.zeros_like(gamma)
    gamma_adj[1::4] = 1
    gamma_adj[2::4] = 1
    gamma += np.pi * (1 - gamma_adj) if (n // 2) % 2 else np.pi * gamma_adj
    gamma = np.mod(gamma, 2 * np.pi)

    return RMNumpy(units=n, theta_init=theta, phi_init=phi, gamma_init=gamma)


def reck_decomposition(u: np.ndarray, pbar_handle: Callable = None) -> RMNumpy:
    """Reck decomposition of unitary matrix :math:`U` to output a NumPy triangular mesh layer

    Args:
        u: unitary matrix :math:`U` to be decomposed into pairwise operators.
        pbar_handle: Useful for larger matrices

    Returns:
        The :code:`TMNumpy` layer that outputs the unitary :math:`U`

    """
    u_hat = u.T.copy()
    n = u.shape[0]
    # odd and even layer dimensions
    theta_checkerboard = np.zeros_like(u, dtype=NP_FLOAT)
    phi_checkerboard = np.zeros_like(u, dtype=NP_FLOAT)
    phi_checkerboard = np.hstack((np.zeros((n, 1)), phi_checkerboard))
    iterator = pbar_handle(range(n - 1)) if pbar_handle else range(n - 1)

    for i in iterator:
        for j in range(i + 1):
            pairwise_index = i - j
            target_row, target_col = n - j - 1, i - j
            theta = np.arctan(np.abs(u_hat[target_row, target_col + 1] / u_hat[target_row, target_col])) * 2
            phi = np.angle(-u_hat[target_row, target_col] / u_hat[target_row, target_col + 1])
            mzi = BlochMZI(theta, phi, hadamard=False, dtype=np.complex128)
            right_multiplier = mzi.givens_rotation(units=n, m=pairwise_index)
            u_hat = u_hat @ right_multiplier.conj().T
            theta_checkerboard[pairwise_index, -j - 1] = theta
            phi_checkerboard[pairwise_index, -j - 1] = phi + np.pi

    diag_phases = np.angle(np.diag(u_hat))
    theta = checkerboard_to_param(np.fliplr(theta_checkerboard), n)
    phi_checkerboard = np.fliplr(phi_checkerboard)
    if n % 2:
        phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    else:
        phi_checkerboard[:, 1:] += np.fliplr(np.diag(diag_phases))
        phi_checkerboard[-1, 2::2] += np.pi / 2  # neurophox layers assume pi / 2 phase shift in even layer "bounces"
        phi_checkerboard[0, 2::2] += np.pi / 2

    gamma = phi_checkerboard[:, 0]
    external_phases = phi_checkerboard[:, 1:]
    phi, gamma = grid_common_mode_flow(external_phases, gamma=gamma)
    phi = checkerboard_to_param(phi, n)

    # for some reason, we need to adjust gamma at the end in this strange way (found via trial and error...):
    gamma_adj = np.zeros_like(gamma)
    gamma_adj[1::4] = 1
    gamma_adj[2::4] = 1
    gamma += np.pi * (1 - gamma_adj) if (n // 2) % 2 else np.pi * gamma_adj
    gamma = np.mod(gamma, 2 * np.pi)

    return RMNumpy(units=n, theta_init=theta, phi_init=phi, gamma_init=gamma)


def checkerboard_to_param(checkerboard: np.ndarray, units: int):
    param = np.zeros((units, units // 2))
    if units % 2:
        param[::2, :] = checkerboard.T[::2, :-1:2]
    else:
        param[::2, :] = checkerboard.T[::2, ::2]
    param[1::2, :] = checkerboard.T[1::2, 1::2]
    return param


def grid_common_mode_flow(external_phases: np.ndarray, gamma: np.ndarray, basis: str = "sm"):
    """In a grid mesh (e.g., triangular, rectangular meshes), arrange phases according to single-mode (:code:`sm`),
       differential mode (:code:`diff`), or max-:math:`\\pi` (:code:`maxpi`, all external phase shifts are at most
       :math:`\\pi`). This is done using a procedure called "common mode flow" where common modes are shifted
       throughout the mesh until phases are correctly set.

    Args:
        external_phases: external phases in the grid mesh
        gamma: input phase shifts
        basis: single-mode (:code:`sm`), differential mode (:code:`diff`), or max-:math:`\\pi` (:code:`maxpi`)

    Returns:
        new external phases shifts and new gamma resulting

    """
    units, num_layers = external_phases.shape
    phase_shifts = np.hstack((gamma[:, np.newaxis], external_phases)).T
    new_phase_shifts = np.zeros_like(external_phases.T)

    for i in range(num_layers):
        current_layer = num_layers - i
        start_idx = (current_layer - 1) % 2
        end_idx = units - (current_layer + units - 1) % 2
        # calculate phase information
        upper_phase = phase_shifts[current_layer][start_idx:end_idx][::2]
        lower_phase = phase_shifts[current_layer][start_idx:end_idx][1::2]
        upper_phase = np.mod(upper_phase, 2 * np.pi)
        lower_phase = np.mod(lower_phase, 2 * np.pi)
        if basis == "sm":
            new_phase_shifts[-i - 1][start_idx:end_idx][::2] = upper_phase - lower_phase
        # assign differential phase to the single mode layer and keep common mode layer
        else:
            phase_diff = upper_phase - lower_phase
            phase_diff[phase_diff > np.pi] -= 2 * np.pi
            phase_diff[phase_diff < -np.pi] += 2 * np.pi
            if basis == "diff":
                new_phase_shifts[-i - 1][start_idx:end_idx][::2] = phase_diff / 2
                new_phase_shifts[-i - 1][start_idx:end_idx][1::2] = -phase_diff / 2
            elif basis == "pimax":
                new_phase_shifts[-i - 1][start_idx:end_idx][::2] = phase_diff * (phase_diff >= 0)
                new_phase_shifts[-i - 1][start_idx:end_idx][1::2] = -phase_diff * (phase_diff < 0)
        # update the previous layer with the common mode calculated for the current layer\
        phase_shifts[current_layer] -= new_phase_shifts[-i - 1]
        phase_shifts[current_layer - 1] += np.mod(phase_shifts[current_layer], 2 * np.pi)
        phase_shifts[current_layer] = 0
    new_gamma = np.mod(phase_shifts[0], 2 * np.pi)
    return np.mod(new_phase_shifts.T, 2 * np.pi), new_gamma


def parallel_nullification(np_layer):
    """Perform parallel nullification

    Args:
        np_layer:

    Returns:

    """
    units, num_layers = np_layer.units, np_layer.num_layers
    nullification_set = np_layer.nullification_set

    # set the mesh to bar state
    theta = []
    phi = []

    perm_idx = np_layer.mesh.model.perm_idx
    num_tunable = np_layer.mesh.model.num_tunable

    # run the real-time O(L) algorithm
    for idx in range(num_layers):
        layer = num_layers - idx - 1
        if idx > 0:
            current_mesh = MeshNumpyLayer(
                MeshModel(perm_idx=perm_idx[layer + 1:],
                          num_tunable=num_tunable[layer + 1:],
                          basis='sm',
                          theta_init=np.asarray(theta),
                          phi_init=np.asarray(phi),
                          gamma_init=np.zeros_like(np_layer.phases.gamma))
            )
            layer_trm = current_mesh.inverse_transform(nullification_set[layer]).squeeze()
        else:
            layer_trm = nullification_set[layer].take(inverse_permutation(perm_idx[-1]))
        upper_inputs = layer_trm[:-1][::2]
        lower_inputs = layer_trm[1:][::2]
        theta.insert(0, np.arctan(np.abs(upper_inputs / lower_inputs)) * 2)
        phi.insert(0, np.angle(upper_inputs / lower_inputs))
    return MeshNumpyLayer(
        MeshModel(perm_idx=perm_idx,
                  num_tunable=num_tunable,
                  basis='sm',
                  theta_init=np.asarray(theta),
                  phi_init=np.asarray(phi),
                  gamma_init=np_layer.phases.gamma.copy())
    )
