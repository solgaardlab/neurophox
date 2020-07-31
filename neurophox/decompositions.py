import numpy as np
from .components import BlochMZI
from .config import NP_FLOAT
from typing import Callable, Tuple


def clements_decomposition(u: np.ndarray, pbar_handle: Callable = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        u: unitary matrix :math:`U` to be decomposed into pairwise operators.
        pbar_handle: Useful for larger matrices

    Returns:
        Clements decomposition of unitary matrix :math:`U` in terms of
        :math:`\\boldsymbol{\\theta}`, :math:`\\boldsymbol{\\phi}`, :math:`\\boldsymbol{\\gamma}`.

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
                theta_checkerboard[pairwise_index, j] = mzi.theta
                phi_checkerboard[pairwise_index, j] = -mzi.phi
                print(phi, phi_checkerboard[pairwise_index, j])
                # phi_checkerboard[pairwise_index + 1, j] = np.pi
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
                phi_checkerboard[pairwise_index, -j - 1] = -phi

    diag_phases = np.angle(np.diag(u_hat))
    theta = checkerboard_to_param(np.fliplr(theta_checkerboard), n)
    phi_checkerboard = np.fliplr(phi_checkerboard)
    phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    gamma = phi_checkerboard[:, 0]
    external_phases = phi_checkerboard[:, 1:]
    phi, gamma = common_mode_flow(external_phases, gamma=gamma)
    return theta, phi, gamma, phi_checkerboard[:, 1:]

    # diag_phases = np.angle(np.diag(u_hat))
    # theta = MeshParam(checkerboard_to_param(np.fliplr(theta_checkerboard), units), units)
    # phi = MeshParam(checkerboard_to_param(np.fliplr(np.mod(phi_checkerboard, 2 * np.pi)), units), units)
    #
    # phi, gamma = common_mode_flow(phi.checkerboard_arrangement,
    #                               gamma=np.zeros_like(diag_phases),
    #                               diag_phase_shifts=diag_phases)
    #
    # return theta.param, phi, None, diag_phases


def checkerboard_to_param(checkerboard: np.ndarray, units: int):
    param = np.zeros((units, units // 2))
    if units % 2:
        param[::2, :] = checkerboard.T[::2, :-1:2]
    else:
        param[::2, :] = checkerboard.T[::2, ::2]
    if units % 2:
        param[1::2, :] = checkerboard.T[1::2, 1::2]
    else:
        param[1::2, :] = checkerboard.T[1::2, 1::2]
    return param


def common_mode_flow(external_phases: np.ndarray, gamma: np.ndarray,
                     differential_mode: bool = False):
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
        # assign differential phase to the single mode layer and keep common mode layer
        if differential_mode:
            upper_phase = np.mod(upper_phase, 2 * np.pi)
            lower_phase = np.mod(lower_phase, 2 * np.pi)
            phase_diff = upper_phase - lower_phase
            phase_common = (upper_phase + lower_phase) / 2
            phase_common[phase_diff > np.pi] -= np.pi
            phase_common[phase_diff < -np.pi] += np.pi
            phase_diff[phase_diff > np.pi] -= 2 * np.pi
            phase_diff[phase_diff < -np.pi] += 2 * np.pi
            phase_shifts[current_layer][start_idx:end_idx][::2] = phase_common
            phase_shifts[current_layer][start_idx:end_idx][1::2] = phase_common
            new_phase_shifts[-i - 1][start_idx:end_idx][::2] = phase_diff / 2
            new_phase_shifts[-i - 1][start_idx:end_idx][1::2] = -phase_diff / 2
        else:
            new_phase_shifts[-i - 1][start_idx:end_idx][::2] = upper_phase - lower_phase
            phase_shifts[current_layer][start_idx:end_idx][::2] = lower_phase
        # update the previous layer with the common mode calculated for the current layer
        phase_shifts[current_layer - 1] += phase_shifts[current_layer]
        phase_shifts[current_layer] = 0

    new_phase_shifts = np.mod(new_phase_shifts.T, 2 * np.pi)
    new_gamma = np.mod(phase_shifts[0], 2 * np.pi)
    new_phi = checkerboard_to_param(new_phase_shifts, units)
    return new_phi, new_gamma
