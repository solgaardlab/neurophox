import tensorflow as tf
import torch
import numpy as np
import itertools
from scipy.stats import unitary_group
from neurophox.config import TF_COMPLEX
from neurophox.helpers import fix_phase_tf, tri_phase_tf, fix_phase_torch
from neurophox.tensorflow import MeshLayer
from neurophox.torch import RMTorch
from neurophox.numpy import RMNumpy
from neurophox.meshmodel import RectangularMeshModel
from neurophox.ml.linear import complex_mse
from torch.autograd.functional import jacobian

TEST_CASES = itertools.product([7, 8], (0, 0.1), (True, False))


class PhaseControlTest(tf.test.TestCase):
    def test_mask_pcf(self):
        def random_mask_init(x, use_torch=False):
            x_mask = np.ones_like(x).flatten()
            x_mask[:x_mask.size // 2] = 0
            np.random.shuffle(x_mask)
            x_mask = np.reshape(x_mask, x.shape)
            fix_phase = fix_phase_torch if use_torch else fix_phase_tf
            return (x, fix_phase(x, x_mask)), x_mask

        for units, bs_error, hadamard in TEST_CASES:
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                np.random.seed(0)
                target = unitary_group.rvs(units, random_state=0)
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                rm_numpy = RMNumpy(units)
                theta_init, theta_mask = random_mask_init(rm_numpy.theta)
                phi_init, phi_mask = random_mask_init(rm_numpy.phi)
                mesh_model = RectangularMeshModel(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error,
                    theta_init=theta_init,
                    phi_init=phi_init,
                    gamma_init=rm_numpy.gamma
                )
                rm = MeshLayer(mesh_model)
                with tf.GradientTape() as tape:
                    loss = complex_mse(rm(identity_matrix), target)
                grads = tape.gradient(loss, rm.trainable_variables)
                t_loss = torch.nn.MSELoss(reduction='mean')
                rm_torch = RMTorch(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error,
                    theta_init=random_mask_init(rm_numpy.theta, use_torch=True)[0],
                    phi_init=random_mask_init(rm_numpy.phi, use_torch=True)[0],
                    gamma_init=rm_numpy.gamma
                )

                torch_loss = t_loss(torch.view_as_real(rm_torch(torch.eye(units, dtype=torch.cfloat))),
                                    torch.view_as_real(torch.as_tensor(target, dtype=torch.cfloat)))
                var = torch_loss.sum()
                var.backward()
                print(torch.autograd.grad(var, [rm_torch.theta]))
                theta_grad, phi_grad = grads[0].numpy(), grads[1].numpy()
                theta_grad_zeros = theta_grad[np.where(theta_mask == 0)]
                phi_grad_zeros = phi_grad[np.where(phi_mask == 0)]
                self.assertAllClose(theta_grad_zeros, np.zeros_like(theta_grad_zeros))
                self.assertAllClose(phi_grad_zeros, np.zeros_like(phi_grad_zeros))

    def test_tri_pcf(self):
        def random_mask_init(x, phase_range):
            return x, tri_phase_tf(phase_range)

        for units, bs_error, hadamard in TEST_CASES:
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                np.random.seed(0)
                rm_numpy = RMNumpy(units)
                theta_init = random_mask_init(rm_numpy.theta, np.pi)
                phi_init = random_mask_init(rm_numpy.phi, 2 * np.pi)
                mesh_model = RectangularMeshModel(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error,
                    theta_init=theta_init,
                    phi_init=phi_init,
                    gamma_init=rm_numpy.gamma
                )
                rm = MeshLayer(mesh_model)
                phases, layers = rm.phases_and_layers
                theta_transformed = phases.theta.param
                phi_transformed = phases.phi.param
                self.assertAllLessEqual(theta_transformed, np.pi)
                self.assertAllLessEqual(phi_transformed, 2 * np.pi)


if __name__ == '__main__':
    tf.test.main()