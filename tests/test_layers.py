import tensorflow as tf
import numpy as np
import pytest

from neurophox.config import TF_COMPLEX, NP_COMPLEX
import itertools

from neurophox.numpy import RMNumpy, TMNumpy, PRMNumpy, BMNumpy, MeshNumpyLayer
from neurophox.tensorflow import RM, TM, PRM, BM, MeshLayer
from neurophox.torch import RMTorch, TMTorch, PRMTorch, BMTorch, MeshTorchLayer

TEST_DIMENSIONS = [7, 8]


class RMLayerTest(tf.test.TestCase):
    def test_tf(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                rm = RM(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                self.assertAllClose(rm.inverse_transform(rm.matrix), identity_matrix)
                self.assertAllClose(rm.matrix.conj().T, rm.inverse_matrix)

    # def test_t(self):
    #     for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
    #         with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
    #             identity_matrix = np.eye(units, dtype=NP_COMPLEX)
    #             rm = RMTorch(
    #                 units=units,
    #                 hadamard=hadamard,
    #                 bs_error=bs_error
    #             )
    #             self.assertAllClose(rm.matrix().conj().T @ rm.matrix(), identity_matrix)
    #             self.assertAllClose(rm.matrix().conj().T, rm.inverse_matrix())


class PRMLayerTest(tf.test.TestCase):
    def test_tf(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                prm = PRM(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                self.assertAllClose(prm.inverse_transform(prm.matrix), identity_matrix)
                self.assertAllClose(prm.matrix.conj().T, prm.inverse_matrix)

    # def test_t(self):
    #     for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
    #         with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
    #             identity_matrix = np.eye(units, dtype=NP_COMPLEX)
    #             prm = PRMTorch(
    #                 units=units,
    #                 hadamard=hadamard,
    #                 bs_error=bs_error
    #             )
    #             self.assertAllClose(prm.matrix().conj().T @ prm.matrix(), identity_matrix)
    #             self.assertAllClose(prm.matrix().conj().T, prm.inverse_matrix())


class TMLayerTest(tf.test.TestCase):
    def test_tf(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                tm = TM(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                self.assertAllClose(tm.inverse_transform(tm.matrix), identity_matrix)
                self.assertAllClose(tm.matrix.conj().T, tm.inverse_matrix)

    # def test_t(self):
    #     for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
    #         with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
    #             identity_matrix = np.eye(units, dtype=NP_COMPLEX)
    #             tm = TMTorch(
    #                 units=units,
    #                 hadamard=hadamard,
    #                 bs_error=bs_error
    #             )
    #             self.assertAllClose(tm.matrix().conj().T @ tm.matrix(), identity_matrix)
    #             self.assertAllClose(tm.matrix().conj().T, tm.inverse_matrix())


class BMLayerTest(tf.test.TestCase):
    def test_tf(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                identity_matrix = tf.eye(2 ** units, dtype=TF_COMPLEX)
                bm = BM(
                    num_layers=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                self.assertAllClose(bm.inverse_transform(bm.matrix), identity_matrix)
                self.assertAllClose(bm.matrix.conj().T, bm.inverse_matrix)

    # def test_t(self):
    #     for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
    #         with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
    #             identity_matrix = np.eye(2 ** units, dtype=NP_COMPLEX)
    #             bm = BMTorch(
    #                 num_layers=units,
    #                 hadamard=hadamard,
    #                 bs_error=bs_error
    #             )
    #             self.assertAllClose(bm.matrix().conj().T @ bm.matrix(), identity_matrix)
    #             self.assertAllClose(bm.matrix().conj().T, bm.inverse_matrix())


class CorrespondenceTest(tf.test.TestCase):
    def test_rm(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                rm_np = RMNumpy(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                test_correspondence(self, rm_np)

    def test_tm(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                tm_np = TMNumpy(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                test_correspondence(self, tm_np)

    def test_prm(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                prm_np = PRMNumpy(
                    units=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                test_correspondence(self, prm_np)

    def test_bm(self):
        for units, bs_error, hadamard in itertools.product(TEST_DIMENSIONS, (0, 0.1), (True, False)):
            with self.subTest(units=units, bs_error=bs_error, hadamard=hadamard):
                bm_np = BMNumpy(
                    num_layers=units,
                    hadamard=hadamard,
                    bs_error=bs_error
                )
                test_correspondence(self, bm_np)


@pytest.mark.skip(reason="helper function")
def test_correspondence(test_case: tf.test.TestCase, np_layer: MeshNumpyLayer):
    # set the testing to true and rebuild layer!
    np_layer._setup(testing=True)
    tf_layer = MeshLayer(np_layer.mesh.model)
    # t_layer = MeshTorchLayer(np_layer.mesh.model)
    test_case.assertAllClose(tf_layer.matrix, np_layer.matrix)
    test_case.assertAllClose(tf_layer.matrix.conj().T, np_layer.inverse_matrix)
    # test_case.assertAllClose(t_layer.matrix(), np_layer.matrix)


if __name__ == '__main__':
    tf.test.main()
