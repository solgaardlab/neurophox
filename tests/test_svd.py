from neurophox.tensorflow import SVD
from neurophox.config import TF_COMPLEX, NP_COMPLEX

import numpy as np
import tensorflow as tf


TEST_DIMENSIONS = [7, 8]


class SVDRMTransformerTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            identity_matrix = np.eye(units, dtype=NP_COMPLEX)
            svd_layer = SVD(units=units, mesh_dict={'name': 'rm'})
            self.assertAllClose(
                svd_layer.inverse_transform(svd_layer.matrix),
                identity_matrix, atol=1e-5
            )


class SVDPRMTransformerTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
            svd_layer = SVD(units=units, mesh_dict={'name': 'prm'})
            self.assertAllClose(
                svd_layer.inverse_transform(svd_layer.matrix),
                identity_matrix, atol=1e-5
            )


class SVDRRMTransformerTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
            svd_layer = SVD(units=units, mesh_dict={'name': 'rm', 'properties': {'num_layers': units * 2}})
            self.assertAllClose(
                svd_layer.inverse_transform(svd_layer.matrix),
                identity_matrix, atol=1e-4
            )


if __name__ == '__main__':
    tf.test.main()
