import tensorflow as tf

from neurophox.config import TF_COMPLEX

from neurophox.numpy import RMNumpy, TMNumpy, PRMNumpy, BMNumpy
from neurophox.tensorflow import RM, TM, PRM, BM, MeshLayer

TEST_DIMENSIONS = [7, 8]


class RMLayerTest(tf.test.TestCase):
    def test_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                rm = RM(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                self.assertAllClose(rm.inverse_transform(rm.matrix), identity_matrix)
                self.assertAllClose(rm.matrix.conj().T, rm.inverse_matrix)

    def test_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                rd = RM(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                self.assertAllClose(rd.inverse_transform(rd.matrix), identity_matrix)
                self.assertAllClose(rd.matrix.conj().T, rd.inverse_matrix)


class PRMLayerTest(tf.test.TestCase):
    def test_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                prm = PRM(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                self.assertAllClose(prm.inverse_transform(prm.matrix), identity_matrix)
                self.assertAllClose(prm.matrix.conj().T, prm.inverse_matrix)

    def test_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                prm = PRM(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                self.assertAllClose(prm.inverse_transform(prm.matrix), identity_matrix)
                self.assertAllClose(prm.matrix.conj().T, prm.inverse_matrix)


class TMLayerTest(tf.test.TestCase):
    def test_beamsplitter(self):
        from neurophox.tensorflow import TM
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                tm = TM(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                self.assertAllClose(tm.inverse_transform(tm.matrix), identity_matrix)
                self.assertAllClose(tm.matrix.conj().T, tm.inverse_matrix)

    def test_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                identity_matrix = tf.eye(units, dtype=TF_COMPLEX)
                tm = TM(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                self.assertAllClose(tm.inverse_transform(tm.matrix), identity_matrix)
                self.assertAllClose(tm.matrix.conj().T, tm.inverse_matrix)


class NumpyCorrespondenceTest(tf.test.TestCase):
    def test_rm_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                rm_np = RMNumpy(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                rm_np._setup(None, testing=True)
                rm_tf = MeshLayer(rm_np.mesh.model)
                self.assertAllClose(rm_tf.matrix, rm_np.matrix)
                self.assertAllClose(rm_tf.matrix.conj().T, rm_np.inverse_matrix)

    def test_rm_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                rm_np = RMNumpy(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                rm_np._setup(None, testing=True)
                rm_tf = MeshLayer(rm_np.mesh.model)
                self.assertAllClose(rm_tf.matrix, rm_np.matrix)
                self.assertAllClose(rm_tf.matrix.conj().T, rm_np.inverse_matrix)

    def test_tm_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                tm_np = TMNumpy(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                tm_np._setup(None, testing=True)
                tm_tf = MeshLayer(tm_np.mesh.model)
                self.assertAllClose(tm_tf.matrix, tm_np.matrix)
                self.assertAllClose(tm_tf.matrix.conj().T, tm_np.inverse_matrix)

    def test_tm_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                tm_np = TMNumpy(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                tm_np._setup(None, testing=True)
                tm_tf = MeshLayer(tm_np.mesh.model)
                self.assertAllClose(tm_tf.matrix, tm_np.matrix)
                self.assertAllClose(tm_tf.matrix.conj().T, tm_np.inverse_matrix)

    def test_prm_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                prm_np = PRMNumpy(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                prm_np._setup(None, testing=True)
                prm_tf = MeshLayer(prm_np.mesh.model)
                self.assertAllClose(prm_tf.matrix, prm_np.matrix)
                self.assertAllClose(prm_tf.matrix.conj().T, prm_np.inverse_matrix)

    def test_prm_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                prm_np = PRMNumpy(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                prm_np._setup(None, testing=True)
                prm_tf = MeshLayer(prm_np.mesh.model)
                self.assertAllClose(prm_tf.matrix, prm_np.matrix)
                self.assertAllClose(prm_tf.matrix.conj().T, prm_np.inverse_matrix)

    def test_bm_beamsplitter(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                bm_np = PRMNumpy(
                    units=units,
                    hadamard=False,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                bm_np._setup(None, testing=True)
                bm_tf = MeshLayer(bm_np.mesh.model)
                self.assertAllClose(bm_tf.matrix, bm_np.matrix)
                self.assertAllClose(bm_tf.matrix.conj().T, bm_np.inverse_matrix)

    def test_bm_hadamard(self):
        for units in TEST_DIMENSIONS:
            for bs_error in (0, 0.1):
                bm_np = PRMNumpy(
                    units=units,
                    hadamard=True,
                    bs_error=bs_error
                )
                # set the testing to true and rebuild layer!
                bm_np._setup(None, testing=True)
                bm_tf = MeshLayer(bm_np.mesh.model)
                self.assertAllClose(bm_tf.matrix, bm_np.matrix)
                self.assertAllClose(bm_tf.matrix.conj().T, bm_np.inverse_matrix)


if __name__ == '__main__':
    tf.test.main()
