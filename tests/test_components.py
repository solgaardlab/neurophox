import numpy as np
import tensorflow as tf

from neurophox.components import *
from neurophox.config import NP_COMPLEX, TEST_SEED

TEST_DIMENSIONS = [7, 8]
IDENTITY = np.eye(2, dtype=NP_COMPLEX)
np.random.seed(TEST_SEED)
RANDOM_PHASE_SHIFT = float(np.pi * 2 * np.random.rand())
np.random.seed(TEST_SEED)
RANDOM_THETA = float(np.pi * 2 * np.random.rand())


class LinearOpticalComponentTest(tf.test.TestCase):
    def test(self):
        loc = PairwiseUnitary()
        self.assertAllClose(loc.matrix @ loc.inverse_matrix, IDENTITY)
        self.assertAllClose(loc.matrix.conj().T, loc.inverse_matrix)


class PhaseShiftTest(tf.test.TestCase):
    def test_upper(self):
        ps = PhaseShiftUpper(RANDOM_PHASE_SHIFT)
        ps_inv = PhaseShiftUpper(-RANDOM_PHASE_SHIFT)
        self.assertAllClose(ps.matrix @ ps.inverse_matrix, IDENTITY)
        self.assertAllClose(ps.matrix.conj(), ps.inverse_matrix)
        self.assertAllClose(ps.matrix.conj(), ps_inv.matrix)

    def test_lower(self):
        ps = PhaseShiftLower(RANDOM_PHASE_SHIFT)
        ps_inv = PhaseShiftLower(-RANDOM_PHASE_SHIFT)
        self.assertAllClose(ps.matrix @ ps.inverse_matrix, IDENTITY)
        self.assertAllClose(ps.matrix.conj(), ps.inverse_matrix)
        self.assertAllClose(ps.matrix.conj(), ps_inv.matrix)

    def test_common_mode(self):
        ps = PhaseShiftCommonMode(RANDOM_PHASE_SHIFT)
        ps_inv = PhaseShiftCommonMode(-RANDOM_PHASE_SHIFT)
        self.assertAllClose(ps.matrix @ ps.inverse_matrix, IDENTITY)
        self.assertAllClose(ps.matrix, np.exp(1j * RANDOM_PHASE_SHIFT) * IDENTITY)
        self.assertAllClose(ps.matrix.conj(), ps.inverse_matrix)
        self.assertAllClose(ps.matrix.conj(), ps_inv.matrix)

    def test_differential_mode(self):
        ps = PhaseShiftDifferentialMode(RANDOM_PHASE_SHIFT)
        ps_inv = PhaseShiftDifferentialMode(-RANDOM_PHASE_SHIFT)
        self.assertAllClose(ps.matrix @ ps.inverse_matrix, IDENTITY)
        self.assertAllClose(ps.matrix.conj(), ps.inverse_matrix)
        self.assertAllClose(ps.matrix.conj(), ps_inv.matrix)


class BeamsplitterTest(tf.test.TestCase):
    def test_(self):
        for hadamard in [True, False]:
            for epsilon in [0, 0.1]:
                bs = Beamsplitter(hadamard=hadamard, epsilon=epsilon)
                self.assertAllClose(bs.matrix @ bs.inverse_matrix, IDENTITY)
                self.assertAllClose(bs.matrix.conj().T, bs.inverse_matrix)
                if epsilon == 0:
                    self.assertAllClose(np.abs(bs.matrix ** 2), 0.5 * np.ones_like(bs.matrix))
                self.assertAllClose(np.linalg.det(bs.matrix), -1 if hadamard else 1)


class MZITest(tf.test.TestCase):
    def test_mzi(self):
        for hadamard in [True, False]:
            for epsilon in [0, 0.1]:
                mzi = BlochMZI(theta=RANDOM_THETA, phi=RANDOM_PHASE_SHIFT, hadamard=hadamard, epsilon=epsilon)
                bs = Beamsplitter(hadamard=hadamard, epsilon=epsilon)
                internal_ps = PhaseShiftDifferentialMode(RANDOM_THETA)
                external_ps_upper = PhaseShiftUpper(RANDOM_PHASE_SHIFT)
                self.assertAllClose(mzi.matrix @ mzi.inverse_matrix, IDENTITY)
                self.assertAllClose(mzi.matrix.conj().T, mzi.inverse_matrix)
                self.assertAllClose(np.linalg.det(mzi.matrix), np.exp(1j * RANDOM_PHASE_SHIFT))
                self.assertAllClose(mzi.matrix, bs.matrix @ internal_ps.matrix @ bs.matrix @ external_ps_upper.matrix)

    def test_smmzi(self):
        for hadamard in [True, False]:
            for epsilon in [0, 0.1]:
                mzi = SMMZI(theta=RANDOM_THETA, phi=RANDOM_PHASE_SHIFT, hadamard=hadamard, epsilon=epsilon)
                bs = Beamsplitter(hadamard=hadamard, epsilon=epsilon)
                internal_ps = PhaseShiftUpper(RANDOM_THETA)
                external_ps_upper = PhaseShiftUpper(RANDOM_PHASE_SHIFT)
                self.assertAllClose(mzi.matrix @ mzi.inverse_matrix, IDENTITY)
                self.assertAllClose(mzi.matrix.conj().T, mzi.inverse_matrix)
                self.assertAllClose(np.linalg.det(mzi.matrix), np.exp(1j * (RANDOM_PHASE_SHIFT + RANDOM_THETA)))
                self.assertAllClose(mzi.matrix, bs.matrix @ internal_ps.matrix @ bs.matrix @ external_ps_upper.matrix)



if __name__ == '__main__':
    tf.test.main()
