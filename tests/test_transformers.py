import tensorflow as tf

from neurophox.tensorflow import Diagonal, DiagonalPhaseLayer, RectangularPerm, ButterflyPerm

TEST_DIMENSIONS = [7, 8]


class DiagonalTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            diag_layer = Diagonal(units=units)
            self.assertAllClose(diag_layer(diag_layer.inverse_matrix), tf.eye(units))


class DiagonalPhaseTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            diag_phase = DiagonalPhaseLayer(units=units)
            self.assertAllClose(diag_phase(diag_phase.inverse_matrix), tf.eye(units))


class RectangularPermutationTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            rp = RectangularPerm(units=units,
                                 frequency=units // 2)
            self.assertAllClose(rp(rp.inverse_matrix), tf.eye(units))


class ButterflyPermutationTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            if not units % 2:  # odd case still needs to be worked out.
                fp = ButterflyPerm(units=units,
                                   frequency=units // 2)
                self.assertAllClose(fp(fp.inverse_matrix), tf.eye(units))
            else:
                self.assertRaises(NotImplementedError)


if __name__ == '__main__':
    tf.test.main()
