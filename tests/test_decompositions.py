import tensorflow as tf

from neurophox.numpy import RMNumpy, TMNumpy, BMNumpy
from neurophox.decompositions import parallel_nullification, clements_decomposition

TEST_DIMENSIONS = [7, 8]
TEST_LAYERS = [3, 4]


class ParallelNullificationTest(tf.test.TestCase):
    def test_rm(self):
        for units in TEST_DIMENSIONS:
            rm = RMNumpy(units=units, basis='sm')
            reconstructed_rm = parallel_nullification(rm)
            self.assertAllClose(rm.matrix, reconstructed_rm.matrix)

    def test_tm(self):
        for units in TEST_DIMENSIONS:
            tm = TMNumpy(units=units, basis='sm')
            reconstructed_tm = parallel_nullification(tm)
            self.assertAllClose(tm.matrix, reconstructed_tm.matrix)

    def test_bm(self):
        for num_layers in TEST_LAYERS:
            bm = BMNumpy(num_layers=num_layers, basis='sm')
            reconstructed_bm = parallel_nullification(bm)
            self.assertAllClose(bm.matrix, reconstructed_bm.matrix)


class ClementsDecompositionTest(tf.test.TestCase):
    def test(self):
        for units in TEST_DIMENSIONS:
            rm = RMNumpy(units=units)
            reconstructed_rm = clements_decomposition(rm.matrix)
            self.assertAllClose(rm.matrix, reconstructed_rm.matrix)



if __name__ == '__main__':
    tf.test.main()
