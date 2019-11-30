import tensorflow as tf
import numpy as np

from neurophox.numpy import MeshNumpyLayer, RMNumpy, TMNumpy, BMNumpy, MeshPhases
from neurophox.meshmodel import MeshModel
from neurophox.helpers import inverse_permutation

TEST_DIMENSIONS = [7, 8]
TEST_LAYERS = [3, 4]


def run_parallel_nullification(np_layer):
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
                          basis='sm'),
                phases=MeshPhases(theta=np.asarray(theta),
                                  phi=np.asarray(phi),
                                  gamma=np.zeros_like(np_layer.phases.gamma))
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
                  basis='sm'),
        phases=MeshPhases(theta=np.asarray(theta),
                          phi=np.asarray(phi),
                          gamma=np_layer.phases.gamma)
    )


class ParallelNullificationTest(tf.test.TestCase):
    def test_rm(self):
        for units in TEST_DIMENSIONS:
            rm = RMNumpy(units=units, basis='sm')
            reconstructed_rm = run_parallel_nullification(rm)
            self.assertAllClose(rm.matrix, reconstructed_rm.matrix)

    def test_tm(self):
        for units in TEST_DIMENSIONS:
            tm = TMNumpy(units=units, basis='sm')
            reconstructed_tm = run_parallel_nullification(tm)
            self.assertAllClose(tm.matrix, reconstructed_tm.matrix)

    def test_bm(self):
        for num_layers in TEST_LAYERS:
            bm = BMNumpy(num_layers=num_layers, basis='sm')
            reconstructed_bm = run_parallel_nullification(bm)
            self.assertAllClose(bm.matrix, reconstructed_bm.matrix)


if __name__ == '__main__':
    tf.test.main()
