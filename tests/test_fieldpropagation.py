import tensorflow as tf
import numpy as np
from neurophox.numpy import RMNumpy
from neurophox.helpers import random_gaussian_batch

DIMS = [7, 8]


class FieldPropagationTest(tf.test.TestCase):
    def test(self):
        for units in DIMS:
            identity = np.eye(units)
            rdlayer = RMNumpy(units=units)
            fields = rdlayer.propagate(identity)
            inverse_fields = rdlayer.inverse_propagate(rdlayer.transform(identity))
            self.assertAllClose(fields, inverse_fields)

    def test_explicit(self):
        for units in DIMS:
            identity = np.eye(units)
            rdlayer = RMNumpy(units=units)
            fields = rdlayer.propagate(identity, explicit=True)
            inverse_fields = rdlayer.inverse_propagate(rdlayer.transform(identity), explicit=True)
            self.assertAllClose(fields, inverse_fields)


class FieldPropagationBatchTest(tf.test.TestCase):
    def test(self):
        for units in DIMS:
            batch = random_gaussian_batch(batch_size=units, units=units)
            rdlayer = RMNumpy(units=units)
            fields = rdlayer.propagate(batch)
            inverse_fields = rdlayer.inverse_propagate(rdlayer.transform(batch))
            self.assertAllClose(fields, inverse_fields)

    def test_explicit(self):
        for units in DIMS:
            identity = random_gaussian_batch(batch_size=units, units=units)
            rdlayer = RMNumpy(units=units)
            fields = rdlayer.propagate(identity, explicit=True)
            inverse_fields = rdlayer.inverse_propagate(rdlayer.transform(identity), explicit=True)
            self.assertAllClose(fields, inverse_fields)
