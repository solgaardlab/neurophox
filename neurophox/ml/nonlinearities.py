import tensorflow as tf
from ..config import TF_COMPLEX


def cnormsq(inputs: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs)), TF_COMPLEX)


def cnorm(inputs: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.sqrt(tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs))), TF_COMPLEX)
