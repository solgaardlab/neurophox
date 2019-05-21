import tensorflow as tf
from ..config import TF_COMPLEX


def cnormsq(inputs: tf.Tensor) -> tf.Tensor:
    """

    Args:
        inputs: The input tensor, :math:`V`.

    Returns:
        An output tensor that performs elementwise absolute value squared operation, :math:`f(V) = |V|^2`.

    """
    return tf.cast(tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs)), TF_COMPLEX)


def cnorm(inputs: tf.Tensor) -> tf.Tensor:
    """

    Args:
        inputs: The input tensor, :math:`V`.

    Returns:
        An output tensor that performs elementwise absolute value operation, :math:`f(V) = |V|`.

    """
    return tf.cast(tf.sqrt(tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs))), TF_COMPLEX)
