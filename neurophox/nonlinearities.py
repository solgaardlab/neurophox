import tensorflow as tf


def cnormsq(inputs: tf.Tensor) -> tf.Tensor:
    return tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs))


def cnorm(inputs: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.square(tf.math.real(inputs)) + tf.square(tf.math.imag(inputs)))


def modrelu(inputs: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
    inputs_abs: tf.Tensor = tf.abs(inputs) + 0.01
    step1 = inputs_abs + bias
    step2 = tf.complex(tf.nn.relu(step1), 0.0)
    step3 = inputs / tf.complex(inputs_abs, 0.0)
    return tf.multiply(step3, step2)


def modrelu_real(z: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    z_norm: tf.Tensor = tf.abs(z) + 0.01
    step1 = z_norm + b
    step2 = tf.nn.relu(step1)
    step3 = tf.sign(z)
    return tf.multiply(step3, step2)


def crelu(inputs: tf.Tensor) -> tf.Tensor:
    return tf.complex(
        tf.nn.relu(tf.math.real(inputs)),
        tf.nn.relu(tf.math.imag(inputs))
    )

