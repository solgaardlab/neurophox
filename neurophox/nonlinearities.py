import tensorflow as tf
import numpy as np


def cnormsq(inputs: tf.Tensor) -> tf.Tensor:
    return tf.square(tf.real(inputs)) + tf.square(tf.imag(inputs))


def cnorm(inputs: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.square(tf.real(inputs)) + tf.square(tf.imag(inputs)))


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
        tf.nn.relu(tf.real(inputs)),
        tf.nn.relu(tf.imag(inputs))
    )


class ParametrizedNonlinearity:
    def __init__(self, scope_name: str, units: int, is_complex: bool):
        self.scope_name = scope_name
        self.units = units
        self.is_complex = is_complex

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs

    def add_summary(self) -> tf.summary.Summary:
        return None  # required to override this if calling add_summary

    def __call__(self, inputs: tf.Tensor):
        self.transform(inputs)


class SelfPhaseModulation(ParametrizedNonlinearity):
    def __init__(self, scope_name: str, units: int, is_complex: bool=True):
        super(SelfPhaseModulation, self).__init__(scope_name, units, is_complex)
        with tf.variable_scope(self.scope_name):
            self.spm_scale = tf.get_variable(
                shape=[units],
                name="spm_scale",
                initializer=tf.random_uniform_initializer(0, 2 * np.pi * np.sqrt(units))
            )

    def add_summary(self) -> tf.summary.Summary:
        return tf.summary.image('spm_scale', self.spm_scale[tf.newaxis, tf.newaxis, :, tf.newaxis])

    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs * tf.exp(tf.abs(inputs) * self.spm_scale * 1j)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.transform(inputs)


NAME_TO_NONLINEARITY = {
    'norm': cnorm,
    'norm_sq': cnormsq,
    'modrelu': modrelu,
    'spm': SelfPhaseModulation,
    'crelu': crelu
}
