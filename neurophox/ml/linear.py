from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import datetime

from ..tensorflow import SVD, MeshLayer

from ..config import TF_COMPLEX, NP_COMPLEX
from ..helpers import random_gaussian_batch


class LinearModelRunner:
    def __init__(self, model_name: str, layer: MeshLayer, optimizer: tf.keras.optimizers.Optimizer,
                 batch_size: int, iterations_per_epoch: int=50, logdir: Optional[str]=None):  # e.g., logdir=/data/tensorboard/neurophox/
        self.losses = []
        self.results = []
        self.layer = layer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.iters = 0
        self.iterations_per_epoch = iterations_per_epoch
        self.model_name = model_name
        self.logdir = logdir
        if self.logdir:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(
                f'{self.logdir}/{self.model_name}/' + current_time + '/'
            )

    def iterate(self, target_unitary: np.ndarray):
        x_train, y_train = generate_keras_batch(self.layer.units, target_unitary, self.batch_size)
        with tf.GradientTape() as tape:
            loss = complex_mse(self.layer(x_train), y_train)
        grads = tape.gradient(loss, self.layer.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.layer.trainable_variables))
        self.losses.append(tf.reduce_sum(complex_mse(self.layer(tf.eye(self.layer.units, dtype=TF_COMPLEX)),
                                       tf.convert_to_tensor(target_unitary.astype(NP_COMPLEX)))).numpy())
        if self.iters % self.iterations_per_epoch == 0:
            if self.logdir:
                self.update_tensorboard()
            if not isinstance(self.layer, SVD):
                phases = self.layer.phases
                mask = self.layer.mesh.model.mask
                estimate = self.layer(np.eye(self.layer.units, dtype=NP_COMPLEX))
                self.results.append({
                    "theta_list": phases.theta.param_list(mask),
                    "phi_list": phases.phi.param_list(mask),
                    "theta_checkerboard": phases.theta.checkerboard_arrangement,
                    "phi_checkerboard": phases.phi.checkerboard_arrangement,
                    "gamma": phases.gamma,
                    "estimate_mag": np.abs(estimate),
                    "error_mag": np.abs(estimate - target_unitary)
                })
        self.iters += 1

    def update_tensorboard(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', self.losses[-1], step=self.iters // self.iterations_per_epoch)

    def run(self, num_epochs, target_unitary: np.ndarray, pbar: Optional[Callable]=None):
        iterator = pbar(range(num_epochs * self.iterations_per_epoch)) if pbar else range(num_epochs * self.iterations_per_epoch)
        for _ in iterator:
            self.iterate(target_unitary)
            if pbar is not None:
                iterator.set_description(f"𝓛: {self.losses[-1]:.5f}")
            else:
                print(f"𝓛: {self.losses[-1]:.5f}")


def generate_keras_batch(units, target_unitary, batch_size):
    x_train = random_gaussian_batch(batch_size=batch_size, units=units)
    y_train = x_train @ target_unitary
    return tf.convert_to_tensor(x_train, dtype=TF_COMPLEX), tf.convert_to_tensor(y_train, dtype=TF_COMPLEX)


def complex_mse(y_true, y_pred):
    real_loss = tf.losses.mse(tf.math.real(y_true), tf.math.real(y_pred))
    imag_loss = tf.losses.mse(tf.math.imag(y_true), tf.math.imag(y_pred))
    return (real_loss + imag_loss) / 2
