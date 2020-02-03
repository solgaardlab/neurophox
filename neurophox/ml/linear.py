from typing import Optional, Callable, List, Union

import numpy as np
import tensorflow as tf
import pickle

from ..config import TF_COMPLEX, NP_COMPLEX
from ..helpers import random_gaussian_batch
from ..tensorflow import MeshLayer, SVD


class LinearMultiModelRunner:
    """
    Complex mean square error linear optimization experiment that can run and track multiple model optimizations in parallel.

    Args:
        experiment_name: Name of the experiment
        layer_names: List of layer names
        layers: List of transformer layers
        optimizer: Optimizer for all layers or list of optimizers for each layer
        batch_size: Batch size for the optimization
        iterations_per_epoch: Iterations per epoch
        iterations_per_tb_update: Iterations per update of TensorBoard
        logdir: Logging directory for TensorBoard to track losses of each layer (default to `None` for no logging)
        train_on_test: Use same training and testing set
        store_params: Store params during the training for visualization later
    """

    def __init__(self, experiment_name: str, layer_names: List[str],
                 layers: List[MeshLayer],
                 optimizer: Union[tf.keras.optimizers.Optimizer, List[tf.keras.optimizers.Optimizer]],
                 batch_size: int, iterations_per_epoch: int = 50, iterations_per_tb_update: int = 5,
                 logdir: Optional[str] = None, train_on_test: bool = False,
                 store_params: bool = True):  # e.g., logdir=/data/tensorboard/neurophox/
        self.losses = {name: [] for name in layer_names}
        self.results = {name: [] for name in layer_names}
        self.layer_names = layer_names
        self.layers = layers
        self.optimizers = optimizer if isinstance(optimizer, List) else [optimizer for _ in layer_names]
        if not (len(layer_names) == len(layers) and len(layers) == len(self.optimizers)):
            raise ValueError("layer_names, layers, and optimizers must all be the same length")
        self.batch_size = batch_size
        self.iters = 0
        self.iterations_per_epoch = iterations_per_epoch
        self.iterations_per_tb_update = iterations_per_tb_update
        self.experiment_name = experiment_name
        self.logdir = logdir
        self.train_on_test = train_on_test
        self.store_params = store_params
        if self.logdir:
            self.summary_writers = {name: tf.summary.create_file_writer(
                f'{self.logdir}/{experiment_name}/{name}/'
            ) for name in layer_names}

    def iterate(self, target_unitary: np.ndarray, learning_rate=1e-2, method=1, density=0.5):
        """
        Run gradient update toward a target unitary :math:`U`.

        Args:
            target_unitary: Target unitary, :math:`U`.

        """
        if self.train_on_test:
            x_train, y_train = tf.eye(self.layers[0].units, dtype=TF_COMPLEX), \
                               tf.convert_to_tensor(target_unitary, dtype=TF_COMPLEX)
        else:
            x_train, y_train = generate_keras_batch(self.layers[0].units, target_unitary, self.batch_size)
        for name, layer, optimizer in zip(self.layer_names, self.layers, self.optimizers):
            if method == 0:
                with tf.GradientTape() as tape:
                    loss = complex_mse(layer(x_train), y_train)
                grads = tape.gradient(loss, layer.trainable_variables)
            else:
                grads = stoch_update(layer, x_train, y_train,
                                     density=density, learning_rate=learning_rate,
                                     method=method)  # weight update already completed in the function call
            optimizer.apply_gradients(grads_and_vars=zip(grads, layer.trainable_variables))
            self.losses[name].append(tf.reduce_sum(
                complex_mse(layer(tf.eye(layer.units, dtype=TF_COMPLEX)),
                            tf.convert_to_tensor(target_unitary.astype(NP_COMPLEX)))).numpy()
                                     )
            if self.iters % self.iterations_per_tb_update and self.logdir:
                self.update_tensorboard(name)
            if self.iters % self.iterations_per_epoch == 0 and self.store_params:
                if not isinstance(layer, SVD):
                    phases = layer.phases
                    mask = layer.mesh.model.mask
                    estimate = layer(tf.eye(layer.units, dtype=TF_COMPLEX)).numpy()
                    self.results[name].append({
                        "theta_list": phases.theta.param_list(mask),
                        "phi_list": phases.phi.param_list(mask),
                        "theta_checkerboard": phases.theta.checkerboard_arrangement,
                        "phi_checkerboard": phases.phi.checkerboard_arrangement,
                        "gamma": phases.gamma,
                        "estimate_mag": np.abs(estimate),
                        "error_mag": np.abs(estimate - target_unitary)
                    })

        self.iters += 1

    def update_tensorboard(self, name: str):
        """
        Update TensorBoard variables.

        Args:
            name: Layer name corresponding to variables that are updated

        """
        with self.summary_writers[name].as_default():
            tf.summary.scalar(f'loss-{self.experiment_name}', self.losses[name][-1], step=self.iters)

    def run(self, num_epochs: int, target_unitary: np.ndarray, pbar: Optional[Callable] = None, learning_rate=1e-2,
            decay_rate=0.999, method=1, density=0.5):
        """

        Args:
            num_epochs: Number of epochs (defined in terms of `iterations_per_epoch`)
            target_unitary: Target unitary, :math:`U`.
            pbar: Progress bar (tqdm recommended)

        """
        print('running')
        iterator = pbar(range(num_epochs * self.iterations_per_epoch)) if pbar else range(
            num_epochs * self.iterations_per_epoch)
        for _ in iterator:
            self.iterate(target_unitary, learning_rate, method, density)
            learning_rate *= decay_rate

    def save(self, savepath: str):
        """
        Save results for the multi-model runner to pickle file.

        Args:
            savepath: Path to save results.
        """
        with open(f"{savepath}/{self.experiment_name}.p", "wb") as f:
            pickle.dump({"losses": self.losses, "results": self.results}, f)


def generate_keras_batch(units, target_unitary, batch_size):
    x_train = random_gaussian_batch(batch_size=batch_size, units=units)
    y_train = x_train @ target_unitary
    return tf.convert_to_tensor(x_train, dtype=TF_COMPLEX), tf.convert_to_tensor(y_train, dtype=TF_COMPLEX)


def complex_mse(y_true, y_pred):
    """

    Args:
        y_true: The true labels, :math:`V \in \mathbb{C}^{B \\times N}`
        y_pred: The true labels, :math:`\\widehat{V} \in \mathbb{C}^{B \\times N}`

    Returns:
        The complex mean squared error :math:`\\boldsymbol{e} \in \mathbb{R}^B`,
        where given example :math:`\\widehat{V}_i \in \mathbb{C}^N`,
        we have :math:`e_i = \\frac{\|V_i - \\widehat{V}_i\|^2}{N}`.

    """
    real_loss = tf.losses.mse(tf.math.real(y_true), tf.math.real(y_pred))
    imag_loss = tf.losses.mse(tf.math.imag(y_true), tf.math.imag(y_pred))
    return (real_loss + imag_loss) / 2


def stoch_update(layer, x_train, y_train, density=0.5, jig_scale=1e-5, learning_rate=1e1, method=1):
    """
        Return the ONN weights by perturbation.

        Args:
            layer: callable layer for loss evaluation
            x_train: 1 batch of data used for training
            y_train: 1 batch of labels used for training
            density: proportion of neurons that jiggle together
            jig_scale: the amount of jiggling (finite difference stepsize)
            learning_rate: learning rate (that determines stepsize of final weight update)
            method: 1. nonuniform jiggle 2. uniform jiggle with uniform dist. 3. uniform jiggle with binary dist.

    """
    # 1. estimating gradient
    loss = complex_mse(layer(x_train), y_train)
    jig_events = []
    jig_amounts = []
    # 2. choose which perturbation scheme:
    for var in layer.variables:
        jig_events.append(np.random.choice([0, 1], size=var.shape, p=[1 - density, density]))
        if method == 1:
            # nonuniform jiggle
            jig_amounts.append(np.random.random(var.shape) - 0.5)
        elif method == 2:
            # uniform jiggle with uniform distribution
            jig_amounts.append((np.random.random() - 0.5) * np.ones(var.shape))
        else:
            # uniform jiggle with binary distribution
            jig_amounts.append((np.random.choice([0, 1]) - 0.5) * np.ones(var.shape))
        var.assign(var.read_value() + jig_scale * jig_events[-1] * jig_amounts[-1])
    # 3. calculate new loss and undo the perturbation
    loss_after_jiggle = complex_mse(layer(x_train), y_train)
    for i, var in enumerate(layer.variables):
        # only update the weights of jiggled neurons
        var.assign(var.read_value() - jig_scale * jig_events[i] * jig_amounts[i])
    dldw = [tf.reduce_sum(loss_after_jiggle - loss).numpy() / i for i in jig_amounts]
    # 4. return the update, while also undoing the perturbation
    return [-layer_gradient * learning_rate * jig_event for layer_gradient, jig_event in zip(dldw, jig_events)]
