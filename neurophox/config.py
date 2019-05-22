import tensorflow as tf
import numpy as np

# Backends

PYTORCH = 'torch'
TFKERAS = 'tf'
NUMPY = 'numpy'

# Types (for memory)

NP_COMPLEX = np.complex128
NP_FLOAT = np.float64

TF_COMPLEX = tf.complex64
TF_FLOAT = tf.float32

# Test seed

TEST_SEED = 31415

# Phase basis

BLOCH = "bloch"
SINGLEMODE = "sm"
DEFAULT_BASIS = BLOCH