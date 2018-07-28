import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy import misc

from tensorflow.examples.tutorials.mnist import input_data

x = input_data.read_data_sets()
