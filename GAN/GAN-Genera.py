import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy import misc

ary = np.random.uniform(0, 1, 49)
img_input = tf.cast(tf.reshape(ary, [-1, 7, 7, 1]), tf.float32)

net = tf.layers.conv2d_transpose(inputs=img_input, filters=1024, kernel_size=0, padding='same')

net = tf.layers.conv2d_transpose(inputs=net, filters=512, kernel_size=5, padding='same')

net = tf.layers.conv2d_transpose(inputs=net, filters=256, kernel_size=5, padding='same')

net = tf.layers.conv2d_transpose(inputs=net, filters=128, kernel_size=5, padding='same', strides=2)

net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=5, padding='same', strides=2)

print(net)

net = tf.layers.flatten(net)
print(net)

net = tf.reshape(net, [28, 28, 1])
print(net)