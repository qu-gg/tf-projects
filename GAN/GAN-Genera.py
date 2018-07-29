import os
import tensorflow as tf
import numpy as np
from scipy import misc

ary = np.random.uniform(0, 1, 49)
img_input = tf.cast(tf.reshape(ary, [-1, 7, 7, 1]), tf.float32)

conv_one = tf.layers.conv2d_transpose(inputs=img_input, filters=1024, kernel_size=0, padding='same')

conv_two = tf.layers.conv2d_transpose(inputs=conv_one, filters=512, kernel_size=5, padding='same')

conv_three = tf.layers.conv2d_transpose(inputs=conv_two, filters=256, kernel_size=5, padding='same')

conv_four = tf.layers.conv2d_transpose(inputs=conv_three, filters=128, kernel_size=5, padding='same', strides=2)

final_layer = tf.layers.conv2d_transpose(inputs=conv_four, filters=1, kernel_size=5, padding='same', strides=2)

flattened = tf.layers.flatten(final_layer)

output = tf.reshape(flattened, [28, 28])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    fake = sess.run(output)
    misc.toimage(fake).show()

sess.close()
