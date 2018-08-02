import tensorflow as tf
import numpy as np
from scipy import misc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=.070))


def bias_var(sh):
    return tf.Variable(tf.constant(value=0.1, shape=[sh]))


def conv_layer(img, weight, bias):
    layer = tf.nn.conv2d(img, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    return layer


x_input = tf.placeholder(tf.float32, [None, 49])
img_input = tf.cast(tf.reshape(x_input, [-1, 7, 7, 1]), tf.float32)


conv1 = conv_layer(img_input, weight_var([5, 5, 1, 32]), bias_var(32))
conv1_t = tf.layers.conv2d_transpose(inputs=conv1, filters=1024, kernel_size=5, padding='same', activation=tf.nn.relu)

conv2 = conv_layer(conv1_t, weight_var([5, 5, 1024, 512]), bias_var(512))
conv2_t = tf.layers.conv2d_transpose(inputs=conv2, filters=512, kernel_size=5, padding='same', activation=tf.nn.relu)

conv3 = conv_layer(conv2_t, weight_var([5, 5, 512, 256]), bias_var(256))
conv3_t = tf.layers.conv2d_transpose(inputs=conv3, filters=256, kernel_size=5, strides=2, padding='same',
                                     activation=tf.nn.relu)

conv4 = conv_layer(conv3_t, weight_var([5, 5, 256, 128]), bias_var(128))
conv4_t = tf.layers.conv2d_transpose(inputs=conv4, filters=128, kernel_size=5, padding='same', activation=tf.nn.relu)

conv5 = conv_layer(conv4_t, weight_var([5, 5, 128, 1]), bias_var(1))
final_layer = tf.layers.conv2d_transpose(inputs=conv5, filters=1, kernel_size=5, padding='same',
                                         strides=2, activation=tf.nn.tanh)

output = tf.layers.flatten(final_layer)
reshaped = tf.reshape(output, [28,28])


def train_gen(loss):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        cost = tf.reduce_min(loss)
        tf.train.AdamOptimizer(learning_rate=.0001).minimize(cost)

        get_img()


def generate_img(num_runs=1):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fakes = [np.random.uniform(-1, 1, 49) for _ in range(num_runs)]
        fake = sess.run(output, feed_dict={x_input: fakes})

    sess.close()
    return fake


def get_img():
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fake = sess.run(reshaped, feed_dict={x_input: [np.random.uniform(-1, 1, 49)]})
        print(fake)
        misc.toimage(fake).show()

    sess.close()
    return fake


get_img()