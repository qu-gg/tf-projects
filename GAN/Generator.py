import tensorflow as tf
import numpy as np
from scipy import misc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

x_input = tf.placeholder(tf.float32, [None, 49])
img_input = tf.cast(tf.reshape(x_input, [-1, 7, 7, 1]), tf.float32)

conv1_w = tf.Variable(tf.truncated_normal([7, 7, 1, 1024], stddev=1))
conv1_b = tf.Variable(tf.zeros([1024]))
conv1 = tf.nn.conv2d(img_input, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

conv_one = tf.layers.conv2d_transpose(inputs=conv1, filters=512, kernel_size=5, padding='same',
                                      activation=tf.nn.relu)

conv_two = tf.layers.conv2d_transpose(inputs=conv_one, filters=256, kernel_size=5, padding='same',
                                      activation=tf.nn.relu)

conv_three = tf.layers.conv2d_transpose(inputs=conv_two, filters=128, kernel_size=5, padding='same', strides=2,
                                        activation=tf.nn.relu)

final_layer = tf.layers.conv2d_transpose(inputs=conv_three, filters=1, kernel_size=5, padding='same',
                                         strides=2, activation=tf.nn.softmax)


output = tf.layers.flatten(final_layer)
reshaped = tf.reshape(output, [28,28])

#  loss = tf.placeholder(tf.float32, [None, 128])
# cost = tf.reduce_min(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(cost)


def train_gen(num, entropy):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(num):
            print()
            #sess.run(optimizer, feed_dict={loss: entropy})


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
        misc.toimage(fake).show()

    sess.close()

