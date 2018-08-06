import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import random
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


"""
Generator Network
"""

g1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 1024], stddev=.1))
g1_b = tf.Variable(tf.ones([1]))

g2_w = tf.Variable(tf.truncated_normal([5, 5, 1024, 512], stddev=.1))
g2_b = tf.Variable(tf.ones([512]))

g3_w = tf.Variable(tf.truncated_normal([5, 5, 512, 256], stddev=.1))
g3_b = tf.Variable(tf.ones([256]))

g4_w = tf.Variable(tf.truncated_normal([5, 5, 256, 128], stddev=.1))
g4_b = tf.Variable(tf.ones([128]))

g5_w = tf.Variable(tf.truncated_normal([5, 5, 128, 1], stddev=.1))
g5_b = tf.Variable(tf.ones([1]))

theta_g = [g1_w, g2_w, g3_w, g4_w, g5_w, g1_b, g2_w, g3_b, g4_b, g5_b]


def generator(noise, reshaped=False):
    inputs = np.asarray(noise)
    img_input = tf.cast(tf.reshape(inputs, [-1, 7, 7, 1]), tf.float32)
    conv1 = tf.nn.conv2d(img_input, g1_w, strides=[1, 1, 1, 1], padding='SAME') + g1_b

    conv2 = tf.nn.conv2d(conv1, g2_w, strides=[1, 1, 1, 1], padding='SAME') + g2_b

    conv3 = tf.nn.conv2d(conv2, g3_w, strides=[1, 1, 1, 1], padding='SAME') + g3_b

    conv4 = tf.nn.conv2d(conv3, g4_w, strides=[1, 1, 1, 1], padding='SAME') + g4_b
    conv4_t = tf.layers.conv2d_transpose(inputs=conv4, filters=128, kernel_size=5, strides=2, padding='same',
                                         activation=tf.nn.relu)

    conv5 = tf.nn.conv2d(conv4_t, g5_w, strides=[1, 1, 1, 1], padding='SAME') + g5_b
    final_layer = tf.layers.conv2d_transpose(inputs=conv5, filters=1, kernel_size=5, padding='same',
                                             strides=2, activation=tf.nn.tanh)

    output = tf.layers.flatten(final_layer)
    if reshaped:
        return tf.reshape(output, [28, 28])
    return output


"""
Discriminator Network
"""

d1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 3], stddev=.1))
d1_b = tf.Variable(tf.ones([3]))

d2_w = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=.1))
d2_b = tf.Variable(tf.ones([64]))

d3_w = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=.1))
d3_b = tf.Variable(tf.ones([128]))

d4_w = tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=.1))
d4_b = tf.Variable(tf.ones([256]))

theta_d = [d1_w, d2_w, d3_w, d4_w, d1_b, d2_b, d3_b, d4_b]


def discriminator(inputs):
    # input_x = tf.cast(np.asarray(inputs), tf.float32)
    x_reshape = tf.reshape(inputs, [-1, 28, 28, 1])

    conv1 = tf.nn.conv2d(x_reshape, d1_w, strides=[1, 1, 1, 1], padding='SAME') + d1_b

    conv2 = tf.nn.conv2d(conv1, d2_w, strides=[1, 2, 2, 1], padding='SAME') + d2_b

    conv3 = tf.nn.conv2d(conv2, d3_w, strides=[1, 2, 2, 1], padding='SAME') + d3_b

    conv4 = tf.nn.conv2d(conv3, d4_w, strides=[1, 2, 2, 1], padding='SAME') + d4_b

    flatten = tf.layers.flatten(conv4)

    pred = tf.layers.dense(inputs=flatten, units=1)
    pred_logit = tf.nn.sigmoid(pred)

    return pred


batch_size = 68

z_noise = [np.random.uniform(-1, 1, 49) for _ in range(batch_size)]
gen_sample = generator(z_noise)

# Probability for real images
images, _ = data.train.next_batch(batch_size)
d_real = discriminator(images)

# Probability for fake images
d_fake = discriminator(gen_sample)

# Loss
d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

# Training
gen_optimizer = tf.train.AdamOptimizer(learning_rate=.00085).minimize(g_loss, var_list=theta_g)
dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=.00085).minimize(d_loss, var_list=theta_d)


# Image Creation
img_noise = [np.random.uniform(-1., 1., 49)]
sample_return = generator(img_noise, True)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(500):
        sess.run(dis_optimizer)
        sess.run(gen_optimizer)

        if _ % 100 == 0:
            fake = sess.run(sample_return)
            # fake = sess.run(image, feed_dict={vec_noise: [np.random.uniform(-1., 1., 49)]})
            print(fake)
            misc.toimage(fake).show()

sess.close()
