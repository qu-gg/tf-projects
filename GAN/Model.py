import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)


def create_batch(batch_size):
    """
    Function for creating a single training batch for the discriminator, mixed of real and fake images
    Class is 0 if it is real and 1 if it is fake
    :param batch_size:
    :return:
    """
    num = random.randint(20, 40)
    fakes = generate_img(num)
    reals, _ = data.train.next_batch(batch_size - num)

    image_batch = []
    class_batch = []

    for i in range(batch_size - num):
        image_batch.append(reals[i])
        class_batch.append([random.uniform(0.0, 0.1)])
    for i in range(num):
        image_batch.append(fakes[i])
        class_batch.append([random.uniform(0.9, 1.0)])

    # Shuffle
    numpy_state = np.random.get_state()
    np.random.shuffle(image_batch)
    np.random.set_state(numpy_state)
    np.random.shuffle(class_batch)

    return image_batch, class_batch


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.5))


def bias_var(sh):
    return tf.Variable(tf.constant(value=0.1, shape=[sh]))


def discrim_layer(img, weight, bias, strides):
    layer = tf.nn.conv2d(img, weight, strides=strides, padding='SAME',) + bias
    result = tf.nn.leaky_relu(layer)
    return result


def gen_layer(img, weight, bias):
    layer = tf.nn.conv2d(img, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    return layer


"""
Generator Network
"""
with tf.name_scope('genera'):
    x_input = tf.placeholder(tf.float32, [None, 49])
    img_input = tf.cast(tf.reshape(x_input, [-1, 7, 7, 1]), tf.float32)


    conv1 = gen_layer(img_input, weight_var([5, 5, 1, 32]), bias_var(32))
    conv1_t = tf.layers.conv2d_transpose(inputs=conv1, filters=1024, kernel_size=5, padding='same', activation=tf.nn.relu)

    conv2 = gen_layer(conv1_t, weight_var([5, 5, 1024, 512]), bias_var(512))
    conv2_t = tf.layers.conv2d_transpose(inputs=conv2, filters=512, kernel_size=5, padding='same', activation=tf.nn.relu)

    conv3 = gen_layer(conv2_t, weight_var([5, 5, 512, 256]), bias_var(256))
    conv3_t = tf.layers.conv2d_transpose(inputs=conv3, filters=256, kernel_size=5, strides=2, padding='same',
                                         activation=tf.nn.relu)

    conv4 = gen_layer(conv3_t, weight_var([5, 5, 256, 128]), bias_var(128))
    conv4_t = tf.layers.conv2d_transpose(inputs=conv4, filters=128, kernel_size=5, padding='same', activation=tf.nn.relu)

    conv5 = gen_layer(conv4_t, weight_var([5, 5, 128, 1]), bias_var(1))
    final_layer = tf.layers.conv2d_transpose(inputs=conv5, filters=1, kernel_size=5, padding='same',
                                             strides=2, activation=tf.nn.tanh)

    output = tf.layers.flatten(final_layer)
    reshaped = tf.reshape(output, [28,28])


def generate_img(num_runs=1):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fakes = [np.random.uniform(-1, 1, 49) for _ in range(num_runs)]
        fake = sess.run(output, feed_dict={x_input: fakes})
    sess.close()
    return fake


"""
Discriminator Network
"""
with tf.name_scope("discrim"):
    input_x = tf.placeholder(tf.float32, [None, 784])
    x_reshape = tf.reshape(input_x, [-1, 28, 28, 1])

    input_cls = tf.placeholder(tf.float32, [None, 1])
    conv1 = discrim_layer(x_reshape, weight_var([5, 5, 1, 3]), bias_var(3), [1, 1, 1, 1])

    conv2 = discrim_layer(conv1, weight_var([5, 5, 3, 64]), bias_var(64), [1, 2, 2, 1])

    conv3 = discrim_layer(conv2, weight_var([5, 5, 64, 128]), bias_var(128), [1, 2, 2, 1])

    conv4 = discrim_layer(conv3, weight_var([5, 5, 128, 256]), bias_var(256), [1, 2, 2, 1])

    flatten = tf.layers.flatten(conv4)

    pred = tf.layers.dense(inputs=flatten, units=1)
    pred = tf.nn.sigmoid(pred)

    # Loss, cost, optimizing
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_cls, logits=pred)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00085).minimize(cost)

    # Accuracy
    correct_prediction = tf.equal(tf.greater(pred, [0.5]), tf.cast(input_cls, 'bool'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


def use_discrim(image_batch, class_batch):
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    prediction = sess.run(pred, feed_dict={input_x: image_batch,
                                           input_cls: class_batch})

    sess.close()
    return prediction


def main():
    batch_size = 68

    images, _ = data.train.next_batch(batch_size)
    classes = [[0] for _ in range(batch_size)]
    dis_real = use_discrim(images, classes)

    g_images, g_classes = create_batch(batch_size)
    dis_fake = use_discrim(g_images, g_classes)

    dis_loss = tf.reduce_mean(np.log(dis_real) + np.log(1 - dis_fake))
    gen_loss = tf.reduce_mean(np.log(dis_fake))

    dis_solver = tf.train.GradientDescentOptimizer(learning_rate=.00085).minimize(
        dis_loss, var_list=[item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discrim')]
    )

