import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from GAN.Generator import generate_img
data = input_data.read_data_sets('data/MNIST', one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 784])
x_reshape = tf.reshape(input_x, [-1, 28, 28, 1])

input_cls = tf.placeholder(tf.float32, [None, 1])

conv1_w = tf.Variable(tf.truncated_normal([28, 28, 1, 3]))
conv1_b = tf.zeros([1])
conv1 = tf.nn.conv2d(x_reshape, conv1_w, padding='SAME', strides=[1, 1, 1, 1]) + conv1_b

conv2 = tf.layers.conv2d(inputs=conv1, padding='same', filters=128, kernel_size=5, strides=2, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=.001))

conv3 = tf.layers.conv2d(inputs=conv2, padding='same', filters=256, kernel_size=5, strides=2, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=.001))

conv4 = tf.layers.conv2d(inputs=conv3, padding='same', filters=512, kernel_size=5, strides=2, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=.001))

flatten = tf.layers.flatten(conv4)

pred = tf.layers.dense(inputs=flatten, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=.001))
pred = tf.nn.sigmoid(pred)

# Loss, cost, optimizing
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_cls, logits=pred)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy
correct_prediction = tf.equal(input_cls, pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def create_batch(mnist, batch_size):
    '''
    Function for creating a single training batch for the discriminator, mixed of real and fake images
    Class is 1 if it is real and 0 if it is fake
    :param mnist:
    :param mnist_cls:
    :param batch_size:
    :return:
    '''
    generated = generate_img(batch_size)

    image_batch = []
    class_batch = []

    even_index = 0
    odd_index = 0
    for i in range(batch_size * 2):
        if i % 2 == 0:
            image_batch.append(mnist[even_index])
            class_batch.append([1])
            even_index += 1
        else:
            image_batch.append(generated[odd_index])
            class_batch.append([0])
            odd_index += 1

    # Shuffle
    numpy_state = np.random.get_state()
    np.random.shuffle(image_batch)
    np.random.set_state(numpy_state)
    np.random.shuffle(class_batch)

    return image_batch, class_batch


def train_discrim(num_iter):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    train_batch_size = 64

    for i in range(num_iter):
        x_batch, y_true = data.train.next_batch(train_batch_size)
        image_batch, class_batch = create_batch(x_batch, train_batch_size)
        feed_dict_train = {input_x: image_batch,
                           input_cls: class_batch}

        sess.run(optimizer, feed_dict=feed_dict_train)

        # accuracy
        print("Result: ", sess.run(pred, feed_dict=feed_dict_train))
        print("Acc on ", i, ": ", sess.run(accuracy, feed_dict=feed_dict_train))

    sess.close()


def use_discrim(num_iter=1):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    batch_size = 64
    for i in range(num_iter):
        x_batch, y_true = data.train.next_batch(batch_size)
        image_batch, class_batch = create_batch(x_batch, batch_size)
        feed_dict_train = {input_x: image_batch,
                           input_cls: class_batch}

        cross_entropy = sess.run(loss, feed_dict=feed_dict_train)

    sess.close()
    return cross_entropy


use_discrim(1)
