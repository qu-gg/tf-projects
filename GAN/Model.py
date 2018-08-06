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


def generator(noise, sess=tf.Session(), shaped=False):
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
        if shaped:
            return tf.reshape(output, [28,28])

    result = sess.run(output, feed_dict={x_input: noise})
    return result


def generate_img(num_runs=1, sess=tf.Session()):
    fakes = [np.random.uniform(-1, 1, 49) for _ in range(num_runs)]
    fake = generator(fakes, sess)

    return fake


def get_img(sess=tf.Session()):
    fake = generator([np.random.uniform(-1, 1, 49) for _ in range(1)], sess, True)
    print(fake)
    misc.toimage(fake).show()

    return fake


"""
Discriminator Network
"""


def discriminator(inputs, classes, sess=tf.Session()):
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

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=input_cls)

    output = sess.run(loss, feed_dict={input_x: inputs, input_cls: classes})
    return output


def use_discrim(image_batch, class_batch):
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    prediction = discriminator(image_batch, class_batch)
    sess.close()
    return prediction


"""
Initializing uninitialized variables
"""


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def main():
    batch_size = 68
    g_sample = generator(np.random.uniform(-1, 1, 49))

    # Scalar of Real Images
    images, _ = data.train.next_batch(batch_size)
    classes = [[0] for _ in range(batch_size)]
    dis_real = use_discrim(images, classes)

    # Scalar of Fake Images
    g_classes = [[1] for _ in range(batch_size)]
    dis_fake = use_discrim(g_sample, g_classes)

    dis_loss = -tf.reduce_mean(np.log(dis_real) + np.log(1 - dis_fake))
    gen_loss = -tf.reduce_mean(np.log(dis_fake))

    dis_solver = tf.train.GradientDescentOptimizer(learning_rate=.00085).minimize(
        dis_loss, var_list=[item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discrim')]
    )

    gen_solver = tf.train.AdamOptimizer(learning_rate=.00085).minimize(
        gen_loss, var_list=[item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='genera')]
    )

    get_img()


def session():
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 68

        gen_sample = generator([np.random.uniform(-1, 1, 49) for _ in range(batch_size)], sess)

        # Probability for real images
        images, _ = data.train.next_batch(batch_size)
        classes = [[0] for _ in range(batch_size)]
        dis_real = discriminator(images, classes, sess)

        # Probability for fake images
        gen_classes = [[1] for _ in range(batch_size)]
        dis_fake = discriminator(gen_sample, gen_classes, sess)

        # Loss
        dis_loss = -tf.reduce_mean(np.log(dis_real) + np.log(1 - dis_fake))
        gen_loss = -tf.reduce_mean(np.log(dis_fake))

        # Training
        dis_solver = tf.train.GradientDescentOptimizer(learning_rate=.00085).minimize(
            dis_loss, var_list=[item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discrim')]
        )

        gen_solver = tf.train.AdamOptimizer(learning_rate=.00085).minimize(
            gen_loss, var_list=[item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='genera')]
        )

        get_img(sess)
    sess.close()



if __name__ == '__main__':
    main()