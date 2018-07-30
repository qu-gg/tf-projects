import tensorflow as tf
import numpy as np
from scipy import misc
from GAN.Discriminator import use_discrim

x_input = tf.placeholder(tf.float32, [None, 49])
img_input = tf.cast(tf.reshape(x_input, [-1, 7, 7, 1]), tf.float32)

conv1_w = tf.Variable(tf.truncated_normal([7, 7, 1, 32]))
conv1_b = tf.zeros([32])
conv1 = tf.nn.conv2d(img_input, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

conv_one = tf.layers.conv2d_transpose(inputs=conv1, filters=1024, kernel_size=5, padding='same',
                                      activation=tf.nn.relu)

conv_two = tf.layers.conv2d_transpose(inputs=conv_one, filters=512, kernel_size=5, padding='same',
                                      activation=tf.nn.relu)

conv_three = tf.layers.conv2d_transpose(inputs=conv_two, filters=256, kernel_size=5, padding='same',
                                        activation=tf.nn.relu)

conv_four = tf.layers.conv2d_transpose(inputs=conv_three, filters=128, kernel_size=5, padding='same', strides=2,
                                       activation=tf.nn.relu)

final_layer = tf.layers.conv2d_transpose(inputs=conv_four, filters=1, kernel_size=5, padding='same', strides=2,
                                         activation=tf.nn.relu)

output = tf.layers.flatten(final_layer)
reshaped = tf.reshape(output, [28,28])

loss = tf.placeholder(tf.float32, [None, 128])
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=.00001).minimize(cost)


def train_gen():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # add loss calculations here
        for _ in range(5):
            entropy = use_discrim()
            sess.run(optimizer, feed_dict={loss: entropy})

        image = sess.run(output, feed_dict={x_input: []})
        misc.toimage(image).show()


def generate_img(num_runs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fakes = []
        for _ in range(num_runs):
            fake = sess.run(output, feed_dict={x_input: [np.random.uniform(0, 1, 49)]})
            fakes.append(fake[0])

    sess.close()
    return fakes


if __name__ == "__main__":
    def main():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            fake = sess.run(reshaped, feed_dict={x_input: [np.random.uniform(0, 1, 49)]})
            misc.toimage(fake).show()


        sess.close()

    main()