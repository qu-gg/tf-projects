import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy import misc

x_size = 1920
y_size = 1080

coords = tf.placeholder(tf.float32, (None, 4), name='features')


def iterative_weights(num_neurons, initial, layers):
    weight_list = []
    initial_weight = tf.Variable(tf.truncated_normal([initial, num_neurons], stddev=.55))
    weight_list.append(initial_weight)

    for _ in range(layers - 2):
        new_weight = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=.220))
        weight_list.append(new_weight)

    final_weight = tf.Variable(tf.random_normal([num_neurons, 3]))
    weight_list.append(final_weight)
    return weight_list


def iterative_layers(weight_list, prev_output, layers):
    output = prev_output
    for step in range(layers):
        output = tf.nn.tanh(tf.matmul(output, weight_list[step]))
    return tf.nn.relu6(output)


num_layers = 6
weights = iterative_weights(3500, 4, num_layers)
pred = iterative_layers(weights, coords, num_layers)


def create_array():
    parameters = [[[] for _ in range(x_size)] for _ in range(y_size)]
    for y in range(y_size):
        for x in range(x_size):
            added_x = abs(x - x_size / 2)
            added_y = abs(y - y_size / 2)

            radius = np.sqrt(added_x ** 2 + added_y ** 2)
            radius = np.cos(10 * radius)
            parameters[y][x] = [x, y, radius, rand]
    return parameters


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    rand = 2000

    features = create_array()

    rgb = [0 for _ in range(y_size)]
    for batch in range(y_size):
        feed = features[batch]
        result = sess.run(pred, feed_dict={coords: feed})
        rgb[batch] = result

    misc.imsave("images/" + "testing.png", np.asarray(rgb))
