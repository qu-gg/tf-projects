import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy import misc

x_size = 250
y_size = 250

coords = tf.placeholder(tf.float32, (None, 4), name='features')


def iterative_weights(num_neurons, initial, layers):
    weight_list = []
    initial_weight = tf.Variable(tf.truncated_normal([initial, num_neurons], stddev=.5500))
    weight_list.append(initial_weight)

    counter = 1
    while counter < layers:
        new_weight = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=.2500))
        weight_list.append(new_weight)
        counter += 1

    final_weight = tf.Variable(tf.truncated_normal([num_neurons, 3], stddev=.0145))
    weight_list.append(final_weight)
    return weight_list


def iterative_layers(weight_list, prev_output, layers):
    calculation = 0
    test = prev_output
    for step in range(layers):
        calculation = tf.nn.tanh(tf.matmul(test, weight_list[step]))
        test = calculation
    return calculation


weights = iterative_weights(1500, 4, 3)
pred = iterative_layers(weights, coords, 3)


def create_array():
    parameters = [[[] for _ in range(x_size)] for _ in range(y_size)]
    for y in range(y_size):
        for x in range(x_size):
            added_x = x + 50
            added_y = y + 50
            radius = np.sqrt(added_x ** 2 + added_y ** 2)
            parameters[y][x] = [added_y, added_x, radius, rand]
    return parameters


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    rand = 1500

    features = create_array()

    rgb = [0 for _ in range(y_size)]
    for batch in range(y_size):
        feed = features[batch]
        result = sess.run(pred, feed_dict={coords: feed})
        rgb[batch] = result

    misc.imsave("images/" + "testing.png", rgb)