import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy import misc

img_size = 1400
coords = tf.placeholder(tf.float32, (None, 4), name='features')

k, l, m = 5000, 5000, 5000
B = 250

w1 = tf.Variable(tf.random_normal([4, k], stddev=.5913, mean=0), name="Weights1")
w2 = tf.Variable(tf.random_normal([k, l], stddev=.35499, mean=0), name="Weights2")
w3 = tf.Variable(tf.random_normal([l, 3], stddev=.2184, mean=0), name="Weights3")

y1 = tf.nn.tanh(tf.matmul(coords, w1), name="HLayer1")
y2 = tf.nn.tanh(tf.matmul(y1, w2), name="HLayer2")
pred = tf.nn.tanh(tf.matmul(y2, w3), name="OutputLayer")


def create_array():
    parameters = [[[] for _ in range(img_size)] for _ in range(img_size)]
    neg_x = -(img_size / 2)
    neg_y = -(img_size / 2)
    for x in range(img_size):
        for y in range(img_size):
            radius = np.sqrt(neg_x ** 2 + neg_y ** 2)
            parameters[x][y] = [x, y, radius, rand]
            neg_y += 1
        neg_x += 1
        neg_y = 0
    return parameters


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    rand = 750

    features = create_array()

    rgb = [0 for _ in range(img_size)]
    for batch in range(img_size):
        feed = features[batch]
        result = sess.run(pred, feed_dict={coords: feed})
        rgb[batch] = result

msave("images/" + "testing.png", rgb)

