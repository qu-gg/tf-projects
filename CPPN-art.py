import tensorflow as tf
import numpy as np
from scipy import misc


coords = tf.placeholder(tf.float32, (None, 4))

k, l, m, n = 100, 50, 30, 15

w1 = tf.Variable(tf.truncated_normal([4, k], stddev=0.1))
b1 = tf.Variable(tf.zeros([k]))
w2 = tf.Variable(tf.truncated_normal([k, l], stddev=0.1))
b2 = tf.Variable(tf.zeros([l]))
w3 = tf.Variable(tf.truncated_normal([l, m], stddev=0.1))
b3 = tf.Variable(tf.zeros([m]))
w4 = tf.Variable(tf.truncated_normal([m, n], stddev=0.1))
b4 = tf.Variable(tf.zeros([n]))
w5 = tf.Variable(tf.truncated_normal([n, 3], stddev=0.1))
b5 = tf.Variable(tf.zeros([3]))

y1 = tf.nn.tanh(tf.matmul(coords, w1) + b1)
y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3)
y4 = tf.nn.tanh(tf.matmul(y3, w4) + b4)
pred = tf.nn.softmax(tf.matmul(y4, w5) + b5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rand = 50

    features = [[[] for _ in range(500)] for _ in range(500)]

    for x_axis in range(500):
        for y_axis in range(500):
            radius = np.sqrt(x_axis**2 + y_axis**2)
            features[x_axis][y_axis] = [x_axis, y_axis, radius, rand]

    reshaped = np.reshape(features, (-1, 4))
    rgb = sess.run(pred, feed_dict={coords: reshaped})
    reshaped_rgb = np.reshape(rgb, (500, 500,3))

    filename = input("Enter filename to save: ")
    misc.imsave("images/" + filename, reshaped_rgb)