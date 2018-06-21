import tensorflow as tf
import numpy as np
from scipy import misc


coords = tf.placeholder(tf.float32, (None, 4))

k, l, m, n = 200, 50, 30, 15
B = 150

w1 = tf.Variable(tf.random_normal([4, k], stddev=(B*(1/4))**2))
b1 = tf.Variable(tf.zeros([k]))
w2 = tf.Variable(tf.random_normal([k, l], stddev=(B*(1/k))**2))
b2 = tf.Variable(tf.zeros([l]))
w3 = tf.Variable(tf.random_normal([l, m], stddev=(B*(1/l))**2))
b3 = tf.Variable(tf.zeros([m]))
w4 = tf.Variable(tf.random_normal([m, n], stddev=(B*(1/m))**2))
b4 = tf.Variable(tf.zeros([n]))
w5 = tf.Variable(tf.random_normal([n, 3], stddev=(B*(1/n))**2))
b5 = tf.Variable(tf.zeros([3]))

y1 = tf.nn.tanh(tf.matmul(coords, w1) + b1)
y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3)
y4 = tf.nn.tanh(tf.matmul(y3, w4) + b4)
pred = tf.nn.tanh(tf.matmul(y4, w5) + b5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rand = 50
    img_size = 1000
    features = [[[] for _ in range(img_size)] for _ in range(img_size)]

    for x_axis in range(img_size):
        for y_axis in range(img_size):
            radius = np.sqrt(x_axis**2 + y_axis**2)
            features[x_axis][y_axis] = [x_axis, y_axis, radius, rand]

    reshaped = np.reshape(features, (-1, 4))
    rgb = sess.run(pred, feed_dict={coords: reshaped})
    reshaped_rgb = np.reshape(rgb, (img_size, img_size,3))

    filename = input("Enter filename to save: ")
    misc.imsave("images/" + filename, reshaped_rgb)