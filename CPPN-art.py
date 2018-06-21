import tensorflow as tf
import numpy as np
from scipy import misc


coords = tf.placeholder(tf.float32, (500, 500, 1, 1))

k, l, m, n = 200, 100, 60, 30
w1 = tf.Variable(tf.truncated_normal([1, k], stddev=0.1))
b1 = tf.Variable(tf.zeros([k]))
w2 = tf.Variable(tf.truncated_normal([k, l], stddev=0.1))
b2 = tf.Variable(tf.zeros([l]))
w3 = tf.Variable(tf.truncated_normal([l, m], stddev=0.1))
b3 = tf.Variable(tf.zeros([m]))
w4 = tf.Variable(tf.truncated_normal([m, n], stddev=0.1))
b4 = tf.Variable(tf.zeros([n]))
w5 = tf.Variable(tf.truncated_normal([n, 3], stddev=0.1))
b5 = tf.Variable(tf.zeros([3]))

y1 = tf.nn.relu(tf.matmul(coords, w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
pred = tf.nn.softmax(tf.matmul(y4, w5) + b5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y = [], []
    for n in range(500):
        x[n] = n
        y[n] = n
    values = np.array(len(x) * len(y) * 3)

    for m in range(500):
        features = [x[m], y[m], 1, 50]
        rgb = sess.run(pred, feed_dict={coords: features})


    filename = input("Enter filename to save: ")
    misc.imsave("images/" + filename, values)