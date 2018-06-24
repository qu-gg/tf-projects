import tensorflow as tf
import numpy as np
from scipy import misc
img_size = 1000

coords = tf.placeholder(tf.float32, (None, 1, 4))

k, l, m, n = 200, 50, 30, 15
B = np.cos(50)
b_w = 3

w1 = tf.Variable(tf.random_normal([img_size, 4, k], stddev=(B*(1/4))**2))
b1 = tf.Variable(tf.zeros([k]))
w2 = tf.Variable(tf.random_normal([img_size, k, l], stddev=(B*(1/k))**2))
b2 = tf.Variable(tf.zeros([l]))
w3 = tf.Variable(tf.random_normal([img_size, l, m], stddev=(B*(1/l))**2))
b3 = tf.Variable(tf.zeros([m]))
w4 = tf.Variable(tf.random_normal([img_size, m, n], stddev=(B*(1/m))**2))
b4 = tf.Variable(tf.zeros([n]))
w5 = tf.Variable(tf.random_normal([img_size, n, b_w], stddev=(B*(1/n))**2))
b5 = tf.Variable(tf.zeros([b_w]))

y1 = tf.nn.tanh(tf.matmul(coords, w1) + b1)
y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3)
y4 = tf.nn.tanh(tf.matmul(y3, w4) + b4)
pred = tf.nn.tanh(tf.matmul(y4, w5) + b5)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    rand = 50
    features = [[[] for _ in range(img_size)] for _ in range(img_size)]

    parameters = [[[] for _ in range(img_size)] for _ in range(img_size)]
    for x in range(img_size):
        for y in range(img_size):
            radius = np.sqrt(x**2 + y**2)
            parameters[x][y] = [[x, y, radius, rand]]

    rgb = [0 for _ in range(img_size)]
    for batch in range(img_size):
        feed = parameters[batch]
        result = sess.run(pred, feed_dict={coords: feed})
        rgb[batch] = result

    print(rgb)
    final = np.zeros((500, 500, 3))

    filename = input("Enter filename to save: ")
    misc.imsave("images/" + filename, final)
