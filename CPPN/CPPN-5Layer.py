import tensorflow as tf
import numpy as np
from scipy import misc
img_size = 500

coords = tf.placeholder(tf.float32, (None, 4), name='features')
tf.summary.histogram("features", coords)

k, l, m, n = 200, 50, 30, 15
B = np.cos(100)

w1 = tf.Variable(tf.random_normal([4, k], stddev=-51.54981, name="Weights1"))
b1 = tf.Variable(tf.zeros([k]), name="Bias1")
w2 = tf.Variable(tf.random_normal([k, l], stddev=.895163, name="Weights2"))
b2 = tf.Variable(tf.zeros([l]), name="Bias2")
w3 = tf.Variable(tf.random_normal([l, m], stddev=0.195461, name="Weights3"))
b3 = tf.Variable(tf.zeros([m]), name="Bias3")
w4 = tf.Variable(tf.random_normal([m, n], stddev=0.98413), name="Weights4")
b4 = tf.Variable(tf.zeros([n]), name="Bias4")
w5 = tf.Variable(tf.random_normal([n, 3], stddev=0.00156), name="Weights5")
b5 = tf.Variable(tf.zeros([3]), name="Bias5")

y1 = tf.nn.tanh(tf.matmul(coords, w1) + b1, name="HLayer1")
y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2, name="HLayer2")
y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3, name="HLayer3")
y4 = tf.nn.tanh(tf.matmul(y3, w4) + b4, name="HLayer4")
pred = tf.nn.tanh(tf.matmul(y4, w5) + b5, name="OutputLayer")

'''Summary'''

tf.summary.histogram("predictions", pred)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("graphs/", sess.graph)
    merge = tf.summary.merge_all()
    rand = 235.9845613

    parameters = [[[] for _ in range(img_size)] for _ in range(img_size)]
    for x in range(img_size):
        for y in range(img_size):
            radius = np.sqrt(x**2 + y**2)
            parameters[x][y] = [x, y, radius, rand]

    rgb = [0 for _ in range(img_size)]
    for batch in range(img_size):
        feed = parameters[batch]
        summary, result = sess.run([merge, pred], feed_dict={coords: feed})
        writer.add_summary(summary)
        rgb[batch] = result

    misc.toimage(rgb).show()
    writer.close()
