import tensorflow as tf
import numpy as np
from scipy import misc

img_size = 256
coords = tf.placeholder(tf.float32, (1, 4), name='features')
tf.summary.histogram("features", coords)

k, l, m, n, p = 5000, 2500, 1000, 2500, 1500
B = np.cos(100)

w1 = tf.Variable(tf.truncated_normal((4, k), stddev=np.random.uniform(50, 150), mean=0), name="Weights1")
w2 = tf.Variable(tf.truncated_normal([k, l], stddev=B/k, mean=0), name="Weights2")
w3 = tf.Variable(tf.truncated_normal([l, m], stddev=B*3/l, mean=0), name="Weights3")
w4 = tf.Variable(tf.truncated_normal([m, n], stddev=B*2/m, mean=0), name="Weights3")
w5 = tf.Variable(tf.truncated_normal([n, 3], stddev=B/n, mean=0), name="Weights3")

y1 = tf.nn.tanh(tf.matmul(coords, w1), name="HLayer1")
y2 = tf.nn.tanh(tf.matmul(y1, w2), name="HLayer2")
y3 = tf.nn.tanh(tf.matmul(y2, w3), name="OutputLayer")
y4 = tf.nn.tanh(tf.matmul(y3, w4), name="OutputLayer")
pred = tf.nn.tanh(tf.matmul(y4, w5), name="OutputLayer")

'''Summary'''
tf.summary.histogram("predictions", pred)


def create_array():
    value_to_return = [[[] for _ in range(img_size)] for _ in range(img_size)]
    for x in range(img_size):
        for y in range(img_size):
            radius = np.sqrt(x ** 2 + y ** 2)
            value_to_return[x][y] = [x, y, radius, rand]
    return value_to_return


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("graphs/", sess.graph)
    merge = tf.summary.merge_all()
    rand = 250

    parameters = create_array()

    new_rgb = [[[] for _ in range(img_size)] for _ in range(img_size)]
    for x in range(img_size):
        print(x)
        for y in range(img_size):
            radius = np.sqrt(x**2 + y**2)
            feed = [[x, y, radius, rand]]
            result = sess.run(pred, feed_dict={coords: feed})
            new_rgb[x][y] = result
    final = np.reshape(new_rgb, [img_size, img_size, 3])

    misc.toimage(final).show()
    writer.close()


