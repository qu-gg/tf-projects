import tensorflow as tf
import numpy as np
from scipy import misc
img_size = 500
coords = tf.placeholder(tf.float32, (None, 4), name='features')
tf.summary.histogram("features", coords)

k, l, m = 2500, 1000, 600
B = np.cos(100)

w1 = tf.Variable(tf.truncated_normal([4, k], stddev=np.random.uniform(50, 150), mean=0), name="Weights1")
w2 = tf.Variable(tf.truncated_normal([k, l], stddev=B/k, mean=0), name="Weights2")
w3 = tf.Variable(tf.truncated_normal([l, 3], stddev=B/l, mean=0), name="Weights3")

y1 = tf.nn.tanh(tf.matmul(coords, w1), name="HLayer1")
y2 = tf.nn.tanh(tf.matmul(y1, w2), name="HLayer2")
pred = tf.nn.tanh(tf.matmul(y2, w3), name="OutputLayer")

'''Summary'''

tf.summary.histogram("predictions", pred)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("graphs/", sess.graph)
    merge = tf.summary.merge_all()
    rand = 250

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
