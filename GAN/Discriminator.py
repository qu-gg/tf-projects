import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

input_x = tf.placeholder(tf.float32, [None, 784])
x_reshape = tf.reshape(input_x, [-1, 28, 28, 1])

input_cls = tf.placeholder(tf.float32, [None, 1])

def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1))


def bias_var(shape):
    return tf.Variable(tf.zeros([shape]))


def conv_layer(img, weight, bias, strides):
    layer = tf.nn.conv2d(img, weight, strides=strides, padding='SAME') + bias
    return layer


conv1 = conv_layer(x_reshape, weight_var([28, 28, 1, 3]), bias_var(3), [1, 1, 1, 1])

conv2 = conv_layer(conv1, weight_var([14, 14, 3, 128]), bias_var(128), [1, 2, 2, 1])

conv3 = conv_layer(conv2, weight_var([7, 7, 128, 256]), bias_var(256), [1, 2, 2, 1])

conv4 = conv_layer(conv3, weight_var([7, 7, 256, 512]), bias_var(512), [1, 2, 2, 1])

flatten = tf.layers.flatten(conv4)

pred = tf.layers.dense(inputs=flatten, units=1)
pred = tf.nn.sigmoid(pred)

# Loss, cost, optimizing
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_cls, logits=pred)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy
# correct_prediction = tf.equal(input_cls, pred)
correct_prediction = tf.equal(tf.greater(pred, [0.5]),tf.cast(input_cls,'bool'))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train_discrim(num_iter, images, classes):
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(num_iter):
        feed_dict_train = {input_x: images[i],
                           input_cls: classes[i]}

        sess.run(optimizer, feed_dict=feed_dict_train)

        # accuracy
        print("Result: ", sess.run(pred, feed_dict=feed_dict_train))
        print("Acc on ", i, ": ", sess.run(accuracy, feed_dict=feed_dict_train))

    sess.close()


def use_discrim(image_batch, class_batch):
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    cross_entropy = sess.run(loss, feed_dict={input_x: image_batch,
                                              input_cls: class_batch})
    return cross_entropy

