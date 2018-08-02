import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

input_x = tf.placeholder(tf.float32, [None, 784])
x_reshape = tf.reshape(input_x, [-1, 28, 28, 1])

input_cls = tf.placeholder(tf.float32, [None, 1])

conv1_w = tf.Variable(tf.truncated_normal([28, 28, 1, 3]))
conv1_b = tf.truncated_normal([3])
conv1 = tf.nn.conv2d(x_reshape, conv1_w, padding='SAME', strides=[1, 1, 1, 1]) + conv1_b

conv2 = tf.layers.conv2d(inputs=conv1, padding='same', filters=16, kernel_size=5, strides=2, activation=tf.nn.relu)

conv3 = tf.layers.conv2d(inputs=conv2, padding='same', filters=32, kernel_size=5, strides=2, activation=tf.nn.relu)

conv4 = tf.layers.conv2d(inputs=conv3, padding='same', filters=64, kernel_size=5, strides=2, activation=tf.nn.relu)

flatten = tf.layers.flatten(conv4)

pred = tf.layers.dense(inputs=flatten, units=1)
pred = tf.nn.sigmoid(pred)

# Loss, cost, optimizing
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_cls, logits=pred)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

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


def use_discrim(image_batch):
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    cross_entropy = sess.run(loss, feed_dict={input_x: image_batch})
    return cross_entropy

