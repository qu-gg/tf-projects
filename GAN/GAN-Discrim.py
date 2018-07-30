import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from GAN.Generator import generate_img
data = input_data.read_data_sets('data/MNIST', one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 784])
x_reshape = tf.reshape(input_x, [-1, 28, 28, 1])

input_cls = tf.placeholder(tf.float32, [None, 1])

# First conv layer, full size with strides 2 to pool size
first_layer = tf.layers.conv2d(inputs=x_reshape, padding='same', filters=16,
                               kernel_size=5, strides=2, activation=tf.nn.relu)

# Second conv layer, half size
second_layer = tf.layers.conv2d(inputs=first_layer, padding='same', filters=32,
                                kernel_size=5, strides=2, activation=tf.nn.relu)

# Flattened second layer for connecting to 2d layer
flatten_second = tf.layers.flatten(second_layer)

# Fully connected layer
pred = tf.layers.dense(inputs=flatten_second, units=1, activation=tf.nn.softmax)

# Loss, cost, optimizing
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_cls, logits=pred)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# Accuracy
correct_prediction = tf.equal(input_cls, pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train function
def train_discrim(num_iter):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())

        train_batch_size = 64

        for i in range(num_iter):
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            feed_dict_train = {input_x: x_batch,
                               input_cls: 0}

            sess.run(optimizer, feed_dict=feed_dict_train)

    sess.close()

