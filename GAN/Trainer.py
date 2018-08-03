import GAN.Discriminator as Dis
import GAN.Generator as Gen
from scipy import misc
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)


def create_batch(batch_size):
    """
    Function for creating a single training batch for the discriminator, mixed of real and fake images
    Class is 0 if it is real and 1 if it is fake
    :param batch_size:
    :return:
    """
    num = random.randint(20, 40)
    fakes = Gen.generate_img(num)
    reals, _ = data.train.next_batch(batch_size - num)

    image_batch = []
    class_batch = []

    for i in range(batch_size - num):
        image_batch.append(reals[i])
        class_batch.append([random.uniform(0.0, 0.1)])
    for i in range(num):
        image_batch.append(fakes[i])
        class_batch.append([random.uniform(0.9, 1.0)])

    # Shuffle
    numpy_state = np.random.get_state()
    np.random.shuffle(image_batch)
    np.random.set_state(numpy_state)
    np.random.shuffle(class_batch)

    return image_batch, class_batch


def generator_train():
    image = Gen.generate_img()
    misc.toimage(image).show()

    test_i = Gen.generate_img(50)
    entropy = Dis.use_discrim(test_i, None)
    Gen.train_gen(10, entropy)

    image = Gen.generate_img()
    misc.toimage(image).show()


images, classes = [], []
for i in range(10):
    i_batch, c_batch = create_batch(68)
    images.append(i_batch)
    classes.append(c_batch)

Dis.train_discrim(10, images, classes)

images = Gen.generate_img(32)
classes = [[0] for i in range(32)]

#loss = Dis.use_discrim(images, classes)


#Gen.train_gen(loss)
Gen.get_img()