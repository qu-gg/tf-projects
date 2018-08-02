import GAN.Discriminator as Dis
import GAN.Generator as Gen
from scipy import misc
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)


def create_batch(batch_size):
    """
    Function for creating a single training batch for the discriminator, mixed of real and fake images
    Class is 1 if it is real and 0 if it is fake
    :param batch_size:
    :return:
    """
    fakes = Gen.generate_img(batch_size / 2)
    reals, _ = data.train.next_batch(batch_size / 2)

    image_batch = []
    class_batch = []

    even_index = 0
    odd_index = 0
    for i in range(batch_size):
        if i % 2 == 0:
            image_batch.append(reals[even_index])
            class_batch.append(1)
            even_index += 1
        else:
            image_batch.append(fakes[odd_index])
            class_batch.append(0)
            odd_index += 1

    # Shuffle
    numpy_state = np.random.get_state()
    np.random.shuffle(image_batch)
    np.random.set_state(numpy_state)
    np.random.shuffle(class_batch)

    return image_batch, class_batch


def generator_train():
    image = GAN.Generator.generate_img()
    misc.toimage(image).show()

    images = GAN.Generator.generate_img(50)
    entropy = GAN.Discriminator.use_discrim(images)
    GAN.Generator.train_gen(10, entropy)

    image = GAN.Generator.generate_img()
    misc.toimage(image).show()


generator_train()
