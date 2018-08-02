import GAN.Discriminator
import GAN.Generator
from scipy import misc
import tensorflow as tf


def generator_train():
    image = GAN.Generator.generate_img()
    misc.toimage(image).show()

    images = GAN.Generator.generate_img(50)
    entropy = GAN.Discriminator.use_discrim(images)
    GAN.Generator.train_gen(10, entropy)

    image = GAN.Generator.generate_img()
    misc.toimage(image).show()


generator_train()
