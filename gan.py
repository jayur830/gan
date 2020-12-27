import numpy as np
import tensorflow as tf
from tqdm import tqdm


class GAN:
    """
    Constructor
    :param kernel_initializer - Initializer for weights and biases
    :param input_dim - Dimension of the input
    """
    def __init__(self,
                 kernel_initializer: str = "he_normal",
                 input_shape: tuple = None):
        # Input dimensions
        self.__input_shape = input_shape

        # Generator
        self.__generator = self.get_generator(kernel_initializer)

        # Discriminator
        self.__discriminator = self.get_discriminator(kernel_initializer)
        # Discriminator is not trainable yet.
        self.__discriminator.trainable = False

        # Create Generative Adversarial Networks
        gan_input = tf.keras.layers.Input(shape=self.__input_shape)
        gan_output = self.__discriminator(self.__generator(gan_input))
        self.__gan = tf.keras.models.Model(gan_input, gan_output)

    def get_generator(self, kernel_initializer: str) -> tf.keras.models.Model:
        pass

    def get_discriminator(self, kernel_initializer: str) -> tf.keras.models.Model:
        pass

    def compile(self,
                discriminator_optimizer: tf.keras.optimizers.Optimizer,
                discriminator_loss: tf.keras.losses.Loss,
                gan_optimizer: tf.keras.optimizers.Optimizer,
                gan_loss: tf.keras.losses.Loss):
        self.__discriminator.compile(discriminator_optimizer, discriminator_loss)
        self.__gan.compile(gan_optimizer, gan_loss)

    """
    :param x - Training dataset X
    :epochs - Training epochs
    :batch_size - Size of the batch in 1 iteration
    """
    def fit(self, x, epochs: int = 1, batch_size: int = 32, callback: list = None, step_callback: int = 10):
        for i in range(epochs):
            print(f"{'-' * 10} {i + 1}/{epochs} Epochs {'-' * 10}")
            for j in tqdm(range(x.shape[0] // batch_size)):
                noise = np.random.uniform(low=-1., size=(batch_size, self.__input_shape[0]))

                y = np.zeros(2 * batch_size)
                y[:batch_size] = 1.

                d_input = np.concatenate([
                    x[np.random.randint(0, x.shape[0], size=batch_size)],
                    self.__generator.predict(noise)
                ])

                self.__discriminator.trainable = True
                print(f"discriminator_loss: {self.__discriminator.train_on_batch(d_input, y, return_dict=True)['loss']}", end="\t")

                self.__discriminator.trainable = False
                print(f"gan_loss: {self.__gan.train_on_batch(noise, np.ones(batch_size), return_dict=True)['loss']}")

                if j % step_callback == 0:
                    for f in callback:
                        f(self.__generator)
