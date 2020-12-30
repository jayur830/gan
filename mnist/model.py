import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def __init__(self, input_shape):
        super(MnistGAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=1024),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=784),
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=1024,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                input_shape=(784,)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=256),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=1),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
