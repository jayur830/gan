import tensorflow as tf

from gan import GAN


class Cifar10GAN(GAN):
    def __init__(self, input_shape):
        super(Cifar10GAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            # (100,) -> (2 * 2 * 128,)
            tf.keras.layers.Dense(
                units=2 * 2 * 128,
                kernel_initializer = kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (2 * 2 * 128,) -> (2, 2, 128)
            tf.keras.layers.Reshape(target_shape=(2, 2, 128)),
            tf.keras.layers.Dropout(rate=.4),
            # (2, 2, 128) -> (2, 2, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (2, 2, 64) -> (4, 4, 64)
            tf.keras.layers.UpSampling2D(),
            # (4, 4, 64) -> (4, 4, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=4,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (4, 4, 32) -> (8, 8, 32)
            tf.keras.layers.UpSampling2D(),
            # (8, 8, 32) -> (8, 8, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (8, 8, 16) -> (16, 16, 16)
            tf.keras.layers.UpSampling2D(),
            # (16, 16, 16) -> (16, 16, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=4,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (16, 16, 8) -> (32, 32, 8)
            tf.keras.layers.UpSampling2D(),
            # (32, 32, 8) -> (32, 32, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            # (32, 32, 3) -> (16, 16, 8)
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (16, 16, 8) -> (8, 8, 16)
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (8, 8, 16) -> (4, 4, 16)
            tf.keras.layers.MaxPool2D(),
            # (4, 4, 16) -> (2, 2, 32)
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # Flatten
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.4),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
